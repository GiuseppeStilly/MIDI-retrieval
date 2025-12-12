import torch
import numpy as np
import os
import tarfile
import zipfile
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer
from midi2audio import FluidSynth
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset

# Importiamo le classi dal tuo file esistente
from ensemble_system import NeuralMidiSearch_V1, NeuralMidiSearch_V2

# --- CONFIGURAZIONE ---
HF_REPO_ID = "GiuseppeStilly/MIDI-Retrieval"
MODEL_V1_FILENAME = "v1_lstm_optimized.pt"
MODEL_V2_FILENAME = "v2_transformer_mpnet.pt"
DATASET_FILENAME = "test_midicaps_tokenized.pt"

SOUNDFONT_URL = "https://schristiancollins.com/generaluser-gs/GeneralUser_GS_1.471.zip"
SOUNDFONT_PATH = "GeneralUser_GS_1.471.sf2"
MIDI_DATA_DIR = "midicaps_data" # Cartella dove scaricare i midi raw per l'audio

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class InferenceDataset(Dataset):
    """Dataset semplice per caricare i token per l'indicizzazione"""
    def __init__(self, data_path):
        self.data = torch.load(data_path, map_location="cpu")
    
    def __len__(self): return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        # Gestione compatibilità nomi chiavi
        m = item.get("m", item.get("midi_ids", [0]))
        if not isinstance(m, torch.Tensor): m = torch.tensor(m, dtype=torch.long)
        return {"midi_ids": m, "path": item.get("path", "unknown")}

class EnsembleSearchEngine:
    def __init__(self):
        print("--- Initializing Ensemble Search Engine ---")
        self.model_v1 = None
        self.model_v2 = None
        self.tokenizer_v1 = None # MiniLM
        self.tokenizer_v2 = None # MPNet
        
        # Database in memoria
        self.db_matrix_v1 = None
        self.db_matrix_v2 = None
        self.db_paths = []
        
        self.setup_resources()
        self.load_models()
        self.build_index() # Pre-calcola gli embedding
        
    def setup_resources(self):
        """Scarica SoundFont, MIDI raw e installa FluidSynth se necessario"""
        # 1. FluidSynth check (Linux)
        if os.name == 'posix':
            if os.system("which fluidsynth > /dev/null 2>&1") != 0:
                print("Installing FluidSynth...")
                os.system("apt-get update -y && apt-get install -y fluidsynth")

        # 2. SoundFont
        if not os.path.exists(SOUNDFONT_PATH):
            print("Downloading SoundFont...")
            os.system(f"curl -L {SOUNDFONT_URL} -o soundfont.zip")
            with zipfile.ZipFile("soundfont.zip", 'r') as zip_ref:
                zip_ref.extractall(".")
            # Trova il file sf2 estratto
            for root, dirs, files in os.walk("."):
                for file in files:
                    if file.endswith(".sf2"):
                        global SOUNDFONT_PATH
                        SOUNDFONT_PATH = os.path.join(root, file)
                        break

        # 3. Raw MIDI Data (per il playback audio)
        if not os.path.exists(MIDI_DATA_DIR):
            print("Downloading Raw MIDI files...")
            try:
                path = hf_hub_download(repo_id="amaai-lab/MidiCaps", filename="midicaps.tar.gz", repo_type="dataset")
                with tarfile.open(path) as tar:
                    tar.extractall(MIDI_DATA_DIR)
            except Exception as e:
                print(f"Warning: Could not download raw MIDIs: {e}")

    def load_models(self):
        print("Loading Models from Hugging Face...")
        
        # --- LOAD V1 (LSTM) ---
        path_v1 = hf_hub_download(repo_id=HF_REPO_ID, filename=MODEL_V1_FILENAME)
        ckpt_v1 = torch.load(path_v1, map_location=DEVICE, weights_only=False)
        self.model_v1 = NeuralMidiSearch_V1(ckpt_v1.get('vocab_size', 3000)).to(DEVICE)
        self.model_v1.load_state_dict(ckpt_v1['model_state'], strict=False)
        self.model_v1.eval()
        self.tokenizer_v1 = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

        # --- LOAD V2 (Transformer) ---
        path_v2 = hf_hub_download(repo_id=HF_REPO_ID, filename=MODEL_V2_FILENAME)
        ckpt_v2 = torch.load(path_v2, map_location=DEVICE, weights_only=False)
        self.model_v2 = NeuralMidiSearch_V2(ckpt_v2.get('vocab_size', 3000)).to(DEVICE)
        self.model_v2.load_state_dict(ckpt_v2['model_state'], strict=False)
        self.model_v2.eval()
        self.tokenizer_v2 = AutoTokenizer.from_pretrained("sentence-transformers/all-mpnet-base-v2")
        
        print("Models Loaded.")

    def build_index(self):
        """Passa tutto il dataset attraverso i modelli per creare l'indice di ricerca."""
        print("Building Search Index (this happens only once at startup)...")
        
        data_path = hf_hub_download(repo_id=HF_REPO_ID, filename=DATASET_FILENAME, repo_type="model")
        ds = InferenceDataset(data_path)
        loader = DataLoader(ds, batch_size=64, shuffle=False)
        
        embs_v1 = []
        embs_v2 = []
        self.db_paths = []
        
        with torch.no_grad():
            for batch in tqdm(loader, desc="Indexing"):
                m_ids = batch["midi_ids"].to(DEVICE)
                self.db_paths.extend(batch["path"])
                
                # Encode V1
                v1_out = self.model_v1.encode_midi(m_ids)
                embs_v1.append(v1_out.cpu())
                
                # Encode V2 (Mixed Precision per velocità)
                with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                    v2_out = self.model_v2.encode_midi(m_ids)
                embs_v2.append(v2_out.float().cpu())

        self.db_matrix_v1 = torch.cat(embs_v1)
        self.db_matrix_v2 = torch.cat(embs_v2)
        print(f"Index built. DB Size: {len(self.db_paths)} items.")

    def search(self, query_text):
        if not self.model_v1 or not self.model_v2:
            return "Models not loaded", None, None

        # 1. Encode Text Query (V1 & V2)
        with torch.no_grad():
            # V1 Input
            i1 = self.tokenizer_v1([query_text], padding=True, truncation=True, return_tensors="pt").to(DEVICE)
            q_emb_v1 = self.model_v1.encode_text(i1["input_ids"], i1["attention_mask"]).cpu()
            
            # V2 Input
            i2 = self.tokenizer_v2([query_text], padding=True, truncation=True, return_tensors="pt").to(DEVICE)
            q_emb_v2 = self.model_v2.encode_text(i2["input_ids"], i2["attention_mask"]).cpu()

        # 2. Compute Similarity
        sim_v1 = q_emb_v1 @ self.db_matrix_v1.T
        sim_v2 = q_emb_v2 @ self.db_matrix_v2.T
        
        # 3. Ensemble Averaging
        final_sim = (sim_v1 + sim_v2) / 2.0
        
        # 4. Get Best Result
        best_idx = torch.argmax(final_sim).item()
        best_score = final_sim[0, best_idx].item()
        best_path_raw = self.db_paths[best_idx] # Path salvato nel tensore (es. "data/name.mid")
        filename = os.path.basename(best_path_raw)
        
        # 5. Audio Generation logic
        # Cerchiamo il file effettivo nella cartella scaricata
        local_midi_path = os.path.join(MIDI_DATA_DIR, "midicaps", filename)
        if not os.path.exists(local_midi_path):
             # Fallback: prova nella root della cartella scaricata
             local_midi_path = os.path.join(MIDI_DATA_DIR, filename)

        audio_out = "result.wav"
        status_msg = f"Found: {filename}\nEnsemble Score: {best_score:.4f}\n(Avg of V1 & V2)"

        if os.path.exists(local_midi_path):
            try:
                fs = FluidSynth(SOUNDFONT_PATH)
                fs.midi_to_audio(local_midi_path, audio_out)
                return status_msg, audio_out, local_midi_path
            except Exception as e:
                return status_msg + f"\nAudio Error: {e}", None, local_midi_path
        else:
            return status_msg + "\n(MIDI File not found locally)", None, None
