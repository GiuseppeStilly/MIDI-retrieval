import os
import torch
import numpy as np
import tarfile
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer
from midi2audio import FluidSynth
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import sys

# Import architectures
try:
    from ensemble_system import NeuralMidiSearch_V1, NeuralMidiSearch_V2
except ImportError:
    # Fallback if the file is named differently
    from evaluation import NeuralMidiSearch_V1, NeuralMidiSearch_V2

# --- CONFIGURATION ---
HF_REPO_ID = "GiuseppeStilly/MIDI-Retrieval"
MODEL_V1_NAME = "v1_lstm_optimized.pt"
MODEL_V2_NAME = "v2_transformer_mpnet.pt"
TOKENIZED_DATASET = "dataset_midicaps_tokenized.pt"
SOUNDFONT_PATH = "soundfont.sf2"
MIDI_DATA_DIR = "midicaps_data"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SearchEngine:
    def __init__(self):
        print(f"Initializing Ensemble Search Engine on {DEVICE}...")
        self._setup_resources()
        self._load_ensemble()
        print("Initialization complete. System Ready.")

    def _setup_resources(self):
        # 1. Download SoundFont
        if not os.path.exists(SOUNDFONT_PATH):
            print("Downloading SoundFont...")
            os.system(f"curl -L https://github.com/urish/cinto/raw/master/media/FluidR3_GM.sf2 -o {SOUNDFONT_PATH}")

        # 2. Download MIDI Dataset (The actual .mid files)
        if not os.path.exists(MIDI_DATA_DIR):
            print("Downloading MIDI Dataset...")
            try:
                path = hf_hub_download(repo_id="amaai-lab/MidiCaps", filename="midicaps.tar.gz", repo_type="dataset")
                with tarfile.open(path) as tar:
                    tar.extractall(MIDI_DATA_DIR)
            except Exception as e:
                print(f"Warning: Could not download dataset. {e}")

    def _load_ensemble(self):
        self.model_v1 = None
        self.model_v2 = None
        self.db_v1 = None
        self.db_v2 = None
        self.db_paths = []

        try:
            # --- LOAD V1 (LSTM) ---
            print(f"Loading V1: {MODEL_V1_NAME}...")
            path_v1 = hf_hub_download(repo_id=HF_REPO_ID, filename=MODEL_V1_NAME)
            ckpt_v1 = torch.load(path_v1, map_location=DEVICE, weights_only=False)
            
            self.model_v1 = NeuralMidiSearch_V1(ckpt_v1.get('vocab_size', 3000)).to(DEVICE)
            self.model_v1.load_state_dict(ckpt_v1['model_state'], strict=False)
            self.model_v1.eval()
            self.tok_v1 = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

            # Check for V1 Database
            if 'db_matrix' in ckpt_v1:
                self.db_v1 = ckpt_v1['db_matrix']
                self.db_paths = ckpt_v1.get('db_paths', [])
            
            # --- LOAD V2 (Transformer) ---
            print(f"Loading V2: {MODEL_V2_NAME}...")
            path_v2 = hf_hub_download(repo_id=HF_REPO_ID, filename=MODEL_V2_NAME)
            ckpt_v2 = torch.load(path_v2, map_location=DEVICE, weights_only=False)
            
            self.model_v2 = NeuralMidiSearch_V2(ckpt_v2.get('vocab_size', 3000)).to(DEVICE)
            self.model_v2.load_state_dict(ckpt_v2['model_state'], strict=False)
            self.model_v2.eval()
            self.tok_v2 = AutoTokenizer.from_pretrained("sentence-transformers/all-mpnet-base-v2")

            # Check for V2 Database
            if 'db_matrix' in ckpt_v2:
                self.db_v2 = ckpt_v2['db_matrix']
                if not self.db_paths:
                    self.db_paths = ckpt_v2.get('db_paths', [])

            # --- AUTO-BUILD CHECK ---
            if self.db_v1 is None or self.db_v2 is None:
                print("Database missing in checkpoints. Auto-building in memory...")
                self._build_database_in_memory()
            else:
                print(f"Ensemble Loaded. Database size: {len(self.db_paths)}")

        except Exception as e:
            print(f"Critical Error loading models: {e}")

    def _build_database_in_memory(self):
        """Reconstructs the vector database by processing the tokenized dataset."""
        try:
            print("   Downloading tokenized dataset...")
            path_tok = hf_hub_download(repo_id=HF_REPO_ID, filename=TOKENIZED_DATASET)
            dataset = torch.load(path_tok)
            
            # Extract data
            all_midi_tokens = [d['m'] for d in dataset]
            self.db_paths = [d['p'] for d in dataset]
            
            print(f"   Processing {len(all_midi_tokens)} songs on {DEVICE}. This takes 2-3 minutes...")
            
            list_v1 = []
            list_v2 = []
            batch_size = 128 
            
            with torch.no_grad():
                for i in tqdm(range(0, len(all_midi_tokens), batch_size), desc="Building DB"):
                    batch = all_midi_tokens[i : i + batch_size]
                    
                    # Process batch items
                    for m in batch:
                        inp = m.unsqueeze(0).long().to(DEVICE)
                        
                        # Compute V1
                        if self.db_v1 is None:
                            emb1 = self.model_v1.encode_midi(inp)
                            list_v1.append(emb1.cpu())
                        
                        # Compute V2
                        if self.db_v2 is None:
                            emb2 = self.model_v2.encode_midi(inp)
                            list_v2.append(emb2.cpu())

            # Finalize matrices
            if self.db_v1 is None:
                self.db_v1 = torch.cat(list_v1, dim=0)
            
            if self.db_v2 is None:
                self.db_v2 = torch.cat(list_v2, dim=0)
                
            print("Database built successfully.")
            
        except Exception as e:
            print(f"Error auto-building database: {e}")

    def search(self, query):
        if not self.model_v1 or not self.model_v2:
            return "Error: Models not loaded.", None, None
        
        try:
            # 1. V1 Inference
            in_v1 = self.tok_v1([query], padding=True, truncation=True, return_tensors="pt").to(DEVICE)
            with torch.no_grad():
                emb_v1 = self.model_v1.encode_text(in_v1["input_ids"], in_v1["attention_mask"]).cpu().numpy()
            scores_v1 = cosine_similarity(emb_v1, self.db_v1)[0]

            # 2. V2 Inference
            in_v2 = self.tok_v2([query], padding=True, truncation=True, return_tensors="pt").to(DEVICE)
            with torch.no_grad():
                emb_v2 = self.model_v2.encode_text(in_v2["input_ids"], in_v2["attention_mask"]).cpu().numpy()
            scores_v2 = cosine_similarity(emb_v2, self.db_v2)[0]

            # 3. Ensemble Fusion (Average)
            final_scores = (scores_v1 + scores_v2) / 2.0
            
            # 4. Retrieval
            best_idx = np.argmax(final_scores)
            best_score = final_scores[best_idx]
            
            if not self.db_paths:
                return "No database paths available.", None, None

            full_path = self.db_paths[best_idx]
            filename = os.path.basename(full_path)
            
            # Locate local file
            local_path = None
            for root, _, files in os.walk(MIDI_DATA_DIR):
                if filename in files:
                    local_path = os.path.join(root, filename)
                    break
            
            # Format output
            score_details = (f"Ensemble Score: {best_score:.4f}\n"
                             f"(LSTM: {scores_v1[best_idx]:.3f} | Transformer: {scores_v2[best_idx]:.3f})")
            
            result_text = f"Found: {filename}\n{score_details}"
            
            if not local_path:
                return result_text + "\n(File not found locally)", None, None

            # 5. Audio Synthesis
            audio_out = "preview.wav"
            try:
                FluidSynth(SOUNDFONT_PATH).midi_to_audio(local_path, audio_out)
                return result_text, audio_out, local_path
            except Exception as e:
                return result_text + f"\n(Audio Error: {e})", None, local_path
        
        except Exception as e:
            return f"Search Error: {e}", None, None
