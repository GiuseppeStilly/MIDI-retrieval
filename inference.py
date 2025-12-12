import os
import torch
import numpy as np
import tarfile
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer
from midi2audio import FluidSynth
from sklearn.metrics.pairwise import cosine_similarity
import sys
from ensemble_system import NeuralMidiSearch_V1, NeuralMidiSearch_V2

HF_REPO_ID = "GiuseppeStilly/MIDI-Retrieval"
MODEL_V1_NAME = "v1_lstm_optimized.pt"
MODEL_V2_NAME = "v2_transformer_mpnet.pt"
SOUNDFONT_PATH = "soundfont.sf2"
MIDI_DATA_DIR = "midicaps_data"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SearchEngine:
    def __init__(self):
        print("Initializing Ensemble Search Engine...")
        self._setup_resources()
        self._load_ensemble()
        print("Initialization complete.")

    def _setup_resources(self):
        if not os.path.exists(SOUNDFONT_PATH):
            print("Downloading SoundFont...")
            os.system(f"curl -L https://github.com/urish/cinto/raw/master/media/FluidR3_GM.sf2 -o {SOUNDFONT_PATH}")

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
        
        try:
            print(f"Loading V1: {MODEL_V1_NAME}...")
            path_v1 = hf_hub_download(repo_id=HF_REPO_ID, filename=MODEL_V1_NAME)
            ckpt_v1 = torch.load(path_v1, map_location=DEVICE, weights_only=False)
            
            print(f"DEBUG: Keys found in V1 checkpoint: {list(ckpt_v1.keys())}")

            if 'db_matrix' not in ckpt_v1:
                print("Error: 'db_matrix' key missing in V1 checkpoint.")
            
            self.model_v1 = NeuralMidiSearch_V1(ckpt_v1.get('vocab_size', 3000)).to(DEVICE)
            self.model_v1.load_state_dict(ckpt_v1['model_state'], strict=False)
            self.model_v1.eval()
            
            self.tok_v1 = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
            self.db_v1 = ckpt_v1.get('db_matrix')

            print(f"Loading V2: {MODEL_V2_NAME}...")
            path_v2 = hf_hub_download(repo_id=HF_REPO_ID, filename=MODEL_V2_NAME)
            ckpt_v2 = torch.load(path_v2, map_location=DEVICE, weights_only=False)
            
            print(f"DEBUG: Keys found in V2 checkpoint: {list(ckpt_v2.keys())}")

            self.model_v2 = NeuralMidiSearch_V2(ckpt_v2.get('vocab_size', 3000)).to(DEVICE)
            self.model_v2.load_state_dict(ckpt_v2['model_state'], strict=False)
            self.model_v2.eval()
            
            self.tok_v2 = AutoTokenizer.from_pretrained("sentence-transformers/all-mpnet-base-v2")
            self.db_v2 = ckpt_v2.get('db_matrix')

            self.db_paths = ckpt_v1.get('db_paths', [])
            if not self.db_paths:
                print("WARNING: 'db_paths' is missing.")
                
            print(f"Ensemble Loaded. Database size: {len(self.db_paths)}")

        except Exception as e:
            print(f"Critical Error loading models: {e}")
            self.model_v1 = None
            self.model_v2 = None

    def search(self, query):
        if not self.model_v1 or not self.model_v2:
            return "Error: Models failed to load. Check logs.", None, None
        
        try:
            in_v1 = self.tok_v1([query], padding=True, truncation=True, return_tensors="pt").to(DEVICE)
            emb_v1 = self.model_v1.encode_text(in_v1["input_ids"], in_v1["attention_mask"]).cpu().detach().numpy()
            scores_v1 = cosine_similarity(emb_v1, self.db_v1)[0]

            in_v2 = self.tok_v2([query], padding=True, truncation=True, return_tensors="pt").to(DEVICE)
            emb_v2 = self.model_v2.encode_text(in_v2["input_ids"], in_v2["attention_mask"]).cpu().detach().numpy()
            scores_v2 = cosine_similarity(emb_v2, self.db_v2)[0]

            final_scores = (scores_v1 + scores_v2) / 2.0
            
            best_idx = np.argmax(final_scores)
            best_score = final_scores[best_idx]
            
            if not self.db_paths:
                return f"Best Match Index: {best_idx} (No filenames)", None, None

            full_path = self.db_paths[best_idx]
            filename = os.path.basename(full_path)
            
            local_path = None
            for root, _, files in os.walk(MIDI_DATA_DIR):
                if filename in files:
                    local_path = os.path.join(root, filename)
                    break
            
            score_details = (f"Final Score: {best_score:.4f}\n"
                             f"(LSTM: {scores_v1[best_idx]:.3f} | Transformer: {scores_v2[best_idx]:.3f})")
            
            result_text = f"Found: {filename}\n{score_details}"
            
            if not local_path:
                return result_text + "\n(File not found locally)", None, None

            audio_out = "preview.wav"
            try:
                FluidSynth(SOUNDFONT_PATH).midi_to_audio(local_path, audio_out)
                return result_text, audio_out, local_path
            except Exception as e:
                return result_text + f"\n(Audio Error: {e})", None, local_path
        
        except Exception as e:
            return f"Search Error: {e}", None, None