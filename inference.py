import os
import torch
import numpy as np
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer
from sklearn.metrics.pairwise import cosine_similarity
from midi2audio import FluidSynth
import config as cfg
from model import NeuralMidiSearchTransformer

class SearchEngine:
    def __init__(self):
        self.model = None
        self.db_matrix = None
        self.db_paths = None
        self.tokenizer = None
        
        self.setup_environment()
        self.load_model()

    def setup_environment(self):
        # Install FluidSynth if missing (Linux/Colab environment)
        if not os.path.exists('/usr/bin/fluidsynth') and not os.path.exists('/usr/local/bin/fluidsynth'):
            print("Installing FluidSynth...")
            os.system("sudo apt-get update -y > /dev/null")
            os.system("sudo apt-get install -y fluidsynth > /dev/null")
        
        # Download SoundFont for audio generation
        if not os.path.exists(cfg.SOUNDFONT_PATH):
            print("Downloading SoundFont...")
            os.system(f"curl -L {cfg.SOUNDFONT_URL} -o {cfg.SOUNDFONT_PATH}")

    def load_model(self):
        print(f"Connecting to Hugging Face Repo: {cfg.HF_REPO_ID}...")
        
        # The filename is defined in config.py (midi_search_MPNET_TRANSFORMER.pt)
        filename = os.path.basename(cfg.SAVE_FILE) 
        
        try:
            model_path = hf_hub_download(repo_id=cfg.HF_REPO_ID, filename=filename)
        except Exception as e:
            print(f"Error downloading model: {e}")
            return

        print(f"Loading Transformer Model from {model_path}...")
        
        # Load Checkpoint
        checkpoint = torch.load(model_path, map_location=cfg.DEVICE, weights_only=False)
        
        # Initialize Architecture (Transformer Only)
        self.model = NeuralMidiSearchTransformer(checkpoint['vocab_size']).to(cfg.DEVICE)
        self.model.load_state_dict(checkpoint['model_state'])
        self.model.eval()
        
        # Load Database Index
        self.db_matrix = checkpoint['db_matrix']
        self.db_paths = checkpoint['db_paths']
        
        # Load Tokenizer (MPNet)
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.MODEL_NAME)
        
        print("Transformer Search Engine Ready.")

    def search(self, query):
        if not self.model:
            return "Model not loaded.", None, None

        # 1. Encode Text Query
        inputs = self.tokenizer([query], padding=True, truncation=True, return_tensors="pt").to(cfg.DEVICE)
        
        with torch.no_grad():
            query_emb = self.model.encode_text(inputs["input_ids"], inputs["attention_mask"]).cpu().numpy()

        # 2. Compute Similarity (Single Model)
        scores = cosine_similarity(query_emb, self.db_matrix)[0]

        # 3. Retrieve Best Match
        best_idx = np.argmax(scores)
        best_path = self.db_paths[best_idx]
        best_score = scores[best_idx]
        
        filename = os.path.basename(best_path)
        info = (f"Found: {filename}\n"
                f"Similarity Score: {best_score:.3f}\n"
                f"Model: Transformer (MPNet)")
        
        # 4. Generate Audio Preview
        audio_out = "result.wav"
        if os.path.exists(best_path):
            FluidSynth(cfg.SOUNDFONT_PATH).midi_to_audio(best_path, audio_out)
            return info, audio_out, best_path
        
        return info + " (MIDI file missing locally)", None, None