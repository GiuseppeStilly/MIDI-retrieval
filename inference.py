import os
import tarfile
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
        # 1. Install FluidSynth (for audio generation)
        if not os.path.exists('/usr/bin/fluidsynth') and not os.path.exists('/usr/local/bin/fluidsynth'):
            print("Installing FluidSynth...")
            os.system("sudo apt-get update -y > /dev/null")
            os.system("sudo apt-get install -y fluidsynth > /dev/null")
        
        # 2. Download SoundFont
        if not os.path.exists(cfg.SOUNDFONT_PATH):
            print("Downloading SoundFont...")
            os.system(f"curl -L {cfg.SOUNDFONT_URL} -o {cfg.SOUNDFONT_PATH}")

        # 3. Download MIDI Dataset (MidiCaps) for playback
        target_dir = getattr(cfg, "MIDI_DATA_DIR", "midicaps_data")
        
        if not os.path.exists(target_dir):
            print("Downloading MIDI files for playback...")
            try:
                path = hf_hub_download(repo_id="amaai-lab/MidiCaps", filename="midicaps.tar.gz", repo_type="dataset")
                with tarfile.open(path) as tar: 
                    tar.extractall(target_dir)
                print("MIDI files downloaded.")
            except Exception as e:
                print(f"Warning: Could not download MIDI dataset: {e}")

    def load_model(self):
        print(f"Connecting to Hugging Face Repo: {cfg.HF_REPO_ID}...")
        
        filename = os.path.basename(cfg.SAVE_FILE) 
        
        try:
            model_path = hf_hub_download(repo_id=cfg.HF_REPO_ID, filename=filename)
        except Exception as e:
            print(f"Error downloading model: {e}")
            return

        print(f"Loading Transformer Model from {model_path}...")
        
        checkpoint = torch.load(model_path, map_location=cfg.DEVICE, weights_only=False)
        
        self.model = NeuralMidiSearchTransformer(checkpoint['vocab_size']).to(cfg.DEVICE)
        self.model.load_state_dict(checkpoint['model_state'])
        self.model.eval()
        
        self.db_matrix = checkpoint['db_matrix']
        self.db_paths = checkpoint['db_paths']
        
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.MODEL_NAME)
        
        print("Transformer Search Engine Ready.")

    def search(self, query):
        if not self.model:
            return "Model not loaded.", None, None

        # Encode Query
        inputs = self.tokenizer([query], padding=True, truncation=True, return_tensors="pt").to(cfg.DEVICE)
        
        with torch.no_grad():
            query_emb = self.model.encode_text(inputs["input_ids"], inputs["attention_mask"]).cpu().numpy()

        # Calculate Similarity
        scores = cosine_similarity(query_emb, self.db_matrix)[0]

        # Retrieve Best Match
        best_idx = np.argmax(scores)
        best_path = self.db_paths[best_idx]
        best_score = scores[best_idx]
        
        filename = os.path.basename(best_path)
        info = (f"Found: {filename}\n"
                f"Similarity Score: {best_score:.3f}\n"
                f"Model: Transformer (MPNet)")
        
        # Audio Generation
        audio_out = "result.wav"
        data_dir = getattr(cfg, "MIDI_DATA_DIR", "midicaps_data")

        # Resolve path: sometimes saved paths differ from local paths
        final_path = best_path
        if not os.path.exists(final_path):
            final_path = os.path.join(data_dir, filename)

        if os.path.exists(final_path):
            FluidSynth(cfg.SOUNDFONT_PATH).midi_to_audio(final_path, audio_out)
            return info, audio_out, final_path
        
        return info + " (MIDI file missing locally)", None, None