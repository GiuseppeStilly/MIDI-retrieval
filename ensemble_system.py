import os
import requests
import importlib.util
import torch
from tqdm import tqdm
from huggingface_hub import hf_hub_download

# --- CONFIGURATION ---
GITHUB_USER = "GiuseppeStilly"
REPO_NAME = "MIDI-retrieval"
HF_REPO_ID = "GiuseppeStilly/MIDI-Retrieval"

# Branch & Weight Definitions
MODEL_CONFIGS = {
    "v1": {
        "branch": "main",
        "file": "model.py",
        "class": "NeuralMidiSearch",
        "weights": "v1_lstm_optimized.pt",
        "desc": "LSTM + MiniLM"
    },
    "v2": {
        "branch": "v2.0-MPNET+Transformer",
        "file": "model.py",
        "class": "NeuralMidiSearchTransformer",
        "weights": "midi_search_MPNET_TRANSFORMER.pt",
        "desc": "Transformer + MPNet"
    }
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class EnsembleEngine:
    """
    Manages the dynamic loading of different model architectures from Git
    and performs inference to generate the ensemble score.
    """
    
    def __init__(self):
        self.temp_files = []

    def _fetch_class(self, version):
        """Dynamically downloads and imports the model class from GitHub."""
        cfg = MODEL_CONFIGS[version]
        local_name = f"arch_{version}_temp.py"
        self.temp_files.append(local_name)
        
        url = f"https://raw.githubusercontent.com/{GITHUB_USER}/{REPO_NAME}/{cfg['branch']}/{cfg['file']}"
        print(f"📥 Fetching {cfg['desc']} source from branch '{cfg['branch']}'...")
        
        try:
            response = requests.get(url)
            if response.status_code != 200:
                raise ConnectionError(f"Failed to download {url}")
            
            with open(local_name, "w") as f:
                f.write(response.text)
                
            spec = importlib.util.spec_from_file_location(f"mod_{version}", local_name)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return getattr(module, cfg['class'])
        except Exception as e:
            print(f"Error importing {version}: {e}")
            return None

    def _compute_embeddings(self, model, loader, desc):
        """Runs the forward pass to get text and MIDI embeddings."""
        text_embs, midi_embs = [], []
        model.eval()
        
        with torch.no_grad():
            for batch in tqdm(loader, desc=desc):
                i_ids = batch["input_ids"].to(DEVICE)
                mask = batch["attention_mask"].to(DEVICE)
                m_ids = batch["midi_ids"].to(DEVICE)
                
                with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                    t_out = model.encode_text(i_ids, mask)
                    m_out = model.encode_midi(m_ids)
                
                text_embs.append(t_out.float().cpu())
                midi_embs.append(m_out.float().cpu())
                
        return torch.cat(text_embs), torch.cat(midi_embs)

    def get_similarity_matrix(self, version, loader):
        """
        Loads the specific model version, runs inference, returns Sim Matrix, 
        and clears GPU memory immediately.
        """
        cfg = MODEL_CONFIGS[version]
        
        # 1. Get Class
        ModelClass = self._fetch_class(version)
        if not ModelClass: return None

        # 2. Download Weights
        print(f"Downloading weights: {cfg['weights']}...")
        try:
            path = hf_hub_download(repo_id=HF_REPO_ID, filename=cfg['weights'])
            ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
            
            # Initialize Model
            vocab_size = ckpt.get('vocab_size', 3000)
            model = ModelClass(vocab_size).to(DEVICE)
            model.load_state_dict(ckpt['model_state'], strict=False)
            
            # 3. Run Inference
            text, midi = self._compute_embeddings(model, loader, desc=f"Running {cfg['desc']}")
            sim_matrix = text @ midi.T
            
            # 4. Cleanup to save memory for the next model
            del model, ckpt, text, midi
            torch.cuda.empty_cache()
            
            return sim_matrix
            
        except Exception as e:
            print(f" Error processing {version}: {e}")
            return None

    def cleanup(self):
        """Removes temporary python files."""
        for f in self.temp_files:
            if os.path.exists(f): os.remove(f)