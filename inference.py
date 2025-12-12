import torch
from torch.utils.data import DataLoader, Dataset
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer
from tqdm import tqdm
import os
import sys

# --- IMPORT MODEL ARCHITECTURES ---
# This requires 'ensemble_system.py' to be in the same directory.
try:
    from ensemble_system import NeuralMidiSearch_V1, NeuralMidiSearch_V2
except ImportError:
    raise ImportError("Critical Error: 'ensemble_system.py' not found. Please clone the repository properly.")

# --- CONFIGURATION ---
HF_REPO_ID = "GiuseppeStilly/MIDI-Retrieval"
DATASET_FILENAME = "dataset_midicaps_tokenized.pt"
MODEL_V1_NAME = "v1_lstm_optimized.pt"
MODEL_V2_NAME = "v2_transformer_mpnet.pt"

# Detect device (GPU is highly recommended for indexing)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 128 

class InferenceDataset(Dataset):
    """
    Wrapper to load the pre-tokenized MIDI database.
    """
    def __init__(self, data_path):
        print(f"Loading database from {data_path}...")
        self.data = torch.load(data_path, map_location="cpu")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        # Handle different naming conventions in the dataset (m vs midi_ids)
        m = item.get("m", item.get("midi_ids", [0]))
        if not isinstance(m, torch.Tensor): 
            m = torch.tensor(m, dtype=torch.long)
        return {"midi_ids": m, "index": idx}

class MidiSearchEngine:
    def __init__(self):
        self.model_v1 = None
        self.model_v2 = None
        self.tokenizer_v1 = None
        self.tokenizer_v2 = None
        self.index_v1 = None
        self.index_v2 = None
        self.raw_data = None
        self.initialized = False

    def initialize(self):
        """
        Main setup function. Downloads models, loads data, and builds the search index.
        """
        print("--- Initializing Search Engine ---")
        
        # 1. Download and Load Dataset
        try:
            print("Downloading dataset from HuggingFace...")
            data_path = hf_hub_download(repo_id=HF_REPO_ID, filename=DATASET_FILENAME, repo_type="dataset")
        except Exception:
            # Fallback: sometimes datasets are stored in the model repo
            print("Dataset not found in 'dataset' repo. Trying 'model' repo...")
            data_path = hf_hub_download(repo_id=HF_REPO_ID, filename=DATASET_FILENAME)
            
        dataset = InferenceDataset(data_path)
        self.raw_data = dataset.data
        print(f"Database loaded successfully: {len(self.raw_data)} items.")

        # 2. Load Neural Models
        print("Loading Neural Models (V1 & V2)...")
        self.model_v1 = self._load_model(NeuralMidiSearch_V1, MODEL_V1_NAME)
        self.model_v2 = self._load_model(NeuralMidiSearch_V2, MODEL_V2_NAME)
        
        if not self.model_v1 or not self.model_v2:
            print("Error: Failed to load models.")
            return False

        # 3. Load Text Tokenizers (Sentence-Transformers)
        # These must match the ones used during training
        self.tokenizer_v1 = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        self.tokenizer_v2 = AutoTokenizer.from_pretrained("sentence-transformers/all-mpnet-base-v2")

        # 4. Build Vector Index (The heavy lifting)
        print("Building Search Index (Computing embeddings for all songs)...")
        self.index_v1 = self._compute_embeddings(self.model_v1, dataset)
        self.index_v2 = self._compute_embeddings(self.model_v2, dataset)
        
        self.initialized = True
        print("--- System Ready for Search ---")
        return True

    def _load_model(self, model_class, filename):
        """Helper to download and load a PyTorch checkpoint."""
        try:
            path = hf_hub_download(repo_id=HF_REPO_ID, filename=filename)
            ckpt = torch.load(path, map_location=DEVICE)
            
            # Retrieve vocab size used during training
            vocab_size = ckpt.get('vocab_size', 3000)
            
            model = model_class(vocab_size).to(DEVICE)
            model.load_state_dict(ckpt['model_state'], strict=False)
            model.eval()
            return model
        except Exception as e:
            print(f"Error loading model {filename}: {e}")
            return None

    def _compute_embeddings(self, model, dataset):
        """Runs the model on the entire dataset to create the search index."""
        loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
        embeddings = []
        
        with torch.no_grad():
            for batch in tqdm(loader, desc="Indexing"):
                m_ids = batch["midi_ids"].to(DEVICE)
                # Project MIDI tokens into the embedding space
                emb = model.encode_midi(m_ids)
                embeddings.append(emb.cpu())
        
        return torch.cat(embeddings).to(DEVICE)

    def search(self, text_query, top_k=5):
        """
        Performs the hybrid search using both V1 and V2 models.
        """
        if not self.initialized:
            raise RuntimeError("Engine not initialized. Call initialize() first.")

        with torch.no_grad():
            # --- V1 Branch ---
            i1 = self.tokenizer_v1(text_query, return_tensors="pt", padding=True, truncation=True).to(DEVICE)
            text_emb_v1 = self.model_v1.encode_text(i1["input_ids"], i1["attention_mask"])
            
            # --- V2 Branch ---
            i2 = self.tokenizer_v2(text_query, return_tensors="pt", padding=True, truncation=True).to(DEVICE)
            text_emb_v2 = self.model_v2.encode_text(i2["input_ids"], i2["attention_mask"])
            
            #
