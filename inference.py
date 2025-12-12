import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer
from tqdm import tqdm
import os

from ensemble_system import NeuralMidiSearch_V1, NeuralMidiSearch_V2

# Configuration
HF_REPO_ID = "GiuseppeStilly/MIDI-Retrieval"
DATASET_FILENAME = "test_midicaps_tokenized.pt"
MODEL_V1_NAME = "v1_lstm_optimized.pt"
MODEL_V2_NAME = "v2_transformer_mpnet.pt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class InferenceDataset(Dataset):
    def __init__(self, data_path):
        self.data = torch.load(data_path, map_location="cpu")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
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
        print("Initializing Search Engine...")
        
        # Load Dataset
        try:
            data_path = hf_hub_download(repo_id=HF_REPO_ID, filename=DATASET_FILENAME, repo_type="model")
            dataset = InferenceDataset(data_path)
            self.raw_data = dataset.data
            print(f"Dataset loaded: {len(self.raw_data)} items")
        except Exception as e:
            print(f"Dataset load error: {e}")
            return False

        # Load Models
        self.model_v1 = self._load_model(NeuralMidiSearch_V1, MODEL_V1_NAME)
        self.model_v2 = self._load_model(NeuralMidiSearch_V2, MODEL_V2_NAME)
        
        # Load Tokenizers
        self.tokenizer_v1 = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        self.tokenizer_v2 = AutoTokenizer.from_pretrained("sentence-transformers/all-mpnet-base-v2")

        if not self.model_v1 or not self.model_v2:
            return False

        # Build Vectors Index
        print("Building Vector Index...")
        self.index_v1 = self._compute_embeddings(self.model_v1, dataset)
        self.index_v2 = self._compute_embeddings(self.model_v2, dataset)
        
        self.initialized = True
        print("System Ready.")
        return True

    def _load_model(self, model_class, filename):
        try:
            path = hf_hub_download(repo_id=HF_REPO_ID, filename=filename)
            ckpt = torch.load(path, map_location=DEVICE)
            vocab_size = ckpt.get('vocab_size', 3000)
            
            model = model_class(vocab_size).to(DEVICE)
            model.load_state_dict(ckpt['model_state'], strict=False)
            model.eval()
            return model
        except Exception as e:
            print(f"Model load error ({filename}): {e}")
            return None

    def _compute_embeddings(self, model, dataset):
        loader = DataLoader(dataset, batch_size=64, shuffle=False)
        embeddings = []
        
        with torch.no_grad():
            for batch in tqdm(loader, desc="Indexing"):
                m_ids = batch["midi_ids"].to(DEVICE)
                emb = model.encode_midi(m_ids)
                embeddings.append(emb.cpu())
        
        return torch.cat(embeddings).to(DEVICE)

    def search(self, text_query, top_k=5):
        if not self.initialized:
            raise RuntimeError("Engine not initialized.")

        with torch.no_grad():
            # Encode Query V1
            inputs_v1 = self.tokenizer_v1(text_query, return_tensors="pt", padding=True, truncation=True).to(DEVICE)
            q_emb_v1 = self.model_v1.encode_text(inputs_v1["input_ids"], inputs_v1["attention_mask"])

            # Encode Query V2
            inputs_v2 = self.tokenizer_v2(text_query, return_tensors="pt", padding=True, truncation=True).to(DEVICE)
            q_emb_v2 = self.model_v2.encode_text(inputs_v2["input_ids"], inputs_v2["attention_mask"])

            # Compute Similarity
            sim_v1 = torch.mm(q_emb_v1, self.index_v1.T).squeeze(0)
            sim_v2 = torch.mm(q_emb_v2, self.index_v2.T).squeeze(0)

            # Ensemble (Mean)
            final_scores = (sim_v1 + sim_v2) / 2.0

            # Ranking
            top_scores, top_indices = torch.topk(final_scores, k=top_k)
            
            results = []
            for score, idx in zip(top_scores, top_indices):
                idx_val = idx.item()
                raw_item = self.raw_data[idx_val]
                results.append({
                    "rank_score": score.item(),
                    "v1_score": sim_v1[idx_val].item(),
                    "v2_score": sim_v2[idx_val].item(),
                    "midi_tokens": raw_item.get("m", raw_item.get("midi_ids")),
                    "metadata": raw_item
                })
            
            return results
