import torch
import sys
from transformers import AutoTokenizer
from huggingface_hub import hf_hub_download
from torch.utils.data import DataLoader
from tqdm import tqdm

# Ensure ensemble_system.py is in the same directory
from ensemble_system import NeuralMidiSearch_V1, NeuralMidiSearch_V2, CachedDataset

# --- CONFIGURATION ---
HF_REPO_ID = "GiuseppeStilly/MIDI-Retrieval"
TEST_FILENAME = "test_midicaps_tokenized.pt"
MODEL_V1_NAME = "v1_lstm_optimized.pt"
MODEL_V2_NAME = "v2_transformer_mpnet.pt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MIDI_Search_Engine:
    def __init__(self):
        print(f"--- Initializing Engine ({DEVICE}) ---")
        self.models = {}
        self.tokenizers = {}
        self.db_embeddings = {} 
        self.dataset = None
        
        # 1. Load Models and Tokenizers
        self._load_resources()
        
        # 2. Build Index (Database)
        self._build_index()

    def _load_resources(self):
        print(">> Loading Tokenizers and Models...")
        try:
            # Tokenizers
            self.tokenizers['v1'] = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
            self.tokenizers['v2'] = AutoTokenizer.from_pretrained("sentence-transformers/all-mpnet-base-v2")

            # Model V1 (LSTM)
            path_v1 = hf_hub_download(repo_id=HF_REPO_ID, filename=MODEL_V1_NAME)
            ckpt_v1 = torch.load(path_v1, map_location=DEVICE)
            vocab_v1 = ckpt_v1.get('vocab_size', 3000)
            self.models['v1'] = NeuralMidiSearch_V1(vocab_v1).to(DEVICE)
            self.models['v1'].load_state_dict(ckpt_v1['model_state'], strict=False)
            self.models['v1'].eval()

            # Model V2 (Transformer)
            path_v2 = hf_hub_download(repo_id=HF_REPO_ID, filename=MODEL_V2_NAME)
            ckpt_v2 = torch.load(path_v2, map_location=DEVICE)
            vocab_v2 = ckpt_v2.get('vocab_size', 3000)
            self.models['v2'] = NeuralMidiSearch_V2(vocab_v2).to(DEVICE)
            self.models['v2'].load_state_dict(ckpt_v2['model_state'], strict=False)
            self.models['v2'].eval()
            
            print(">> Models loaded successfully.")
            
        except Exception as e:
            print(f"CRITICAL ERROR loading resources: {e}")
            sys.exit(1)

    def _build_index(self):
        print(">> Indexing MIDI Dataset (this may take a moment)...")
        try:
            data_path = hf_hub_download(repo_id=HF_REPO_ID, filename=TEST_FILENAME, repo_type="model")
            self.dataset = CachedDataset(data_path)
            loader = DataLoader(self.dataset, batch_size=32, shuffle=False)
            
            embs_v1, embs_v2 = [], []

            with torch.no_grad():
                for batch in tqdm(loader, desc="Encoding MIDI"):
                    m_ids = batch["midi_ids"].to(DEVICE)
                    
                    # Encode V1
                    e1 = self.models['v1'].encode_midi(m_ids)
                    embs_v1.append(e1.cpu())
                    
                    # Encode V2
                    e2 = self.models['v2'].encode_midi(m_ids)
                    embs_v2.append(e2.cpu())

            self.db_embeddings['v1'] = torch.cat(embs_v1)
            self.db_embeddings['v2'] = torch.cat(embs_v2)
            print(f">> Database ready: {len(self.dataset)} items indexed.\n")
            
        except Exception as e:
            print(f"Error building index: {e}")
            sys.exit(1)

    def search(self, query_text, top_k=5):
        if not query_text.strip():
            return []

        with torch.no_grad():
            # 1. Encode Text V1
            inputs_v1 = self.tokenizers['v1'](query_text, return_tensors='pt', padding=True, truncation=True).to(DEVICE)
            t_emb_v1 = self.models['v1'].encode_text(inputs_v1['input_ids'], inputs_v1['attention_mask'])

            # 2. Encode Text V2
            inputs_v2 = self.tokenizers['v2'](query_text, return_tensors='pt', padding=True, truncation=True).to(DEVICE)
            t_emb_v2 = self.models['v2'].encode_text(inputs_v2['input_ids'], inputs_v2['attention_mask'])

            # 3. Calculate Similarity (Move to CPU for DB comparison)
            sim_v1 = (t_emb_v1.cpu() @ self.db_embeddings['v1'].T).squeeze()
            sim_v2 = (t_emb_v2.cpu() @ self.db_embeddings['v2'].T).squeeze()

            # 4. Ensemble (Average)
            sim_final = (sim_v1 + sim_v2) / 2.0
            
            # 5. Ranking
            scores, indices = torch.topk(sim_final, k=top_k)
            
            results = []
            for score, idx in zip(scores, indices):
                idx = idx.item()
                results.append({
                    "idx": idx,
                    "score": score.item(),
                    "score_v1": sim_v1[idx].item(),
                    "score_v2": sim_v2[idx].item()
                })
            return results

# --- TERMINAL MODE BLOCK ---
if __name__ == "__main__":
    engine = MIDI_Search_Engine()
    
    print("="*50)
    print("TERMINAL INFERENCE MODE (Type 'q' to exit)")
    print("="*50)
    
    while True:
        text = input("\nDescribe the music: ")
        if text.lower() in ['q', 'exit', 'quit']:
            break
            
        res = engine.search(text, top_k=3)
        
        print("\nRESULTS:")
        for r in res:
            print(f"[{r['idx']}] Score: {r['score']:.4f} (V1: {r['score_v1']:.2f} | V2: {r['score_v2']:.2f})")
