import os
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from huggingface_hub import hf_hub_download
import config as cfg
from model import NeuralMidiSearchTransformer

# --- CONFIGURATION ---
REPO_ID = "GiuseppeStilly/MIDI-retrieval"
MODEL_FILENAME = "v2_transformer_mpnet.pt"
CACHE_FILENAME = "test_midicaps_tokenized.pt"
BATCH_SIZE = 128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CachedDataset(Dataset):
    def __init__(self, cache_path):
        print(f"Loading data from: {cache_path}")
        self.data = torch.load(cache_path, map_location="cpu")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            "midi_ids": item["m"].long(),
            "input_ids": item["i"].long(),
            "attention_mask": item["a"].long()
        }

def official_test():
    print("STARTING OFFICIAL TEST (FULL CLOUD)")
    
    # 1. DOWNLOAD FROM HUGGING FACE
    try:
        print("1. Downloading Model...")
        model_path = hf_hub_download(repo_id=REPO_ID, filename=MODEL_FILENAME)
        
        print("2. Downloading Test Set...")
        cache_path = hf_hub_download(repo_id=REPO_ID, filename=CACHE_FILENAME)
        
        print("Downloads completed successfully.")
    except Exception as e:
        print(f"Error downloading from HF: {e}")
        print("Ensure you have uploaded both the model and the test set to Hugging Face.")
        return

    # 2. LOAD MODEL
    print("Initializing Neural Network...")
    # Using weights_only=False to support numpy arrays in the checkpoint
    checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=False)
    
    model = NeuralMidiSearchTransformer(checkpoint['vocab_size']).to(DEVICE)
    model.load_state_dict(checkpoint['model_state'])
    model.eval()

    # 3. LOAD DATA
    ds = CachedDataset(cache_path)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False)

    # 4. COMPUTE METRICS
    print(f"Evaluating on {len(ds)} examples...")
    text_embs, midi_embs = [], []
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Generating Embeddings"):
            i_ids = batch["input_ids"].to(DEVICE)
            mask = batch["attention_mask"].to(DEVICE)
            m_ids = batch["midi_ids"].to(DEVICE)

            text_embs.append(model.encode_text(i_ids, mask))
            midi_embs.append(model.encode_midi(m_ids))

    text_tensor = torch.cat(text_embs)
    midi_tensor = torch.cat(midi_embs)
    
    # Calculate Similarity Matrix
    print("Calculating Ranking...")
    sim_matrix = text_tensor @ midi_tensor.T
    sim_matrix = sim_matrix.cpu().float()
    
    ranks = []
    n = sim_matrix.shape[0]
    
    for i in range(n):
        target_score = sim_matrix[i, i]
        # Rank: how many scores are higher than the target? + 1
        rank = (sim_matrix[i] > target_score).sum().item() + 1
        ranks.append(rank)
    
    ranks = np.array(ranks)
    mrr = np.mean(1.0 / ranks)

    print("\n" + "="*40)
    print("OFFICIAL RESULTS (Hugging Face)")
    print("="*40)
    print(f"R@1:  {np.mean(ranks <= 1) * 100:.2f}%")
    print(f"R@5:  {np.mean(ranks <= 5) * 100:.2f}%")
    print(f"R@10: {np.mean(ranks <= 10) * 100:.2f}%")
    print(f"MRR:       {mrr:.4f}")
    print("="*40)

if __name__ == "__main__":
    official_test()