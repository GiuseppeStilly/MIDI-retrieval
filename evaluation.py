import os
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from huggingface_hub import hf_hub_download
import config as cfg

from model import NeuralMidiSearch

# --- CONFIGURATION ---
HF_REPO_ID = "GiuseppeStilly/MIDI-Retrieval"
TEST_FILENAME = "test_midicaps_tokenized.pt"

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

def evaluate():
    print("="*60)
    print("STARTING EVALUATION (V1 LSTM)")
    print("="*60)
    
    # 1. DOWNLOAD TEST SET
    try:
        print(f"Downloading {TEST_FILENAME}...")
        cache_path = hf_hub_download(repo_id=HF_REPO_ID, filename=TEST_FILENAME, repo_type="model")
    except Exception as e:
        print(f"Error downloading test set: {e}")
        return

    # 2. LOAD LOCAL MODEL
    if not os.path.exists(cfg.SAVE_FILE):
        print(f"Error: Model not found at {cfg.SAVE_FILE}")
        return

    print(f"Loading Model from: {cfg.SAVE_FILE}")
    checkpoint = torch.load(cfg.SAVE_FILE, map_location=cfg.DEVICE)
    
    model = NeuralMidiSearch(checkpoint['vocab_size']).to(cfg.DEVICE)
    model.load_state_dict(checkpoint['model_state'])
    model.eval()

    # 3. PREPARE DATA
    ds = CachedDataset(cache_path)
    loader = DataLoader(ds, batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=0)

    # 4. COMPUTE EMBEDDINGS
    print(f"Evaluating on {len(ds)} examples...")
    text_embs, midi_embs = [], []
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Generating Embeddings"):
            i_ids = batch["input_ids"].to(cfg.DEVICE)
            mask = batch["attention_mask"].to(cfg.DEVICE)
            m_ids = batch["midi_ids"].to(cfg.DEVICE)

            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                if hasattr(model, 'encode_text'):
                    t_out = model.encode_text(i_ids, mask)
                    m_out = model.encode_midi(m_ids)
                else:
                    t_out, m_out = model(i_ids, mask, m_ids)

            text_embs.append(t_out.float().cpu())
            midi_embs.append(m_out.float().cpu())

    text_tensor = torch.cat(text_embs)
    midi_tensor = torch.cat(midi_embs)
    
    # 5. CALCULATE METRICS
    print("Calculating Similarity Matrix...")
    sim_matrix = text_tensor @ midi_tensor.T
    
    ranks = []
    n = sim_matrix.shape[0]
    
    for i in range(n):
        target_score = sim_matrix[i, i]
        rank = (sim_matrix[i] > target_score).sum().item() + 1
        ranks.append(rank)
    
    ranks = np.array(ranks)
    mrr = np.mean(1.0 / ranks)

    print("\n" + "="*40)
    print("FINAL RESULTS")
    print("="*40)
    print(f"R@1:  {np.mean(ranks <= 1) * 100:.2f}%")
    print(f"R@5:  {np.mean(ranks <= 5) * 100:.2f}%")
    print(f"R@10: {np.mean(ranks <= 10) * 100:.2f}%")
    print(f"MRR:  {mrr:.4f}")
    print(f"Median Rank: {np.median(ranks):.0f}")
    print("="*40)

if __name__ == "__main__":
    evaluate()