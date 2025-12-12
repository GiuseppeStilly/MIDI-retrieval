import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from huggingface_hub import hf_hub_download
from tqdm import tqdm
import sys
import os

from ensemble_system import NeuralMidiSearch_V1, NeuralMidiSearch_V2

# --- CONFIGURATION ---
HF_REPO_ID = "GiuseppeStilly/MIDI-Retrieval"
TEST_FILENAME = "test_midicaps_tokenized.pt"
MODEL_V1_NAME = "v1_lstm_optimized.pt"
MODEL_V2_NAME = "v2_transformer_mpnet.pt"
BATCH_SIZE = 64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CachedDataset(Dataset):
    def __init__(self, cache_path):
        print(f"Loading Dataset: {cache_path}")
        self.data = torch.load(cache_path, map_location="cpu")
    
    def __len__(self): return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        m = item.get("m", item.get("midi_ids", [0]))
        i = item.get("i", item.get("input_ids", [0]))
        a = item.get("a", item.get("attention_mask", [0]))
        
        if not isinstance(m, torch.Tensor): m = torch.tensor(m, dtype=torch.long)
        if not isinstance(i, torch.Tensor): i = torch.tensor(i, dtype=torch.long)
        if not isinstance(a, torch.Tensor): a = torch.tensor(a, dtype=torch.long)

        return {"midi_ids": m, "input_ids": i, "attention_mask": a}

def compute_sim_matrix(model_class, model_name, loader, desc):
    print(f"\nProcessing {desc}...")
    try:
        path = hf_hub_download(repo_id=HF_REPO_ID, filename=model_name)
        ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
        
        vocab_size = ckpt.get('vocab_size', 3000)
        model = model_class(vocab_size).to(DEVICE)
        
        # Load state dict (strict=False to allow minor mismatches like temperature)
        model.load_state_dict(ckpt['model_state'], strict=False)
        model.eval()
        
        text_embs, midi_embs = [], []
        with torch.no_grad():
            for batch in tqdm(loader, desc="Inference"):
                i_ids = batch["input_ids"].to(DEVICE)
                mask = batch["attention_mask"].to(DEVICE)
                m_ids = batch["midi_ids"].to(DEVICE)
                
                # Autocast for GPU efficiency
                with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                    t = model.encode_text(i_ids, mask)
                    m = model.encode_midi(m_ids)
                
                text_embs.append(t.float().cpu())
                midi_embs.append(m.float().cpu())
        
        T = torch.cat(text_embs)
        M = torch.cat(midi_embs)
        sim = T @ M.T
        
        # Quick Sanity Check (Standalone MRR)
        ranks = []
        for i in range(len(sim)):
            rank = (sim[i] > sim[i, i]).sum().item() + 1
            ranks.append(rank)
        mrr = np.mean(1.0 / np.array(ranks))
        print(f"   {desc} Standalone MRR: {mrr:.4f}")
        
        del model, ckpt, T, M, text_embs, midi_embs
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        return sim

    except Exception as e:
        print(f"Error with {model_name}: {e}")
        return None

def main():
    print("="*60)
    print("DIAGNOSTIC ENSEMBLE EVALUATION")
    print(f"Device: {DEVICE}")
    print("="*60)
    
    # 1. Load Data
    try:
        data_path = hf_hub_download(repo_id=HF_REPO_ID, filename=TEST_FILENAME, repo_type="model")
        ds = CachedDataset(data_path)
        loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    except Exception as e:
        print(f"Critical Data Error: {e}")
        return

    # 2. V1 Inference
    sim_v1 = compute_sim_matrix(NeuralMidiSearch_V1, MODEL_V1_NAME, loader, "V1 (LSTM)")
    
    # 3. V2 Inference
    sim_v2 = compute_sim_matrix(NeuralMidiSearch_V2, MODEL_V2_NAME, loader, "V2 (Transformer)")

    if sim_v1 is None or sim_v2 is None:
        print("Aborting: One or both models failed to load.")
        return

    # 4. Ensemble
    print("\nCalculating Ensemble Metrics (Average)...")
    sim_final = (sim_v1 + sim_v2) / 2.0
    
    ranks = []
    n = sim_final.shape[0]
    for i in range(n):
        target = sim_final[i, i]
        rank = (sim_final[i] > target).sum().item() + 1
        ranks.append(rank)
    
    ranks = np.array(ranks)
    mrr = np.mean(1.0 / ranks)

    print("\n" + "="*40)
    print("FINAL ENSEMBLE RESULTS")
    print("="*40)
    print(f"R@1:       {np.mean(ranks <= 1) * 100:.2f}%")
    print(f"R@5:       {np.mean(ranks <= 5) * 100:.2f}%")
    print(f"R@10:      {np.mean(ranks <= 10) * 100:.2f}%")
    print(f"MRR:       {mrr:.4f}")
    print("="*40)

if __name__ == "__main__":
    main()