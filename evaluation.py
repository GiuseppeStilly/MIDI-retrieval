import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from huggingface_hub import hf_hub_download

# Import from local file 'ensemble.py'
from ensemble import load_model, compute_similarity_matrix, HF_REPO_ID

# --- CONFIGURATION ---
TEST_FILENAME = "test_midicaps_tokenized.pt"
MODEL_V1_NAME = "v1_lstm_optimized.pt"
MODEL_V2_NAME = "v2_transformer_mpnet.pt"
BATCH_SIZE = 64

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

def calculate_metrics(sim_matrix):
    """Computes R@1, R@5, R@10 and MRR."""
    ranks = []
    n = sim_matrix.shape[0]
    
    for i in range(n):
        target_score = sim_matrix[i, i]
        # Rank is the count of scores higher than the target + 1
        rank = (sim_matrix[i] > target_score).sum().item() + 1
        ranks.append(rank)
    
    ranks = np.array(ranks)
    return {
        "R@1": np.mean(ranks <= 1) * 100,
        "R@5": np.mean(ranks <= 5) * 100,
        "R@10": np.mean(ranks <= 10) * 100,
        "MRR": np.mean(1.0 / ranks)
    }

def print_metrics(name, metrics):
    print(f"\n--- {name} RESULTS ---")
    print(f"R@1:  {metrics['R@1']:.2f}%")
    print(f"R@5:  {metrics['R@5']:.2f}%")
    print(f"R@10: {metrics['R@10']:.2f}%")
    print(f"MRR:  {metrics['MRR']:.4f}")

def main():
    print("="*60)
    print("STARTING DIAGNOSTIC EVALUATION")
    print("="*60)
    
    # 1. Load Data
    try:
        data_path = hf_hub_download(repo_id=HF_REPO_ID, filename=TEST_FILENAME, repo_type="model")
        ds = CachedDataset(data_path)
        loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    except Exception as e:
        print(f"Critical Data Error: {e}")
        return

    sim_v1 = None
    sim_v2 = None

    # 2. Evaluate V1 (LSTM) Only
    print("\n[1/3] Evaluating V1 (LSTM)...")
    model_v1 = load_model("V1", MODEL_V1_NAME)
    if model_v1:
        sim_v1 = compute_similarity_matrix(model_v1, loader, "Inference V1")
        metrics_v1 = calculate_metrics(sim_v1)
        print_metrics("V1 (LSTM)", metrics_v1)
        del model_v1
        torch.cuda.empty_cache()
    
    # 3. Evaluate V2 (Transformer) Only
    print("\n[2/3] Evaluating V2 (Transformer)...")
    model_v2 = load_model("V2", MODEL_V2_NAME)
    if model_v2:
        sim_v2 = compute_similarity_matrix(model_v2, loader, "Inference V2")
        metrics_v2 = calculate_metrics(sim_v2)
        print_metrics("V2 (Transformer)", metrics_v2)
        del model_v2
        torch.cuda.empty_cache()

    # 4. Evaluate Ensemble
    print("\n[3/3] Evaluating Ensemble (Average)...")
    if sim_v1 is not None and sim_v2 is not None:
        sim_final = (sim_v1 + sim_v2) / 2.0
        metrics_ens = calculate_metrics(sim_final)
        print_metrics("ENSEMBLE", metrics_ens)
    else:
        print("Cannot compute ensemble: one model failed to load.")

if __name__ == "__main__":
    main()