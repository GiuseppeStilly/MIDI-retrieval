import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from huggingface_hub import hf_hub_download

# Import the engine from our system file
from ensemble_system import EnsembleEngine, HF_REPO_ID

# --- CONFIGURATION ---
TEST_FILENAME = "test_midicaps_tokenized.pt"
BATCH_SIZE = 64

# --- DATA LOADING ---
class CachedDataset(Dataset):
    def __init__(self, cache_path):
        print(f"📂 Loading Test Data: {cache_path}")
        self.data = torch.load(cache_path, map_location="cpu")
    
    def __len__(self): return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        m_ids = item.get("m", item.get("midi_ids", [0]))
        i_ids = item.get("i", item.get("input_ids", [0]))
        a_mask = item.get("a", item.get("attention_mask", [0]))

        if not isinstance(m_ids, torch.Tensor): m_ids = torch.tensor(m_ids, dtype=torch.long)
        if not isinstance(i_ids, torch.Tensor): i_ids = torch.tensor(i_ids, dtype=torch.long)
        if not isinstance(a_mask, torch.Tensor): a_mask = torch.tensor(a_mask, dtype=torch.long)

        return {"midi_ids": m_ids, "input_ids": i_ids, "attention_mask": a_mask}

# --- METRICS CALCULATOR ---
def calculate_metrics(sim_matrix):
    ranks = []
    n = sim_matrix.shape[0]
    
    for i in range(n):
        target_score = sim_matrix[i, i]
        # Rank = count of scores greater than target + 1
        rank = (sim_matrix[i] > target_score).sum().item() + 1
        ranks.append(rank)
    
    ranks = np.array(ranks)
    return {
        "R@1": np.mean(ranks <= 1) * 100,
        "R@5": np.mean(ranks <= 5) * 100,
        "R@10": np.mean(ranks <= 10) * 100,
        "MRR": np.mean(1.0 / ranks),
        "Mean Rank": np.mean(ranks)
    }

# --- MAIN ---
def main():
    print("="*60)
    print("STARTING ENSEMBLE EVALUATION")
    print("="*60)
    
    # 1. Prepare Data
    try:
        data_path = hf_hub_download(repo_id=HF_REPO_ID, filename=TEST_FILENAME, repo_type="model")
        ds = CachedDataset(data_path)
        loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    except Exception as e:
        print(f"Critical Error loading data: {e}")
        return

    # 2. Initialize Engine
    engine = EnsembleEngine()

    # 3. Get Scores
    print("\n--- Phase 1: V1 Inference ---")
    sim_v1 = engine.get_similarity_matrix("v1", loader)
    
    print("\n--- Phase 2: V2 Inference ---")
    sim_v2 = engine.get_similarity_matrix("v2", loader)
    
    engine.cleanup()

    if sim_v1 is None or sim_v2 is None:
        print("Error: Could not compute scores for both models.")
        return

    # 4. Ensemble (Average)
    print("\n--- Phase 3: Calculating Metrics ---")
    sim_final = (sim_v1 + sim_v2) / 2.0
    
    metrics = calculate_metrics(sim_final)

    print("\n" + "="*40)
    print("FINAL RESULTS (V1 + V2 ENSEMBLE)")
    print("="*40)
    for k, v in metrics.items():
        if "Rank" in k or "MRR" in k:
            print(f"{k:<10}: {v:.4f}")
        else:
            print(f"{k:<10}: {v:.2f}%")
    print("="*40)

if __name__ == "__main__":
    main()