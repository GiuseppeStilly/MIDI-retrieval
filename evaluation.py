import os
import torch
import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd 
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

# --- PLOTTING FUNCTION (Updated for VS Code) ---
def plot_training_loss(loss_history):
    """Generates and saves the loss graph to a file."""
    if not loss_history:
        print("No loss history found to plot.")
        return

    df = pd.Series(loss_history)
    
    plt.figure(figsize=(10, 6), dpi=100)
    
    # 1. Batch Loss
    plt.plot(df, label='Batch Loss', alpha=0.3, color='lightblue')
    
    # 2. Moving Average (Trend)
    if len(df) > 50:
        window = max(5, int(len(df) / 50))
        moving_avg = df.rolling(window=window).mean()
        plt.plot(moving_avg, label=f'Trend (Avg {window} steps)', color='blue', linewidth=2)

    plt.xlabel('Training Steps (Batches)')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # CHANGED: Save to file instead of showing window
    output_file = "training_loss.png"
    plt.savefig(output_file)
    plt.close()
    print(f"Graph saved to {output_file}")

# --- DATASET CLASS ---
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
    
    # --- PLOT LOSS HISTORY ---
    if 'loss_history' in checkpoint:
        print("\nSaving Training Loss History...")
        plot_training_loss(checkpoint['loss_history'])
    else:
        print("\nWarning: 'loss_history' not found in checkpoint. Skipping plot.")
    # -------------------------

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

            text_embs.append(model.encode_text(i_ids, mask).cpu())
            midi_embs.append(model.encode_midi(m_ids).cpu())

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