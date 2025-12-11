import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from huggingface_hub import hf_hub_download
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
import config as cfg

HF_REPO_ID = "GiuseppeStilly/MIDI-Retrieval"
MODEL_FILENAME = "v1_lstm_optimized.pt" 
TEST_FILENAME = "test_midicaps_tokenized.pt"

print(f"Preparing to download model from Hugging Face: {HF_REPO_ID} ...")

try:
    MODEL_PATH = hf_hub_download(repo_id=HF_REPO_ID, filename=MODEL_FILENAME)
    print(f"Model downloaded to: {MODEL_PATH}")
except Exception as e:
    print(f"Error downloading model from HF: {e}")
    print("Falling back to local config path...")
    MODEL_PATH = cfg.SAVE_FILE

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = getattr(cfg, 'BATCH_SIZE', 64)

# --- CLASS DEFINITIONS ---
class NeuralMidiSearch(nn.Module):
    def __init__(self, midi_vocab_size):
        super().__init__()
        embed_dim = 512
        
        self.bert = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        self.text_proj = nn.Linear(self.bert.config.hidden_size, embed_dim)

        self.midi_emb = nn.Embedding(midi_vocab_size, 256)
        self.lstm = nn.LSTM(256, 256, batch_first=True, bidirectional=True)
        self.midi_proj = nn.Linear(512, embed_dim)

        self.temperature = nn.Parameter(torch.tensor(0.07))

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def encode_midi(self, midi_ids):
        with torch.no_grad():
            x = self.midi_emb(midi_ids)
            x, _ = self.lstm(x)
            vec = torch.mean(x, dim=1)
            return F.normalize(self.midi_proj(vec), p=2, dim=1)

    def encode_text(self, input_ids, attention_mask):
        with torch.no_grad():
            bert_out = self.bert(input_ids, attention_mask)
            t_vec = self.mean_pooling(bert_out, attention_mask)
            return F.normalize(self.text_proj(t_vec), p=2, dim=1)

class CachedDataset(Dataset):
    def __init__(self, cache_path):
        print(f"Loading test data from: {cache_path}")
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

def plot_training_loss(loss_history):
    if not loss_history:
        print("No loss history found to plot.")
        return

    plt.figure(figsize=(10, 6))
    
    # Plot raw data
    plt.plot(loss_history, label='Batch Loss', alpha=0.3, color='lightblue')
    
    # Plot moving average if data is sufficient
    if len(loss_history) > 50:
        window = 50
        moving_avg = np.convolve(loss_history, np.ones(window)/window, mode='valid')
        plt.plot(range(window-1, len(loss_history)), moving_avg, label='Moving Avg (50)', color='blue', linewidth=2)
    
    plt.xlabel('Training Steps')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()
# --- EXECUTION ---

def main():
    print("="*60)
    print("STARTING EVALUATION + PLOTTING")
    print("="*60)
    
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file not found at {MODEL_PATH}")
        return

    # 1. Load Checkpoint
    print(f"Loading Model from: {MODEL_PATH}")
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    
    # 2. Plot Loss History
    if 'loss_history' in checkpoint:
        print("Found loss history. Generating plot...")
        plot_training_loss(checkpoint['loss_history'])
    else:
        print("Warning: 'loss_history' key not found in checkpoint.")

    # 3. Initialize Model
    model = NeuralMidiSearch(checkpoint['vocab_size']).to(DEVICE)
    model.load_state_dict(checkpoint['model_state'])
    model.eval()

    # 4. Download Test Data
    try:
        print(f"Downloading {TEST_FILENAME}...")
        cache_path = hf_hub_download(repo_id=HF_REPO_ID, filename=TEST_FILENAME, repo_type="model")
    except Exception as e:
        print(f"Error downloading test set: {e}")
        return

    # 5. Prepare Evaluation
    ds = CachedDataset(cache_path)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    print(f"Evaluating on {len(ds)} examples...")
    text_embs, midi_embs = [], []
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Generating Embeddings"):
            i_ids = batch["input_ids"].to(DEVICE)
            mask = batch["attention_mask"].to(DEVICE)
            m_ids = batch["midi_ids"].to(DEVICE)

            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                t_out = model.encode_text(i_ids, mask)
                m_out = model.encode_midi(m_ids)

            text_embs.append(t_out.float().cpu())
            midi_embs.append(m_out.float().cpu())

    text_tensor = torch.cat(text_embs)
    midi_tensor = torch.cat(midi_embs)
    
    # 6. Metrics
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
    print(f"R@1:       {np.mean(ranks <= 1) * 100:.2f}%")
    print(f"R@5:       {np.mean(ranks <= 5) * 100:.2f}%")
    print(f"R@10:      {np.mean(ranks <= 10) * 100:.2f}%")
    print(f"MRR:       {mrr:.4f}")
    print("="*40)

if __name__ == "__main__":
    main()