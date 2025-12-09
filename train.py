import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from miditok import REMI, TokenizerConfig
from tqdm import tqdm
import numpy as np
import config as cfg

try:
    from model import NeuralMidiSearchTransformer
except ImportError:
    from model import NeuralMidiSearchTransformer

# --- LOSS FUNCTION ---

class MarginRankingLoss(nn.Module):
    """
    Main Loss for Text-to-MIDI Retrieval.
    Objective: Sim(Text, CorrectMusic) > Sim(Text, WrongMusic) + Margin
    """
    def __init__(self, margin=0.3):
        super().__init__()
        self.margin = margin

    def forward(self, text_features, midi_features):
        """
        Computes ranking loss.
        text_features: [batch_size, embed_dim]
        midi_features: [batch_size, embed_dim]
        """
        # Compute similarity matrix [Batch x Batch]
        sim_matrix = torch.matmul(text_features, midi_features.T)
        
        # Diagonal contains the correct pairs (Positives)
        pos_sim = torch.diag(sim_matrix)
        
        batch_size = sim_matrix.size(0)
        loss = 0.0
        
        for i in range(batch_size):
            # For each item, take all similarities EXCEPT the correct one (Negatives)
            # sim_matrix[i] is the i-th row. We remove the i-th element.
            neg_sims = torch.cat([sim_matrix[i, :i], sim_matrix[i, i+1:]])
            
            # Loss = max(0, margin - (Positive - Negative))
            # We want Positive similarity to be higher than Negative by at least 'margin'
            loss += torch.sum(torch.clamp(self.margin - (pos_sim[i] - neg_sims), min=0.0))
        
        # Average over batch
        return loss / batch_size

# --- DATASET ---

class MidiCapsDataset(Dataset):
    """
    Standard Dataset: Loads (Caption, MIDI) pairs.
    Includes Random Crop for Data Augmentation to prevent overfitting.
    """
    def __init__(self, preloaded_data, midi_tok, text_tok, is_train=False):
        self.data = preloaded_data
        self.midi_tok = midi_tok
        self.text_tok = text_tok
        self.is_train = is_train

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        
        # 1. MIDI Handling
        ids = item["m"].tolist() if isinstance(item["m"], torch.Tensor) else item["m"]
        
        # Random Crop (Essential Data Augmentation)
        seq_len = len(ids)
        if self.is_train and seq_len > cfg.MAX_SEQ_LEN:
            # Pick a random segment of the song (not always the start)
            start_idx = np.random.randint(0, seq_len - cfg.MAX_SEQ_LEN)
            ids = ids[start_idx : start_idx + cfg.MAX_SEQ_LEN]
        else:
            # Padding or standard truncation
            if seq_len < cfg.MAX_SEQ_LEN:
                ids = ids + [0] * (cfg.MAX_SEQ_LEN - seq_len)
            else:
                ids = ids[:cfg.MAX_SEQ_LEN]

        # 2. Text Handling
        if "i" in item:
            # If already tokenized in cache file
            txt_input = item["i"]
            txt_mask = item["a"]
        else:
            # On-the-fly tokenization (fallback)
            txt = self.text_tok(item["caption"], padding="max_length", truncation=True, max_length=64, return_tensors="pt")
            txt_input = txt["input_ids"].squeeze(0)
            txt_mask = txt["attention_mask"].squeeze(0)
            
        return {
            "midi_ids": torch.tensor(ids, dtype=torch.long),
            "input_ids": txt_input,
            "attention_mask": txt_mask,
            "path": item.get("path", "unknown")
        }

# --- TRAINING LOOP ---

def train_text_to_midi(model, train_loader, optimizer, scheduler, scaler):
    print("\n" + "="*60)
    print("📝 STARTING TRAINING: Text-to-MIDI Alignment")
    print("="*60)
    
    loss_fn = MarginRankingLoss(margin=cfg.MARGIN)
    model.train()
    
    total_epochs = cfg.EPOCHS 
    
    for epoch in range(total_epochs):
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{total_epochs}")
        epoch_loss = 0
        
        for batch in loop:
            # Move data to GPU
            i_ids = batch["input_ids"].to(cfg.DEVICE)
            mask = batch["attention_mask"].to(cfg.DEVICE)
            m_ids = batch["midi_ids"].to(cfg.DEVICE)
            
            optimizer.zero_grad()
            
            # Mixed Precision Training
            with torch.amp.autocast('cuda'):
                t_emb, m_emb = model(i_ids, mask, m_ids)
                loss = loss_fn(t_emb, m_emb)
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            
            epoch_loss += loss.item()
            loop.set_postfix(loss=loss.item())
        
        # Log End of Epoch
        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch+1} Avg Loss: {avg_loss:.4f}")
        
        # Save Checkpoint
        torch.save({
            "epoch": epoch,
            "model_state": model.state_dict(),
            "config": "V4.0_Student_Direct",
            "loss": avg_loss
        }, cfg.SAVE_FILE)

def main():
    torch.cuda.empty_cache()
    
    # 1. Load Data
    DATA_FILE = cfg.CACHE_FILE
    if not os.path.exists(DATA_FILE):
        print(f"Dataset file not found at {DATA_FILE}!")
        return
        
    print(f"📂 Loading dataset from {DATA_FILE}...")
    data = torch.load(DATA_FILE)
    
    # 2. Initialize Tokenizers
    midi_tok = REMI(TokenizerConfig(num_velocities=16, use_chords=True))
    text_tok = AutoTokenizer.from_pretrained(cfg.MODEL_NAME)
    
    # 3. Initialize Model
    print(f"Initializing V4.0 Model (Layers={cfg.NUM_LAYERS}, Dim={cfg.EMBED_DIM})...")
    model = NeuralMidiSearchTransformer(len(midi_tok)).to(cfg.DEVICE)
    
    # 4. Optimizer & Scaler
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.LEARNING_RATE, weight_decay=0.01)
    scaler = torch.cuda.amp.GradScaler()
    
    # 5. Prepare DataLoader
    dataset = MidiCapsDataset(preloaded_data=data, midi_tok=midi_tok, text_tok=text_tok, is_train=True)
    loader = DataLoader(dataset, batch_size=cfg.BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    
    total_steps = len(loader) * cfg.EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer, int(0.1*total_steps), total_steps)
    
    # 6. RUN TRAINING
    train_text_to_midi(model, loader, optimizer, scheduler, scaler)
    
    # 7. FINAL INDEXING (Create the database for search)
    print("\nIndexing database...")
    model.eval()
    db_embs, db_paths = [], []
    
    idx_loader = DataLoader(dataset, batch_size=cfg.BATCH_SIZE, shuffle=False)
    with torch.no_grad():
        for batch in tqdm(idx_loader, desc="Encoding DB"):
            m_ids = batch["midi_ids"].to(cfg.DEVICE)
            with torch.amp.autocast('cuda'):
                embs = model.encode_midi(m_ids)
            db_embs.append(embs.float().cpu().numpy())
            db_paths.extend(batch["path"])
    
    # Save Model + Index
    torch.save({
        "model_state": model.state_dict(),
        "vocab_size": len(midi_tok),
        "db_matrix": np.vstack(db_embs),
        "db_paths": db_paths,
    }, cfg.SAVE_FILE)
    
    print(f"DONE! Model and Search Index saved to {cfg.SAVE_FILE}")

if __name__ == "__main__":
    main()