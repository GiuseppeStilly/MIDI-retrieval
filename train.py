import os
import tarfile
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from miditok import REMI, TokenizerConfig
from huggingface_hub import hf_hub_download
from datasets import load_dataset
from tqdm import tqdm
import numpy as np

# Local imports
import config as cfg
from model import NeuralMidiSearch

# --- DATASET CLASS ---
class MidiCapsDataset(Dataset):
    def __init__(self, examples, midi_tok, text_tok):
        self.examples = examples
        self.midi_tok = midi_tok
        self.text_tok = text_tok

    def __len__(self): 
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        
        # Tokenize MIDI
        try:
            tokens = self.midi_tok(ex["path"])
            # Handle different MidiTok versions return types
            ids = tokens.ids if hasattr(tokens, "ids") else tokens[0].ids
        except:
            # Fallback for corrupted files
            ids = [0] 

        # Manual Padding / Truncation
        if len(ids) < cfg.MAX_SEQ_LEN:
            ids = ids + [0] * (cfg.MAX_SEQ_LEN - len(ids))
        else:
            ids = ids[:cfg.MAX_SEQ_LEN]
            
        midi_ids = torch.tensor(ids, dtype=torch.long)

        # Tokenize Text
        txt = self.text_tok(ex["caption"], padding="max_length", truncation=True, 
                            max_length=64, return_tensors="pt")
        
        return {
            "midi_ids": midi_ids,
            "input_ids": txt["input_ids"].squeeze(0),
            "attention_mask": txt["attention_mask"].squeeze(0),
            "path": ex["path"]
        }

def main():
    # 1. Data Preparation
    print("📥 Loading Metadata...")
    ds = load_dataset("amaai-lab/MidiCaps", split="train")
    train_ds = ds.filter(lambda ex: not ex["test_set"])
    
    # Download physical MIDI files
    midi_root = Path(cfg.MIDI_DATA_DIR)
    if not midi_root.exists():
        print("⬇️ Downloading MIDI archive...")
        path = hf_hub_download(repo_id="amaai-lab/MidiCaps", filename="midicaps.tar.gz", repo_type="dataset")
        with tarfile.open(path) as tar: 
            tar.extractall(midi_root)
    
    # Build valid example list
    examples = []
    for item in tqdm(train_ds, desc="Indexing files"):
        p = midi_root / item["location"]
        if p.exists(): 
            examples.append({"path": str(p), "caption": item["caption"]})

    # 2. Initialize Tokenizers
    midi_tok = REMI(TokenizerConfig(num_velocities=16, use_chords=True))
    text_tok = AutoTokenizer.from_pretrained(cfg.MODEL_NAME)
    
    dataset = MidiCapsDataset(examples, midi_tok, text_tok)
    # num_workers > 0 speeds up data loading
    loader = DataLoader(dataset, batch_size=cfg.BATCH_SIZE, shuffle=True, num_workers=2)

    # 3. Initialize Model
    model = NeuralMidiSearch(len(midi_tok)).to(cfg.DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.LEARNING_RATE)

    # 4. Training Loop
    print(f"🚀 Starting training on {cfg.DEVICE}...")
    model.train()
    for epoch in range(cfg.EPOCHS):
        total_loss = 0
        loop = tqdm(loader, desc=f"Epoch {epoch+1}/{cfg.EPOCHS}")
        
        for batch in loop:
            optimizer.zero_grad()
            i_ids = batch["input_ids"].to(cfg.DEVICE)
            mask = batch["attention_mask"].to(cfg.DEVICE)
            m_ids = batch["midi_ids"].to(cfg.DEVICE)

            t_emb, m_emb = model(i_ids, mask, m_ids)
            
            # --- Contrastive Loss Calculation ---
            # Calculate similarity matrix
            logits = (t_emb @ m_emb.T) / model.temperature
            labels = torch.arange(logits.shape[0]).to(cfg.DEVICE)
            
            # Symmetric loss (Text->Midi & Midi->Text)
            loss = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2.0
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            loop.set_postfix(loss=loss.item())
        
        print(f"Epoch {epoch+1} Avg Loss: {total_loss/len(loader):.4f}")

    # 5. Final Indexing (Create Vector Database)
    print("🗂️ Creating Search Index...")
    model.eval()
    db_embs, db_paths = [], []
    idx_loader = DataLoader(dataset, batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=2)
    
    with torch.no_grad():
        for batch in tqdm(idx_loader, desc="Encoding DB"):
            m_ids = batch["midi_ids"].to(cfg.DEVICE)
            embs = model.encode_midi(m_ids)
            db_embs.append(embs.cpu().numpy())
            db_paths.extend(batch["path"])
            
    # 6. Save Everything
    torch.save({
        "model_state": model.state_dict(),
        "vocab_size": len(midi_tok),
        "db_matrix": np.vstack(db_embs),
        "db_paths": db_paths
    }, cfg.SAVE_FILE)
    
    print(f"✅ Model and Index saved to {cfg.SAVE_FILE}")

if __name__ == "__main__":
    main()