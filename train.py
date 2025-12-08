import os
import tarfile
import random
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from miditok import REMI, TokenizerConfig
from huggingface_hub import hf_hub_download
from datasets import load_dataset
from tqdm import tqdm
import numpy as np

import config as cfg
try:
    from model import NeuralMidiSearchTransformer
except ImportError:
    from model import NeuralMidiSearchTransformer

class InfoNCELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, text_features, midi_features, temperature):
        text_features = F.normalize(text_features, p=2, dim=1)
        midi_features = F.normalize(midi_features, p=2, dim=1)

        logits = torch.matmul(text_features, midi_features.T) / temperature
        
        labels = torch.arange(logits.size(0)).to(logits.device)
        
        loss_text = F.cross_entropy(logits, labels)
        loss_midi = F.cross_entropy(logits.T, labels)
        
        return (loss_text + loss_midi) / 2.0

class MidiCapsDataset(Dataset):
    def __init__(self, examples=None, preloaded_data=None, midi_tok=None, text_tok=None, is_train=False):
        self.examples = examples
        self.preloaded_data = preloaded_data
        self.midi_tok = midi_tok
        self.text_tok = text_tok
        self.is_train = is_train

    def __len__(self): 
        if self.preloaded_data:
            return len(self.preloaded_data)
        return len(self.examples)

    def __getitem__(self, idx):
        if self.preloaded_data:
            item = self.preloaded_data[idx]
            ids = item["m"].tolist() if isinstance(item["m"], torch.Tensor) else item["m"]
            
            if "i" in item:
                txt_input = item["i"]
                txt_mask = item["a"]
            else:
                txt = self.text_tok(item["caption"], padding="max_length", truncation=True, max_length=64, return_tensors="pt")
                txt_input = txt["input_ids"].squeeze(0)
                txt_mask = txt["attention_mask"].squeeze(0)
            
            path = item.get("path", "unknown")

        else:
            ex = self.examples[idx]
            try:
                tokens = self.midi_tok(ex["path"])
                ids = tokens.ids if hasattr(tokens, "ids") else tokens[0].ids
            except:
                ids = [0]
            
            path = ex["path"]
            txt = self.text_tok(ex["caption"], padding="max_length", truncation=True, max_length=64, return_tensors="pt")
            txt_input = txt["input_ids"].squeeze(0)
            txt_mask = txt["attention_mask"].squeeze(0)

        seq_len = len(ids)
        if self.is_train and seq_len > cfg.MAX_SEQ_LEN:
            start_idx = random.randint(0, seq_len - cfg.MAX_SEQ_LEN)
            ids = ids[start_idx : start_idx + cfg.MAX_SEQ_LEN]
        else:
            if seq_len < cfg.MAX_SEQ_LEN:
                ids = ids + [0] * (cfg.MAX_SEQ_LEN - seq_len)
            else:
                ids = ids[:cfg.MAX_SEQ_LEN]
            
        return {
            "midi_ids": torch.tensor(ids, dtype=torch.long),
            "input_ids": txt_input,
            "attention_mask": txt_mask,
            "path": path
        }

def main():
    torch.cuda.empty_cache()
    
    # 1. Dataset Selection
    AUGMENTED_FILE = os.path.join(cfg.BASE_DIR, "dataset_midicaps_AUGMENTED.pt")
    NORMAL_FILE = cfg.CACHE_FILE

    dataset = None
    midi_tok = REMI(TokenizerConfig(num_velocities=16, use_chords=True))
    text_tok = AutoTokenizer.from_pretrained(cfg.MODEL_NAME)

    if os.path.exists(AUGMENTED_FILE):
        print(f"Loading Augmented Dataset from {AUGMENTED_FILE}...")
        data = torch.load(AUGMENTED_FILE)
        dataset = MidiCapsDataset(preloaded_data=data, midi_tok=midi_tok, text_tok=text_tok, is_train=True)
    elif os.path.exists(NORMAL_FILE):
        print(f"Loading Standard Dataset from {NORMAL_FILE}...")
        data = torch.load(NORMAL_FILE)
        dataset = MidiCapsDataset(preloaded_data=data, midi_tok=midi_tok, text_tok=text_tok, is_train=True)
    else:
        print("No cache found. Please run prepare_augmentation.py first.")
        return

    loader = DataLoader(
        dataset, batch_size=cfg.BATCH_SIZE, shuffle=True, 
        num_workers=2, pin_memory=True
    )

    # 3. Model & Loss
    model = NeuralMidiSearchTransformer(len(midi_tok)).to(cfg.DEVICE)
    loss_fn = InfoNCELoss()
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.LEARNING_RATE, weight_decay=0.01)

    # 4. Resume Logic
    start_epoch = 0
    loss_history = []
    
    if os.path.exists(cfg.SAVE_FILE):
        print(f"Resuming from checkpoint: {cfg.SAVE_FILE}")
        try:
            checkpoint = torch.load(cfg.SAVE_FILE, map_location=cfg.DEVICE, weights_only=False)
            model.load_state_dict(checkpoint['model_state'])
            if 'epoch' in checkpoint: start_epoch = checkpoint['epoch'] + 1
            if 'loss_history' in checkpoint: loss_history = checkpoint['loss_history']
        except Exception as e:
            print(f"Checkpoint error: {e}. Starting fresh.")

    # 5. Training Loop
    print(f"\nStarting Training (V3.0) on {cfg.DEVICE}...")
    model.train()
    
    total_steps = len(loader) * cfg.EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer, int(0.1*total_steps), total_steps)

    for epoch in range(start_epoch, cfg.EPOCHS):
        loop = tqdm(loader, desc=f"Epoch {epoch+1}/{cfg.EPOCHS}")
        epoch_loss = 0
        
        for batch in loop:
            optimizer.zero_grad()
            i_ids = batch["input_ids"].to(cfg.DEVICE)
            mask = batch["attention_mask"].to(cfg.DEVICE)
            m_ids = batch["midi_ids"].to(cfg.DEVICE)

            t_emb, m_emb = model(i_ids, mask, m_ids)
            loss = loss_fn(t_emb, m_emb, model.temperature)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            
            loss_history.append(loss.item())
            epoch_loss += loss.item()
            loop.set_postfix(loss=loss.item())
        
        print(f"Avg Loss: {epoch_loss/len(loader):.4f}")
        
        torch.save({
            "epoch": epoch,
            "model_state": model.state_dict(),
            "vocab_size": len(midi_tok),
            "loss_history": loss_history
        }, cfg.SAVE_FILE)

    # 6. Indexing (Original Data Only)
    print("Indexing database (Original data only)...")
    model.eval()
    db_embs, db_paths = [], []
    
    # Force load of the standard non-augmented dataset for the search index
    if os.path.exists(NORMAL_FILE):
        print(f"Loading standard cache from {NORMAL_FILE} for indexing.")
        idx_data = torch.load(NORMAL_FILE)
        # is_train=False ensures no random cropping
        idx_dataset = MidiCapsDataset(preloaded_data=idx_data, midi_tok=midi_tok, text_tok=text_tok, is_train=False)
    else:
        # Fallback to current dataset if normal file missing
        print("Warning: Standard cache not found. Indexing current loaded data.")
        idx_dataset = dataset

    idx_loader = DataLoader(idx_dataset, batch_size=cfg.BATCH_SIZE, shuffle=False)
    
    with torch.no_grad():
        for batch in tqdm(idx_loader, desc="Encoding"):
            m_ids = batch["midi_ids"].to(cfg.DEVICE)
            embs = model.encode_midi(m_ids)
            db_embs.append(embs.cpu().numpy())
            if "path" in batch:
                db_paths.extend(batch["path"])
            
    torch.save({
        "epoch": cfg.EPOCHS,
        "model_state": model.state_dict(),
        "vocab_size": len(midi_tok),
        "db_matrix": np.vstack(db_embs),
        "db_paths": db_paths,
        "loss_history": loss_history
    }, cfg.SAVE_FILE)
    
    print(f"Done! Saved to {cfg.SAVE_FILE}")

if __name__ == "__main__":
    main()