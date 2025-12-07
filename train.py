import os
import tarfile
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from miditok import REMI, TokenizerConfig
from huggingface_hub import hf_hub_download
from datasets import load_dataset
from tqdm import tqdm
import numpy as np

# Local imports
import config as cfg
from model import NeuralMidiSearchTransformer

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
            ids = tokens.ids if hasattr(tokens, "ids") else tokens[0].ids
        except:
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
    torch.cuda.empty_cache()
    
    # 1. Data Preparation
    print("Loading Metadata...")
    ds = load_dataset("amaai-lab/MidiCaps", split="train")
    train_ds = ds.filter(lambda ex: not ex["test_set"])
    
    # Download physical MIDI files
    midi_root = Path(cfg.MIDI_DATA_DIR)
    if not midi_root.exists():
        print(" Downloading MIDI archive...")
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
    loader = DataLoader(
        dataset, batch_size=cfg.BATCH_SIZE, shuffle=True, 
        num_workers=2, pin_memory=True
    )

    # 3. Initialize Model
    model = NeuralMidiSearchTransformer(len(midi_tok)).to(cfg.DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.LEARNING_RATE, weight_decay=0.01)

    # 4. Checkpoint & Resume Logic
    start_epoch = 0
    loss_history = []
    
    if os.path.exists(cfg.SAVE_FILE):
        print(f"Found checkpoint: {cfg.SAVE_FILE}")
        try:
            checkpoint = torch.load(cfg.SAVE_FILE, map_location=cfg.DEVICE, weights_only=False)
            
            # Check for collapsed model
            should_restart = False
            if 'loss_history' in checkpoint and len(checkpoint['loss_history']) > 0:
                if checkpoint['loss_history'][-1] > 4.0:
                    should_restart = True
            
            if should_restart:
                print(" Detected collapsed model (High Loss). Restarting from scratch.")
                start_epoch = 0
                loss_history = []
            else:
                model.load_state_dict(checkpoint['model_state'])
                if 'epoch' in checkpoint:
                    start_epoch = checkpoint['epoch'] + 1
                if 'loss_history' in checkpoint:
                    loss_history = checkpoint['loss_history']
                print(f"✅ Resuming from Epoch {start_epoch + 1}...")

        except Exception as e:
            print(f"Error loading checkpoint: {e}. Starting from scratch.")

    # 5. Setup Scheduler
    total_steps = len(loader) * cfg.EPOCHS
    warmup_steps = int(0.1 * total_steps)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    # Skip steps if resuming
    if start_epoch > 0:
        steps_to_skip = start_epoch * len(loader)
        for _ in range(steps_to_skip):
            scheduler.step()

    # 6. Training Loop
    print(f"\n Starting Stabilized Training on {cfg.DEVICE}...")
    model.train()
    
    for epoch in range(start_epoch, cfg.EPOCHS):
        loop = tqdm(loader, desc=f"Epoch {epoch+1}/{cfg.EPOCHS}")
        
        epoch_loss = 0
        for batch in loop:
            optimizer.zero_grad()
            i_ids = batch["input_ids"].to(cfg.DEVICE)
            mask = batch["attention_mask"].to(cfg.DEVICE)
            m_ids = batch["midi_ids"].to(cfg.DEVICE)

            t_emb, m_emb = model(i_ids, mask, m_ids)
            
            # Contrastive Loss
            logits = (t_emb @ m_emb.T) / model.temperature
            labels = torch.arange(logits.shape[0]).to(cfg.DEVICE)
            loss = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2.0
            
            loss.backward()
            
            # --- STABILITY: Gradient Clipping ---
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            scheduler.step()
            
            loss_history.append(loss.item())
            epoch_loss += loss.item()
            loop.set_postfix(loss=loss.item())
        
        print(f"Epoch {epoch+1} Avg Loss: {epoch_loss/len(loader):.4f}")
        
        # Save Checkpoint
        torch.save({
            "epoch": epoch,
            "model_state": model.state_dict(),
            "vocab_size": len(midi_tok),
            "loss_history": loss_history
        }, cfg.SAVE_FILE)

    # 7. Final Indexing
    print("Creating Search Index...")
    model.eval()
    db_embs, db_paths = [], []
    idx_loader = DataLoader(dataset, batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=2)
    
    with torch.no_grad():
        for batch in tqdm(idx_loader, desc="Encoding DB"):
            m_ids = batch["midi_ids"].to(cfg.DEVICE)
            embs = model.encode_midi(m_ids)
            db_embs.append(embs.cpu().numpy())
            db_paths.extend(batch["path"])
            
    # Save Final Model with Index
    torch.save({
        "epoch": cfg.EPOCHS,
        "model_state": model.state_dict(),
        "vocab_size": len(midi_tok),
        "db_matrix": np.vstack(db_embs),
        "db_paths": db_paths,
        "loss_history": loss_history
    }, cfg.SAVE_FILE)
    
    print(f"✅ Training Complete! Saved to {cfg.SAVE_FILE}")

if __name__ == "__main__":
    main()