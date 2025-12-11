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

import config as cfg
from model import NeuralMidiSearchTransformer

# --- DATASET CLASSES ---

class PreTokenizedDataset(Dataset):
    def __init__(self, data):
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Handle short keys from compressed dataset
        m_ids = item.get("m", item.get("midi_ids", [0]))
        i_ids = item.get("i", item.get("input_ids", [0]))
        a_mask = item.get("a", item.get("attention_mask", [0]))

        if not isinstance(m_ids, torch.Tensor): m_ids = torch.tensor(m_ids, dtype=torch.long)
        if not isinstance(i_ids, torch.Tensor): i_ids = torch.tensor(i_ids, dtype=torch.long)
        if not isinstance(a_mask, torch.Tensor): a_mask = torch.tensor(a_mask, dtype=torch.long)

        return {
            "midi_ids": m_ids,
            "input_ids": i_ids,
            "attention_mask": a_mask,
            "path": item.get("path", "unknown")
        }

class MidiCapsDataset(Dataset):
    def __init__(self, examples, midi_tok, text_tok):
        self.examples = examples
        self.midi_tok = midi_tok
        self.text_tok = text_tok

    def __len__(self): return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        try:
            tokens = self.midi_tok(ex["path"])
            ids = tokens.ids if hasattr(tokens, "ids") else tokens[0].ids
        except: ids = [0] 

        if len(ids) < cfg.MAX_SEQ_LEN: ids = ids + [0] * (cfg.MAX_SEQ_LEN - len(ids))
        else: ids = ids[:cfg.MAX_SEQ_LEN]
            
        midi_ids = torch.tensor(ids, dtype=torch.long)
        txt = self.text_tok(ex["caption"], padding="max_length", truncation=True, max_length=64, return_tensors="pt")
        
        return {
            "midi_ids": midi_ids,
            "input_ids": txt["input_ids"].squeeze(0),
            "attention_mask": txt["attention_mask"].squeeze(0),
            "path": ex["path"]
        }

def main():
    torch.cuda.empty_cache()
    
    # 1. Data Preparation
    dataset = None
    print("Checking for pre-tokenized dataset on Hugging Face...")
    
    try:
        file_path = hf_hub_download(repo_id="GiuseppeStilly/MIDI-Retrieval", filename="dataset_midicaps_tokenized.pt", repo_type="model")
        print(f"Found pre-tokenized data at: {file_path}")
        preloaded_data = torch.load(file_path, map_location="cpu")
        dataset = PreTokenizedDataset(preloaded_data)
        print(f"Loaded {len(dataset)} examples directly.")
    except Exception as e:
        print(f"Pre-tokenized download failed ({e}). Falling back to raw files.")
        dataset = None

    # Fallback to raw processing if download fails
    midi_tok = REMI(TokenizerConfig(num_velocities=16, use_chords=True))
    
    if dataset is None:
        print("Loading Metadata from HuggingFace...")
        ds = load_dataset("amaai-lab/MidiCaps", split="train")
        train_ds = ds.filter(lambda ex: not ex["test_set"])
        
        midi_root = Path(cfg.MIDI_DATA_DIR)
        if not midi_root.exists():
            print("Downloading MIDI archive...")
            path = hf_hub_download(repo_id="amaai-lab/MidiCaps", filename="midicaps.tar.gz", repo_type="dataset")
            with tarfile.open(path) as tar: tar.extractall(midi_root)
        
        examples = []
        for item in tqdm(train_ds, desc="Indexing files"):
            p = midi_root / item["location"]
            if p.exists(): examples.append({"path": str(p), "caption": item["caption"]})

        text_tok = AutoTokenizer.from_pretrained(cfg.MODEL_NAME)
        dataset = MidiCapsDataset(examples, midi_tok, text_tok)

    loader = DataLoader(dataset, batch_size=cfg.BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)

    # 2. Model Initialization (V2)
    print(f"Initializing V2 Model (Transformer) on {cfg.DEVICE}...")
    model = NeuralMidiSearchTransformer(len(midi_tok)).to(cfg.DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.LEARNING_RATE, weight_decay=0.01)

    # 3. Resume Logic
    start_epoch = 0
    loss_history = []
    
    if os.path.exists(cfg.SAVE_FILE):
        print(f"Found checkpoint: {cfg.SAVE_FILE}")
        try:
            checkpoint = torch.load(cfg.SAVE_FILE, map_location=cfg.DEVICE, weights_only=False)
            
            # Safety check for loss explosion
            if 'loss_history' in checkpoint and len(checkpoint['loss_history']) > 0:
                if checkpoint['loss_history'][-1] > 6.0:
                    print("Detected collapsed model. Restarting.")
                    start_epoch = 0
                    loss_history = []
                else:
                    model.load_state_dict(checkpoint['model_state'])
                    start_epoch = checkpoint.get('epoch', 0) + 1
                    loss_history = checkpoint.get('loss_history', [])
                    print(f"Resuming from Epoch {start_epoch + 1}...")
            else:
                model.load_state_dict(checkpoint['model_state'])
                start_epoch = checkpoint.get('epoch', 0) + 1
                loss_history = checkpoint.get('loss_history', [])

        except Exception as e:
            print(f"Error loading checkpoint: {e}. Starting fresh.")

    # 4. Scheduler
    total_steps = len(loader) * cfg.EPOCHS
    warmup_steps = int(0.1 * total_steps)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    if start_epoch > 0:
        steps_to_skip = start_epoch * len(loader)
        for _ in range(steps_to_skip): scheduler.step()

    # 5. Training Loop
    print(f"Starting Training V2...")
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
            
            logits = (t_emb @ m_emb.T) / model.temperature
            labels = torch.arange(logits.shape[0]).to(cfg.DEVICE)
            loss = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2.0
            
            loss.backward()
            
            # Gradient Clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            scheduler.step()
            
            loss_history.append(loss.item())
            epoch_loss += loss.item()
            loop.set_postfix(loss=loss.item())
        
        avg_loss = epoch_loss / len(loader)
        print(f"Epoch {epoch+1} Avg Loss: {avg_loss:.4f}")
        
        # Save Checkpoint
        torch.save({
            "epoch": epoch,
            "model_state": model.state_dict(),
            "vocab_size": len(midi_tok),
            "loss_history": loss_history,
            "config": "V2_Transformer"
        }, cfg.SAVE_FILE)
        print(f"Saved checkpoint to {cfg.SAVE_FILE}")

    print("Training Complete.")

if __name__ == "__main__":
    main()