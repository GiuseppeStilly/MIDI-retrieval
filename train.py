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
import config as cfg

try:
    from model import NeuralMidiSearch
except ImportError:
    print("Warning: 'model.py' not found.")

class MidiCapsDataset(Dataset):
    def __init__(self, examples, midi_tok, text_tok):
        self.examples = examples
        self.midi_tok = midi_tok
        self.text_tok = text_tok

    def __len__(self): 
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        
        try:
            tokens = self.midi_tok(ex["path"])
            ids = tokens.ids if hasattr(tokens, "ids") else tokens[0].ids
        except:
            ids = [0] 

        if len(ids) < cfg.MAX_SEQ_LEN:
            ids = ids + [0] * (cfg.MAX_SEQ_LEN - len(ids))
        else:
            ids = ids[:cfg.MAX_SEQ_LEN]
            
        midi_ids = torch.tensor(ids, dtype=torch.long)

        txt = self.text_tok(ex["caption"], padding="max_length", truncation=True, 
                            max_length=64, return_tensors="pt")
        
        return {
            "midi_ids": midi_ids,
            "input_ids": txt["input_ids"].squeeze(0),
            "attention_mask": txt["attention_mask"].squeeze(0),
            "path": ex["path"]
        }

class PreTokenizedDataset(Dataset):
    def __init__(self, data):
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Handle compressed keys from .pt file ('m', 'i', 'a')
        m_ids = item.get("m", item.get("midi_ids", [0]))
        i_ids = item.get("i", item.get("input_ids", [0]))
        a_mask = item.get("a", item.get("attention_mask", [0]))

        if not isinstance(m_ids, torch.Tensor):
            m_ids = torch.tensor(m_ids, dtype=torch.long)
        if not isinstance(i_ids, torch.Tensor):
            i_ids = torch.tensor(i_ids, dtype=torch.long)
        if not isinstance(a_mask, torch.Tensor):
            a_mask = torch.tensor(a_mask, dtype=torch.long)

        return {
            "midi_ids": m_ids,
            "input_ids": i_ids,
            "attention_mask": a_mask,
            "path": item.get("path", "unknown")
        }

def main():
    dataset = None
    midi_tok = REMI(TokenizerConfig(num_velocities=16, use_chords=True))
    
    print("Checking for pre-tokenized dataset on Hugging Face...")
    try:
        file_path = hf_hub_download(
            repo_id="GiuseppeStilly/MIDI-Retrieval", 
            filename="test_midicaps_tokenized.pt", 
            repo_type="model" 
        )
        print(f"Found pre-tokenized data at: {file_path}")
        preloaded_data = torch.load(file_path, map_location="cpu")
        dataset = PreTokenizedDataset(preloaded_data)
        print(f"Loaded {len(dataset)} examples.")
        
    except Exception as e:
        print(f"Pre-tokenized download failed ({e}). Using standard processing.")
        dataset = None

    if dataset is None:
        print("Loading Metadata from HuggingFace...")
        ds = load_dataset("amaai-lab/MidiCaps", split="train")
        train_ds = ds.filter(lambda ex: not ex["test_set"])
        
        midi_root = Path(cfg.MIDI_DATA_DIR)
        if not midi_root.exists():
            print("Downloading MIDI archive...")
            path = hf_hub_download(repo_id="amaai-lab/MidiCaps", filename="midicaps.tar.gz", repo_type="dataset")
            with tarfile.open(path) as tar: 
                tar.extractall(midi_root)
        
        examples = []
        for item in tqdm(train_ds, desc="Indexing files"):
            p = midi_root / item["location"]
            if p.exists(): 
                examples.append({"path": str(p), "caption": item["caption"]})

        text_tok = AutoTokenizer.from_pretrained(cfg.MODEL_NAME)
        dataset = MidiCapsDataset(examples, midi_tok, text_tok)

    loader = DataLoader(dataset, batch_size=cfg.BATCH_SIZE, shuffle=True, num_workers=0)

    print(f"Initializing Model on {cfg.DEVICE}...")
    model = NeuralMidiSearch(len(midi_tok)).to(cfg.DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.LEARNING_RATE)
    
    loss_history = []

    print(f"Starting training...")
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
            
            logits = (t_emb @ m_emb.T) / model.temperature
            labels = torch.arange(logits.shape[0]).to(cfg.DEVICE)
            
            loss = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2.0
            
            loss.backward()
            optimizer.step()
            
            current_loss = loss.item()
            loss_history.append(current_loss)
            total_loss += current_loss
            
            loop.set_postfix(loss=current_loss)
        
        print(f"Epoch {epoch+1} Avg Loss: {total_loss/len(loader):.4f}")

    print("Saving model and history...")
    torch.save({
        "model_state": model.state_dict(),
        "vocab_size": len(midi_tok),
        "loss_history": loss_history,
        "config": "V1_Optimized"
    }, cfg.SAVE_FILE)
    
    print(f"Saved to {cfg.SAVE_FILE}")

if __name__ == "__main__":
    main()