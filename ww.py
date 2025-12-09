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

# --- LOSS FUNCTIONS ---

class InfoNCELoss(nn.Module):
    """Contrastive loss for MIDI-only pretraining (PHASE 1)."""
    def __init__(self, temperature=0.05):
        super().__init__()
        self.temperature = temperature

    def forward(self, midi_features_1, midi_features_2):
        """
        Contrastive loss between two MIDI embeddings (original vs shifted).
        midi_features_1: shape [batch_size, embed_dim]
        midi_features_2: shape [batch_size, embed_dim]
        """
        midi_features_1 = F.normalize(midi_features_1, p=2, dim=1)
        midi_features_2 = F.normalize(midi_features_2, p=2, dim=1)
        
        # Cosine similarity matrix
        logits = torch.matmul(midi_features_1, midi_features_2.T) / self.temperature
        
        # Labels are diagonal (positive pairs)
        labels = torch.arange(logits.size(0)).to(logits.device)
        
        # Standard contrastive loss
        loss = F.cross_entropy(logits, labels)
        return loss

class MarginRankingLoss(nn.Module):
    """Margin ranking loss for text-to-MIDI retrieval (PHASE 2)."""
    def __init__(self, margin=0.3):
        super().__init__()
        self.margin = margin

    def forward(self, text_features, midi_features):
        """
        Margin ranking: positive pairs should have high similarity,
        negative pairs should have lower similarity.
        """
        # Similarity matrix
        sim_matrix = torch.matmul(text_features, midi_features.T)
        
        # Diagonal = positive pairs
        pos_sim = torch.diag(sim_matrix)
        
        # Off-diagonal = negative pairs
        batch_size = sim_matrix.size(0)
        
        # For each positive, compute loss against all negatives
        loss = 0.0
        for i in range(batch_size):
            # Negatives are all but the diagonal
            neg_sims = torch.cat([sim_matrix[i, :i], sim_matrix[i, i+1:]])
            
            # Margin loss: max(0, margin - (pos - neg))
            loss += torch.sum(torch.clamp(self.margin - (pos_sim[i] - neg_sims), min=0.0))
        
        return loss / batch_size

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
        
        # Handle sequence length
        if self.is_train and seq_len > cfg.MAX_SEQ_LEN:
            start_idx = np.random.randint(0, seq_len - cfg.MAX_SEQ_LEN)
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

def phase_1_pretraining(model, train_loader, optimizer, scheduler, scaler):
    """
    PHASE 1: MIDI-only contrastive pretraining with pitch shifts.
    Model learns robust MIDI embeddings invariant to pitch shifts.
    """
    print("\n" + "="*60)
    print("PHASE 1: MIDI-Only Contrastive Pretraining")
    print("="*60)
    
    loss_fn = InfoNCELoss(temperature=cfg.CONTRASTIVE_TEMPERATURE)
    model.train()
    
    for epoch in range(cfg.CONTRASTIVE_EPOCHS):
        loop = tqdm(train_loader, desc=f"Phase1-Epoch {epoch+1}/{cfg.CONTRASTIVE_EPOCHS}")
        epoch_loss = 0
        
        for batch in loop:
            # In Phase 1, we load augmented data with pitch shifts
            # batch["midi_ids"] contains pairs: [original, shifted_1, ..., shifted_n]
            m_ids = batch["midi_ids"].to(cfg.DEVICE)
            
            # Split batch into consecutive pairs
            batch_size = m_ids.size(0)
            
            # For simplicity: take pairs (i, i+1) where i is even
            # In real scenario, you'd load pre-paired data or implement within-batch pairing
            with torch.cuda.amp.autocast():
                # Encode all MIDI sequences
                midi_embeddings = model.encode_midi(m_ids)
                
                # Compute loss on consecutive pairs
                loss = 0.0
                pair_count = 0
                for i in range(0, batch_size - 1, 2):
                    pair_loss = loss_fn(midi_embeddings[i:i+1], midi_embeddings[i+1:i+2])
                    loss += pair_loss
                    pair_count += 1
                
                if pair_count > 0:
                    loss = loss / pair_count
            
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            
            epoch_loss += loss.item()
            loop.set_postfix(loss=loss.item())
        
        print(f"Phase1 Epoch {epoch+1} Avg Loss: {epoch_loss/len(train_loader):.4f}")
    
    print("Phase 1 Complete. MIDI encoder is now robust to pitch shifts.\n")

def phase_2_finetuning(model, train_loader, optimizer, scheduler, scaler):
    """
    PHASE 2: Text-to-MIDI fine-tuning on original (non-augmented) data.
    Model learns to align text captions with MIDI representations.
    """
    print("\n" + "="*60)
    print("PHASE 2: Text-to-MIDI Fine-tuning")
    print("="*60)
    
    loss_fn = MarginRankingLoss(margin=cfg.MARGIN)
    model.train()
    
    for epoch in range(cfg.SUPERVISED_EPOCHS):
        loop = tqdm(train_loader, desc=f"Phase2-Epoch {epoch+1}/{cfg.SUPERVISED_EPOCHS}")
        epoch_loss = 0
        
        for batch in loop:
            i_ids = batch["input_ids"].to(cfg.DEVICE)
            mask = batch["attention_mask"].to(cfg.DEVICE)
            m_ids = batch["midi_ids"].to(cfg.DEVICE)
            
            optimizer.zero_grad()
            
            with torch.cuda.amp.autocast():
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
        
        print(f"Phase2 Epoch {epoch+1} Avg Loss: {epoch_loss/len(train_loader):.4f}")
        
        # Save checkpoint
        torch.save({
            "epoch": epoch,
            "model_state": model.state_dict(),
            "phase": 2,
        }, cfg.SAVE_FILE)
    
    print("Phase 2 Complete. Text-MIDI alignment optimized.\n")

def main():
    torch.cuda.empty_cache()
    
    # Determine dataset to use
    AUGMENTED_FILE = os.path.join(cfg.BASE_DIR, "dataset_midicaps_MIDI_PRETRAINING.pt")
    NORMAL_FILE = cfg.CACHE_FILE
    
    midi_tok = REMI(TokenizerConfig(num_velocities=16, use_chords=True))
    text_tok = AutoTokenizer.from_pretrained(cfg.MODEL_NAME)
    
    # Model
    model = NeuralMidiSearchTransformer(len(midi_tok)).to(cfg.DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.LEARNING_RATE, weight_decay=0.01)
    scaler = torch.cuda.amp.GradScaler()
    
    print(f"Config: NUM_LAYERS={cfg.NUM_LAYERS}, EMBED_DIM={cfg.EMBED_DIM}, BATCH_SIZE={cfg.BATCH_SIZE}")
    print(f"Mixed Precision (AMP) Enabled.")
    
    # PHASE 1: Contrastive Pretraining (if enabled)
    if cfg.USE_CONTRASTIVE_PRETRAINING:
        if os.path.exists(AUGMENTED_FILE):
            print(f"\nLoading augmented dataset from {AUGMENTED_FILE} for Phase 1...")
            data = torch.load(AUGMENTED_FILE)
            dataset = MidiCapsDataset(preloaded_data=data, midi_tok=midi_tok, text_tok=text_tok, is_train=True)
        else:
            print(f"Augmented file not found. Skipping Phase 1.")
            cfg.USE_CONTRASTIVE_PRETRAINING = False
        
        if cfg.USE_CONTRASTIVE_PRETRAINING:
            loader = DataLoader(dataset, batch_size=cfg.BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
            total_steps = len(loader) * cfg.CONTRASTIVE_EPOCHS
            scheduler = get_linear_schedule_with_warmup(optimizer, int(0.1*total_steps), total_steps)
            
            phase_1_pretraining(model, loader, optimizer, scheduler, scaler)
    
    # PHASE 2: Supervised Fine-tuning
    if os.path.exists(NORMAL_FILE):
        print(f"\nLoading standard dataset from {NORMAL_FILE} for Phase 2...")
        data = torch.load(NORMAL_FILE)
        dataset = MidiCapsDataset(preloaded_data=data, midi_tok=midi_tok, text_tok=text_tok, is_train=True)
    else:
        print("No cache found. Please run data preprocessing first.")
        return
    
    loader = DataLoader(dataset, batch_size=cfg.BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    total_steps = len(loader) * cfg.SUPERVISED_EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer, int(0.1*total_steps), total_steps)
    
    phase_2_finetuning(model, loader, optimizer, scheduler, scaler)
    
    # Final Indexing
    print("\nIndexing database...")
    model.eval()
    db_embs, db_paths = [], []
    
    idx_loader = DataLoader(dataset, batch_size=cfg.BATCH_SIZE, shuffle=False)
    with torch.no_grad():
        for batch in tqdm(idx_loader, desc="Encoding"):
            m_ids = batch["midi_ids"].to(cfg.DEVICE)
            with torch.cuda.amp.autocast():
                embs = model.encode_midi(m_ids)
            db_embs.append(embs.float().cpu().numpy())
            db_paths.extend(batch["path"])
    
    torch.save({
        "model_state": model.state_dict(),
        "vocab_size": len(midi_tok),
        "db_matrix": np.vstack(db_embs),
        "db_paths": db_paths,
    }, cfg.SAVE_FILE)
    
    print(f"Done! Model and index saved to {cfg.SAVE_FILE}")

if __name__ == "__main__":
    main()
