import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from transformers import AutoModel

# --- CONFIGURATION CONSTANTS (V2) ---
V2_EMBED_DIM = 512
V2_MIDI_EMBED_DIM = 256
V2_FF_DIM = 1024
V2_NUM_HEADS = 4
V2_NUM_LAYERS = 4
V2_DROPOUT = 0.1

# ARCHITECTURE V1 (LSTM + MiniLM)
class NeuralMidiSearch_V1(nn.Module):
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
        token = model_output.last_hidden_state
        mask = attention_mask.unsqueeze(-1).expand(token.size()).float()
        return torch.sum(token * mask, 1) / torch.clamp(mask.sum(1), min=1e-9)

    def encode_midi(self, midi_ids):
        with torch.no_grad():
            x = self.midi_emb(midi_ids)
            x, _ = self.lstm(x)
            vec = torch.mean(x, dim=1)
            return F.normalize(self.midi_proj(vec), p=2, dim=1)

    def encode_text(self, input_ids, attention_mask):
        with torch.no_grad():
            out = self.bert(input_ids, attention_mask)
            vec = self.mean_pooling(out, attention_mask)
            return F.normalize(self.text_proj(vec), p=2, dim=1)

# ARCHITECTURE V2 (Transformer + MPNet)
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div)
        pe[:, 1::2] = torch.cos(position * div)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class NeuralMidiSearch_V2(nn.Module):
    def __init__(self, midi_vocab_size):
        super().__init__()
        
        # Text Encoder (MPNet)
        self.bert = AutoModel.from_pretrained("sentence-transformers/all-mpnet-base-v2")
        self.text_proj = nn.Linear(768, V2_EMBED_DIM) 

        # MIDI Encoder
        self.midi_emb = nn.Embedding(midi_vocab_size, V2_MIDI_EMBED_DIM)
        self.pos_encoder = PositionalEncoding(d_model=V2_MIDI_EMBED_DIM, dropout=V2_DROPOUT)
        
        layer = nn.TransformerEncoderLayer(
            d_model=V2_MIDI_EMBED_DIM, 
            nhead=V2_NUM_HEADS, 
            dim_feedforward=V2_FF_DIM, 
            dropout=V2_DROPOUT,
            batch_first=True,
            activation="gelu"
        )
        self.midi_transformer = nn.TransformerEncoder(layer, num_layers=V2_NUM_LAYERS)
        self.midi_proj = nn.Linear(V2_MIDI_EMBED_DIM, V2_EMBED_DIM)
        self.temperature = nn.Parameter(torch.tensor(0.07))

    def mean_pooling(self, model_output, attention_mask):
        token = model_output.last_hidden_state
        mask = attention_mask.unsqueeze(-1).expand(token.size()).float()
        return torch.sum(token * mask, 1) / torch.clamp(mask.sum(1), min=1e-9)

    def encode_midi(self, midi_ids):
        with torch.no_grad():
            # CRITICAL FIX: Scaling by sqrt(d_model) before positional encoding
            x = self.midi_emb(midi_ids) * math.sqrt(V2_MIDI_EMBED_DIM)
            
            x = self.pos_encoder(x)
            x = self.midi_transformer(x)
            vec = torch.mean(x, dim=1)
            return F.normalize(self.midi_proj(vec), p=2, dim=1)

    def encode_text(self, input_ids, attention_mask):
        with torch.no_grad():
            out = self.bert(input_ids, attention_mask)
            vec = self.mean_pooling(out, attention_mask)
            return F.normalize(self.text_proj(vec), p=2, dim=1)