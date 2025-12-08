import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
import math
import config as cfg

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class NeuralMidiSearchTransformer(nn.Module):
    def __init__(self, midi_vocab_size):
        super().__init__()
        
        # Text Encoder
        self.bert = AutoModel.from_pretrained(cfg.MODEL_NAME)
        self.text_hidden_size = self.bert.config.hidden_size
        self.text_proj = nn.Linear(self.text_hidden_size, cfg.EMBED_DIM)

        # MIDI Encoder
        # Added padding_idx=0 to ensure zero token remains zero vector
        self.midi_emb = nn.Embedding(midi_vocab_size, cfg.MIDI_EMBED_DIM, padding_idx=0)
        
        # Positional Encoding
        self.pos_encoder = PositionalEncoding(d_model=cfg.MIDI_EMBED_DIM, dropout=cfg.DROPOUT)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.MIDI_EMBED_DIM, 
            nhead=cfg.NUM_HEADS, 
            dim_feedforward=cfg.FF_DIM, 
            dropout=cfg.DROPOUT, 
            batch_first=True, 
            activation="gelu"
        )
        self.midi_transformer = nn.TransformerEncoder(encoder_layer, num_layers=cfg.NUM_LAYERS)
        
        self.midi_proj = nn.Linear(cfg.MIDI_EMBED_DIM, cfg.EMBED_DIM)
        
        self.temperature = nn.Parameter(torch.tensor(0.07))

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )

    def midi_mean_pooling(self, midi_output, padding_mask):
        # Create mask: 1 for valid tokens, 0 for padding
        mask_expanded = (~padding_mask).unsqueeze(-1).float()
        
        sum_embeddings = torch.sum(midi_output * mask_expanded, 1)
        sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    def forward(self, input_ids, attention_mask, midi_ids):
        # Text
        bert_out = self.bert(input_ids, attention_mask)
        t_vec = self.mean_pooling(bert_out, attention_mask)
        t_emb = F.normalize(self.text_proj(t_vec), p=2, dim=1)

        # MIDI
        # Create padding mask (True where value is 0) to ignore padding in attention
        padding_mask = (midi_ids == 0)
        
        x = self.midi_emb(midi_ids) * math.sqrt(cfg.MIDI_EMBED_DIM)
        x = self.pos_encoder(x)
        
        # Pass padding_mask to transformer
        x = self.midi_transformer(x, src_key_padding_mask=padding_mask)
        
        # Use masked pooling instead of simple mean
        m_vec = self.midi_mean_pooling(x, padding_mask)
        m_emb = F.normalize(self.midi_proj(m_vec), p=2, dim=1)

        return t_emb, m_emb

    def encode_midi(self, midi_ids):
        with torch.no_grad():
            padding_mask = (midi_ids == 0)
            
            x = self.midi_emb(midi_ids) * math.sqrt(cfg.MIDI_EMBED_DIM)
            x = self.pos_encoder(x)
            x = self.midi_transformer(x, src_key_padding_mask=padding_mask)
            
            m_vec = self.midi_mean_pooling(x, padding_mask)
            return F.normalize(self.midi_proj(m_vec), p=2, dim=1)

    def encode_text(self, input_ids, attention_mask):
        with torch.no_grad():
            bert_out = self.bert(input_ids, attention_mask)
            t_vec = self.mean_pooling(bert_out, attention_mask)
            return F.normalize(self.text_proj(t_vec), p=2, dim=1)