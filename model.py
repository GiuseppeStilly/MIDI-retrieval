import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
import config as cfg

class NeuralMidiSearch(nn.Module):
    def __init__(self, midi_vocab_size):
        super().__init__()
        embed_dim = getattr(cfg, 'EMBED_DIM', 512)
        
        # Text Encoder (MiniLM)
        self.bert = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        self.text_proj = nn.Linear(self.bert.config.hidden_size, embed_dim)

        # MIDI Encoder (Bi-LSTM)
        self.midi_emb = nn.Embedding(midi_vocab_size, 256)
        self.lstm = nn.LSTM(256, 256, batch_first=True, bidirectional=True)
        self.midi_proj = nn.Linear(512, embed_dim)

        # Learnable Temperature parameter (Essential for InfoNCE loss)
        self.temperature = nn.Parameter(torch.tensor(0.07))

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def forward(self, input_ids, attention_mask, midi_ids):
        # Text Branch
        bert_out = self.bert(input_ids, attention_mask)
        t_vec = self.mean_pooling(bert_out, attention_mask)
        t_emb = F.normalize(self.text_proj(t_vec), p=2, dim=1)

        # MIDI Branch
        x = self.midi_emb(midi_ids)
        x, _ = self.lstm(x)
        vec = torch.mean(x, dim=1) # Global Average Pooling
        m_emb = F.normalize(self.midi_proj(vec), p=2, dim=1)

        return t_emb, m_emb

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