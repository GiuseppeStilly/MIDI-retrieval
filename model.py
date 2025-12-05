import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
import config as cfg  # Imports our configuration

class NeuralMidiSearch(nn.Module):
    def __init__(self, midi_vocab_size):
        super().__init__()

        # --- TEXT ENCODER (Transformer) ---
        self.bert = AutoModel.from_pretrained(cfg.MODEL_NAME)
        # Dynamically get the output size of the text model (e.g., 384 for MiniLM)
        self.text_hidden_size = self.bert.config.hidden_size 
        # Projection layer to map text to the shared embedding space
        self.text_proj = nn.Linear(self.text_hidden_size, cfg.EMBED_DIM)

        # --- MIDI ENCODER (LSTM) ---
        self.midi_emb = nn.Embedding(midi_vocab_size, cfg.MIDI_EMBED_DIM)
        # Bidirectional LSTM captures context from both directions
        self.lstm = nn.LSTM(cfg.MIDI_EMBED_DIM, cfg.MIDI_HIDDEN_SIZE, 
                            batch_first=True, bidirectional=True)
        # Projection layer (LSTM output is doubled due to bidirectionality)
        self.midi_proj = nn.Linear(cfg.MIDI_HIDDEN_SIZE * 2, cfg.EMBED_DIM)

        # Learnable temperature parameter for Contrastive Loss scaling
        self.temperature = nn.Parameter(torch.tensor(0.07))

    def mean_pooling(self, model_output, attention_mask):
        """Standard mean pooling for sentence-transformers."""
        token_embeddings = model_output.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )

    def forward(self, input_ids, attention_mask, midi_ids):
        """Training forward pass: computes both embeddings."""
        # 1. Text Processing
        bert_out = self.bert(input_ids, attention_mask)
        t_vec = self.mean_pooling(bert_out, attention_mask)
        t_emb = F.normalize(self.text_proj(t_vec), p=2, dim=1)

        # 2. MIDI Processing
        x = self.midi_emb(midi_ids)
        x, _ = self.lstm(x)
        # Global Average Pooling over time dimension
        m_vec = torch.mean(x, dim=1) 
        m_emb = F.normalize(self.midi_proj(m_vec), p=2, dim=1)

        return t_emb, m_emb

    def encode_text(self, input_ids, attention_mask):
        """Inference helper: Encode query text."""
        with torch.no_grad():
            bert_out = self.bert(input_ids, attention_mask)
            t_vec = self.mean_pooling(bert_out, attention_mask)
            return F.normalize(self.text_proj(t_vec), p=2, dim=1)

    def encode_midi(self, midi_ids):
        """Inference helper: Encode MIDI sequence."""
        with torch.no_grad():
            x = self.midi_emb(midi_ids)
            x, _ = self.lstm(x)
            vec = torch.mean(x, dim=1)
            return F.normalize(self.midi_proj(vec), p=2, dim=1)