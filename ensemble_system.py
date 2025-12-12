import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from transformers import AutoModel
from huggingface_hub import hf_hub_download
from tqdm import tqdm

# --- CONFIGURATION ---
HF_REPO_ID = "GiuseppeStilly/MIDI-Retrieval"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EMBED_DIM = 512

# ARCHITECTURE V1 (LSTM + MiniLM)
class NeuralMidiSearch_V1(nn.Module):
    def __init__(self, midi_vocab_size):
        super().__init__()
        # Text Encoder
        self.bert = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        self.text_proj = nn.Linear(self.bert.config.hidden_size, EMBED_DIM)
        
        # MIDI Encoder
        self.midi_emb = nn.Embedding(midi_vocab_size, 256)
        self.lstm = nn.LSTM(256, 256, batch_first=True, bidirectional=True)
        self.midi_proj = nn.Linear(512, EMBED_DIM) 
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
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div)
        pe[:, 1::2] = torch.cos(position * div)
        pe = pe.unsqueeze(0) 
        self.register_buffer('pe', pe)

    def forward(self, x): return x + self.pe[:, :x.size(1)]

class NeuralMidiSearch_V2(nn.Module):
    def __init__(self, midi_vocab_size):
        super().__init__()
        midi_hidden = 256
        
        # Text Encoder
        self.bert = AutoModel.from_pretrained("sentence-transformers/all-mpnet-base-v2")
        self.text_proj = nn.Linear(768, EMBED_DIM) 

        # MIDI Encoder
        self.midi_emb = nn.Embedding(midi_vocab_size, midi_hidden)
        self.pos_encoder = PositionalEncoding(midi_hidden)
        
        layer = nn.TransformerEncoderLayer(
            d_model=midi_hidden, nhead=4, 
            dim_feedforward=1024, # Critical fix for V2
            batch_first=True
        )
        self.midi_transformer = nn.TransformerEncoder(layer, num_layers=4)
        self.midi_proj = nn.Linear(midi_hidden, EMBED_DIM)
        self.temperature = nn.Parameter(torch.tensor(0.07))

    def mean_pooling(self, model_output, attention_mask):
        token = model_output.last_hidden_state
        mask = attention_mask.unsqueeze(-1).expand(token.size()).float()
        return torch.sum(token * mask, 1) / torch.clamp(mask.sum(1), min=1e-9)

    def encode_midi(self, midi_ids):
        with torch.no_grad():
            x = self.midi_emb(midi_ids)
            x = self.pos_encoder(x)
            x = self.midi_transformer(x)
            vec = torch.mean(x, dim=1)
            return F.normalize(self.midi_proj(vec), p=2, dim=1)

    def encode_text(self, input_ids, attention_mask):
        with torch.no_grad():
            out = self.bert(input_ids, attention_mask)
            vec = self.mean_pooling(out, attention_mask)
            return F.normalize(self.text_proj(vec), p=2, dim=1)

# ==========================================
# HELPER FUNCTIONS
# ==========================================
def load_model(version, filename):
    print(f"Loading {version} model from {filename}...")
    try:
        path = hf_hub_download(repo_id=HF_REPO_ID, filename=filename)
        ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
        vocab_size = ckpt.get('vocab_size', 3000)
        
        if version == "V1":
            model = NeuralMidiSearch_V1(vocab_size)
        else:
            model = NeuralMidiSearch_V2(vocab_size)
            
        model.to(DEVICE)
        model.load_state_dict(ckpt['model_state'], strict=False)
        model.eval()
        return model
    except Exception as e:
        print(f"Error loading {version}: {e}")
        return None

def compute_similarity_matrix(model, loader, desc):
    text_embs, midi_embs = [], []
    
    with torch.no_grad():
        for batch in tqdm(loader, desc=desc):
            i_ids = batch["input_ids"].to(DEVICE)
            mask = batch["attention_mask"].to(DEVICE)
            m_ids = batch["midi_ids"].to(DEVICE)
            
            # Autocast for performance
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                t = model.encode_text(i_ids, mask)
                m = model.encode_midi(m_ids)
            
            text_embs.append(t.float().cpu())
            midi_embs.append(m.float().cpu())
            
    T = torch.cat(text_embs)
    M = torch.cat(midi_embs)
    
    return T @ M.T