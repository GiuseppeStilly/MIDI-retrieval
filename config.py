
import torch
import os

# --- PATHS ---
# Automatic Google Drive detection
DRIVE_ROOT = "/content/drive/MyDrive"
if not os.path.exists(DRIVE_ROOT):
    # Fallback for local testing
    DRIVE_ROOT = "."

SAVE_FILE = os.path.join(DRIVE_ROOT, "midi_search_model.pt")
MIDI_DATA_DIR = "midicaps_data"

# --- MODEL HYPERPARAMETERS ---
# We use MiniLM because it's fast and effective for semantic search
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2" 
EMBED_DIM = 512          # Dimension of the shared latent space
MIDI_HIDDEN_SIZE = 256   # LSTM hidden size
MIDI_EMBED_DIM = 256     # MIDI token embedding size
MAX_SEQ_LEN = 256        # Max length of MIDI token sequence

# --- TRAINING HYPERPARAMETERS ---
BATCH_SIZE = 128
LEARNING_RATE = 1e-4
EPOCHS = 2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"⚙️ Config loaded. Device: {DEVICE}")