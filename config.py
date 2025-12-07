import torch
import os

# --- PATHS ---
# Automatic detection for local or Drive environment
if os.path.exists("/content/drive/MyDrive"):
    DRIVE_ROOT = "/content/drive/MyDrive"
else:
    DRIVE_ROOT = "."

# Updated filename for the Transformer version
SAVE_FILE = os.path.join(DRIVE_ROOT, "midi_search_MPNET_TRANSFORMER.pt")
CACHE_FILE = os.path.join(DRIVE_ROOT, "dataset_midicaps_tokenized.pt")
MIDI_DATA_DIR = "midicaps_data"

# --- MODEL HYPERPARAMETERS ---
# Using MPNet (768 hidden size) instead of MiniLM
MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
EMBED_DIM = 512          # Dimension of the shared latent space

# Transformer Architecture Settings
MIDI_EMBED_DIM = 256     # MIDI token embedding size
NUM_LAYERS = 4           # Number of Transformer Encoder layers
NUM_HEADS = 4            # Number of Attention Heads
FF_DIM = 1024            # Dimension of Feed Forward network
DROPOUT = 0.1            # Dropout rate
MAX_SEQ_LEN = 512        # Increased sequence length for Transformer

# --- TRAINING HYPERPARAMETERS ---
# Adjusted for MPNet stability
BATCH_SIZE = 64
LEARNING_RATE = 5e-5     # Lower learning rate to prevent divergence
EPOCHS = 20
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Config loaded. Device: {DEVICE}")