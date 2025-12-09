import torch
import os

# --- PATH CONFIGURATION ---
COLAB_DRIVE_PATH = "/content/drive/MyDrive"

if os.path.exists(COLAB_DRIVE_PATH):
    # Google Colab
    PROJECT_NAME = "MIDI_Retrieval_Project"
    BASE_DIR = os.path.join(COLAB_DRIVE_PATH, PROJECT_NAME)
    os.makedirs(BASE_DIR, exist_ok=True)
    print(f"Environment: Google Colab. Working directory: {BASE_DIR}")
else:
    # Local/Server
    BASE_DIR = "."
    print(f"Environment: Local/Server. Working directory: Current Folder")

# --- FILE PATHS ---
SAVE_FILE = os.path.join(BASE_DIR, "midi_search_MPNET_TRANSFORMER.pt")
CACHE_FILE = os.path.join(BASE_DIR, "dataset_midicaps_tokenized.pt")

# --- MODEL HYPERPARAMETERS (V4.0 STUDENT) ---
MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
EMBED_DIM = 384  # Reduced from 512

# Transformer Architecture - STUDENT
MIDI_EMBED_DIM = 384  # Matching Embedding Dim
NUM_LAYERS = 4        # 4 Layers d
NUM_HEADS = 6         # 384 / 64 = 6 Heads
FF_DIM = 1024         # Compact Feed Forward
DROPOUT = 0.15        # Regularization
MAX_SEQ_LEN = 512     # Focus on the first ~20 seconds of music

# --- TRAINING HYPERPARAMETERS ---
BATCH_SIZE = 256      # High Batch Size for better Hard Negative Mining
LEARNING_RATE = 2e-4
EPOCHS = 15           # Direct Fine-tuning Epochs

# --- MARGIN RANKING LOSS ---
MARGIN = 0.3          # Distance required between correct and incorrect song

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Config loaded. Device: {DEVICE}")