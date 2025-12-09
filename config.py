import torch
import os

# --- PATH CONFIGURATION ---
COLAB_DRIVE_PATH = "/content/drive/MyDrive"

if os.path.exists(COLAB_DRIVE_PATH):
    # Google Colab Environment
    PROJECT_NAME = "MIDI_Retrieval_Project"
    BASE_DIR = os.path.join(COLAB_DRIVE_PATH, PROJECT_NAME)
    os.makedirs(BASE_DIR, exist_ok=True)
    print(f"Environment: Google Colab. Working directory: {BASE_DIR}")
else:
    # Local or Server Environment
    BASE_DIR = "."
    print(f"Environment: Local/Server. Working directory: Current Folder")

# --- FILE PATHS ---
# ✅ RENAMED: V4 Optimized (Distinct from V3 Teacher for future Ensemble)
SAVE_FILE = os.path.join(BASE_DIR, "midi_search_V4_OPTIMIZED.pt")
CACHE_FILE = os.path.join(BASE_DIR, "dataset_midicaps_tokenized.pt")

# --- MODEL HYPERPARAMETERS (V4.0) ---
MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
EMBED_DIM = 384  # Matches Text Embedding Dimension

# Transformer Architecture - V4.0 OPTIMIZED
MIDI_EMBED_DIM = 384  # Reduced from 512 for better generalization
NUM_LAYERS = 4        # Optimized depth (prevents overfitting)
NUM_HEADS = 6         # 384 / 64 = 6 heads
FF_DIM = 1024         # Efficient Feed Forward
DROPOUT = 0.15        # Regularization to improve test performance
MAX_SEQ_LEN = 512     # Focus on the most semantically dense part of the song

# --- TRAINING HYPERPARAMETERS ---
BATCH_SIZE = 256      # High batch size for effective Hard Negative Mining
LEARNING_RATE = 2e-4  # Optimized for fine-tuning
EPOCHS = 15           # Direct Text-to-MIDI training epochs

# --- MARGIN RANKING LOSS ---
MARGIN = 0.3          # Minimum distance required between correct and incorrect song

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Config loaded. Device: {DEVICE}")