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
SAVE_FILE = os.path.join(BASE_DIR, "midi_search_MPNET_TRANSFORMER.pt")
CACHE_FILE = os.path.join(BASE_DIR, "dataset_midicaps_tokenized.pt")
MIDI_DATA_DIR = "midicaps_data"
HF_REPO_ID = "GiuseppeStilly/MIDI-Retrieval"

# --- AUDIO CONFIGURATION ---
SOUNDFONT_PATH = "soundfont.sf2"
SOUNDFONT_URL = "https://raw.githubusercontent.com/urish/cinto/master/media/FluidR3_GM.sf2"

# --- MODEL HYPERPARAMETERS ---
MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
EMBED_DIM = 384  # Reduced from 512 to save memory

# Transformer Architecture - OPTIMIZED
MIDI_EMBED_DIM = 384  # Reduced from 512
NUM_LAYERS = 4  # Reduced from 6 (faster + smaller memory footprint)
NUM_HEADS = 6  # Reduced from 8 (compatible with reduced embedding)
FF_DIM = 1024  # Reduced from 2048
DROPOUT = 0.15  # Slightly increased for regularization
MAX_SEQ_LEN = 512  # Reduced from 1024 (MIDI sequences rarely need >512)

# --- TRAINING HYPERPARAMETERS ---
BATCH_SIZE = 256  # Increased to take advantage of reduced model size
LEARNING_RATE = 2e-4  # Slightly increased for better convergence
EPOCHS = 30  # Increased to benefit from better pretraining

# --- CONTRASTIVE LEARNING (for Phase 1: MIDI-only pretraining) ---
CONTRASTIVE_TEMPERATURE = 0.05  # Lower temp = sharper distinctions (tuned for batch size 256)
USE_CONTRASTIVE_PRETRAINING = True  # Toggle: True = 2-phase, False = supervised only
CONTRASTIVE_EPOCHS = 15  # Phase 1 epochs (MIDI pretraining with pitch shifts)

# --- MARGIN RANKING LOSS (for Phase 2: Text-MIDI fine-tuning) ---
MARGIN = 0.3  # Margin for ranking loss (balanced penalty)
SUPERVISED_EPOCHS = 15  # Phase 2 epochs (text-MIDI alignment)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Config loaded. Device: {DEVICE}")
