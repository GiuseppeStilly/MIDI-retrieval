import torch
import os

# --- PATH CONFIGURATION ---
COLAB_DRIVE_PATH = "/content/drive/MyDrive"

if os.path.exists(COLAB_DRIVE_PATH):
    PROJECT_NAME = "MIDI_Retrieval_Project"
    BASE_DIR = os.path.join(COLAB_DRIVE_PATH, PROJECT_NAME)
    os.makedirs(BASE_DIR, exist_ok=True)
    print(f"Environment: Google Colab. Working directory: {BASE_DIR}")
else:
    BASE_DIR = "."
    print(f"Environment: Local/Server. Working directory: Current Folder")

# --- FILE PATHS ---
SAVE_FILE = os.path.join(BASE_DIR, "midi_search_V3.pt")
CACHE_FILE = os.path.join(BASE_DIR, "dataset_midicaps_tokenized.pt")

# --- MODEL HYPERPARAMETERS (Optimized Student Architecture) ---
MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
EMBED_DIM = 384  

# Transformer Architecture
MIDI_EMBED_DIM = 384  
NUM_LAYERS = 4        
NUM_HEADS = 6         
FF_DIM = 1024         
DROPOUT = 0.15        
MAX_SEQ_LEN = 512     

# --- TRAINING HYPERPARAMETERS ---
BATCH_SIZE = 256      
LEARNING_RATE = 5e-5  # Reduced to prevent gradient explosion
EPOCHS = 15           

# --- MARGIN RANKING LOSS ---
MARGIN = 0.3          

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Config loaded. Device: {DEVICE}")