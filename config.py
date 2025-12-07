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
EMBED_DIM = 512          

# Transformer Architecture
MIDI_EMBED_DIM = 256     
NUM_LAYERS = 4           
NUM_HEADS = 4            
FF_DIM = 1024            
DROPOUT = 0.1            
MAX_SEQ_LEN = 512        

# --- TRAINING HYPERPARAMETERS ---
BATCH_SIZE = 64
LEARNING_RATE = 5e-5     
EPOCHS = 20
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Config loaded. Device: {DEVICE}")