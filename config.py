import torch
import os

# --- PATHS ---
if os.path.exists("/content/drive/MyDrive"):
    BASE_ROOT = "/content/drive/MyDrive"
else:
    BASE_ROOT = "."

PROJECT_FOLDER = "MIDI_Retrieval_Project"
PROJECT_PATH = os.path.join(BASE_ROOT, PROJECT_FOLDER)

os.makedirs(PROJECT_PATH, exist_ok=True)

SAVE_FILE = os.path.join(PROJECT_PATH, "v2_transformer_mpnet.pt")
CACHE_FILE = os.path.join(PROJECT_PATH, "dataset_midicaps_tokenized.pt")

MIDI_DATA_DIR = "midicaps_data"
HF_REPO_ID = "GiuseppeStilly/MIDI-Retrieval" 

# --- AUDIO CONFIGURATION ---
SOUNDFONT_PATH = "soundfont.sf2"
SOUNDFONT_URL = "https://github.com/urish/cinto/raw/master/media/FluidR3_GM.sf2"

# --- MODEL HYPERPARAMETERS ---
MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
EMBED_DIM = 512          

# Transformer Architecture Settings
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