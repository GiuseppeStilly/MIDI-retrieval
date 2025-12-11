import os
import torch

# --- PATH CONFIGURATION ---
DRIVE_ROOT = "/content/drive/MyDrive"

# Check environment (Colab vs Local)
if not os.path.exists(DRIVE_ROOT):
    DRIVE_ROOT = "." 

# Project directory setup
PROJECT_PATH = os.path.join(DRIVE_ROOT, "MIDI_Retrieval_Project")

if DRIVE_ROOT != ".":
    os.makedirs(PROJECT_PATH, exist_ok=True)
    SAVE_FILE = os.path.join(PROJECT_PATH, "v1_lstm_optimized.pt")
else:
    # Local save
    os.makedirs("runs", exist_ok=True)
    SAVE_FILE = os.path.join("runs", "v1_lstm_optimized.pt")

MIDI_DATA_DIR = "midicaps_data"

# --- MODEL HYPERPARAMETERS (Version 1) ---
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2" 
EMBED_DIM = 512          
MIDI_HIDDEN_SIZE = 256   
MIDI_EMBED_DIM = 256     
MAX_SEQ_LEN = 256        

# --- TRAINING HYPERPARAMETERS ---
BATCH_SIZE = 128
LEARNING_RATE = 1e-4
EPOCHS = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Config loaded. Device: {DEVICE}")
print(f"Target Save File: {SAVE_FILE}")