import os
import torch

# --- PATH CONFIGURATION ---
# Logic to automatically switch between Google Colab and Local VS Code

# 1. Define the Google Drive root
COLAB_DRIVE_ROOT = "/content/drive/MyDrive"

# 2. Check if we are running on Colab (Drive exists)
if os.path.exists(COLAB_DRIVE_ROOT):
    print("Environment detected: Google Colab")
    # Define the project folder on Drive
    PROJECT_DIR = os.path.join(COLAB_DRIVE_ROOT, "MIDI_Retrieval_Project")
    
    # Create the folder if it doesn't exist
    os.makedirs(PROJECT_DIR, exist_ok=True)
    
    BASE_DIR = PROJECT_DIR
    
    # Keep data local on Colab for speed (do not save dataset to Drive)
    MIDI_DATA_DIR = "./midicaps_data"
else:
    print("Environment detected: Local (VS Code)")
    # Save in the current directory
    BASE_DIR = "."
    MIDI_DATA_DIR = "midicaps_data"

# --- FILE PATHS ---
# The model will be saved here
SAVE_FILE = os.path.join(BASE_DIR, "midi_search_model_v1.pt")

# --- MODEL HYPERPARAMETERS ---
# Using the exact parameters from your recovered code
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

print(f"Config loaded. Device: {DEVICE}")
print(f"Target Save File: {SAVE_FILE}")