import torch
import os

# --- PATH CONFIGURATION ---
COLAB_DRIVE_PATH = "/content/drive/MyDrive"

# Check if running on Colab with Drive mounted
if os.path.exists(COLAB_DRIVE_PATH):
    PROJECT_NAME = "MIDI_Retrieval_Project"
    BASE_DIR = os.path.join(COLAB_DRIVE_PATH, PROJECT_NAME)
    os.makedirs(BASE_DIR, exist_ok=True)
    print(f"Environment: Google Colab. Working directory: {BASE_DIR}")
else:
    # Fallback for local environment
    BASE_DIR = "."
    print(f"Environment: Local/Server. Working directory: Current Folder")

# --- FILE PATHS ---
# The model will be saved with this specific name
SAVE_FILE = os.path.join(BASE_DIR, "v1-lstm.pt")

# Temporary directory for raw MIDI files (local storage is faster than Drive)
MIDI_DATA_DIR = "./data_midi_temp"

# --- MODEL HYPERPARAMETERS ---
MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"

# Sequence settings
MAX_SEQ_LEN = 512

# --- TRAINING HYPERPARAMETERS ---
BATCH_SIZE = 256
LEARNING_RATE = 5e-5
EPOCHS = 15

# --- HARDWARE ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Config loaded. Output target: {SAVE_FILE}")