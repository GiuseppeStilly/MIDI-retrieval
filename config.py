import os
import torch

# --- PATH CONFIGURATION ---
# Check if running on Colab
COLAB_DRIVE_PATH = "/content/drive/MyDrive"

if os.path.exists(COLAB_DRIVE_PATH):
    PROJECT_NAME = "MIDI_Retrieval_Project"
    BASE_DIR = os.path.join(COLAB_DRIVE_PATH, PROJECT_NAME)
    os.makedirs(BASE_DIR, exist_ok=True)
    print(f"Environment: Google Colab. Working directory: {BASE_DIR}")
else:
    # Environment: Local (VSCode)
    BASE_DIR = "."
    print(f"Environment: Local/Server. Working directory: Current Folder")

# --- FILE PATHS ---
# Saved model name matches the branch name
SAVE_FILE = os.path.join(BASE_DIR, "v1.0-lstm-minilm.pt")
MIDI_DATA_DIR = "./data_midi_temp"

# --- MODEL HYPERPARAMETERS ---
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# MiniLM has an embedding dimension of 384
EMBED_DIM = 384
MIDI_EMBED_DIM = 384
MAX_SEQ_LEN = 512

# --- TRAINING HYPERPARAMETERS ---
BATCH_SIZE = 256
LEARNING_RATE = 5e-5
EPOCHS = 15

# --- HARDWARE ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Config loaded. Model: {MODEL_NAME} | Output: {SAVE_FILE}")