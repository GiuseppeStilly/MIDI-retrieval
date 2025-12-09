import os
import torch
import numpy as np
import gradio as gr
from midi2audio import FluidSynth
from transformers import AutoTokenizer
from sklearn.metrics.pairwise import cosine_similarity
import tarfile
from huggingface_hub import hf_hub_download

# Local imports
import config as cfg
from model import NeuralMidiSearch

# --- SYSTEM SETUP ---
# Install FluidSynth if missing (Linux/Colab specific)
if os.system("which fluidsynth") != 0:
    print("FluidSynth not found. Installing...")
    os.system("sudo apt-get update && sudo apt-get install -y fluidsynth fluid-soundfont-gm")

SOUNDFONT = "/usr/share/sounds/sf2/FluidR3_GM.sf2"

# --- RESTORE DATASET (If needed for playback) ---
if not os.path.exists(cfg.MIDI_DATA_DIR):
    print("Restoring MIDI files for playback...")
    try:
        os.makedirs(cfg.MIDI_DATA_DIR, exist_ok=True)
        path = hf_hub_download(repo_id="amaai-lab/MidiCaps", filename="midicaps.tar.gz", repo_type="dataset")
        with tarfile.open(path) as tar: 
            tar.extractall(cfg.MIDI_DATA_DIR)
    except Exception as e:
        print(f"Warning: Could not download MIDI files. Playback won't work. {e}")

# --- LOAD SYSTEM ---
def load_system():
    if not os.path.exists(cfg.SAVE_FILE):
        return None, None, None, None
        
    print(f"📂 Loading checkpoint from: {cfg.SAVE_FILE}")
    # Load weights_only=False to support the dictionary structure
    checkpoint = torch.load(cfg.SAVE_FILE, map_location=cfg.DEVICE, weights_only=False)
    
    # Reconstruct the model using the saved vocab size
    model = NeuralMidiSearch(checkpoint['vocab_size']).to(cfg.DEVICE)
    model.load_state_dict(checkpoint['model_state'])
    model.eval()
    
    tokenizer = AutoTokenizer.from_pretrained(cfg.MODEL_NAME)
    
    return model, tokenizer, checkpoint['db_matrix'], checkpoint['db_paths']

# Global Load
MODEL, TOKENIZER, DB_MATRIX, DB_PATHS = load_system()

# --- SEARCH LOGIC ---
def search(query):
    if MODEL is None:
        return "Error: Model not found. Please run train.py first!", None, None

    # 1. Encode Text Query
    inputs = TOKENIZER([query], padding=True, truncation=True, return_tensors="pt").to(cfg.DEVICE)
    query_vec = MODEL.encode_text(inputs["input_ids"], inputs["attention_mask"]).cpu().numpy()

    # 2. Calculate Cosine Similarity
    scores = cosine_similarity(query_vec, DB_MATRIX)[0]
    best_idx = np.argmax(scores)
    best_score = scores[best_idx]
    
    # 3. Retrieve File
    full_path = DB_PATHS[best_idx]
    filename = os.path.basename(full_path)
    
    # Handle path differences if running on a different machine than training
    local_path = None
    for root, _, files in os.walk(cfg.MIDI_DATA_DIR):
        if filename in files:
            local_path = os.path.join(root, filename)
            break
            
    result_text = f"Result: {filename}\n Match Score: {best_score:.4f}"

    if not local_path:
         return result_text + "\n(Original MIDI file not found on disk)", None, None

    # 4. Generate Audio Preview
    audio_out = "output.wav"
    try:
        FluidSynth(SOUNDFONT).midi_to_audio(local_path, audio_out)
        return result_text, audio_out, local_path
    except Exception as e:
        return result_text + f"\n(Audio Render Error: {e})", None, local_path

# --- INTERFACE ---
with gr.Blocks(theme=gr.themes.Soft(), title="AI Music Search") as demo:
    gr.Markdown("# Neural MIDI Search")
    gr.Markdown("Search for music using natural language.")
    
    with gr.Row():
        txt_input = gr.Textbox(placeholder="Describe the music...", label="Query")
        btn_search = gr.Button("Search", variant="primary")
    
    with gr.Row():
        lbl_info = gr.Textbox(label="Details")
        aud_player = gr.Audio(label="Preview")
        file_down = gr.File(label="Download MIDI")
    
    btn_search.click(search, inputs=txt_input, outputs=[lbl_info, aud_player, file_down])

if __name__ == "__main__":
    demo.launch(share=True, debug=True)