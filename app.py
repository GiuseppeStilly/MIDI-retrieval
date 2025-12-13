import os
import sys
import tarfile
import tempfile
from pathlib import Path

import torch
import gradio as gr
from transformers import AutoTokenizer
from miditok import REMI, TokenizerConfig
from huggingface_hub import hf_hub_download
from midi2audio import FluidSynth
from datasets import load_dataset

# IMPORT ARCHITECTURES FROM LOCAL FILE
from ensemble_system import NeuralMidiSearch_V1, NeuralMidiSearch_V2

# 1. SYSTEM CONFIGURATION
HF_REPO = "GiuseppeStilly/MIDI-Retrieval"
DATASET_FILENAME = "dataset_midicaps_tokenized.pt"
OFFICIAL_CAPTION_REPO = "amaai-lab/MidiCaps"

# Model Filenames
MODEL_V1_FILENAME = "v1_lstm_optimized.pt"
INDEX_V1_FILENAME = "midi_index_v1.pt"

MODEL_V2_FILENAME = "v2_transformer_mpnet.pt"
INDEX_V2_FILENAME = "midi_index_v2.pt"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Running on device: {DEVICE}")

# Environment Setup
SOUNDFONT_PATH = "font.sf3"
if not os.path.exists(SOUNDFONT_PATH):
    print("Downloading SoundFont...")
    os.system('wget -q https://github.com/musescore/MuseScore/raw/master/share/sound/FluidR3Mono_GM.sf3 -O font.sf3')

# 2. UTILITIES
def download_real_midi_data():
    """Downloads and extracts the raw MIDI dataset (1.6GB)."""
    midi_data_dir = Path("./midicaps_data")
    if not midi_data_dir.exists() or not any(midi_data_dir.iterdir()):
        print("Downloading Real MIDI Archive (1.6GB)...")
        try:
            tar_path = hf_hub_download(repo_id="amaai-lab/MidiCaps", filename="midicaps.tar.gz", repo_type="dataset")
            print("Extracting archive...")
            with tarfile.open(tar_path) as tar:
                tar.extractall(path=midi_data_dir)
            print("Extraction complete.")
        except Exception as e:
            print(f"Download error: {e}")
            return None
    return midi_data_dir

def index_local_files(midi_dir):
    """Maps filenames to local paths."""
    file_map = {}
    if midi_dir:
        for root, _, files in os.walk(midi_dir):
            for file in files:
                if file.endswith((".mid", ".midi")):
                    file_map[file] = str(Path(root) / file)
    return file_map

def load_models(midi_vocab_size):
    """Loads V1 and V2 models and indices using imported classes."""
    print("Loading Models & Indices...")
    
    # --- Load V1 ---
    tok_v1 = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    try:
        path_v1 = hf_hub_download(repo_id=HF_REPO, filename=MODEL_V1_FILENAME)
        model_v1 = NeuralMidiSearch_V1(midi_vocab_size).to(DEVICE)
        model_v1.eval()
        chk = torch.load(path_v1, map_location=DEVICE)
        model_v1.load_state_dict(chk['model_state'] if 'model_state' in chk else chk, strict=False)
        
        idx_v1 = torch.load(hf_hub_download(repo_id=HF_REPO, filename=INDEX_V1_FILENAME), map_location=DEVICE)
        bank_v1 = idx_v1['embeddings'].to(DEVICE)
        print("V1 (Bi-LSTM) Loaded.")
    except Exception as e:
        print(f"V1 Load Failed: {e}")
        model_v1, bank_v1 = None, None

    # --- Load V2 ---
    tok_v2 = AutoTokenizer.from_pretrained("sentence-transformers/all-mpnet-base-v2")
    try:
        path_v2 = hf_hub_download(repo_id=HF_REPO, filename=MODEL_V2_FILENAME)
        model_v2 = NeuralMidiSearch_V2(midi_vocab_size).to(DEVICE)
        model_v2.eval()
        chk = torch.load(path_v2, map_location=DEVICE)
        model_v2.load_state_dict(chk['model_state'] if 'model_state' in chk else chk, strict=False)
        
        idx_v2 = torch.load(hf_hub_download(repo_id=HF_REPO, filename=INDEX_V2_FILENAME), map_location=DEVICE)
        bank_v2 = idx_v2['embeddings'].to(DEVICE)
        print("V2 (Transformer) Loaded.")
    except Exception as e:
        print(f"V2 Load Failed: {e}")
        model_v2, bank_v2 = None, None

    return (model_v1, bank_v1, tok_v1), (model_v2, bank_v2, tok_v2)

def load_aux_data():
    """Loads mapping datasets."""
    print("Loading Auxiliary Data...")
    try:
        path = hf_hub_download(repo_id=HF_REPO, filename=DATASET_FILENAME)
        raw_ds = torch.load(path, map_location="cpu")
        
        captions = {}
        ds_cat = load_dataset(OFFICIAL_CAPTION_REPO, split="train")
        for row in ds_cat:
            full_path = row.get('location', row.get('path', ''))
            if full_path:
                captions[os.path.basename(full_path).strip()] = row.get('caption', '')
        return raw_ds, captions
    except Exception as e:
        print(f"Aux Data Error: {e}")
        return [], {}

# 3. INITIALIZATION
midi_tok = REMI(TokenizerConfig(num_velocities=16, use_chords=True, use_programs=False))

midi_dir = download_real_midi_data()
real_file_map = index_local_files(midi_dir)
(v1_res, v2_res) = load_models(len(midi_tok))
raw_dataset, filename_to_caption = load_aux_data()
fs = FluidSynth(SOUNDFONT_PATH)

# 4. SEARCH LOGIC
def search_ensemble(query):
    print(f"\nQuery: '{query}'")
    
    # V1 Calculation
    scores_v1 = None
    if v1_res[0] and v1_res[1]:
        m1, b1, t1 = v1_res
        inp = t1(query, padding=True, truncation=True, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            emb = m1.encode_text(inp["input_ids"], inp["attention_mask"])
            scores_v1 = torch.matmul(emb, b1.T).squeeze(0)

    # V2 Calculation
    scores_v2 = None
    if v2_res[0] and v2_res[1]:
        m2, b2, t2 = v2_res
        inp = t2(query, padding=True, truncation=True, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            emb = m2.encode_text(inp["input_ids"], inp["attention_mask"])
            scores_v2 = torch.matmul(emb, b2.T).squeeze(0)

    # Fusion
    fusion_note = ""
    if scores_v1 is not None and scores_v2 is not None:
        if scores_v1.shape == scores_v2.shape:
            final_scores = (scores_v1 + scores_v2) / 2
            fusion_note = "Ensemble (Avg)"
        else:
            final_scores = scores_v1
            fusion_note = "V1 Only (Mismatch)"
    elif scores_v1 is not None:
        final_scores = scores_v1
        fusion_note = "V1 Only"
    elif scores_v2 is not None:
        final_scores = scores_v2
        fusion_note = "V2 Only"
    else:
        return [None] * 15

    # Retrieval
    best_scores, best_idxs = torch.topk(final_scores, k=30)
    results = []
    
    for score, idx in zip(best_scores, best_idxs):
        if len(results) >= 5: break
        idx = idx.item()
        
        fname = "unknown"
        if idx < len(raw_dataset):
            item = raw_dataset[idx]
            raw_path = item.get("p", "")
            fname = os.path.basename(str(raw_path)).strip()
        
        local_path = real_file_map.get(fname)
        if local_path and os.path.exists(local_path) and fname != "unknown":
            try:
                temp_wav = os.path.join(tempfile.gettempdir(), f"res_{idx}_{fname}.wav")
                fs.midi_to_audio(local_path, temp_wav)
                caption = filename_to_caption.get(fname, "No description available.")
                info = (f"**File:** `{fname}`\n"
                        f"**Score:** `{score:.4f}` ({fusion_note})\n\n"
                        f"_{caption}_")
                results.append((temp_wav, local_path, info))
                print(f"Match: {fname} ({score:.3f})")
            except Exception as e:
                print(f"Error: {e}")

    while len(results) < 5:
        results.append((None, None, "No result found."))

    return [item for sublist in results for item in sublist]

# 5. UI
custom_css = """
.result-box {background-color: #2b2b2b; padding: 15px; border-radius: 10px; border: 1px solid #444; margin-bottom: 10px;}
h1 {text-align: center; color: #a29bfe;}
"""

with gr.Blocks(theme=gr.themes.Soft(), css=custom_css, title="Ensemble MIDI Search") as demo:
    gr.Markdown("# Neural MIDI Search: Ensemble System")
    gr.Markdown("Retrieving Real MIDI files using V1 (Bi-LSTM) + V2 (Transformer) Late Fusion.")

    with gr.Row():
        txt_input = gr.Textbox(label="Query", placeholder="e.g., Jazz piano", scale=4)
        btn_submit = gr.Button("Search", variant="primary", scale=1)

    outputs = []
    for i in range(1, 6):
        with gr.Group(elem_classes="result-box"):
            gr.Markdown(f"### Result #{i}")
            with gr.Row():
                outputs.extend([
                    gr.Audio(label="Preview", type="filepath"),
                    gr.File(label="Download"),
                    gr.Markdown()
                ])

    btn_submit.click(search_ensemble, inputs=txt_input, outputs=outputs)
    txt_input.submit(search_ensemble, inputs=txt_input, outputs=outputs)

if __name__ == "__main__":
    demo.launch(share=True)
