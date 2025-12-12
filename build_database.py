import torch
import os
from huggingface_hub import hf_hub_download
from ensemble_system import NeuralMidiSearch_V1, NeuralMidiSearch_V2
from tqdm import tqdm

# --- CONFIGURATION ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TOKENIZED_FILE = "dataset_midicaps_tokenized.pt"
MODEL_V1_NAME = "v1_lstm_optimized.pt"
MODEL_V2_NAME = "v2_transformer_mpnet.pt"
REPO_ID = "GiuseppeStilly/MIDI-Retrieval"

def build_db():
    print(f"Starting Database Builder on {DEVICE}...")

    # 1. Load the Tokenized Dataset
    print("Loading tokenized dataset...")
    try:
        path = hf_hub_download(repo_id=REPO_ID, filename=TOKENIZED_FILE)
        # The file is a list of dictionaries
        dataset = torch.load(path)
        print(f"Dataset loaded: {len(dataset)} items.")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    # 2. Extract Data
    # We extract the tokens ('m') and the file paths ('p')
    all_midi_tokens = [d['m'] for d in dataset]
    all_paths = [d['p'] for d in dataset]

    # 3. Process V1 (LSTM)
    print("\n--- Processing V1 (LSTM) ---")
    try:
        path_v1 = hf_hub_download(repo_id=REPO_ID, filename=MODEL_V1_NAME)
        ckpt_v1 = torch.load(path_v1, map_location=DEVICE)
        
        # Initialize Model
        model_v1 = NeuralMidiSearch_V1(ckpt_v1.get('vocab_size', 3000)).to(DEVICE)
        model_v1.load_state_dict(ckpt_v1['model_state'], strict=False)
        model_v1.eval()

        # Generate Database
        db_matrix_v1 = []
        print("Generating V1 embeddings (this may take time)...")
        
        with torch.no_grad():
            for m_tokens in tqdm(all_midi_tokens):
                # Add batch dimension (1, seq_len)
                inp = m_tokens.unsqueeze(0).long().to(DEVICE) 
                emb = model_v1.encode_midi(inp)
                db_matrix_v1.append(emb.cpu())
        
        # Stack into a single matrix
        final_db_v1 = torch.cat(db_matrix_v1, dim=0)
        print(f"V1 Matrix shape: {final_db_v1.shape}")

        # Save FIXED Checkpoint
        new_ckpt_v1 = {
            'model_state': ckpt_v1['model_state'],
            'vocab_size': ckpt_v1.get('vocab_size', 3000),
            'db_matrix': final_db_v1,
            'db_paths': all_paths
        }
        torch.save(new_ckpt_v1, "v1_lstm_optimized_FIXED.pt")
        print("Saved v1_lstm_optimized_FIXED.pt")

    except Exception as e:
        print(f"Error processing V1: {e}")


    # 4. Process V2 (Transformer)
    print("\n--- Processing V2 (Transformer) ---")
    try:
        path_v2 = hf_hub_download(repo_id=REPO_ID, filename=MODEL_V2_NAME)
        ckpt_v2 = torch.load(path_v2, map_location=DEVICE)
        
        model_v2 = NeuralMidiSearch_V2(ckpt_v2.get('vocab_size', 3000)).to(DEVICE)
        model_v2.load_state_dict(ckpt_v2['model_state'], strict=False)
        model_v2.eval()

        db_matrix_v2 = []
        print("Generating V2 embeddings...")
        
        with torch.no_grad():
            for m_tokens in tqdm(all_midi_tokens):
                inp = m_tokens.unsqueeze(0).long().to(DEVICE)
                emb = model_v2.encode_midi(inp)
                db_matrix_v2.append(emb.cpu())
        
        final_db_v2 = torch.cat(db_matrix_v2, dim=0)
        print(f"V2 Matrix shape: {final_db_v2.shape}")

        new_ckpt_v2 = {
            'model_state': ckpt_v2['model_state'],
            'vocab_size': ckpt_v2.get('vocab_size', 3000),
            'db_matrix': final_db_v2,
            'db_paths': all_paths
        }
        torch.save(new_ckpt_v2, "v2_transformer_mpnet_FIXED.pt")
        print("Saved v2_transformer_mpnet_FIXED.pt")

    except Exception as e:
        print(f"Error processing V2: {e}")

if __name__ == "__main__":
    build_db()