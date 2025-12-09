import torch
import os
from tqdm import tqdm
from miditok import REMI, TokenizerConfig
import config as cfg

# CONFIGURATION
INPUT_CACHE = cfg.CACHE_FILE
OUTPUT_CACHE_PRETRAINING = os.path.join(cfg.BASE_DIR, "dataset_midicaps_MIDI_PRETRAINING.pt")

# The pitch variations we want to learn
PITCH_SHIFTS = [-3, -1, 1, 3]

def get_pitch_shift_map(tokenizer, semitones):
    """Creates the map to translate MIDI tokens (Pitch Shift)."""
    mapping = {}
    vocab = tokenizer.vocab
    id_to_token = {v: k for k, v in vocab.items()}
    
    for old_id, token_str in id_to_token.items():
        if token_str.startswith("Pitch_"):
            try:
                val = int(token_str.split("_")[1])
                new_val = val + semitones
                new_token_str = f"Pitch_{new_val}"
                
                # Keep within valid piano range [0, 127]
                if new_token_str in vocab and 0 <= new_val <= 127:
                    mapping[old_id] = vocab[new_token_str]
                else:
                    mapping[old_id] = old_id
            except:
                mapping[old_id] = old_id
        else:
            mapping[old_id] = old_id
    
    return mapping

def main():
    print(f"Loading original cache: {INPUT_CACHE}...")
    if not os.path.exists(INPUT_CACHE):
        print("Error: Input cache not found. Run training once to generate the base cache first.")
        return

    original_data = torch.load(INPUT_CACHE)
    print(f"Original samples: {len(original_data)}")

    # Tokenizer
    tokenizer = REMI(TokenizerConfig(num_velocities=16, use_chords=True))

    # Pre-calculate translation maps for speed
    maps = {}
    for s in PITCH_SHIFTS:
        maps[s] = get_pitch_shift_map(tokenizer, s)

    augmented_data = []
    pair_indices = [] 
    
    print("Generating MIDI-only pretraining data...")
    
    # Use tqdm with enumerate to keep track of indices
    for i, item in enumerate(tqdm(original_data)):
        
        # 1. Save the Original in the giant list
        augmented_data.append(item)
        orig_data_idx = len(augmented_data) - 1
        
        # Retrieve tokens
        original_ids = item["m"].tolist() if isinstance(item["m"], torch.Tensor) else item["m"]
        
        # 2. Generate Shifted variants
        for s in PITCH_SHIFTS:
            shift_map = maps[s]
            new_ids = [shift_map.get(tid, tid) for tid in original_ids]
            
            new_item = item.copy()
            new_item["m"] = torch.tensor(new_ids, dtype=torch.long)
            
            # Add variant to the giant list
            augmented_data.append(new_item)
            shifted_data_idx = len(augmented_data) - 1
            
            # 3. SAVE THE SAFE PAIR (Original Index, Shifted Index)
            pair_indices.append((orig_data_idx, shifted_data_idx))

    print(f"Augmentation complete. Total items: {len(augmented_data)}")
    print(f"Total Pairs generated: {len(pair_indices)}")
    
    # Save everything in a structured dictionary
    output_dict = {
        "data": augmented_data,
        "pairs": pair_indices 
    }
    
    print(f"Saving to {OUTPUT_CACHE_PRETRAINING}...")
    torch.save(output_dict, OUTPUT_CACHE_PRETRAINING)
    print("Done. Ready for Phase 1 Training.")

if __name__ == "__main__":
    main()