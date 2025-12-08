import torch
import os
from tqdm import tqdm
from miditok import REMI, TokenizerConfig
import config as cfg

# CONFIGURATION
INPUT_CACHE = cfg.CACHE_FILE  # e.g., dataset_midicaps_tokenized.pt
OUTPUT_CACHE = os.path.join(cfg.BASE_DIR, "dataset_midicaps_AUGMENTED.pt")
PITCH_SHIFTS = [-2, 2] # Shift down 2 semitones, shift up 2 semitones

def get_pitch_shift_map(tokenizer, semitones):
    """
    Creates a mapping dictionary to translate token IDs for pitch shifting.
    """
    mapping = {}
    vocab = tokenizer.vocab
    
    # Invert vocab to find token string by ID
    id_to_token = {v: k for k, v in vocab.items()}
    
    for old_id, token_str in id_to_token.items():
        if token_str.startswith("Pitch_"):
            try:
                # Extract pitch value (e.g., Pitch_60 -> 60)
                val = int(token_str.split("_")[1])
                new_val = val + semitones
                
                # Construct new token string
                new_token_str = f"Pitch_{new_val}"
                
                # If new token exists in vocab, map it
                if new_token_str in vocab:
                    mapping[old_id] = vocab[new_token_str]
                else:
                    # If out of piano range, keep original or map to silence
                    mapping[old_id] = old_id
            except:
                mapping[old_id] = old_id
        else:
            # Keep non-pitch tokens (Velocity, Duration, etc.) unchanged
            mapping[old_id] = old_id
            
    return mapping

def main():
    print(f"Loading original cache: {INPUT_CACHE}...")
    if not os.path.exists(INPUT_CACHE):
        print("Error: Input cache not found. Run training once to generate the base cache first.")
        return

    original_data = torch.load(INPUT_CACHE)
    print(f"Original samples: {len(original_data)}")

    # Initialize Tokenizer to understand token IDs
    tokenizer = REMI(TokenizerConfig(num_velocities=16, use_chords=True))
    
    # Pre-calculate translation maps
    maps = {}
    for s in PITCH_SHIFTS:
        maps[s] = get_pitch_shift_map(tokenizer, s)

    augmented_data = []
    
    print("Generating augmented data...")
    for item in tqdm(original_data):
        # 1. Keep Original
        augmented_data.append(item)
        
        # 2. Generate Shifted Versions
        original_ids = item["m"].tolist()
        
        for s in PITCH_SHIFTS:
            shift_map = maps[s]
            # Apply mapping: if ID is in map use new ID, else use old ID
            new_ids = [shift_map.get(tid, tid) for tid in original_ids]
            
            new_item = item.copy()
            new_item["m"] = torch.tensor(new_ids, dtype=torch.long)
            # Caption remains the same
            augmented_data.append(new_item)

    print(f"Augmentation complete. New size: {len(augmented_data)}")
    
    print(f"Saving to {OUTPUT_CACHE}...")
    torch.save(augmented_data, OUTPUT_CACHE)
    print("Done.")

if __name__ == "__main__":
    main()