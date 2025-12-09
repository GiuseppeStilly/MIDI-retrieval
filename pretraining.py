import torch
import os
from tqdm import tqdm
from miditok import REMI, TokenizerConfig
import config as cfg

# CONFIGURATION
INPUT_CACHE = cfg.CACHE_FILE
OUTPUT_CACHE_PRETRAINING = os.path.join(cfg.BASE_DIR, "dataset_midicaps_MIDI_PRETRAINING.pt")

PITCH_SHIFTS = [-3, -1, 1, 3]  # Expanded: more diverse pitch shifts (semantic-preserving for MIDI)

def get_pitch_shift_map(tokenizer, semitones):
    """
    Creates a mapping dictionary to translate token IDs for pitch shifting.
    """
    mapping = {}
    vocab = tokenizer.vocab
    id_to_token = {v: k for k, v in vocab.items()}
    
    for old_id, token_str in id_to_token.items():
        if token_str.startswith("Pitch_"):
            try:
                val = int(token_str.split("_")[1])
                new_val = val + semitones
                new_token_str = f"Pitch_{new_val}"
                
                # Keep within piano range [0, 127]
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

    # Initialize Tokenizer
    tokenizer = REMI(TokenizerConfig(num_velocities=16, use_chords=True))

    # Pre-calculate translation maps
    maps = {}
    for s in PITCH_SHIFTS:
        maps[s] = get_pitch_shift_map(tokenizer, s)

    augmented_data = []

    print("Generating MIDI-only pretraining data (with pitch shifts as positive pairs)...")
    for item in tqdm(original_data):
        # 1. Keep Original
        augmented_data.append(item)

        # 2. Generate Shifted Versions (POSITIVE PAIRS for contrastive learning)
        original_ids = item["m"].tolist() if isinstance(item["m"], torch.Tensor) else item["m"]
        
        for s in PITCH_SHIFTS:
            shift_map = maps[s]
            new_ids = [shift_map.get(tid, tid) for tid in original_ids]
            
            new_item = item.copy()
            new_item["m"] = torch.tensor(new_ids, dtype=torch.long)
            # Keep caption the same (we'll use it in Phase 2)
            augmented_data.append(new_item)

    print(f"Augmentation complete. New size: {len(augmented_data)} (original {len(original_data)} + {len(PITCH_SHIFTS)} shifts each)")
    print(f"Saving to {OUTPUT_CACHE_PRETRAINING}...")
    torch.save(augmented_data, OUTPUT_CACHE_PRETRAINING)
    print("Done.")

if __name__ == "__main__":
    main()
