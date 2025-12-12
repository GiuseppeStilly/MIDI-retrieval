import sys
import os
from inference import MidiSearchEngine  # Imports your engine
from miditok import REMI, TokenizerConfig      # Imports the decoder

def main():
    # 1. Initialize the Search Engine
    # This will load the database and the models (takes ~1 minute)
    engine = MidiSearchEngine()
    success = engine.initialize()
    
    if not success:
        print("Failed to initialize engine.")
        return

    # 2. Setup the Decoder (Must match Training Config!)
    # The model speaks "REMI" language with specific settings.
    tokenizer = REMI(TokenizerConfig(num_velocities=16, use_chords=True))

    while True:
        # 3. Ask user for input
        print("\n" + "="*40)
        query = input("Describe the music you want (or 'q' to quit): ")
        
        if query.lower() == 'q':
            break
            
        print(f"Searching for: '{query}'...")
        
        try:
            # 4. Perform Search
            results = engine.search(query, top_k=1) # Get the best match
            
            if not results:
                print("No results found.")
                continue
                
            best_match = results[0]
            tokens = best_match['midi_tokens']
            score = best_match['rank_score']
            
            print(f"Found match! Score: {score:.4f}")
            
            # 5. Convert Tokens -> MIDI File
            # miditok expects a list of tracks, so we wrap tokens in []
            if hasattr(tokens, 'tolist'):
                tokens = tokens.tolist()
                
            midi_obj = tokenizer.tokens_to_midi([tokens])
            
            # 6. Save the file
            filename = "output_result.mid"
            midi_obj.dump(filename)
            
            print(f"SUCCESS: Saved music to '{filename}'")
            print("Open this file with your media player to listen!")
            
        except Exception as e:
            print(f"Error during processing: {e}")

if __name__ == "__main__":
    main()
