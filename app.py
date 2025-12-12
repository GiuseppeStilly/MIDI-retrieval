import gradio as gr
import os
import tempfile
import mido
from midi2audio import FluidSynth
from inference import MidiSearchEngine

# --- CONFIGURATION ---
SOUNDFONT_PATH = "soundfont.sf2"  # Path to your .sf2 file
engine = MidiSearchEngine()
engine.initialize()

def tokens_to_midi(tokens, output_path):
    """
    Converts token IDs to a MIDI file.
    Note: Replace the logic inside the loop with your specific Detokenizer/Vocabulary 
    mapping for accurate musical reconstruction.
    """
    try:
        mid = mido.MidiFile()
        track = mido.MidiTrack()
        mid.tracks.append(track)
        
        # Default placeholder logic: map tokens to notes
        # Assumes tokens correlate to pitch/duration. Adjust based on your vocabulary.
        time_accum = 0
        for t in tokens:
            if isinstance(t, int):
                # Safe mapping to MIDI range 0-127
                note = t % 128
                # Note On
                track.append(mido.Message('note_on', note=note, velocity=64, time=0))
                # Note Off (Arbitrary duration)
                track.append(mido.Message('note_off', note=note, velocity=64, time=240))
        
        mid.save(output_path)
        return True
    except Exception as e:
        print(f"MIDI Generation Error: {e}")
        return False

def render_audio(midi_path, output_wav_path):
    if not os.path.exists(SOUNDFONT_PATH):
        return None
    try:
        fs = FluidSynth(SOUNDFONT_PATH)
        fs.midi_to_audio(midi_path, output_wav_path)
        return output_wav_path
    except Exception as e:
        print(f"Audio Render Error: {e}")
        return None

def search_interface(query):
    if not query.strip():
        return [None] * 9
    
    results = engine.search(query, top_k=3)
    outputs = []
    
    for i in range(3):
        if i < len(results):
            res = results[i]
            tokens = res['midi_tokens'].tolist() if hasattr(res['midi_tokens'], 'tolist') else res['midi_tokens']
            
            # Create temp files
            temp_mid = tempfile.NamedTemporaryFile(delete=False, suffix=".mid").name
            temp_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
            
            # Generate MIDI and Audio
            has_midi = tokens_to_midi(tokens, temp_mid)
            wav_path = render_audio(temp_mid, temp_wav) if has_midi else None
            
            # Format Output
            info = f"### Result {i+1}\n**Consensus Score:** {res['rank_score']:.4f}"
            
            outputs.append(info)
            outputs.append(wav_path)
            outputs.append(temp_mid)
        else:
            outputs.extend([None, None, None])
            
    return outputs

# --- GRADIO UI ---
with gr.Blocks(title="Neural MIDI Search") as demo:
    gr.Markdown("# Ensemble Neural MIDI Retrieval")
    
    with gr.Row():
        txt_input = gr.Textbox(lines=1, placeholder="Describe the music...", label="Search Query")
        btn_submit = gr.Button("Search", variant="primary")

    # Result Slots
    for i in range(1, 4):
        with gr.Group():
            with gr.Row():
                gr.Markdown(f"### Rank {i}")
            with gr.Row():
                info_box = gr.Markdown()
                audio_player = gr.Audio(label="Preview", type="filepath")
                file_download = gr.File(label="Download MIDI", file_count="single")
                
                # Dynamic variable assignment for output mapping
                if i == 1: r1_out = [info_box, audio_player, file_download]
                elif i == 2: r2_out = [info_box, audio_player, file_download]
                elif i == 3: r3_out = [info_box, audio_player, file_download]

    all_outputs = r1_out + r2_out + r3_out
    
    btn_submit.click(search_interface, inputs=txt_input, outputs=all_outputs)
    txt_input.submit(search_interface, inputs=txt_input, outputs=all_outputs)

if __name__ == "__main__":
    demo.launch(share=True)
