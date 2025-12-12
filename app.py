import gradio as gr
from inference import EnsembleSearchEngine
import os

# 1. Initialize Logic
print("booting up AI...")
# Inizializziamo l'engine una volta sola all'avvio dell'app
try:
    engine = EnsembleSearchEngine()
    engine_status = "System Ready."
except Exception as e:
    print(f"CRITICAL ERROR: {e}")
    engine = None
    engine_status = f"System Error: {e}"

# 2. Define Custom Theme (Cyberpunk Style)
custom_theme = gr.themes.Default(
    primary_hue="pink",
    neutral_hue="slate",
).set(
    body_background_fill="#090918",
    body_background_fill_dark="#090918",
    block_background_fill="#14142B",
    block_background_fill_dark="#14142B",
    body_text_color="#FFFFFF",
    body_text_color_dark="#FFFFFF",
    block_label_text_color="#FF66C4", # Pink labels
    input_background_fill="#1E1E3F",
    input_border_color="#4B4B8B",
    button_primary_background_fill="#D6008D", # Magenta Button
    button_primary_background_fill_hover="#B50077",
    button_primary_text_color="#FFFFFF",
    block_border_width="1px",
    block_border_color="#2E2E5C"
)

# Wrapper per la funzione di ricerca
def run_search(query):
    if engine is None:
        return "Engine Initialization Failed. Check logs.", None, None
    return engine.search(query)

# 3. Define UI Structure
with gr.Blocks(theme=custom_theme, title="Neural MIDI Search") as demo:
    
    # Header
    with gr.Row():
        gr.Markdown(
            """
            # 🎹 Neural MIDI Search (Ensemble)
            Retrieving music using **Dual-Encoder Ensemble**:
            * **Model V1:** Bi-LSTM + MiniLM (Seq-level)
            * **Model V2:** Transformer + MPNet (Attentional)
            """
        )
    
    # Input Area
    with gr.Row():
        with gr.Column(scale=4):
            txt_input = gr.Textbox(
                label="Describe the music", 
                placeholder="A sad piano melody in A minor, slow tempo, expressive...", 
                lines=1,
                autofocus=True
            )
        with gr.Column(scale=1):
            btn_search = gr.Button("🔍 SEARCH", variant="primary", size="lg")
    
    # Divider
    gr.HTML("<hr style='border-color: #2E2E5C; margin: 20px 0;'>")
    
    # Output Area
    with gr.Row():
        # Left: Info
        with gr.Column(scale=1):
            lbl_info = gr.Textbox(label="Result Metadata", lines=5, interactive=False)
        
        # Right: Media
        with gr.Column(scale=1):
            aud_out = gr.Audio(label="Audio Preview", type="filepath")
            file_out = gr.File(label="Download MIDI")

    # Footer
    gr.Markdown(f"_{engine_status}_")

    # Event Listeners
    # Premendo invio nella casella di testo o cliccando il bottone
    txt_input.submit(run_search, inputs=txt_input, outputs=[lbl_info, aud_out, file_out])
    btn_search.click(run_search, inputs=txt_input, outputs=[lbl_info, aud_out, file_out])

if __name__ == "__main__":
    demo.launch(share=True, debug=True)
