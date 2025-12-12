import gradio as gr
from inference import SearchEngine

# 1. Initialize Logic
print("Initializing Transformer Search Engine...")
engine = SearchEngine()

# 2. Define Custom Theme
# We create a custom theme to match the "Neural MIDI Search" presentation:
# - Background: Deep Dark Blue (representing the slide background)
# - Accents: Neon Pink (representing arrows and highlights)
custom_theme = gr.themes.Default(
    primary_hue="pink",   # Base color for primary actions (Pink)
    neutral_hue="slate",  # Base color for structure (Cool dark grays)
).set(
    # --- Background Colors ---
    body_background_fill="#090918",        # Very dark blue/purple (Main background)
    body_background_fill_dark="#090918",   
    block_background_fill="#14142B",       # Slightly lighter blue (Container background)
    block_background_fill_dark="#14142B",
    
    # --- Text Colors ---
    body_text_color="#FFFFFF",             # White text for readability
    body_text_color_dark="#FFFFFF",
    block_label_text_color="#FF66C4",      # Light pink for labels (Query, Result Info)
    
    # --- Input Fields ---
    input_background_fill="#1E1E3F",       # Dark blue for textboxes
    input_background_fill_dark="#1E1E3F",
    input_border_color="#4B4B8B",
    
    # --- Buttons (The "Pop" Color) ---
    button_primary_background_fill="#D6008D",       # Vibrant Magenta/Pink 
    button_primary_background_fill_hover="#B50077", # Darker pink on hover
    button_primary_text_color="#FFFFFF",
    
    # --- Borders ---
    block_border_width="1px",
    block_border_color="#2E2E5C"
)

# 3. Define UI
with gr.Blocks(theme=custom_theme) as demo:
    # Header Section
    gr.Markdown("# AI MIDI Search (Transformer Edition)")
    gr.Markdown("Searching using **MPNet** + **4-Layer Transformer** architecture.")
    
    # Input Section
    with gr.Row():
        txt = gr.Textbox(
            label="Text Query", 
            placeholder="A sad piano melody in A minor...", 
            lines=1
        )
        btn = gr.Button("SEARCH", variant="primary", scale=0) # scale=0 keeps button small
    
    # Output Section
    with gr.Row():
        lbl = gr.Textbox(label="Result Info")
        aud = gr.Audio(label="Preview")
        file = gr.File(label="Download MIDI")
        
    # Interaction Logic
    btn.click(engine.search, inputs=txt, outputs=[lbl, aud, file])

if __name__ == "__main__":
    demo.launch(share=True)
