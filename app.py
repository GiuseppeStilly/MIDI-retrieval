import gradio as gr
from inference import SearchEngine

# 1. Initialize Logic
# This will trigger the database build process (wait ~2 minutes)
engine = SearchEngine()

# 2. Define UI
custom_css = """
body { background-color: #0b0f19; }
"""

with gr.Blocks(theme=gr.themes.Soft(), css=custom_css) as demo:
    gr.Markdown("# Neural MIDI Search (Ensemble)")
    gr.Markdown("Search for music using natural language. This system uses an ensemble of **Bi-LSTM** and **Transformer** models.")
    
    with gr.Row():
        txt_input = gr.Textbox(
            label="Describe the music", 
            placeholder="A fast rock song with electric guitar...", 
            lines=1
        )
        btn_search = gr.Button("Search", variant="primary")
    
    with gr.Row():
        with gr.Column():
            lbl_output = gr.Textbox(label="Result Info")
        with gr.Column():
            audio_output = gr.Audio(label="Audio Preview")
            file_output = gr.File(label="Download MIDI")
        
    # Interaction Logic
    btn_search.click(
        fn=engine.search, 
        inputs=txt_input, 
        outputs=[lbl_output, audio_output, file_output]
    )

if __name__ == "__main__":
    demo.launch(share=True)
    
