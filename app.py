import gradio as gr
from inference import SearchEngine

# Initialize Logic
print("Initializing Transformer Search Engine...")
engine = SearchEngine()

# Define UI
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# AI MIDI Search (Transformer Edition)")
    gr.Markdown("Searching using MPNet + 4-Layer Transformer architecture.")
    
    with gr.Row():
        txt = gr.Textbox(label="Query", placeholder="A sad piano melody in A minor...")
        btn = gr.Button("SEARCH", variant="primary")
    
    with gr.Row():
        lbl = gr.Textbox(label="Result Info")
        aud = gr.Audio(label="Preview")
        file = gr.File(label="Download MIDI")
        
    btn.click(engine.search, inputs=txt, outputs=[lbl, aud, file])

if __name__ == "__main__":
    demo.launch(share=True)