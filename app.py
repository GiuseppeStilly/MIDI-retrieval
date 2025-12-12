import gradio as gr
from inference import MIDI_Search_Engine

# 1. Initialize the Engine (Runs once on startup)
print("Starting Gradio server...")
engine = MIDI_Search_Engine()

# 2. Define the Interface Function
def gradio_search(text):
    results = engine.search(text, top_k=10)
    
    # Format data for Gradio Table
    # Columns: Index, Total Score, V1 Score, V2 Score
    output_data = []
    for r in results:
        output_data.append([
            r['idx'], 
            f"{r['score']:.4f}", 
            f"{r['score_v1']:.4f}", 
            f"{r['score_v2']:.4f}"
        ])
    return output_data

# 3. Build the UI
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# Neural MIDI Search (Ensemble V1 + V2)")
    gr.Markdown("Retrieve MIDI files from the dataset using natural language descriptions.")
    
    with gr.Row():
        with gr.Column(scale=1):
            txt_input = gr.Textbox(
                label="Description", 
                placeholder="e.g., A fast rock song with electric guitar...", 
                lines=2
            )
            btn_search = gr.Button("Search MIDI", variant="primary")
        
        with gr.Column(scale=2):
            output_table = gr.Dataframe(
                headers=["Dataset Index", "Ensemble Score", "V1 Score", "V2 Score"],
                label="Top Results",
                datatype=["number", "str", "str", "str"],
                interactive=False
            )

    # Triggers (Button click or Enter key)
    btn_search.click(fn=gradio_search, inputs=txt_input, outputs=output_table)
    txt_input.submit(fn=gradio_search, inputs=txt_input, outputs=output_table)

# 4. Launch App
if __name__ == "__main__":
    demo.launch()
