import gradio as gr
from inference import MidiSearchEngine

# Initialize Engine
engine = MidiSearchEngine()
engine.initialize()

def format_results(results):
    output = []
    for i, res in enumerate(results):
        score_percentage = res['rank_score'] * 100
        
        # Determine which model contributed more
        v1_conf = res['v1_score']
        v2_conf = res['v2_score']
        dominant = "Balanced"
        if v1_conf > v2_conf + 0.05: dominant = "Temporal (V1)"
        if v2_conf > v1_conf + 0.05: dominant = "Semantic (V2)"

        md = f"""
        ### Result {i+1}
        * **Consensus Score:** {res['rank_score']:.4f}
        * **V1 Score:** {res['v1_score']:.4f} | **V2 Score:** {res['v2_score']:.4f}
        * **Dominant Model:** {dominant}
        * **Token Count:** {len(res['midi_tokens'])}
        ---
        """
        output.append(md)
    return "\n".join(output)

def search_interface(query):
    if not query.strip():
        return "Please enter a search query."
    
    try:
        results = engine.search(query, top_k=5)
        return format_results(results)
    except Exception as e:
        return f"Error: {str(e)}"

# Gradio UI
with gr.Blocks(title="Ensemble MIDI Retrieval") as demo:
    gr.Markdown("# Ensemble Neural MIDI Retrieval")
    gr.Markdown("Bi-LSTM + Transformer Late Fusion Strategy")
    
    with gr.Row():
        txt_input = gr.Textbox(lines=1, placeholder="Describe the music (e.g., 'A fast jazz piano solo')...", label="Search Query")
        btn_submit = gr.Button("Search", variant="primary")
    
    with gr.Row():
        txt_output = gr.Markdown(label="Retrieval Results")
        
    btn_submit.click(search_interface, inputs=txt_input, outputs=txt_output)
    txt_input.submit(search_interface, inputs=txt_input, outputs=txt_output)

if __name__ == "__main__":
    demo.launch()
