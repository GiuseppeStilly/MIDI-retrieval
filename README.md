# MIDI-retrieval
# AI MIDI Retrieval System (Text-to-MIDI)

This repository hosts a **Neural MIDI Retrieval System** that allows users to search for MIDI files using natural language queries (e.g., *"A sad piano melody in A minor"*). 

The system projects both **Text** and **Music** into a shared embedding space, enabling semantic search via Cosine Similarity.

##  Project Goal
The objective was to deploy a pre-trained cross-modal search engine capable of retrieving music from the **MidiCaps dataset** (approx. 150k MIDI files). 

### The Challenge & Solution
During deployment, we identified that the pre-trained model checkpoints contained the neural network weights (the "Brain") but lacked the pre-computed vector database (the "Memory/Index" of the songs).

**Our Solution:**
We implemented a robust **Auto-Build Mechanism** in the inference engine.
1. On startup, the system checks if the model checkpoint contains the database matrix.
2. If missing, it automatically downloads the `tokenized_dataset` from Hugging Face.
3. It uses the model to process the 150k songs in real-time (in memory) to reconstruct the vector index.
4. This ensures the app works "out of the box" without requiring manual preprocessing scripts.

##  Architecture

The system supports two architectures, though **V2** is the recommended default:

* **V2 (Transformer):**
    * **Text Encoder:** `sentence-transformers/all-mpnet-base-v2`
    * **Music Encoder:** A 4-layer Transformer trained to align with MPNet embeddings.
    * **Performance:** Higher accuracy in semantic understanding.

* **V1 (Bi-LSTM):**
    * **Text Encoder:** `sentence-transformers/all-MiniLM-L6-v2`
    * **Music Encoder:** Bidirectional LSTM.
    * **Status:** Legacy/Optimized for speed.

##  Installation

### Prerequisites
* Python 3.8+
* **FluidSynth** (Required for converting MIDI to Audio for preview)

### 1. System Dependencies (Linux/Colab)
```bash
sudo apt-get install -y fluidsynth
