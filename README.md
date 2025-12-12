# Neural MIDI Retrieval System

This repository implements a **Cross-Modal Information Retrieval** system capable of searching for MIDI music files using natural language queries. 

The core of the project is a Deep Learning architecture that learns a joint embedding space for both **Text** and **Symbolic Music** (MIDI), allowing users to find music based on semantic descriptions (e.g., *"A melancholic piano melody in A minor"*).

## Methodology

The system relies on a **Bi-Encoder Architecture** designed to map two different modalities (Text and Music) into a shared high-dimensional vector space.

### 1. Model Architecture (V2)
The architecture consists of two parallel streams:

* **Text Encoder:** We utilize a pre-trained **MPNet** (`sentence-transformers/all-mpnet-base-v2`), which is frozen or fine-tuned to extract rich semantic features from the text queries.
* **Music Encoder:** A custom **Transformer-based Neural Network**. It processes MIDI files as sequences of tokens (pitch, velocity, duration) to generate a dense vector representation of the musical content.

### 2. Training Objective: Contrastive Loss
To align the text and music representations, the model is trained using **Symmetric Contrastive Loss** (CLIP-style loss). 

The objective function works by:
* **Maximizing** the cosine similarity between matched pairs (Correct Text, Correct Music).
* **Minimizing** the similarity between unmatched pairs (Correct Text, Random Music) within the same batch.

This forces the model to "pull" the vectors of a song and its correct description closer together while "pushing" unrelated songs away.

$$ 
\mathcal{L} = - \frac{1}{N} \sum_{i=1}^{N} \log \frac{\exp(sim(T_i, M_i) / \tau)}{\sum_{j=1}^{N} \exp(sim(T_i, M_j) / \tau)} 
$$

*(Where $T$ and $M$ are text and music embeddings, and $\tau$ is the temperature parameter).*

## Inference Pipeline

The inference engine performs efficient similarity search without requiring a pre-indexed database file.

1.  **Dynamic Indexing:** Upon initialization, the system automatically loads the tokenized dataset and computes the embeddings for over 150,000 songs in real-time. This ensures the search index is always synchronized with the current model weights.
2.  **Retrieval:** When a user inputs a query:
    * The text is encoded into a vector $V_{text}$.
    * The system calculates the **Cosine Similarity** between $V_{text}$ and the entire music database matrix.
    * The top-k results are retrieved and synthesized into audio for preview.

## Installation & Usage

### Prerequisites
* Python 3.8+
* FluidSynth (for MIDI-to-Audio rendering)

```bash
# Install FluidSynth (Linux/Colab)
sudo apt-get install -y fluidsynth

# Install Python dependencies
pip install -r requirements.txt
