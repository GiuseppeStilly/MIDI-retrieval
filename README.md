# Ensemble Neural MIDI Retrieval System

This repository implements a **Cross-Modal Information Retrieval System** designed to search for MIDI music files using natural language queries.

The core of the project is an **Ensemble Architecture** that combines the strengths of two distinct neural network models (Recurrent and Transformer-based) to map text and symbolic music into a unified vector space.

## System Architecture: The Ensemble Approach

To achieve robust retrieval performance, the system does not rely on a single model. Instead, it utilizes an **Ensemble Strategy** that fuses predictions from two different architectures. This allows the system to capture both the long-range semantic dependencies and the sequential rhythmic nuances of music.

### 1. Model V1: The Temporal Expert (Bi-LSTM)
* **Architecture:** Bidirectional LSTM (Long Short-Term Memory).
* **Role:** Optimized for processing sequential data, this model excels at capturing the temporal flow, rhythm, and local melodic structures of the MIDI files.
* **Text Encoder:** `sentence-transformers/all-MiniLM-L6-v2` (Optimized for speed/efficiency).

### 2. Model V2: The Semantic Expert (Transformer)
* **Architecture:** Custom 4-Layer Transformer.
* **Role:** Utilizing Self-Attention mechanisms, this model captures global context and complex harmonic relationships within the music, aligning them with high-level semantic descriptions.
* **Text Encoder:** `sentence-transformers/all-mpnet-base-v2` (State-of-the-art semantic embedding).

### 3. Ensemble Fusion (Late Fusion)
During inference, the system processes the user's query through both V1 and V2 pipelines simultaneously.
1.  **V1 Inference:** Calculates the Cosine Similarity score between the query and the V1 database.
2.  **V2 Inference:** Calculates the Cosine Similarity score between the query and the V2 database.
3.  **Score Averaging:** The final relevance score is computed as the arithmetic mean of the two:
    $$Score_{final} = \frac{Score_{V1} + Score_{V2}}{2}$$

This "consensus" approach reduces variance and improves the ranking of retrieved results.

## Training Methodology

Both models were trained using **Symmetric Contrastive Loss** (InfoNCE), a technique popularized by models like CLIP.

The objective is to learn a joint embedding space where:
* The embedding of a song and its correct text description are pulled close together (Maximizing Similarity).
* The embedding of a song and unrelated text descriptions are pushed apart (Minimizing Similarity).

$$ 
\mathcal{L} = - \frac{1}{N} \sum_{i=1}^{N} \log \frac{\exp(sim(T_i, M_i) / \tau)}{\sum_{j=1}^{N} \exp(sim(T_i, M_j) / \tau)} 
$$

*(Where $T$ is the text vector, $M$ is the music vector, and $\tau$ is a learnable temperature parameter).*

## Dynamic Inference Engine

The system features a **Dynamic Indexing** mechanism to ensure flexibility and consistency:

* **Auto-Build:** Upon initialization, the inference engine loads the raw tokenized dataset (approx. 150k songs). It processes this data in real-time using the loaded model weights to construct the vector database in memory.
* **Retrieval:** The system performs a dual-search (V1 + V2) over this generated index to retrieve the top-k most relevant MIDI files.

Quick Start (Google Colab)
'''python
import os

# Configuration
GITHUB_USERNAME = "GiuseppeStilly" 
REPO_NAME = "MIDI-Retrieval" 

# 1. Install System Audio Drivers (FluidSynth)
print("Installing FluidSynth...")
!apt-get update -qq
!apt-get install -y fluidsynth fluid-soundfont-gm -qq

# 2. Clone Repository
if not os.path.exists(REPO_NAME):
    print(f"Cloning {REPO_NAME}...")
    !git clone https://github.com/{GITHUB_USERNAME}/{REPO_NAME}.git
else:
    %cd {REPO_NAME}
    !git pull
    %cd ..

# 3. Install Python Dependencies
print("Installing Python libraries...")
%cd {REPO_NAME}
!pip install -q -r requirements.txt

# 4. Launch Application
print("Launching Application...")
!python app.py
'''
## Installation & Usage

### Prerequisites
* Python 3.8+
* **FluidSynth** (Required for rendering MIDI preview audio)

```bash
# Linux / Google Colab System Dependency
sudo apt-get install -y fluidsynth

# Python Dependencies
pip install torch transformers huggingface_hub midi2audio sklearn gradio pyfluidsynth tqdm
