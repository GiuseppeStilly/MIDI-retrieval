# MIDI-retrieval

# Neural MIDI Search (Transformer Edition)

Team composition: [Tuo Nome], [Nome Amico]

This repository contains the source code for a Cross-Modal Information Retrieval system designed to align Natural Language with Symbolic Music (MIDI). The system enables semantic search of musical files using descriptive text queries.

The goal is to bridge the gap between semantic descriptions (e.g., "A sad piano melody in A minor") and symbolic music representations by mapping both into a shared 384-dimensional embedding space.

## Approach

The project leverages a **Two-Tower (Dual Encoder)** architecture trained using Contrastive Learning. The system consists of a "Teacher" text encoder and a "Student" MIDI encoder optimized via Margin Ranking Loss.

### 1. Architecture & Encoders

The model aligns two distinct modalities into a joint embedding space:

* **Text Encoder (Teacher):** We utilize `sentence-transformers/all-mpnet-base-v2`, a pre-trained Transformer that produces robust semantic embeddings (Dimension: 384). This branch is frozen during training to serve as a stable target.
* **MIDI Encoder (Student):** A custom 4-layer Transformer Encoder designed from scratch.
    * **Tokenization:** Uses the REMI (Revamped MIDI) representation via `miditok`, capturing Pitch, Velocity, Duration, and Chords.
    * **Specs:** 6 Attention Heads, 384 Embedding Dimension, 1024 Feed-Forward dimension.
    * **Augmentation:** Implements Random Cropping during training to improve generalization on variable-length sequences.

### 2. Training & Inference

The model is trained to minimize the distance between a MIDI file and its correct caption while maximizing the distance from incorrect ones.

* **Loss Function:** A **Margin Ranking Loss** (margin=0.3) is used with Hard Negative Mining to penalize the model only when incorrect matches are ranked higher than the correct one.
* **Optimization:** Trained using AdamW with a linear warmup scheduler and Mixed Precision (AMP) for efficiency.
* **Retrieval:** At inference time, both the query and the database are encoded. We compute Cosine Similarity between the query vector and all MIDI vectors to rank the results.

## Repository Structure

* `config.py`: Contains all global constants, hyperparameters (e.g., `EMBED_DIM`, `BATCH_SIZE`), and path configurations.
* `model.py`: Contains the PyTorch implementation of the `NeuralMidiSearchTransformer` class, including the Positional Encoding and the Text/MIDI projection layers.
* `train.py`: Contains the training loop, the `MidiCapsDataset` class for data loading, and the validation logic using Hard Negative Mining.
* `inference.py`: Contains the `SearchEngine` class that handles model loading, embedding generation, and the audio rendering pipeline using FluidSynth.
* `app.py`: Contains the Gradio web application definition, implementing the custom UI theme and interaction logic.
* `test.py`: Contains the official evaluation script that downloads the trained model from Hugging Face and computes retrieval metrics (R@1, R@5, R@10).
