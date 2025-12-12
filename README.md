
# Neural MIDI Search 
Team composition: Edoardo Besteghi, Riccoardo Bucchi D'Incecco, Giuseppe Stillitano

This repository contains the source code for a **Cross-Modal Information Retrieval** system designed to align Natural Language with Symbolic Music (MIDI). The system enables semantic search of musical files using descriptive text queries, leveraging state-of-the-art Transformer architectures.

The goal is to bridge the "modality gap" between semantic descriptions (e.g., *"A sad piano melody in A minor with a slow build-up"*) and symbolic music representations by mapping both into a shared, high-dimensional embedding space.

## Approach

The project leverages a **Two-Tower (Dual Encoder)** architecture trained using Contrastive Learning. This approach treats the problem as a representation learning task, where a "Student" MIDI model learns to align its output space with a pre-trained "Teacher" Text model.

### 1. Feature Engineering & Tokenization
Unlike raw audio, MIDI is symbolic. We process the data using the **REMI (Revamped MIDI)** tokenization scheme via `miditok`.
* **Event-Based Representation:** Music is converted into a sequence of tokens representing Pitch, Velocity (dynamics), Duration, and Chords.
* **Data Augmentation:** To improve generalization and handle variable-length sequences, we implement **Random Cropping** during the training phase, sampling fixed-length windows (512 tokens) from longer musical compositions.

### 2. Model Architecture
The system consists of two parallel encoders projecting data into a shared **384-dimensional embedding space**:

* **Text Encoder (The Teacher):**
    * We utilize `sentence-transformers/all-mpnet-base-v2`, a powerful pre-trained Transformer.
    * This branch produces robust semantic embeddings and is **frozen** during training to serve as a stable target for the MIDI encoder.
* **MIDI Encoder (The Student):**
    * A custom **4-layer Transformer Encoder** designed from scratch for this task.
    * **Specifications:** 6 Attention Heads, 384 Embedding Dimension, 1024 Feed-Forward dimension, GeLU activation.
    * **Positional Encoding:** Standard Sinusoidal encoding is added to preserve the temporal order of musical events.

### 3. Training Strategy
The model is trained to minimize the distance between a MIDI file and its correct caption while maximizing the distance from incorrect ones (Contrastive Learning).

* **Objective Function:** We employ **Margin Ranking Loss** (margin = 0.3). This loss function ensures that the similarity score of a positive pair (Text, MIDI) is higher than any negative pair by at least the specified margin.
* **Hard Negative Mining:** Instead of random negatives, the training loop computes the similarity matrix for the entire batch (Batch Size = 256) and treats all non-diagonal elements as negative samples. This pushes the model to distinguish between closely related musical concepts.
* **Optimization:** The training utilizes the **AdamW** optimizer with a linear warmup scheduler and **Mixed Precision (AMP)** to accelerate training on GPU.

## Repository Structure

### Core Configuration
* `config.py`: Contains all global constants, hyperparameters (e.g., `EMBED_DIM`, `BATCH_SIZE`, `LEARNING_RATE`), and dynamic path configurations for local vs. Colab environments.

### Modeling & Training
* `model.py`: Defines the `NeuralMidiSearchTransformer` class. It includes the logic for the frozen MPNet text encoder, the learnable MIDI Transformer, and the custom Positional Encoding.
* `train.py`: The main training script. It implements:
    * The `MidiCapsDataset` class for efficient data loading.
    * The training loop with **Gradient Scaling** (for AMP).
    * Checkpoint saving and validation logic using the Margin Ranking Loss.

### Inference & Application
* `inference.py`: Contains the `SearchEngine` class. This module handles:
    * Loading the trained model weights.
    * Tokenizing input text queries on the fly.
    * Performing Cosine Similarity search against the indexed database.
    * **Audio Rendering:** A pipeline that converts retrieved MIDI tokens to WAV audio using **FluidSynth** for immediate playback.
* `app.py`: A **Gradio** web application that provides a user-friendly interface. It features a custom "Neon/Dark" theme and allows users to input text, view retrieval results, and listen to the generated audio.
* `test.py`: The official evaluation script. It downloads the trained model from Hugging Face, runs it on the held-out test set, and computes standard Information Retrieval metrics (**Recall@1**, **Recall@5**, **Recall@10** and **Median Rank**).
