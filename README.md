## Model V2: Transformer High-Capacity

This version represents the **advanced implementation** of our system, leveraging attention mechanisms to capture global musical structures and complex semantic nuances.

* **Text Encoder:** We upgrade to **`all-mpnet-base-v2`**, a powerful Transformer model that provides a deeper understanding of complex natural language descriptions.
* **MIDI Encoder:** We replace the recurrent layers with a custom **Transformer Encoder** (trained from scratch). Thanks to Multi-Head Self-Attention, this model can process the entire sequence in parallel, effectively capturing long-range dependencies and overall song structure.
* **Use Case:** Designed for maximum retrieval accuracy (Recall@K) and handling complex, descriptive queries.
