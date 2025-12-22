## Model V1: Recurrent Baseline (Bi-LSTM)

This version implements the **baseline dual-encoder architecture** for text-to-MIDI retrieval, focusing on computational efficiency and sequential modeling.

* **Text Encoder:** We utilize **`all-MiniLM-L6-v2`**, a lightweight Sentence-BERT model, to encode user queries into compact semantic embeddings.
* **MIDI Encoder:** The symbolic music is processed by a **Bidirectional LSTM (Bi-LSTM)**. This recurrent network is optimized for capturing local temporal dependencies, such as rhythmic patterns and short melodic motifs.
* **Use Case:** Ideal for scenarios requiring low latency and efficient training on limited hardware.
