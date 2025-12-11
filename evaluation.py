import torch
import torch.nn.functional as F
import numpy as np
import os
from tqdm import tqdm
from transformers import AutoTokenizer
from datasets import load_dataset
import config as cfg

# Safely import the model class
try:
    from model import NeuralMidiSearch
except ImportError:
    print("Error: 'model.py' not found or 'NeuralMidiSearch' class missing.")

def compute_retrieval_metrics(sim_matrix, ground_truth_indices):
    """
    Computes R@1, R@5, R@10 and MRR.
    
    Args:
        sim_matrix: Tensor of shape (num_queries, num_database_items) containing similarity scores.
        ground_truth_indices: List of integers, where the i-th integer is the column index 
                              of the correct MIDI file in sim_matrix for the i-th query.
    """
    r1, r5, r10 = 0, 0, 0
    mrr = 0
    num_queries = len(ground_truth_indices)

    # Convert to numpy for easier handling
    sim_matrix = sim_matrix.cpu().numpy()
    
    print(f"Calculating metrics for {num_queries} queries...")
    
    for i in tqdm(range(num_queries)):
        target_idx = ground_truth_indices[i]
        
        # If target is not in the database (e.g. data mismatch), skip
        if target_idx == -1:
            continue

        # Get the scores for this query
        scores = sim_matrix[i]
        
        # Get the indices of the top 10 scores (descending order)
        # We use argpartition for speed, then sort the top K
        unsorted_top_k_indices = np.argpartition(scores, -10)[-10:]
        top_k_indices = unsorted_top_k_indices[np.argsort(scores[unsorted_top_k_indices])][::-1]
        
        # Check Recalls
        if target_idx in top_k_indices[:1]:
            r1 += 1
        if target_idx in top_k_indices[:5]:
            r5 += 1
        if target_idx in top_k_indices[:10]:
            r10 += 1
            
        # Check MRR (Scan strictly the sorted ranks)
        # We need the full rank for MRR, or at least search until we find the target
        # For efficiency, we check if it's in top 100 first, else we consider rank > 100
        rank = np.where(np.argsort(scores)[::-1] == target_idx)[0][0] + 1
        mrr += 1.0 / rank

    return {
        "R@1": r1 / num_queries,
        "R@5": r5 / num_queries,
        "R@10": r10 / num_queries,
        "MRR": mrr / num_queries
    }

def main():
    print("="*60)
    print("STARTING EVALUATION")
    print("="*60)

    # 1. Load Checkpoint
    if not os.path.exists(cfg.SAVE_FILE):
        print(f"Error: Checkpoint file {cfg.SAVE_FILE} not found. Run training first.")
        return

    print(f"Loading checkpoint from {cfg.SAVE_FILE}...")
    checkpoint = torch.load(cfg.SAVE_FILE, map_location=cfg.DEVICE)
    
    # Extract Database (MIDI Embeddings)
    db_matrix = torch.tensor(checkpoint["db_matrix"]).to(cfg.DEVICE)
    db_paths = checkpoint["db_paths"]
    
    # Normalize DB embeddings for Cosine Similarity
    db_matrix = F.normalize(db_matrix, p=2, dim=1)
    
    print(f"Database loaded: {len(db_paths)} MIDI files indexed.")

    # 2. Load Model (Text Encoder only needed)
    print("Loading Text Encoder...")
    tokenizer = AutoTokenizer.from_pretrained(cfg.MODEL_NAME)
    
    # We initialize the model structure to load state dict, even if we only need text part mostly
    model = NeuralMidiSearch(checkpoint["vocab_size"]).to(cfg.DEVICE)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    # 3. Load Test Data
    print("Loading Test Dataset from HuggingFace...")
    ds = load_dataset("amaai-lab/MidiCaps", split="test") # Using 'test' split implies separate evaluation
    # If no test split exists in your specific version, use validation or filter
    # ds = ds.filter(lambda ex: ex["test_set"]) 
    
    print(f"Found {len(ds)} test captions.")

    # 4. Generate Query Embeddings (Text)
    print("Encoding text queries...")
    query_embs = []
    ground_truth_indices = []
    
    # Map db_paths to indices for quick lookup
    # We normalize paths to just filenames to avoid directory mismatches
    db_filename_map = {os.path.basename(p): i for i, p in enumerate(db_paths)}
    
    missing_files = 0
    
    with torch.no_grad():
        for item in tqdm(ds, desc="Processing Queries"):
            # 1. Encode Text
            txt_inputs = tokenizer(item["caption"], padding="max_length", truncation=True, 
                                   max_length=64, return_tensors="pt")
            
            input_ids = txt_inputs["input_ids"].to(cfg.DEVICE)
            attention_mask = txt_inputs["attention_mask"].to(cfg.DEVICE)
            
            # Forward pass through the text branch of the model
            # Assuming model has a method or sub-module for text. 
            # If V1 model calls 'model(input_ids, ...)', we use that but ignore midi output
            # But we need to isolate the text embedding generation.
            
            # Based on standard Dual Encoder architecture:
            if hasattr(model, 'encode_text'):
                txt_emb = model.encode_text(input_ids, attention_mask)
            else:
                # Fallback: Perform forward pass with dummy MIDI and extract text part
                # This is inefficient but works if the model class isn't split
                dummy_midi = torch.zeros((1, 1), dtype=torch.long).to(cfg.DEVICE)
                txt_emb, _ = model(input_ids, attention_mask, dummy_midi)

            # Normalize query
            txt_emb = F.normalize(txt_emb, p=2, dim=1)
            query_embs.append(txt_emb)
            
            # 2. Find Ground Truth Index
            # The test item has a 'location' (e.g., 'path/to/song.mid')
            # We check where this file is in our 'db_paths' list
            filename = os.path.basename(item["location"])
            
            if filename in db_filename_map:
                ground_truth_indices.append(db_filename_map[filename])
            else:
                # This happens if the test set refers to a file that wasn't included 
                # in the training/indexing phase (or download failed)
                ground_truth_indices.append(-1)
                missing_files += 1

    if missing_files > 0:
        print(f"Warning: {missing_files} test queries point to MIDI files not present in the database index.")

    # Stack queries
    query_matrix = torch.cat(query_embs, dim=0)

    # 5. Compute Metrics
    print("Computing Similarity Matrix...")
    # Matrix Multiplication (Cosine Similarity since vectors are normalized)
    # Shape: [num_queries, num_database_items]
    sim_matrix = torch.matmul(query_matrix, db_matrix.T)
    
    results = compute_retrieval_metrics(sim_matrix, ground_truth_indices)
    
    print("\n" + "="*30)
    print("FINAL RESULTS")
    print("="*30)
    print(f"R@1:  {results['R@1']:.4f}")
    print(f"R@5:  {results['R@5']:.4f}")
    print(f"R@10: {results['R@10']:.4f}")
    print(f"MRR:  {results['MRR']:.4f}")
    print("="*30)

if __name__ == "__main__":
    main()