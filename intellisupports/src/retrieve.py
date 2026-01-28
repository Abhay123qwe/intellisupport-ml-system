# src/retrieve.py

import faiss
import os
import pickle
import numpy as np
from intellisupports.src.embeddings import EmbeddingGenerator

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")

class TicketRetriever:
    def __init__(self):
        print("Loading embeddings and metadata...")
        with open(f"{ARTIFACTS_DIR}/embeddings.pkl", "rb") as f:
            self.embeddings = pickle.load(f)

        with open(f"{ARTIFACTS_DIR}/metadata.pkl", "rb") as f:
            self.metadata = pickle.load(f)

        self.embeddings = np.array(self.embeddings).astype("float32")

        dim = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)  # cosine similarity (normalized)
        self.index.add(self.embeddings) # type: ignore

        self.embedder = EmbeddingGenerator()

    def search(self, query: str, top_k: int = 5):
        query_emb = self.embedder.encode([query]).astype("float32")
        scores, indices = self.index.search(query_emb, top_k) # type: ignore

        results = []
        for idx, score in zip(indices[0], scores[0]):
            item = self.metadata[idx]
            results.append({
                "category": item["category"],
                "score": float(score),
                "original_index": item["original_index"]
            })

        return results
