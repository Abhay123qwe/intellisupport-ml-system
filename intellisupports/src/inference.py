# src/inference.py

import torch
import pickle
import numpy as np
import os

from intellisupports.src.model import TicketClassifier
from intellisupports.src.embeddings import EmbeddingGenerator
from intellisupports.src.retrieve import TicketRetriever


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")

class IntelliSupportPredictor:
    def __init__(self):
        print("Loading model and artifacts...")

        with open(os.path.join(ARTIFACTS_DIR, "label_encoder.pkl"), "rb") as f:
            label_encoder = pickle.load(f)

        self.label_encoder = label_encoder

        input_dim = 384  # all-MiniLM-L6-v2 embedding size
        num_classes = len(self.label_encoder.classes_)

        self.model = TicketClassifier(input_dim, num_classes)
        self.model.load_state_dict(
            torch.load(f"{ARTIFACTS_DIR}/model.pt", map_location="cpu")
        )
        self.model.eval()

        self.embedder = EmbeddingGenerator()
        self.retriever = TicketRetriever()

    def predict(self, text: str, top_k: int = 5):
        embedding = self.embedder.encode([text])
        embedding_tensor = torch.tensor(embedding, dtype=torch.float32)
        
        

        with torch.no_grad():
            logits = self.model(embedding_tensor)
            probs = torch.softmax(logits, dim=1)   # convert logits → probabilities
            confidence, pred_class = torch.max(probs, dim=1)

            confidence = confidence.item() * 100    # convert to percentage
            pred_class = pred_class.item()

        category = self.label_encoder.inverse_transform([pred_class])[0]

        if confidence < 50:
            category = "Uncertain - Requires Human Review"
        
        similar_tickets = self.retriever.search(text, top_k=top_k)
        
        return {
            "predicted_category": category,
            "confidence": round(confidence, 2),
            "similar_tickets": similar_tickets
        }
