# src/train.py

import os
import pickle
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score

from model import TicketClassifier

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")

class TicketDataset(Dataset):
    def __init__(self, embeddings, labels):
        self.X = torch.tensor(embeddings, dtype=torch.float32)
        self.y = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def main():
    print("Loading embeddings & metadata...")
    with open(f"{ARTIFACTS_DIR}/embeddings.pkl", "rb") as f:
        embeddings = pickle.load(f)

    with open(f"{ARTIFACTS_DIR}/metadata.pkl", "rb") as f:
        metadata = pickle.load(f)

    category = [item["category"] for item in metadata]

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(category)

    with open(f"{ARTIFACTS_DIR}/label_encoder.pkl", "wb") as f:
        pickle.dump(label_encoder, f)

    dataset = TicketDataset(embeddings, y)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    input_dim = embeddings.shape[1]
    num_classes = len(label_encoder.classes_)

    model = TicketClassifier(input_dim, num_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    print("Training classifier...")
    model.train()
    for epoch in range(10):
        epoch_loss = 0
        preds, targets = [], []

        for X_batch, y_batch in dataloader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            preds.extend(outputs.argmax(1).tolist())
            targets.extend(y_batch.tolist())

        acc = accuracy_score(targets, preds)
        f1 = f1_score(targets, preds, average="weighted")

        print(
            f"Epoch {epoch+1} | "
            f"Loss: {epoch_loss:.4f} | "
            f"Acc: {acc:.4f} | "
            f"F1: {f1:.4f}"
        )

    torch.save(model.state_dict(), f"{ARTIFACTS_DIR}/model.pt")
    print("Model saved successfully ✅")


if __name__ == "__main__":
    main()
