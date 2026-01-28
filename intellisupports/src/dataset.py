import os
import re
import pandas as pd
from typing import List

def clean_text(text: str) -> str:
    """
    Basic text cleaning for support train
    """
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_documents(csv_path: str) -> pd.DataFrame:
    """
    Load support train dataset
    Expected columns: ['text', 'category']
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"{csv_path} not found")

    df = pd.read_csv(csv_path)
    if "text" not in df.columns or "category" not in df.columns:
        raise ValueError("CSV must contain 'text' and 'category' columns")
    df["text"] = df["text"].astype(str).apply(clean_text)

    return df


def chunk_text(text: str, chunk_size: int = 100, overlap: int = 20) -> List[str]:
    """
    Chunk long tickets into overlapping pieces
    """
    words = text.split()
    chunks = []

    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = words[start:end]
        chunks.append(" ".join(chunk))
        start += chunk_size - overlap

    return chunks
