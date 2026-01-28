import os
import pickle
from dataset import load_documents, chunk_text
from embeddings import EmbeddingGenerator

# ✅ project root = intellisupports/
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(BASE_DIR, "data", "raw", "train.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "artifacts")

os.makedirs(OUTPUT_DIR, exist_ok=True)

def main():
    print("Loading dataset...")
    df = load_documents(DATA_PATH)

    all_chunks = []
    metadata = []

    print("Chunking documents...")
    for idx, row in df.iterrows():
        chunks = chunk_text(row["text"])
        for chunk in chunks:
            all_chunks.append(chunk)
            metadata.append({
                "category": row["category"],
                "original_index": idx
            })

    print(f"Total chunks: {len(all_chunks)}")

    print("Generating embeddings...")
    embedder = EmbeddingGenerator()
    embeddings = embedder.encode(all_chunks)

    print("Saving artifacts...")
    with open(os.path.join(OUTPUT_DIR, "embeddings.pkl"), "wb") as f:
        pickle.dump(embeddings, f)

    with open(os.path.join(OUTPUT_DIR, "metadata.pkl"), "wb") as f:
        pickle.dump(metadata, f)

    print("Day 1 completed successfully ✅")

if __name__ == "__main__":
    main()
