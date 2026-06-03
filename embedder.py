import os
import pickle
import tiktoken
import faiss
import numpy as np
from openai import OpenAI
from loader import load_docs
from dotenv import load_dotenv

load_dotenv()

client = OpenAI()

def chunk_text(text, filename, chunk_size=200, overlap=20):
    encoder = tiktoken.get_encoding("cl100k_base")
    tokens = encoder.encode(text, disallowed_special=())
    chunks = []
    start = 0

    while start < len(tokens):
        end = start + chunk_size
        chunk_tokens = tokens[start:end]
        chunk_text = encoder.decode(chunk_tokens)
        chunks.append({
            "text": chunk_text,
            "filename": filename
        })
        start = end - overlap

    return chunks

def get_embeddings(texts):
    response = client.embeddings.create(
        input=texts,
        model="text-embedding-3-small"
    )
    return [item.embedding for item in response.data]

def build_index(docs_folder="docs"):
    documents = load_docs(docs_folder)
    all_chunks = []

    for doc in documents:
        chunks = chunk_text(doc["text"], doc["filename"])
        all_chunks.extend(chunks)

    texts = [chunk["text"] for chunk in all_chunks]
    embeddings = get_embeddings(texts)

    dimension = len(embeddings[0])
    index = faiss.IndexFlatL2(dimension)
    vectors = np.array(embeddings).astype("float32")
    index.add(vectors)

    faiss.write_index(index, "index.faiss")
    with open("chunks.pkl", "wb") as f:
        pickle.dump(all_chunks, f)

    print(f"Index built. {len(all_chunks)} chunks stored.")

if __name__ == "__main__":
    build_index()