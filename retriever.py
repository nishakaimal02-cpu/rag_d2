import pickle
import faiss
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI()

def load_index():
    index = faiss.read_index("index.faiss")
    with open("chunks.pkl", "rb") as f:
        chunks = pickle.load(f)
    return index, chunks

def get_query_embedding(question):
    response = client.embeddings.create(
        input=[question],
        model="text-embedding-3-small"
    )
    return response.data[0].embedding

def retrieve_chunks(question, index, chunks, k=3):
    query_vector = get_query_embedding(question)
    query_array = np.array([query_vector]).astype("float32")
    distances, indices = index.search(query_array, k)
    
    results = []
    for i, idx in enumerate(indices[0]):
        results.append({
            "text": chunks[idx]["text"],
            "filename": chunks[idx]["filename"],
            "distance": distances[0][i]
        })
    return results

def classify_intent(question):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "You are an intent classifier. Respond with exactly one word — either 'product' or 'casual'. 'product' means the user is asking about a product, feature, issue, or policy. 'casual' means it is a greeting, thanks, or small talk."
            },
            {
                "role": "user",
                "content": question
            }
        ],
        temperature=0
    )
    return response.choices[0].message.content.strip().lower()

def answer_question(question):
    intent = classify_intent(question)
    
    if intent == "casual":
        return {
            "answer": "Hi! I'm your AI research assistant. Ask anything about transformers, GPT-4, or RAG.",
            "sources": []
        }
    index, chunks = load_index()
    relevant_chunks = retrieve_chunks(question, index, chunks)
    
    context = "\n\n".join([
        f"From {chunk['filename']}:\n{chunk['text']}"
        for chunk in relevant_chunks
    ])
    
    prompt = f"""You are a helpful product assistant. 
Answer the question using only the context provided below.
If the answer is not in the context, say "I don't have that information."

Context:
{context}

Question: {question}
Answer:"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    
    return {
        "answer": response.choices[0].message.content,
        #"sources": [chunk["filename"] for chunk in relevant_chunks]
        "sources": list(dict.fromkeys([chunk["filename"] for chunk in relevant_chunks]))
    }