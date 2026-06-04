# RAG Chatbot — Built from Scratch

Day 2 of my 20-day AI learning journey. A retrieval-augmented generation chatbot built in raw Python — no frameworks, every step written by hand.

## What it does
Ask questions over a set of documents and get answers grounded in those documents, with source citations showing exactly which file the answer came from.

## How it works
1. `loader.py` — reads TXT and PDF files from the docs folder
2. `embedder.py` — chunks text, calls OpenAI embeddings API, stores vectors in FAISS
3. `retriever.py` — embeds the question, searches FAISS, builds prompt, calls GPT-4o-mini
4. `app.py` — Streamlit chat UI with intent classification and source citations

## Key concepts implemented
- Chunking with overlap — 200 token chunks, 20 token overlap
- FAISS vector store — local similarity search
- Intent classification — greetings skip the database
- Source deduplication — same file never cited twice
- PDF + TXT support

## Stack
- Python, OpenAI API, FAISS, Streamlit
- GPT-4o-mini for generation, text-embedding-3-small for embeddings

## Running locally
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
# Add OPENAI_API_KEY to .env
streamlit run app.py
```

## Note on deployment
Click 'Build / Rebuild Index' after opening the app — the index rebuilds each session due to Streamlit Cloud's ephemeral filesystem. Pinecone integration coming in a future day for persistent cloud storage.

