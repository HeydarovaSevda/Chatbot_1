# chat.py
import os, json, pickle
from pathlib import Path
from dotenv import load_dotenv

import faiss
import numpy as np
from mistralai import Mistral
from rank_bm25 import BM25Okapi

load_dotenv("MISTRAL_API_KEY.env")
API_KEY = os.getenv("MISTRAL_API_KEY")
assert API_KEY, "MISTRAL_API_KEY not found. Check MISTRAL_API_KEY.env."

EMBED_MODEL = "mistral-embed"
CHAT_MODEL  = "mistral-small-latest"   
STORE_DIR = Path("rag_store")

def embed_query(client: Mistral, text: str) -> np.ndarray:
    resp = client.embeddings.create(model=EMBED_MODEL, inputs=[text])
    v = np.array(resp.data[0].embedding, dtype="float32")
    faiss.normalize_L2(v.reshape(1, -1))
    return v

def load_store():
    index = faiss.read_index(str(STORE_DIR / "index.faiss"))
    with open(STORE_DIR / "records.pkl", "rb") as f:
        records = pickle.load(f)
    meta = json.loads((STORE_DIR / "meta.json").read_text(encoding="utf-8"))
    return index, records, meta

def rerank_bm25(query: str, candidates: list[dict], top_k: int = 3):
    tokenized = [c["text"].split() for c in candidates]
    bm25 = BM25Okapi(tokenized)
    scores = bm25.get_scores(query.split())
    order = np.argsort(scores)[::-1][:top_k]
    return [candidates[i] for i in order]

def build_prompt(user_question: str, top_chunks: list[dict]) -> list[dict]:
    context_blocks = []
    for c in top_chunks:
        ref_id = c["id"]  
        header = f"[{ref_id}]"
        block = f"{header}\n{c['text']}".strip()
        context_blocks.append(block)

    context_text = "\n\n---\n\n".join(context_blocks)

    system = {
        "role": "system",
        "content": (
            "You are a RAG assistant. Answer strictly based on the provided CONTEXT blocks; do not fabricate. "
            "Write clear, concise English. If the answer is not present in the context, state that explicitly. "
            "At the end of your answer, include the exact context IDs you used as citations, in square brackets, "
            "for example: [data/file.md::chunk0][data/orders.md::chunk2]."
        ),
    }

    user = {
        "role": "user",
        "content": (
            f"Question: {user_question}\n\n"
            f"CONTEXT BLOCKS (each block starts with an exact ID in brackets):\n"
            f"{context_text}\n\n"
            "Task: Answer the question using ONLY the information in the context. "
            "If the answer cannot be found in the context, reply: 'No specific information found in the provided context.' "
            "At the end of your answer, append the exact IDs of the blocks you used as citations, in square brackets."
        ),
    }

    return [system, user]

def main():
    index, records, _ = load_store()
    client = Mistral(api_key=API_KEY)

    print("Type your question (press Enter on empty line to exit):")
    while True:
        q = input("\n> ").strip()
        if not q:
            break

        qvec = embed_query(client, q)
        D, I = index.search(qvec.reshape(1, -1), k=5)
        candidates = [records[i] for i in I[0]]
        top3 = rerank_bm25(q, candidates, top_k=3)

    
        messages = build_prompt(q, top3)
        resp = client.chat.complete(model=CHAT_MODEL, messages=messages, temperature=0.2)
        answer = resp.choices[0].message.content

        print("\n--- ANSWER ---\n")
        print(answer)

        print("\nCitations (exact IDs used):")
        for c in top3:
            print(f"- [{c['id']}]  -> source: {c['source']}, chunk_idx: {c['chunk_idx']}")

if __name__ == "__main__":
    main()
