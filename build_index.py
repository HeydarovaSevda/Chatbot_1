# build_index.py
import os, json, pickle
from pathlib import Path
from dotenv import load_dotenv
from tqdm import tqdm

import faiss
import numpy as np
from mistralai import Mistral

from utils import read_markdown_files, simple_chunk

load_dotenv("MISTRAL_API_KEY.env")
API_KEY = os.getenv("MISTRAL_API_KEY")
assert API_KEY, "MISTRAL_API_KEY tapılmadı. MISTRAL_API_KEY.env daxilində export et."

EMBED_MODEL = "mistral-embed"
STORE_DIR = Path("rag_store")
STORE_DIR.mkdir(exist_ok=True)

def embed_texts(client: Mistral, texts: list[str], batch_size: int = 64):
    """Mistral embeddings API ilə partiya-əməliyyat."""
    vecs = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding"):
        batch = texts[i:i+batch_size]
        resp = client.embeddings.create(model=EMBED_MODEL, inputs=batch)
        for item in resp.data:
            vecs.append(item.embedding)
    return np.array(vecs, dtype="float32")

def main():
    client = Mistral(api_key=API_KEY)

    docs = read_markdown_files("data")
    records = []  
    all_texts = []
    for src_path, txt in docs:
        chunks = simple_chunk(txt, chunk_size=1000, overlap=200)
        for c_idx, c in enumerate(chunks):
            rec_id = f"{src_path}::chunk{c_idx}"
            records.append({"id": rec_id, "text": c, "source": src_path, "chunk_idx": c_idx})
            all_texts.append(c)

    if not records:
        print("Heç bir .md tapılmadı. data/ qovluğunu yoxla.")
        return

    vectors = embed_texts(client, all_texts)  
    dim = vectors.shape[1]

    faiss.normalize_L2(vectors)
    index = faiss.IndexFlatIP(dim) 
    index.add(vectors)

    faiss.write_index(index, str(STORE_DIR / "index.faiss"))
    with open(STORE_DIR / "records.pkl", "wb") as f:
        pickle.dump(records, f)
    meta = {"embed_model": EMBED_MODEL}
    (STORE_DIR / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[OK] {len(records)} chunk indeksləndi. -> rag_store/")

if __name__ == "__main__":
    main()

