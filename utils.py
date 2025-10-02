import os
from pathlib import Path

def read_markdown_files(data_dir: str):
    texts = []
    for p in Path(data_dir).glob("*.md"):

        try:
            txt = p.read_text(encoding="utf-8")
            texts.append((str(p), txt))
        except Exception as e:
            print(f"[WARN] {p} oxunmadı: {e}")
    return texts

def simple_chunk(text: str, chunk_size: int = 1000, overlap: int = 200):
    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        chunks.append(text[start:end])
        if end == n: break
        start = end - overlap
        if start < 0: start = 0
    return chunks
