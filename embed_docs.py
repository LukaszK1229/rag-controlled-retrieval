import os
import json
import hashlib
import requests
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("OPENROUTER_API_KEY")
EMBED_URL = "https://openrouter.ai/api/v1/embeddings"
MODEL = "text-embedding-3-small"

DOCS_DIR = "docs"
OUTPUT_FILE = "embeddings.json"


def embed_text(text: str) -> list:
    payload = {
        "model": MODEL,
        "input": text
    }
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    r = requests.post(EMBED_URL, json=payload, headers=headers)
    r.raise_for_status()
    return r.json()["data"][0]["embedding"]


all_chunks = []

for filename in os.listdir(DOCS_DIR):
    if not filename.endswith(".md"):
        continue

    path = os.path.join(DOCS_DIR, filename)
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    # JEDYNY delimiter – zgodnie z tym, co miałeś OD POCZĄTKU
    parts = content.split("##")

    chunk_index = 0

    for part in parts:
        part = part.strip()
        if not part:
            continue

        lines = part.splitlines()

        if len(lines) == 1:
            text = lines[0].strip()
        else:
            text = "\n".join(lines[1:]).strip()

        if not text:
            continue

        # STABILNY HASH TREŚCI (POD SQL)
        content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()

        embedding = embed_text(text)

        all_chunks.append({
            "file": filename,
            "section": f"chunk_{chunk_index}",
            "text": text,
            "content_hash": content_hash,
            "embedding_dim": embedding
        })

        chunk_index += 1


print(f"Zrobiono embeddingów: {len(all_chunks)}")
print("Przykład:", all_chunks[0])

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(all_chunks, f, ensure_ascii=False, indent=2)

print(f"Zapisano do {OUTPUT_FILE}")
