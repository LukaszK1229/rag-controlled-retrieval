import json
import sqlite3
from pathlib import Path
from datetime import datetime, UTC

from vector_store_faiss import FaissVectorStore

# =========================
# CONFIG
# =========================

DB_PATH = Path(r"C:\Users\ComPP\Desktop\Praca\rag_mvp\rag.db")
INGESTED_FILE = Path("embeddings_ingested.json")

FAISS_DIM = 1536
FAISS_INDEX = "faiss.index"
FAISS_META = "faiss_meta.pkl"


# =========================
# LOAD INGESTED EMBEDDINGS
# =========================

def load_embeddings():
    with INGESTED_FILE.open("r", encoding="utf-8") as f:
        data = json.load(f)

    # map: chunk_id -> embedding
    return {
        r["chunk_id"]: r["embedding_dim"]
        for r in data
        if "embedding_dim" in r
    }


# =========================
# LOAD ACTIVE CHUNKS FROM DB
# =========================

def load_active_chunk_ids(conn):
    cur = conn.cursor()
    rows = cur.execute("""
        SELECT chunk_id
        FROM chunks
        WHERE is_active = 1
    """).fetchall()

    return {row[0] for row in rows}


# =========================
# UPDATE DB FLAG
# =========================

def mark_embedded(conn, chunk_ids):
    if not chunk_ids:
        return

    cur = conn.cursor()
    cur.executemany(
        """
        UPDATE chunks
        SET has_embedding = 1,
            updated_at = ?
        WHERE chunk_id = ?
        """,
        [(datetime.now(UTC).isoformat(), cid) for cid in chunk_ids]
    )
    conn.commit()


# =========================
# MAIN
# =========================

def main():
    print("▶ FAISS incremental rebuild")

    embeddings = load_embeddings()
    print(f"Loaded embeddings: {len(embeddings)}")

    store = FaissVectorStore(
        dim=FAISS_DIM,
        index_path=FAISS_INDEX,
        meta_path=FAISS_META
    )

    already_indexed = set(store.id_map)

    with sqlite3.connect(DB_PATH) as conn:
        active_chunk_ids = load_active_chunk_ids(conn)

        
        
        
        
        to_add = [
            cid for cid in active_chunk_ids
            if cid in embeddings and cid not in already_indexed
        ]

        print(f"Embedding {len(to_add)} chunks...")

        for cid in to_add:
            store.add(cid, embeddings[cid])

        store.save()

        # zsynchronizuj DB
        mark_embedded(conn, to_add)

    print("✔ FAISS rebuild completed.")
    print("FAISS vectors count:", store.index.ntotal)


if __name__ == "__main__":
    main()
