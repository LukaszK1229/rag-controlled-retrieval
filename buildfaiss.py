import sqlite3
import pickle
from vector_store_faiss import FaissVectorStore

DB_PATH = "rag.db"
DIM = 1536  # embedding-3-small
INDEX_PATH = "faiss.index"
META_PATH = "faiss_meta.pkl"

def main():
    print("▶ Loading chunks from DB...")
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.execute("SELECT chunk_id, embedding_dim FROM chunks")
    rows = cur.fetchall()

    print(f"▶ Rows: {len(rows)}")

    store = FaissVectorStore(
        dim=DIM,
        index_path=INDEX_PATH,
        meta_path=META_PATH
    )

    for chunk_id, embedding_blob in rows:
        embedding = pickle.loads(embedding_blob)
        store.add(chunk_id, embedding)

    faiss.write_index(store.index, INDEX_PATH)
    with open(META_PATH, "wb") as f:
        pickle.dump(store.id_map, f)

    print(f"✔ FAISS built with {store.index.ntotal} vectors")

if __name__ == "__main__":
    main()
