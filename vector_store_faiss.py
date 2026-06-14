import faiss
import numpy as np
import pickle
from pathlib import Path


class FaissVectorStore:
    def __init__(self, dim: int, index_path="faiss.index", meta_path="faiss_meta.pkl"):
        self.dim = dim
        self.index_path = Path(index_path)
        self.meta_path = Path(meta_path)

        # zawsze startujemy z pustym indeksem
        self.index = faiss.IndexFlatL2(dim)
        self.id_map: list[str] = []

    # =========================
    # IO
    # =========================

    def load(self):
        if not self.index_path.exists() or not self.meta_path.exists():
            raise FileNotFoundError("FAISS index or metadata file not found")

        self.index = faiss.read_index(str(self.index_path))
        with open(self.meta_path, "rb") as f:
            self.id_map = pickle.load(f)

        if self.index.ntotal != len(self.id_map):
            raise RuntimeError("FAISS index and id_map size mismatch")

    def save(self):
        faiss.write_index(self.index, str(self.index_path))
        with open(self.meta_path, "wb") as f:
            pickle.dump(self.id_map, f)

    # =========================
    # WRITE
    # =========================

    def add(self, chunk_id: str, vector: list[float]):
        vec = np.array(vector, dtype="float32").reshape(1, -1)
        faiss.normalize_L2(vec)

        self.index.add(vec)
        self.id_map.append(chunk_id)

    # =========================
    # READ
    # =========================

    def search(self, query_vector: list[float], k: int = 5):
        if self.index.ntotal == 0:
            return []

        q = np.array(query_vector, dtype="float32").reshape(1, -1)
        faiss.normalize_L2(q)

        distances, indices = self.index.search(q, k)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:
                continue

            results.append({
                "chunk_id": self.id_map[idx],
                "score": float(dist)  
            })

        return results
