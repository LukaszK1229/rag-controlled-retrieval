import os
import uuid
import sqlite3
import requests
import pickle
from dataclasses import dataclass
from typing import List, Optional
from pathlib import Path
from datetime import datetime, UTC
import json
import faiss
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

# =========================
# CONFIG
# =========================

DB_PATH = Path("rag.db")

FAISS_INDEX_PATH = "faiss.index"
FAISS_META_PATH = "faiss_meta.pkl"

EMBED_DIM = 1536
EMBED_MODEL = "text-embedding-3-small"
EMBED_URL = "https://openrouter.ai/api/v1/embeddings"

TOP_K = 5
FETCH_K = 50  # pobieramy więcej z FAISS, filtrujemy metadata, potem tniemy do TOP_K
SCORE_THRESHOLD = 0.85  # UWAGA: dystans L2 (im mniejszy, tym lepiej)
SCORE_GAP_THRESHOLD = 0.05  # L2: top2 - top1; mały gap = retriever mniej pewny

load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# =========================
# STATE
# =========================

@dataclass
class RetrievalResult:
    chunk_id: str
    score: float


@dataclass
class QueryState:
    query_id: str
    question: str
    embedding: Optional[list] = None
    retrieval_results: Optional[List[RetrievalResult]] = None
    context: Optional[str] = None
    answer: Optional[str] = None
    stop_reason: Optional[str] = None



def get_allowed_chunk_ids(
    conn: sqlite3.Connection,
    doc_types: list[str] | None = None,
    user_role: str | None = None,
) -> set[str]:
    """
    Zwraca zbiór chunk_id, które:
    - są aktywne
    - należą do aktywnego dokumentu
    - są w oknie ważności (valid_from / valid_to)
    - pasują do doc_type (opcjonalnie)
    - pasują do role_scope (opcjonalnie)
    """

    base_query = """
        SELECT c.chunk_id
        FROM chunks c
        JOIN documents d ON d.doc_id = c.doc_id
        WHERE c.is_active = 1
          AND d.is_active = 1
          AND (
                d.valid_from IS NULL
                OR d.valid_from <= datetime('now')
              )
          AND (
                d.valid_to IS NULL
                OR d.valid_to >= datetime('now')
              )
    """

    params = []

    if doc_types:
        placeholders = ",".join("?" for _ in doc_types)
        base_query += f" AND d.doc_type IN ({placeholders})"
        params.extend(doc_types)

    if user_role:
        base_query += " AND (d.role_scope IS NULL OR d.role_scope = ?)"
        params.append(user_role)

    rows = conn.execute(base_query, params).fetchall()

    return {row[0] for row in rows}



def log_event(event: str, state: QueryState, payload: dict | None = None):
    record = {
        "ts": datetime.now(UTC).isoformat(),
        "event": event,
        "query_id": state.query_id,
        "payload": payload,
    }

    # stdout (jak było)
    print("[LOG]", json.dumps(record, ensure_ascii=False))

    # >>> ZAPIS DO SQLITE <<<
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            INSERT INTO rag_logs (ts, query_id, event, payload)
            VALUES (?, ?, ?, ?)
            """,
            (
                record["ts"],
                record["query_id"],
                record["event"],
                json.dumps(payload, ensure_ascii=False) if payload else None,
            ),
        )
        conn.commit()


# =========================
# VECTOR STORE
# =========================

class FaissVectorStore:
    def __init__(self, dim: int):
        self.dim = dim
        self.index_path = Path(FAISS_INDEX_PATH)
        self.meta_path = Path(FAISS_META_PATH)

        self.index = faiss.IndexFlatL2(dim)
        self.id_map: list[str] = []

    def load(self):
        if not self.index_path.exists() or not self.meta_path.exists():
            raise FileNotFoundError("FAISS index or metadata file not found")

        self.index = faiss.read_index(str(self.index_path))
        with open(self.meta_path, "rb") as f:
            self.id_map = pickle.load(f)

        if self.index.ntotal != len(self.id_map):
            raise RuntimeError("FAISS index and id_map size mismatch")

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
                "score": float(dist)  # dystans L2
            })

        return results


# =========================
# VECTOR STORE SINGLETON
# =========================

_VECTOR_STORE: Optional[FaissVectorStore] = None


def get_vector_store() -> FaissVectorStore:
    """
    Ładuje FAISS tylko raz na czas życia procesu.
    W CLI to mały zysk, ale pod API nie ładujemy indeksu per request.
    """
    global _VECTOR_STORE

    if _VECTOR_STORE is None:
        store = FaissVectorStore(dim=EMBED_DIM)
        store.load()
        _VECTOR_STORE = store

    return _VECTOR_STORE


# =========================
# EMBEDDING (RUNTIME)
# =========================

def embed(state: QueryState) -> QueryState:
    print("[EMBED] start")

    payload = {
        "model": EMBED_MODEL,
        "input": state.question,
    }
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }

    r = requests.post(EMBED_URL, json=payload, headers=headers)
    r.raise_for_status()

    state.embedding = r.json()["data"][0]["embedding"]
    return state


# =========================
# RETRIEVAL
# =========================

def retrieve(state: QueryState, top_k: int, fetch_k: int = FETCH_K) -> QueryState:
    print("[RETRIEVAL] start")

    if state.embedding is None:
        return state

    store = get_vector_store()

    # [ETAP 2] metadata pre-filter
    with sqlite3.connect(DB_PATH) as conn:
        allowed_chunk_ids = get_allowed_chunk_ids(
            conn,
            doc_types=None,
            user_role="public"
        )

    # Pobieramy więcej wyników z FAISS, bo część może odpaść po metadata filter.
    raw = store.search(state.embedding, k=fetch_k)

    filtered = [
        r for r in raw
        if r["chunk_id"] in allowed_chunk_ids
    ]

    # Finalnie zostawiamy tylko TOP_K wyników dla LLM.
    final_results = filtered[:top_k]

    state.retrieval_results = [
        RetrievalResult(chunk_id=r["chunk_id"], score=r["score"])
        for r in final_results
    ]

    print(f"[RETRIEVAL] results={len(state.retrieval_results)}")

    # [OBS] log ile odpadło
    log_event(
        "retrieval_filtered",
        state,
        {
            "fetch_k": fetch_k,
            "top_k": top_k,
            "raw_count": len(raw),
            "filtered_count": len(filtered),
            "final_count": len(final_results),
        }
    )

    return state


# =========================
# CONTEXT
# =========================
def build_context(state: QueryState) -> QueryState:
    print("[CONTEXT] start")

    if not state.retrieval_results:
        state.context = ""
        return state

  
    ids = [r.chunk_id for r in state.retrieval_results]
    placeholders = ",".join("?" for _ in ids)

    with sqlite3.connect(DB_PATH) as conn:
        rows = conn.execute(
            f"""
            SELECT chunk_id, text
            FROM chunks
            WHERE chunk_id IN ({placeholders})
            """,
            ids,
        ).fetchall()

    
    rows_by_id = {cid: text for cid, text in rows}

    
    ordered_rows = [
        (r.chunk_id, rows_by_id.get(r.chunk_id, ""))
        for r in state.retrieval_results
    ]

    state.context = "\n\n".join(
        f"[{cid}]\n{text}"
        for cid, text in ordered_rows
        if text
    )

    print("[CONTEXT] built")
    return state

# =========================
# GATING
# =========================

def decide(
    state: QueryState,
    score_threshold: float,
    gap_threshold: float = SCORE_GAP_THRESHOLD,
) -> QueryState:
    print("[GATING] start")

    if not state.retrieval_results:
        state.stop_reason = "no_retrieval_results"
        return state

    # L2 distance: im mniejszy score, tym lepszy wynik.
    top_score = state.retrieval_results[0].score

    if top_score > score_threshold:
        state.stop_reason = "low_confidence"
        return state

    # Drugi sygnał: czy top-1 wyraźnie wygrywa z top-2.
    # Jeśli różnica jest mała, retriever może być niepewny.
    if len(state.retrieval_results) >= 2:
        second_score = state.retrieval_results[1].score
        score_gap = second_score - top_score

        if score_gap < gap_threshold:
            state.stop_reason = "low_confidence_gap"
            return state

    return state


# =========================
# LLM
# =========================

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
    default_headers={
        "HTTP-Referer": "http://localhost",
        "X-Title": "rag-runtime",
    },
)

def call_llm(state: QueryState) -> QueryState:
    print("[LLM] call")

    resp = client.chat.completions.create(
        model="openai/gpt-4.1-mini",
        messages=[
            {
                "role": "system",
                "content": "Answer only using the provided context. If missing, say you don't know.",
            },
            {
                "role": "user",
                "content": f"Context:\n{state.context}\n\nQuestion:\n{state.question}",
            },
        ],
        temperature=0.0,
    )

    state.answer = resp.choices[0].message.content.strip()
    return state


# =========================
# FINALIZE
# =========================

def finalize(state: QueryState):
    print("[RESULT]")

    if state.stop_reason:
        print("STOP:", state.stop_reason)

        return {
            "query_id": state.query_id,
            "status": "stopped",
            "reason": state.stop_reason,
        }

    print("\n--- ANSWER ---")
    print(state.answer)

    return {
        "query_id": state.query_id,
        "status": "ok",
        "answer": state.answer,
    }

def get_chunk_snippets(chunk_ids: list[str], limit: int = 120) -> dict[str, str]:
    """
    Pobiera krótkie fragmenty tekstu dla top-k chunków.
    Używane tylko do logów/debuggingu retrievalu.
    """
    if not chunk_ids:
        return {}

    placeholders = ",".join("?" for _ in chunk_ids)

    with sqlite3.connect(DB_PATH) as conn:
        rows = conn.execute(
            f"""
            SELECT chunk_id, text
            FROM chunks
            WHERE chunk_id IN ({placeholders})
            """,
            chunk_ids,
        ).fetchall()

    snippets = {}
    for chunk_id, text in rows:
        clean_text = " ".join((text or "").split())
        snippets[chunk_id] = clean_text[:limit]

    return snippets


# =========================
# ENTRYPOINT
# =========================

def run_query(question: str):
    state = QueryState(
        query_id=str(uuid.uuid4()),
        question=question,
    )

    # [OBS] log zapytania
    log_event(
        "query_received",
        state,
        {"question": state.question}
    )

    print("[ENTRYPOINT] question received")

    state = embed(state)
    state = retrieve(state, top_k=TOP_K)

    # [OBS] log top-k FAISS + snippety do debugowania jakości retrievalu
    if state.retrieval_results:
        chunk_ids = [r.chunk_id for r in state.retrieval_results]
        snippets = get_chunk_snippets(chunk_ids)

        top_score = state.retrieval_results[0].score
        score_gap = None
        if len(state.retrieval_results) >= 2:
            score_gap = state.retrieval_results[1].score - top_score

        log_event(
            "retrieval_results",
            state,
            {
                "score_metric": "L2_distance_lower_is_better",
                "score_threshold": SCORE_THRESHOLD,
                "gap_threshold": SCORE_GAP_THRESHOLD,
                "top_score": top_score,
                "score_gap": score_gap,
                "top_k": [
                    {
                        "rank": idx + 1,
                        "chunk_id": r.chunk_id,
                        "score": r.score,
                        "snippet": snippets.get(r.chunk_id, ""),
                    }
                    for idx, r in enumerate(state.retrieval_results)
                ]
            }
        )

    state = build_context(state)
    state = decide(state, score_threshold=SCORE_THRESHOLD, gap_threshold=SCORE_GAP_THRESHOLD)

    # STOP
    if state.stop_reason:
        stop_payload = {"reason": state.stop_reason}

        if state.retrieval_results:
            stop_payload["top_score"] = state.retrieval_results[0].score
            stop_payload["score_threshold"] = SCORE_THRESHOLD

            if len(state.retrieval_results) >= 2:
                stop_payload["score_gap"] = state.retrieval_results[1].score - state.retrieval_results[0].score
                stop_payload["gap_threshold"] = SCORE_GAP_THRESHOLD

        log_event(
            "query_stopped",
            state,
            stop_payload
        )
        return finalize(state)

    # LLM
    state = call_llm(state)

    # [OBS] answer generated
    log_event(
        "answer_generated",
        state,
        {"answer_length": len(state.answer or "")}
    )

    return finalize(state)



if __name__ == "__main__":
    while True:
        q = input("Q: ").strip()
        if not q:
            break
        run_query(q)
