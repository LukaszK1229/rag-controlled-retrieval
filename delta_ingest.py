import json
import sqlite3
import hashlib
from pathlib import Path
from datetime import datetime

DB_PATH = Path(r"C:\Users\ComPP\Desktop\Praca\rag_mvp\rag.db")
INPUT_FILE = Path("embeddings_ingested.json")


# =========================
# Helpers
# =========================

def hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_ingested(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def ensure_document(conn: sqlite3.Connection, doc_id: str):
    """
    Minimalny upsert dokumentu, wymagany przez runtime (ETAP 2).
    Dodane: role_scope, valid_from, valid_to
    """
    now = datetime.utcnow().isoformat() + "Z"
    cur = conn.cursor()

    row = cur.execute(
        "SELECT doc_id FROM documents WHERE doc_id = ?",
        (doc_id,)
    ).fetchone()

    if row is None:
        cur.execute(
            """
            INSERT INTO documents (
                doc_id,
                source,
                version,
                content_hash,
                doc_type,
                status,
                role_scope,
                valid_from,
                valid_to,
                is_active,
                created_at,
                updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                doc_id,
                "ingest",
                1,
                "unknown",
                "unknown",
                "active",
                "public",   # domyślny scope
                None,       # valid_from
                None,       # valid_to
                1,
                now,
                now,
            ),
        )


# =========================
# Delta ingest (DB-only)
# =========================

def upsert_chunk(conn: sqlite3.Connection, record: dict) -> str:
    """
    Returns: inserted | updated | skipped
    """
    chunk_id = record["chunk_id"]
    doc_id = record["doc_id"]
    text = record["text"]
    section = record.get("metadata", {}).get("section")
    created_at = record["created_at"]

    new_hash = hash_text(text)
    now = datetime.utcnow().isoformat() + "Z"

    cur = conn.cursor()

    row = cur.execute(
        """
        SELECT content_hash, version
        FROM chunks
        WHERE chunk_id = ?
          AND is_active = 1
        """,
        (chunk_id,)
    ).fetchone()

    # --- NEW CHUNK ---
    if row is None:
        cur.execute(
            """
            INSERT INTO chunks (
                chunk_id,
                doc_id,
                text,
                section,
                content_hash,
                version,
                is_active,
                has_embedding,
                created_at,
                updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, 1, 0, ?, ?)
            """,
            (
                chunk_id,
                doc_id,
                text,
                section,
                new_hash,
                1,
                created_at,
                now,
            ),
        )
        return "inserted"

    old_hash, old_version = row

    # --- CHANGED CHUNK ---
    if old_hash != new_hash:
        cur.execute(
            """
            UPDATE chunks
            SET
                text = ?,
                content_hash = ?,
                version = ?,
                has_embedding = 0,
                updated_at = ?
            WHERE chunk_id = ?
              AND is_active = 1
            """,
            (
                text,
                new_hash,
                old_version + 1,
                now,
                chunk_id,
            ),
        )
        return "updated"

    # --- NO CHANGE ---
    return "skipped"


# =========================
# Main
# =========================

def main():
    records = load_ingested(INPUT_FILE)

    conn = sqlite3.connect(DB_PATH)
    stats = {"inserted": 0, "updated": 0, "skipped": 0}

    try:
        # --- METADATA: ensure documents ---
        doc_ids = {r["doc_id"] for r in records}
        for doc_id in doc_ids:
            ensure_document(conn, doc_id)

        # --- chunks ingest ---
        for r in records:
            result = upsert_chunk(conn, r)
            stats[result] += 1

        conn.commit()
    finally:
        conn.close()

    print("Delta ingest result:")
    for k, v in stats.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()