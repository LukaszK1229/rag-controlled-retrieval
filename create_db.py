import sqlite3
from pathlib import Path

DB_PATH = Path("rag.db")


def create_tables(conn: sqlite3.Connection) -> None:
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS documents (
        doc_id TEXT PRIMARY KEY,
        source TEXT NOT NULL,
        version INTEGER NOT NULL,
        content_hash TEXT NOT NULL,
        doc_type TEXT,
        status TEXT NOT NULL,
        role_scope TEXT,
        valid_from TEXT,
        valid_to TEXT,
        is_active INTEGER NOT NULL DEFAULT 1,
        created_at TEXT NOT NULL,
        updated_at TEXT
    );
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS chunks (
        chunk_id TEXT PRIMARY KEY,
        doc_id TEXT NOT NULL,
        text TEXT NOT NULL,
        section TEXT,
        content_hash TEXT NOT NULL,
        version INTEGER NOT NULL,
        is_active INTEGER NOT NULL DEFAULT 1,
        has_embedding INTEGER NOT NULL DEFAULT 0,
        created_at TEXT NOT NULL,
        updated_at TEXT,
        FOREIGN KEY (doc_id) REFERENCES documents(doc_id)
    );
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS rag_logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ts TEXT NOT NULL,
        query_id TEXT NOT NULL,
        event TEXT NOT NULL,
        payload TEXT
    );
    """)

    cursor.execute("CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_id);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_chunks_active ON chunks(is_active);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_rag_logs_query ON rag_logs(query_id);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_rag_logs_event ON rag_logs(event);")

    conn.commit()


def ensure_new_columns(conn):
    cursor = conn.cursor()

    columns = [row[1] for row in cursor.execute("PRAGMA table_info(documents)")]

    if "role_scope" not in columns:
        cursor.execute("ALTER TABLE documents ADD COLUMN role_scope TEXT")

    if "valid_from" not in columns:
        cursor.execute("ALTER TABLE documents ADD COLUMN valid_from TEXT")

    if "valid_to" not in columns:
        cursor.execute("ALTER TABLE documents ADD COLUMN valid_to TEXT")

    conn.commit()


def main():
    print("▶ Creating SQLite database...")
    conn = sqlite3.connect(DB_PATH)

    create_tables(conn)

    # <<< TU BYŁ BRAK >>>
    ensure_new_columns(conn)

    conn.close()
    print(f"✔ Database initialized: {DB_PATH.resolve()}")


if __name__ == "__main__":
    main()