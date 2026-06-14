import json
import re
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any


INPUT_FILE = Path("embeddings.json")
OUTPUT_FILE = Path("embeddings_ingested.json")
SOURCE_NAME = "manual_upload"


def slugify(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[\s-]+", "_", text)
    return text


def validate_record(record: Dict[str, Any], idx: int) -> None:
    required_fields = ["file", "section", "text"]
    for field in required_fields:
        if field not in record:
            raise ValueError(f"Record #{idx} missing required field: {field}")


def load_input(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("Input JSON must be a list of records")

    return data


def migrate(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    migrated = []

    for idx, record in enumerate(records):
        validate_record(record, idx)

        file_name = record["file"]
        section = record["section"]

        doc_id = Path(file_name).stem
        section_slug = slugify(section)
        chunk_id = f"{doc_id}::{section_slug}"

        migrated.append({
            "doc_id": doc_id,
            "chunk_id": chunk_id,
            "source": SOURCE_NAME,
            "text": record["text"],
            "embedding_dim": record["embedding_dim"],
            "metadata": {
                "file": file_name,
                "section": section
            },
            "created_at": datetime.utcnow().isoformat() + "Z"
        })

    return migrated


def save_output(data: List[Dict[str, Any]], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def main():
    print("▶ Loading input data...")
    records = load_input(INPUT_FILE)

    print(f"▶ Migrating {len(records)} records...")
    migrated = migrate(records)

    print("▶ Saving ingested data...")
    save_output(migrated, OUTPUT_FILE)

    print("✔ Ingest completed successfully")
    print(f"→ Output: {OUTPUT_FILE.resolve()}")


if __name__ == "__main__":
    main()
