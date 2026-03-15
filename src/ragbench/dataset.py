from __future__ import annotations

import json
from pathlib import Path

from ragbench.types import Document, EvalQuestion


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
EVAL_PATH = DATA_DIR / "eval" / "questions.json"


def _parse_front_matter(text: str) -> tuple[dict[str, str], str]:
    if not text.startswith("---\n"):
        return {}, text

    _, rest = text.split("---\n", 1)
    front_matter, body = rest.split("\n---\n", 1)
    metadata: dict[str, str] = {}
    for line in front_matter.splitlines():
        if not line.strip() or ":" not in line:
            continue
        key, value = line.split(":", 1)
        metadata[key.strip()] = value.strip()
    return metadata, body.strip()


def load_documents(raw_dir: Path = RAW_DIR) -> list[Document]:
    documents: list[Document] = []
    for path in sorted(raw_dir.glob("*.md")):
        raw_text = path.read_text(encoding="utf-8")
        metadata, body = _parse_front_matter(raw_text)
        tags = [tag.strip() for tag in metadata.get("tags", "").split(",") if tag.strip()]
        documents.append(
            Document(
                id=path.stem,
                title=metadata.get("title", path.stem.replace("_", " ").title()),
                source=metadata.get("source", ""),
                tags=tags,
                content=body,
            )
        )
    return documents


def load_questions(path: Path = EVAL_PATH) -> list[EvalQuestion]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    return [EvalQuestion(**row) for row in rows]
