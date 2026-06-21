"""Dataset registry and Hugging Face text import helpers for QLLM Lab."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

from ..resultsdb import ResultsDB

DEFAULT_DATASET = {
    "name": "default-text",
    "source_type": "local",
    "source": "data/input.txt",
    "split": None,
    "text_column": None,
    "corpus_path": "data/input.txt",
    "n_rows": None,
    "n_chars": None,
    "preview": "Local corpus at data/input.txt; falls back to synthetic text if missing.",
}


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9._-]+", "-", value.strip().lower()).strip("-")
    return slug or "dataset"


def list_datasets(db: ResultsDB) -> list[dict]:
    seen = {DEFAULT_DATASET["name"]}
    rows = [DEFAULT_DATASET]
    for row in db.fetch_lab_datasets():
        if row["name"] not in seen:
            rows.append(row)
            seen.add(row["name"])
    return rows


def get_dataset(db: ResultsDB, name: str) -> dict | None:
    if name == DEFAULT_DATASET["name"]:
        return DEFAULT_DATASET
    return db.get_lab_dataset(name)


def _loader_for_source(source: str) -> tuple[str, dict]:
    lower = source.lower()
    if lower.startswith(("http://", "https://", "hf://")):
        if ".csv" in lower:
            return "csv", {"data_files": source}
        if ".json" in lower or ".jsonl" in lower:
            return "json", {"data_files": source}
        if ".parquet" in lower:
            return "parquet", {"data_files": source}
        if ".txt" in lower:
            return "text", {"data_files": source}
    return source, {}


def _iter_rows(ds) -> Iterable[dict]:
    for row in ds:
        yield row


def import_hf_text_dataset(
    db: ResultsDB,
    source: str,
    split: str,
    text_column: str,
    display_name: str | None = None,
    row_limit: int = 5000,
    cache_dir: str | Path = "data/imported",
) -> dict:
    """Import a public Hugging Face text dataset into a local corpus file."""
    source = source.strip()
    split = (split or "train").strip()
    text_column = text_column.strip()
    if not source:
        raise ValueError("Dataset id or URL is required.")
    if not text_column:
        raise ValueError("Text column is required.")
    row_limit = 5000 if row_limit is None or row_limit == "" else int(row_limit)
    if row_limit < 1:
        raise ValueError("Row limit must be at least 1.")
    if row_limit > 200_000:
        raise ValueError("Row limit must be 200000 or lower for local imports.")

    try:
        from datasets import load_dataset
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Hugging Face imports require the optional dependency: "
            "pip install -e .[hf]"
        ) from exc

    loader, kwargs = _loader_for_source(source)
    try:
        ds = load_dataset(loader, split=split, **kwargs)
    except Exception as exc:  # pragma: no cover - exact HF errors vary
        raise RuntimeError(f"Could not load Hugging Face dataset: {exc}") from exc

    rows: list[str] = []
    n_seen = 0
    for row in _iter_rows(ds):
        if text_column not in row:
            available = ", ".join(row.keys())
            raise ValueError(
                f"Column '{text_column}' was not found. Available columns: {available}"
            )
        value = row[text_column]
        if value is None:
            continue
        text = str(value).strip()
        if text:
            rows.append(text)
        n_seen += 1
        if n_seen >= row_limit:
            break
    corpus = "\n\n".join(rows).strip()
    if not corpus:
        raise ValueError("Import produced no non-empty text.")

    name = slugify(display_name or source.split("/")[-1])
    out_dir = Path(cache_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{name}.txt"
    suffix = 2
    while out_path.exists() and db.get_lab_dataset(name) is not None:
        name = f"{slugify(display_name or source.split('/')[-1])}-{suffix}"
        out_path = out_dir / f"{name}.txt"
        suffix += 1
    out_path.write_text(corpus, encoding="utf-8")

    payload = {
        "name": name,
        "source_type": "huggingface",
        "source": source,
        "split": split,
        "text_column": text_column,
        "corpus_path": str(out_path),
        "n_rows": len(rows),
        "n_chars": len(corpus),
        "preview": corpus[:500],
    }
    db.upsert_lab_dataset(payload)
    return payload
