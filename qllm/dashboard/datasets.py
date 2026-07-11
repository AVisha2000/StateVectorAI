"""Dataset registry and Hugging Face text import helpers for QLLM Lab."""
from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Iterable

from ..resultsdb import ResultsDB

DEFAULT_ROW_LIMIT = 5_000
MAX_ROW_LIMIT = 200_000
DEFAULT_CHAR_LIMIT = 5_000_000
MAX_CHAR_LIMIT = 50_000_000
DEFAULT_BYTE_LIMIT = 10_000_000
MAX_BYTE_LIMIT = 100_000_000

DEFAULT_DATASET = {
    "name": "default-text",
    "source_type": "local",
    "source": "data/input.txt",
    "split": None,
    "text_column": None,
    "corpus_path": "data/input.txt",
    "n_rows": None,
    "n_chars": None,
    "n_bytes": None,
    "preview": "Local corpus at data/input.txt; falls back to synthetic text if missing.",
    "requested_revision": None,
    "resolved_fingerprint": None,
    "revision_applicable": False,
    "row_limit": None,
    "char_limit": None,
    "byte_limit": None,
    "rows_examined": None,
    "sha256": None,
    "truncated": False,
    "truncation_reason": None,
    "warnings": [],
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


def _bounded_limit(
    value: int | str | None,
    *,
    default: int,
    maximum: int,
    label: str,
) -> int:
    limit = default if value is None or value == "" else int(value)
    if limit < 1:
        raise ValueError(f"{label} must be at least 1.")
    if limit > maximum:
        raise ValueError(f"{label} must be {maximum} or lower for local imports.")
    return limit


def _bounded_utf8_prefix(
    value: str, remaining_chars: int, remaining_bytes: int
) -> tuple[str, bool, bool]:
    """Return a UTF-8-safe prefix plus character/byte truncation flags."""
    char_limited = len(value) > remaining_chars
    prefix = value[:remaining_chars]
    encoded = prefix.encode("utf-8")
    byte_limited = len(encoded) > remaining_bytes
    if byte_limited:
        prefix = encoded[:remaining_bytes].decode("utf-8", errors="ignore")
    return prefix, char_limited, byte_limited


def import_hf_text_dataset(
    db: ResultsDB,
    source: str,
    split: str,
    text_column: str,
    display_name: str | None = None,
    row_limit: int = DEFAULT_ROW_LIMIT,
    cache_dir: str | Path = "data/imported",
    revision: str | None = None,
    char_limit: int = DEFAULT_CHAR_LIMIT,
    byte_limit: int = DEFAULT_BYTE_LIMIT,
) -> dict:
    """Import a bounded public text dataset with reproducible provenance."""
    source = str(source or "").strip()
    split = str(split or "train").strip()
    text_column = str(text_column or "").strip()
    if not source:
        raise ValueError("Dataset id or URL is required.")
    if not text_column:
        raise ValueError("Text column is required.")
    row_limit = _bounded_limit(
        row_limit,
        default=DEFAULT_ROW_LIMIT,
        maximum=MAX_ROW_LIMIT,
        label="Row limit",
    )
    char_limit = _bounded_limit(
        char_limit,
        default=DEFAULT_CHAR_LIMIT,
        maximum=MAX_CHAR_LIMIT,
        label="Character limit",
    )
    byte_limit = _bounded_limit(
        byte_limit,
        default=DEFAULT_BYTE_LIMIT,
        maximum=MAX_BYTE_LIMIT,
        label="Byte limit",
    )
    revision = str(revision).strip() if revision is not None else ""
    revision = revision or None
    direct_url = source.lower().startswith(("http://", "https://", "hf://"))
    if direct_url and revision is not None:
        raise ValueError("Revision is not applicable to direct URL imports.")

    try:
        from datasets import load_dataset
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Hugging Face imports require the optional dependency: "
            "pip install -e .[hf]"
        ) from exc

    loader, kwargs = _loader_for_source(source)
    # Streaming makes the row/character/byte limits bound local transfer and
    # materialization rather than merely truncating an already-downloaded set.
    kwargs["streaming"] = True
    if revision is not None:
        kwargs["revision"] = revision
    try:
        ds = load_dataset(loader, split=split, **kwargs)
    except Exception as exc:  # pragma: no cover - exact HF errors vary
        raise RuntimeError(f"Could not load Hugging Face dataset: {exc}") from exc

    resolved_fingerprint = getattr(ds, "_fingerprint", None)
    if resolved_fingerprint is not None:
        resolved_fingerprint = str(resolved_fingerprint)

    parts: list[str] = []
    n_rows = 0
    current_chars = 0
    current_bytes = 0
    rows_examined = 0
    truncation_reason = None
    iterator = iter(_iter_rows(ds))
    while rows_examined < row_limit:
        try:
            row = next(iterator)
        except StopIteration:
            break
        rows_examined += 1
        if text_column not in row:
            available = ", ".join(map(str, row.keys()))
            raise ValueError(
                f"Column '{text_column}' was not found. Available columns: {available}"
            )
        value = row[text_column]
        if value is None:
            continue
        text = str(value).strip()
        if not text:
            continue

        separator = "\n\n" if parts else ""
        segment = separator + text
        prefix, char_limited, byte_limited = _bounded_utf8_prefix(
            segment,
            char_limit - current_chars,
            byte_limit - current_bytes,
        )
        contributed_text = prefix[len(separator):] if len(prefix) >= len(separator) else ""
        if contributed_text:
            parts.append(prefix)
            n_rows += 1
            current_chars += len(prefix)
            current_bytes += len(prefix.encode("utf-8"))
        if char_limited or byte_limited:
            if char_limited and byte_limited:
                truncation_reason = "character_and_byte_limit"
            elif char_limited:
                truncation_reason = "character_limit"
            else:
                truncation_reason = "byte_limit"
            break

    if truncation_reason is None and rows_examined >= row_limit:
        try:
            next(iterator)
        except StopIteration:
            pass
        else:
            # This single provenance lookahead is examined but never imported.
            rows_examined += 1
            truncation_reason = "row_limit"

    corpus = "".join(parts).strip()
    if not corpus:
        raise ValueError("Import produced no non-empty text.")

    corpus_bytes = corpus.encode("utf-8")
    digest = hashlib.sha256(corpus_bytes).hexdigest()
    warnings: list[str] = []
    if truncation_reason == "row_limit":
        warnings.append(
            f"Import truncated at the {row_limit}-row processing limit; "
            "one additional source row confirmed more data was available."
        )
    elif truncation_reason == "character_limit":
        warnings.append(f"Import truncated at the {char_limit}-character limit.")
    elif truncation_reason == "byte_limit":
        warnings.append(f"Import truncated at the {byte_limit}-byte UTF-8 limit.")
    elif truncation_reason == "character_and_byte_limit":
        warnings.append(
            "Import truncated where both the character and UTF-8 byte limits "
            "were exceeded."
        )
    if not direct_url and resolved_fingerprint is None:
        warnings.append(
            "The dataset loader did not expose a resolved fingerprint; retain "
            "the requested revision and content SHA-256 for provenance."
        )

    base_name = slugify(str(display_name or source.split("/")[-1]))
    name = base_name
    out_dir = Path(cache_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = 2
    while True:
        out_path = out_dir / f"{name}.txt"
        if (
            name == DEFAULT_DATASET["name"]
            or db.get_lab_dataset(name) is not None
        ):
            name = f"{base_name}-{suffix}"
            suffix += 1
            continue
        try:
            with out_path.open("xb") as handle:
                handle.write(corpus_bytes)
        except FileExistsError:
            name = f"{base_name}-{suffix}"
            suffix += 1
            continue
        break

    payload = {
        "name": name,
        "source_type": "huggingface",
        "source": source,
        "split": split,
        "text_column": text_column,
        "corpus_path": str(out_path),
        "n_rows": n_rows,
        "n_chars": len(corpus),
        "n_bytes": len(corpus_bytes),
        "preview": corpus[:500],
        "requested_revision": revision,
        "resolved_fingerprint": resolved_fingerprint,
        "revision_applicable": not direct_url,
        "row_limit": row_limit,
        "char_limit": char_limit,
        "byte_limit": byte_limit,
        "rows_examined": rows_examined,
        "sha256": digest,
        "truncated": truncation_reason is not None,
        "truncation_reason": truncation_reason,
        "warnings": warnings,
    }
    db.upsert_lab_dataset(payload)
    return payload
