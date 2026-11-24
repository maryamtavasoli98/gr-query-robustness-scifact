"""Load BEIR SciFact (mteb/scifact mirror on the Hugging Face hub, 8 MB, no login)."""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path

BASE = "https://huggingface.co/datasets/mteb/scifact/resolve/main/"
FILES = ["corpus.jsonl", "queries.jsonl", "qrels/test.jsonl", "qrels/train.jsonl"]


def download(raw_dir: Path) -> None:
    for f in FILES:
        dst = raw_dir / f
        if dst.exists():
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(BASE + f, dst)


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def load(raw_dir: Path) -> dict:
    """Return corpus (ids, texts), queries and qrels {qid: {docid: grade}} for train and test."""
    corpus = _jsonl(raw_dir / "corpus.jsonl")
    queries = {q["_id"]: q["text"] for q in _jsonl(raw_dir / "queries.jsonl")}
    out = {"doc_ids": [d["_id"] for d in corpus], "titles": [d.get("title", "") for d in corpus]}
    out["texts"] = [d.get("text", "") for d in corpus]
    for split in ("train", "test"):
        qrels: dict[str, dict[str, int]] = {}
        for r in _jsonl(raw_dir / "qrels" / f"{split}.jsonl"):
            qrels.setdefault(r["query-id"], {})[r["corpus-id"]] = int(r["score"])
        out[f"qrels_{split}"] = qrels
    out["queries"] = queries
    return out
