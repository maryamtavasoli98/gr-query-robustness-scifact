"""Download BEIR SciFact (8 MB) into data/raw and print basic counts."""

from pathlib import Path

from gr_query_robustness_scifact import data

raw = Path("data/raw")
data.download(raw)
d = data.load(raw)
print(f"docs={len(d['doc_ids'])} test_queries={len(d['qrels_test'])} train_queries={len(d['qrels_train'])}")
