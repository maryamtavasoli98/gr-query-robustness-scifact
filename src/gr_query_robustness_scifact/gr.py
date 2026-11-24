"""Mini generative retriever: docid tree, GRU docid decoder, trie-constrained beam search.

This is a small DSI-style model, NOT PAG. The query is a frozen sentence embedding; the decoder emits a docid
(a path in a hierarchical k-means tree over TF-IDF/LSA document vectors) one code at a time. Decoding is
trie-constrained beam search, so a beam that prunes the right prefix early can never recover it.
"""

from __future__ import annotations

import numpy as np
import torch
from sklearn.cluster import KMeans
from torch import nn


def build_docids(vecs: np.ndarray, k: int = 16, depth: int = 3, seed: int = 0) -> np.ndarray:
    """Recursive k-means. Returns int array (n_docs, depth + 1); the last code disambiguates inside a leaf."""
    n = len(vecs)
    codes = np.zeros((n, depth + 1), dtype=np.int64)

    def rec(idx: np.ndarray, level: int) -> None:
        if level == depth:
            codes[idx, depth] = np.arange(len(idx))
            return
        kk = min(k, len(idx))
        if kk <= 1:
            codes[idx, level] = 0
            rec(idx, level + 1)
            return
        lab = KMeans(n_clusters=kk, n_init=2, random_state=seed).fit_predict(vecs[idx])
        for c in range(kk):
            sub = idx[lab == c]
            codes[sub, level] = c
            if len(sub):
                rec(sub, level + 1)

    rec(np.arange(n), 0)
    return codes


class Trie:
    """Prefix -> sorted list of allowed next codes."""

    def __init__(self, codes: np.ndarray):
        self.children: dict[tuple, list[int]] = {}
        self.leaf: dict[tuple, int] = {}
        for d, row in enumerate(codes):
            t = tuple(int(x) for x in row)
            for i in range(len(t)):
                self.children.setdefault(t[:i], set()).add(t[i])
            self.leaf[t] = d
        self.children = {p: sorted(v) for p, v in self.children.items()}
        self.depth = codes.shape[1]


class Decoder(nn.Module):
    def __init__(self, in_dim: int, sizes: list[int], hidden: int = 512):
        super().__init__()
        self.sizes = sizes
        self.inp = nn.Sequential(nn.Linear(in_dim, hidden), nn.GELU(), nn.Dropout(0.1), nn.Linear(hidden, hidden))
        self.emb = nn.ModuleList([nn.Embedding(s, hidden) for s in sizes[:-1]])
        self.cell = nn.GRUCell(hidden, hidden)
        self.out = nn.ModuleList([nn.Linear(hidden, s) for s in sizes])

    def forward(self, q: torch.Tensor, codes: torch.Tensor) -> torch.Tensor:
        """Teacher-forced mean cross-entropy over code positions."""
        h = self.inp(q)
        loss = 0.0
        for t, head in enumerate(self.out):
            loss = loss + nn.functional.cross_entropy(head(h), codes[:, t])
            if t < len(self.emb):
                h = self.cell(self.emb[t](codes[:, t]), h)
        return loss / len(self.out)

    @torch.no_grad()
    def step_logp(self, h: torch.Tensor, t: int) -> torch.Tensor:
        return torch.log_softmax(self.out[t](h), dim=-1)


@torch.no_grad()
def beam_search(model: Decoder, trie: Trie, q: torch.Tensor, beam: int, device: str = "cpu"):
    """Trie-constrained beam search for ONE query embedding. Returns (docs ranked, per-depth beams of prefixes)."""
    h0 = model.inp(q.to(device).unsqueeze(0))
    beams = [((), 0.0)]
    hs = h0
    history = []
    for t in range(trie.depth):
        logp = model.step_logp(hs, t).cpu().numpy()
        cand = []
        for bi, (prefix, score) in enumerate(beams):
            for c in trie.children[prefix]:
                cand.append((score + float(logp[bi, c]), bi, c))
        cand.sort(key=lambda x: -x[0])
        cand = cand[:beam]
        beams = [(beams[bi][0] + (c,), s) for s, bi, c in cand]
        history.append([b[0] for b in beams])
        if t < trie.depth - 1:
            prev = torch.tensor([bi for _, bi, _ in cand], device=device)
            cs = torch.tensor([c for _, _, c in cand], device=device)
            hs = model.cell(model.emb[t](cs), hs[prev])
    return [trie.leaf[p] for p, _ in beams], history
