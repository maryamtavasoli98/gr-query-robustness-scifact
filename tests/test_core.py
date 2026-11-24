import itertools

import numpy as np
import torch

from gr_query_robustness_scifact import gr, metrics, perturb


def test_ndcg_perfect_and_zero():
    rel = {"a": 1, "b": 1}
    assert metrics.ndcg_at_k(["a", "b", "c"], rel) == 1.0
    assert metrics.ndcg_at_k(["c", "d"], rel) == 0.0
    assert 0 < metrics.ndcg_at_k(["c", "a", "b"], rel) < 1


def test_recall_and_overlap():
    assert metrics.recall_at_k(["a", "x"], {"a": 1, "b": 1}, 2) == 0.5
    assert metrics.overlap_at_k(["a", "b", "c"], ["c", "b", "z"], 3) == 2 / 3


def test_typos_seeded_and_bounded():
    q = "monoclonal antibodies reduce inflammation in rheumatoid arthritis"
    assert perturb.typos(q, 0.5, 7) == perturb.typos(q, 0.5, 7)
    out = perturb.typos(q, 0.5, 7)
    assert out != q and len(out.split()) == len(q.split())
    assert perturb.typos("a to be", 1.0, 1) == "a to be"  # nothing eligible


def test_first_letter_kept_and_edit_distance_one():
    for s in range(50):
        w = perturb.typo_word("inflammation", __import__("random").Random(s))
        assert w[0] == "i" and abs(len(w) - 12) <= 1


def test_reorder_truncate_dropout():
    q = "one two three four five six"
    assert sorted(perturb.reorder(q, 3).split()) == sorted(q.split())
    assert perturb.truncate(q, 0.5) == "one two three"
    assert 1 <= len(perturb.word_dropout(q, 0.9, 1).split()) <= 6


def test_bootstrap_ci_contains_mean():
    v = np.random.default_rng(0).normal(0.5, 0.1, 200)
    m, lo, hi = metrics.bootstrap_ci(v)
    assert lo < m < hi and hi - lo < 0.1


def test_retention_identity():
    x = np.array([0.2, 0.4, 0.6])
    r, lo, hi = metrics.relative_retention_ci(x, x)
    assert r == 1.0 and lo == 1.0 and hi == 1.0


def test_docids_unique_and_trie_complete():
    rng = np.random.default_rng(0)
    vecs = rng.normal(size=(200, 8))
    codes = gr.build_docids(vecs, k=4, depth=2, seed=0)
    assert len({tuple(r) for r in codes}) == 200
    trie = gr.Trie(codes)
    assert len(trie.leaf) == 200 and trie.depth == 3


def test_beam_search_only_valid_docs_and_wide_beam_is_exhaustive():
    rng = np.random.default_rng(1)
    codes = gr.build_docids(rng.normal(size=(60, 6)), k=3, depth=2, seed=0)
    trie = gr.Trie(codes)
    sizes = [int(codes[:, i].max()) + 1 for i in range(codes.shape[1])]
    torch.manual_seed(0)
    model = gr.Decoder(5, sizes, hidden=16).eval()
    q = torch.randn(5)
    docs, hist = gr.beam_search(model, trie, q, beam=1000)
    assert sorted(docs) == list(range(60))  # exhaustive beam returns every doc once
    lp = {}
    h = model.inp(q.unsqueeze(0))
    for d in range(60):
        c = codes[d]
        hh, s = h, 0.0
        for t in range(len(c)):
            s += float(model.step_logp(hh, t)[0, c[t]])
            if t < len(c) - 1:
                hh = model.cell(model.emb[t](torch.tensor([int(c[t])])), hh)
        lp[d] = s
    assert docs == sorted(lp, key=lambda d: -lp[d])  # exact ordering by sequence log-prob
    small, _ = gr.beam_search(model, trie, q, beam=3)
    assert len(small) == 3 and set(small) <= set(range(60))
    assert all(len(set(b)) == len(b) for b in hist)


def test_conditions_cover_all_names():
    for c, s in itertools.product(perturb.CONDITIONS, [1, 2]):
        assert isinstance(perturb.perturb("tumor necrosis factor signalling pathways", c, s), str)
