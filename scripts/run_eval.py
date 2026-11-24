"""Run all experiments and write results/metrics/*.json (everything in the README comes from these files)."""

from __future__ import annotations

import json
import platform
import re
import time
from pathlib import Path

import numpy as np
import torch
from rank_bm25 import BM25Okapi
from scipy.stats import spearmanr
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

from gr_query_robustness_scifact import data, gr, metrics, perturb

OUT = Path("results/metrics")
CACHE = Path("data/cache")
SEEDS = [1, 2, 3, 4, 5]
BEAM = 32
EPOCHS = 25
tok = lambda s: re.findall(r"\w+", s.lower())  # noqa: E731
t_start = time.time()
timings: dict[str, float] = {}


def tick(name: str, t0: float) -> None:
    timings[name] = round(time.time() - t0, 1)
    print(f"[{timings[name]:7.1f}s] {name}", flush=True)


def qseed(seed: int, i: int) -> int:
    return seed * 100003 + i


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    np.random.seed(0)
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    d = data.load(Path("data/raw"))
    ids = d["doc_ids"]
    doc_texts = [f"{t}. {x}" for t, x in zip(d["titles"], d["texts"], strict=True)]
    qids = sorted(d["qrels_test"], key=int)
    queries = [d["queries"][q] for q in qids]
    rels = [d["qrels_test"][q] for q in qids]
    nq = len(qids)
    print(f"{len(ids)} docs, {nq} test queries, device={dev}")

    # ---- systems: BM25, dense, docid tree
    t0 = time.time()
    bm25 = BM25Okapi([tok(t) for t in doc_texts])
    vocab = {w for t in doc_texts for w in tok(t)}
    tick("bm25_index", t0)
    t0 = time.time()
    enc = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=dev)
    enc.max_seq_length = 256
    tokenizer = enc.tokenizer

    def embed(texts: list[str], bs: int = 256) -> np.ndarray:
        return enc.encode(texts, batch_size=bs, normalize_embeddings=True, show_progress_bar=False)

    dpath = CACHE / "doc_emb.npy"
    if dpath.exists():
        doc_emb = np.load(dpath)
    else:
        doc_emb = embed(doc_texts, 64)
        np.save(dpath, doc_emb)
    tick("dense_doc_embeddings", t0)
    t0 = time.time()
    lsa = TruncatedSVD(128, random_state=0).fit_transform(
        TfidfVectorizer(sublinear_tf=True, stop_words="english").fit_transform(doc_texts)
    )
    lsa /= np.linalg.norm(lsa, axis=1, keepdims=True) + 1e-9
    codes = gr.build_docids(lsa, k=16, depth=3, seed=0)
    trie = gr.Trie(codes)
    sizes = [int(codes[:, i].max()) + 1 for i in range(codes.shape[1])]
    tick("docid_tree", t0)
    id2idx = {x: i for i, x in enumerate(ids)}

    # ---- GR training data: pseudo-queries (title + sentences of each doc) + SciFact train claims
    t0 = time.time()
    pq, pdoc = [], []
    for i, (title, text) in enumerate(zip(d["titles"], d["texts"], strict=True)):
        sents = [s for s in re.split(r"(?<=[.!?])\s+", text) if len(s) >= 20]
        for s in ([title] if title else []) + sents:
            pq.append(s)
            pdoc.append(i)
    for q, rel in d["qrels_train"].items():
        for doc, g in rel.items():
            if g > 0 and doc in id2idx:
                pq.append(d["queries"][q])
                pdoc.append(id2idx[doc])
    pdoc = np.array(pdoc)
    print(f"{len(pq)} training pseudo-queries")
    rng = np.random.default_rng(123)
    aug_texts = []
    for rep in range(2):
        for j, s in enumerate(pq):
            rate = float(rng.choice([0.1, 0.25, 0.5]))
            aug_texts.append(perturb.typos(s, rate, 10_000_000 + rep * 1_000_003 + j))
    cpath = CACHE / "train_emb.npz"
    if cpath.exists():
        z = np.load(cpath)
        e_clean, e_aug = z["clean"], z["aug"]
    else:
        e_clean, e_aug = embed(pq), embed(aug_texts)
        np.savez(cpath, clean=e_clean, aug=e_aug)
    tick("train_embeddings", t0)

    def train(x: np.ndarray, doc_idx: np.ndarray, name: str) -> gr.Decoder:
        t1 = time.time()
        torch.manual_seed(0)
        model = gr.Decoder(x.shape[1], sizes).to(dev)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
        xt = torch.tensor(x, device=dev)
        ct = torch.tensor(codes[doc_idx], device=dev)
        n = len(xt)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-3, total_steps=EPOCHS * ((n + 255) // 256))
        for ep in range(EPOCHS):
            model.train()
            perm = torch.randperm(n, device=dev)
            tot = 0.0
            for s in range(0, n, 256):
                b = perm[s : s + 256]
                loss = model(xt[b], ct[b])
                opt.zero_grad()
                loss.backward()
                opt.step()
                sched.step()
                tot += float(loss) * len(b)
            if ep % 5 == 4 or ep == EPOCHS - 1:
                print(f"  {name} epoch {ep + 1} loss {tot / n:.4f}", flush=True)
        model.eval()
        tick(f"train_{name}", t1)
        return model.cpu()

    gr_plain = train(e_clean, pdoc, "gr_plain")
    gr_aug = train(np.concatenate([e_clean, e_aug]), np.concatenate([pdoc, pdoc, pdoc]), "gr_typoaug")

    # ---- ranking functions (return list of doc ids)
    def rank_bm25(qs: list[str]) -> list[list[str]]:
        out = []
        for q in qs:
            sc = bm25.get_scores(tok(q))
            out.append([ids[i] for i in np.argsort(-sc)[:100]])
        return out

    def rank_dense(qe: np.ndarray) -> list[list[str]]:
        sc = qe @ doc_emb.T
        return [[ids[i] for i in np.argsort(-row)[:100]] for row in sc]

    def rank_gr(model: gr.Decoder, qe: np.ndarray, beam: int = BEAM):
        res, hist = [], []
        for v in qe:
            docs, h = gr.beam_search(model, trie, torch.tensor(v), beam)
            res.append([ids[i] for i in docs])
            hist.append(h)
        return res, hist

    def run_all(qs: list[str], beam: int = BEAM):
        qe = embed(qs)
        g1, h1 = rank_gr(gr_plain, qe, beam)
        g2, _ = rank_gr(gr_aug, qe, beam)
        return {"bm25": rank_bm25(qs), "dense": rank_dense(qe), "gr": g1, "gr_typoaug": g2}, qe, h1

    def scores(rk: list[list[str]]):
        return (
            np.array([metrics.ndcg_at_k(r, rel, 10) for r, rel in zip(rk, rels, strict=True)]),
            np.array([metrics.recall_at_k(r, rel, 10) for r, rel in zip(rk, rels, strict=True)]),
        )

    # ---- E1: clean + 6 conditions x 5 seeds
    t0 = time.time()
    clean_rk, clean_qe, clean_hist = run_all(queries)
    systems = list(clean_rk)
    res: dict = {
        "clean": {
            s: {"ndcg10": [scores(clean_rk[s])[0].tolist()], "recall10": [scores(clean_rk[s])[1].tolist()]}
            for s in systems
        }
    }
    stab: dict = {}
    infl: dict = {}
    nwp = lambda s: len(tokenizer.tokenize(s))  # noqa: E731
    clean_wp = np.array([nwp(q) for q in queries])
    clean_words = [tok(q) for q in queries]
    cond_info: dict = {}
    for cond in perturb.CONDITIONS:
        seeds = SEEDS[:1] if cond in perturb.DETERMINISTIC else SEEDS
        res[cond] = {s: {"ndcg10": [], "recall10": []} for s in systems}
        stab[cond] = {s: [] for s in systems}
        infl[cond] = {"wp_ratio": [], "cos": [], "ndcg_drop": {s: [] for s in systems}}
        oov = []
        for seed in seeds:
            pq_ = [perturb.perturb(q, cond, qseed(seed, i)) for i, q in enumerate(queries)]
            rk, qe, _ = run_all(pq_)
            for s in systems:
                nd, rc = scores(rk[s])
                res[cond][s]["ndcg10"].append(nd.tolist())
                res[cond][s]["recall10"].append(rc.tolist())
                stab[cond][s].append(
                    float(np.mean([metrics.overlap_at_k(a, b) for a, b in zip(clean_rk[s], rk[s], strict=True)]))
                )
                infl[cond]["ndcg_drop"][s].append((np.array(res["clean"][s]["ndcg10"][0]) - nd).tolist())
            infl[cond]["wp_ratio"].append((np.array([nwp(q) for q in pq_]) / clean_wp).tolist())
            infl[cond]["cos"].append((qe * clean_qe).sum(1).tolist())
            oov.append(np.mean([np.mean([w not in vocab for w in tok(q)]) for q in pq_]))
        cond_info[cond] = {
            "oov_rate": float(np.mean(oov)),
            "mean_cosine_to_clean": float(np.mean(infl[cond]["cos"])),
            "mean_wordpiece_ratio": float(np.mean(infl[cond]["wp_ratio"])),
            "n_seeds": len(seeds),
        }
        print(cond, {s: round(float(np.mean(res[cond][s]["ndcg10"])), 3) for s in systems}, flush=True)
    clean_oov = float(np.mean([np.mean([w not in vocab for w in c]) for c in clean_words]))
    tick("e1_conditions", t0)

    # ---- E2: dose-response of per-query nDCG drop on wordpiece inflation (typo conditions pooled)
    dose = {}
    for s in systems:
        x = np.concatenate([np.array(infl[c]["wp_ratio"]).ravel() for c in ("typo10", "typo25", "typo50")])
        y = np.concatenate([np.array(infl[c]["ndcg_drop"][s]).ravel() for c in ("typo10", "typo25", "typo50")])
        rho, p = spearmanr(x, y)
        rng2 = np.random.default_rng(0)
        bs = []
        for _ in range(500):
            ix = rng2.integers(0, len(x), len(x))
            bs.append(spearmanr(x[ix], y[ix])[0])
        dose[s] = {
            "spearman_rho": float(rho),
            "p": float(p),
            "ci": [float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))],
            "n": int(len(x)),
        }

    # ---- E4: beam size and early pruning (clean vs typo25, seed 1)
    t0 = time.time()
    beam_res = {}
    pq25 = [perturb.perturb(q, "typo25", qseed(1, i)) for i, q in enumerate(queries)]
    qe25 = embed(pq25)
    for name, qe_ in (("clean", clean_qe), ("typo25", qe25)):
        for beam in (1, 4, 16, 32, 64):
            rk, hist = rank_gr(gr_plain, qe_, beam)
            nd = np.array([metrics.ndcg_at_k(r, rel, 10) for r, rel in zip(rk, rels, strict=True)])
            surv = []
            for depth in range(trie.depth):
                ok = []
                for h, rel in zip(hist, rels, strict=True):
                    pref = {
                        tuple(int(x) for x in codes[id2idx[dd]][: depth + 1])
                        for dd, g in rel.items()
                        if g > 0 and dd in id2idx
                    }
                    ok.append(float(any(p in pref for p in h[depth])))
                surv.append(float(np.mean(ok)))
            beam_res.setdefault(name, {})[str(beam)] = {"ndcg10": float(nd.mean()), "prefix_survival_by_depth": surv}
        print("beam", name, {b: round(v["ndcg10"], 3) for b, v in beam_res[name].items()}, flush=True)
    tick("e4_beams", t0)

    # ---- aggregate
    agg: dict = {}
    for cond, per in res.items():
        agg[cond] = {}
        for s, v in per.items():
            arr = np.array(v["ndcg10"])  # seeds x queries
            seed_means = arr.mean(axis=1)
            m, lo, hi = metrics.bootstrap_ci(arr.mean(axis=0), seed=0)
            cl = np.array(res["clean"][s]["ndcg10"][0])
            ret, rlo, rhi = metrics.relative_retention_ci(cl, arr.mean(axis=0))
            agg[cond][s] = {
                "ndcg10": m, "ci_lo": lo, "ci_hi": hi,
                "seed_std": float(seed_means.std(ddof=1)) if len(seed_means) > 1 else 0.0,
                "recall10": float(np.mean(v["recall10"])),
                "retention": ret, "retention_lo": rlo, "retention_hi": rhi,
                "top10_overlap_with_clean": float(np.mean(stab[cond][s])) if cond != "clean" else 1.0,
            }  # fmt: skip
    pair = {}
    for cond in perturb.CONDITIONS:
        pair[cond] = {}
        for a, b in (("dense", "bm25"), ("gr", "dense"), ("gr_typoaug", "gr")):
            ra = np.array(res[cond][a]["ndcg10"]).mean(0) / max(np.mean(res["clean"][a]["ndcg10"][0]), 1e-9)
            rb = np.array(res[cond][b]["ndcg10"]).mean(0) / max(np.mean(res["clean"][b]["ndcg10"][0]), 1e-9)
            # paired bootstrap of the difference in per-query retained nDCG (normalised by each system's clean mean)
            m, lo, hi = metrics.paired_bootstrap_diff(ra, rb)
            pair[cond][f"{a}_minus_{b}"] = {"diff_in_retention": m, "ci_lo": lo, "ci_hi": hi}
    env = {"python": platform.python_version(), "torch": torch.__version__, "device": dev, "machine": platform.machine(), "seeds": SEEDS, "beam": BEAM, "epochs": EPOCHS, "n_docs": len(ids), "n_test_queries": nq, "n_train_pseudo_queries": len(pq), "docid_vocab_sizes": sizes, "timings_s": timings, "total_s": round(time.time() - t_start, 1)}  # fmt: skip
    (OUT / "main.json").write_text(
        json.dumps(
            {"env": env, "agg": agg, "paired": pair, "conditions": cond_info, "clean_oov_rate": clean_oov}, indent=1
        )
    )
    (OUT / "dose_response.json").write_text(json.dumps(dose, indent=1))
    (OUT / "beam.json").write_text(json.dumps(beam_res, indent=1))
    pqd = {
        c: {s: np.round(np.array(v["ndcg10"]).mean(0), 4).tolist() for s, v in per.items()} for c, per in res.items()
    }
    (OUT / "per_query_ndcg.json").write_text(json.dumps(pqd))
    wp = {c: {"wp_ratio": np.round(np.array(infl[c]["wp_ratio"]).mean(0), 3).tolist()} for c in perturb.CONDITIONS}
    (OUT / "per_query_wordpiece_ratio.json").write_text(json.dumps(wp))
    print("done", env["total_s"], "s")


if __name__ == "__main__":
    main()
