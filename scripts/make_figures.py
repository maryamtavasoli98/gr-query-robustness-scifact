"""Figures from results/metrics/*.json."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

M = Path("results/metrics")
F = Path("results/figures")
F.mkdir(parents=True, exist_ok=True)
main = json.loads((M / "main.json").read_text())
beam = json.loads((M / "beam.json").read_text())
agg = main["agg"]
SYS = {"bm25": "BM25", "dense": "MiniLM dense", "gr": "mini-GR", "gr_typoaug": "mini-GR + typo training"}
COL = {"bm25": "#4c78a8", "dense": "#f58518", "gr": "#54a24b", "gr_typoaug": "#b279a2"}
conds = ["typo10", "typo25", "typo50", "reorder", "truncate50", "dropout30"]

# Fig 1: retention (perturbed / clean nDCG@10) with bootstrap CI
fig, ax = plt.subplots(figsize=(9, 4))
w = 0.2
for k, s in enumerate(SYS):
    r = np.array([agg[c][s]["retention"] for c in conds])
    lo = np.array([agg[c][s]["retention_lo"] for c in conds])
    hi = np.array([agg[c][s]["retention_hi"] for c in conds])
    ax.bar(np.arange(len(conds)) + (k - 1.5) * w, r, w, yerr=[r - lo, hi - r], color=COL[s], label=SYS[s], capsize=2)
ax.axhline(1, color="k", lw=0.6)
ax.set_xticks(range(len(conds)))
ax.set_xticklabels(conds)
ax.set_ylabel("nDCG@10 retained (perturbed / clean)")
ax.set_title("SciFact: how much effectiveness survives each perturbation (95% bootstrap CI)")
ax.set_ylim(0, 1.25)
ax.legend(fontsize=8, ncol=4, loc="upper center")
fig.tight_layout()
fig.savefig(F / "retention.png", dpi=150)
plt.close(fig)

# Fig 2: dose-response, retention vs mean wordpiece inflation for typo conditions
fig, ax = plt.subplots(figsize=(5.5, 4))
for s in SYS:
    xs = [1.0] + [main["conditions"][c]["mean_wordpiece_ratio"] for c in ("typo10", "typo25", "typo50")]
    ys = [1.0] + [agg[c][s]["retention"] for c in ("typo10", "typo25", "typo50")]
    ax.plot(xs, ys, "o-", color=COL[s], label=SYS[s])
ax.set_xlabel("mean MiniLM wordpiece count, perturbed / clean")
ax.set_ylabel("nDCG@10 retained")
ax.set_title("Typo dose-response")
ax.legend(fontsize=7)
fig.tight_layout()
fig.savefig(F / "dose_response.png", dpi=150)
plt.close(fig)

# Fig 3: early pruning, prefix survival of a relevant document by depth, clean vs typo25
fig, axs = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
for ax, name in zip(axs, ("clean", "typo25")):
    for b, v in beam[name].items():
        ax.plot(range(1, 5), v["prefix_survival_by_depth"], "o-", label=f"beam {b}")
    ax.set_title(name + " queries")
    ax.set_xlabel("decoding depth (docid code position)")
    ax.set_xticks(range(1, 5))
axs[0].set_ylabel("P(a relevant prefix is still in the beam)")
axs[1].legend(fontsize=7)
fig.tight_layout()
fig.savefig(F / "prefix_survival.png", dpi=150)
plt.close(fig)
