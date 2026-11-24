"""Render tables from results/metrics/*.json into README.md and report/report.md between <!-- table:NAME --> markers."""

import json
import re
from pathlib import Path

M = Path("results/metrics")
main = json.loads((M / "main.json").read_text())
dose = json.loads((M / "dose_response.json").read_text())
beam = json.loads((M / "beam.json").read_text())
agg, conds = main["agg"], ["typo10", "typo25", "typo50", "reorder", "truncate50", "dropout30"]
SYS = {"bm25": "BM25", "dense": "MiniLM dense", "gr": "mini-GR", "gr_typoaug": "mini-GR + typo training"}


def md(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def t_main():
    h = ["system", "clean"] + conds
    rows = []
    for s, name in SYS.items():
        c = agg["clean"][s]
        rows.append(
            [name, f"{c['ndcg10']:.3f} [{c['ci_lo']:.3f}, {c['ci_hi']:.3f}]"]
            + [f"{agg[k][s]['ndcg10']:.3f}" for k in conds]
        )
    return md(h, rows)


def t_ret():
    rows = []
    for s, name in SYS.items():
        rows.append(
            [name]
            + [
                f"{agg[k][s]['retention']:.2f} [{agg[k][s]['retention_lo']:.2f}, {agg[k][s]['retention_hi']:.2f}]"
                for k in conds
            ]
        )
    return md(["system"] + conds, rows)


def t_stab():
    rows = [[name] + [f"{agg[k][s]['top10_overlap_with_clean']:.2f}" for k in conds] for s, name in SYS.items()]
    return md(["system"] + conds, rows)


def t_seed():
    rows = [[name] + [f"{agg[k][s]['seed_std']:.4f}" for k in conds] for s, name in SYS.items()]
    return md(["system"] + conds, rows)


def t_calib():
    rows = [["clean", "1.00", f"{main['clean_oov_rate']:.3f}", "1.000", "-"]]
    for k in conds:
        c = main["conditions"][k]
        rows.append(
            [
                k,
                f"{c['mean_wordpiece_ratio']:.2f}",
                f"{c['oov_rate']:.3f}",
                f"{c['mean_cosine_to_clean']:.3f}",
                c["n_seeds"],
            ]
        )
    return md(["condition", "wordpiece ratio", "query OOV rate", "MiniLM cosine to clean", "seeds"], rows)


def t_dose():
    rows = [[SYS[s], f"{v['spearman_rho']:.3f} [{v['ci'][0]:.3f}, {v['ci'][1]:.3f}]", v["n"]] for s, v in dose.items()]
    return md(["system", "Spearman rho (wordpiece ratio vs per-query nDCG drop)", "n query-perturbations"], rows)


def t_beam():
    rows = []
    for b in ["1", "4", "16", "32", "64"]:
        c, t = beam["clean"][b], beam["typo25"][b]
        rows.append(
            [
                b,
                f"{c['ndcg10']:.3f}",
                f"{t['ndcg10']:.3f}",
                " / ".join(f"{x:.2f}" for x in c["prefix_survival_by_depth"]),
                " / ".join(f"{x:.2f}" for x in t["prefix_survival_by_depth"]),
            ]
        )
    return md(
        ["beam", "nDCG@10 clean", "nDCG@10 typo25", "prefix survival clean (d1/d2/d3/d4)", "prefix survival typo25"],
        rows,
    )


def t_pair():
    p = main["paired"]
    names = {
        "dense_minus_bm25": "dense - BM25",
        "gr_minus_dense": "mini-GR - dense",
        "gr_typoaug_minus_gr": "typo-trained GR - mini-GR",
    }
    rows = []
    for key, nm in names.items():
        rows.append(
            [nm]
            + [
                f"{p[k][key]['diff_in_retention']:+.2f} [{p[k][key]['ci_lo']:+.2f}, {p[k][key]['ci_hi']:+.2f}]"
                for k in conds
            ]
        )
    return md(["difference in retained nDCG@10"] + conds, rows)


def t_env():
    e = main["env"]
    return (
        f"{e['n_docs']} documents, {e['n_test_queries']} test queries, {e['n_train_pseudo_queries']} GR training pseudo-queries; docid code vocabularies {tuple(e['docid_vocab_sizes'])}; "
        f"beam {e['beam']}; seeds {tuple(e['seeds'])}; torch {e['torch']} on {e['device']} ({e['machine']}), Python {e['python']}; total run time {e['total_s']} s."
    )


TABLES = {
    "main": t_main,
    "retention": t_ret,
    "stability": t_stab,
    "seed": t_seed,
    "calib": t_calib,
    "dose": t_dose,
    "beam": t_beam,
    "paired": t_pair,
    "env": t_env,
}
for path in (Path("README.md"), Path("report/report.md")):
    if not path.exists():
        continue
    txt = path.read_text()
    for name, fn in TABLES.items():
        txt = re.sub(
            rf"(<!-- table:{name} -->)(.*?)(<!-- /table:{name} -->)",
            lambda m, fn=fn: f"{m.group(1)}\n{fn()}\n{m.group(3)}",
            txt,
            flags=re.S,
        )
    path.write_text(txt)
    print("updated", path)
