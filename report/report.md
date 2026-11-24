# Query-perturbation robustness of BM25, dense and trie-constrained generative retrieval on SciFact

Maryam Tavasoli. Technical report, November 2026. Independent project; code and metrics: https://github.com/maryamtavasoli98/gr-query-robustness-scifact

## Abstract
Generative retrieval decodes document identifiers with trie-constrained beam search, and a recent reproduction study reports that intent-preserving typos can make the look-ahead planning of PAG collapse [1]. I ask how general this fragility is in a controlled, laptop-scale setting. On BEIR-format SciFact (5,183 abstracts, 300 test claims) I compare BM25, a MiniLM dense retriever and a small generative retriever (mini-GR: frozen MiniLM query encoder, GRU docid decoder, 4-code docids from a k-means tree, trie-constrained beam search) under typos, reordering, truncation and word dropout, using five perturbation seeds, bootstrap intervals and retention (perturbed / clean nDCG@10). At a 25 % typo rate mini-GR retains 0.68 of its clean nDCG@10, against 0.94 for BM25 and 0.91 for the dense model that shares its encoder. Training on typo-augmented queries lifts retention to 0.81. Per-query losses correlate only weakly with wordpiece inflation (Spearman 0.23 for mini-GR, about 0.1 for BM25 and dense), so the subword-fragmentation explanation suggested in [1] receives weak, correlational support at best. mini-GR is not PAG and the results do not transfer to it.

## 1. Introduction
Generative retrieval (GR) replaces the index with a model that generates document identifiers. Because many GR methods use trie-constrained beam search, a relevant prefix pruned early cannot be recovered. Mekonnen et al. [1] reproduce PAG at inference time and stress-test it with query variations (misspellings, reordering, synonyms, paraphrases, naturality shifts), finding that typos can trigger plan collapse and noting subword fragmentation as a plausible but untested contributor. I read the abstract and the arXiv HTML of [1], not their code. This report asks three questions on a much smaller system: (1) how does robustness compare across BM25, dense and generative retrieval under equal-strength perturbations; (2) is the loss related to wordpiece inflation; (3) does typo-augmented training or a wider beam help.

## 2. Related work
The only related work used here is [1], whose perturbation families and seed-averaged protocol (five seeds) and candidate-overlap diagnostic motivate the design; the data are SciFact in BEIR format [2]. I do not claim a broader literature review.

## 3. Method
**Systems.** BM25 (`rank_bm25`, lower-cased word tokens). Dense: `all-MiniLM-L6-v2` [3] cosine over title and abstract. mini-GR: a frozen MiniLM query embedding feeds a two-layer MLP and a GRU cell that emits four codes; docids are paths in a recursive 16-way k-means tree over TF-IDF/LSA document vectors (code vocabularies of size 16, 16, 16 and 10, the last disambiguating documents inside a leaf), so the identifier space does not depend on the dense encoder. Decoding is trie-constrained beam search (beam 32). Training: 25 epochs, AdamW with one-cycle schedule, on 51,539 pseudo-queries (titles and sentences of every document plus the SciFact train claims). The typo-trained variant adds two typo-corrupted copies. mini-GR and the dense model share the query encoder, so differences between them reflect the identifier and decoding layer.

**Perturbations.** `typoN`: each word of at least four letters receives one character edit (swap, delete, insert, keyboard-neighbour substitution) with probability N %, first letter kept; `reorder`; `truncate50`; `dropout30`. Five seeds, except the deterministic truncation.

**Measurement.** nDCG@10 and recall@10 with percentile bootstrap intervals over queries; retention with a joint bootstrap; seed standard deviation; paired bootstrap of retention differences; applied perturbation strength (wordpiece ratio, OOV rate, embedding cosine); top-10 overlap with the clean ranking; Spearman correlation of per-query nDCG drop with wordpiece ratio; prefix survival of relevant docid prefixes by decoding depth.

## 4. Experiments
Setup: <!-- table:env -->
5183 documents, 300 test queries, 51539 GR training pseudo-queries; docid code vocabularies (16, 16, 16, 10); beam 32; seeds (1, 2, 3, 4, 5); torch 2.15.0 on mps (arm64), Python 3.13.7; total run time 526.4 s.
<!-- /table:env -->

**nDCG@10.**
<!-- table:main -->
| system | clean | typo10 | typo25 | typo50 | reorder | truncate50 | dropout30 |
|---|---|---|---|---|---|---|---|
| BM25 | 0.652 [0.605, 0.699] | 0.628 | 0.614 | 0.571 | 0.652 | 0.493 | 0.570 |
| MiniLM dense | 0.648 [0.602, 0.695] | 0.610 | 0.587 | 0.526 | 0.639 | 0.416 | 0.505 |
| mini-GR | 0.556 [0.509, 0.604] | 0.444 | 0.378 | 0.285 | 0.494 | 0.262 | 0.353 |
| mini-GR + typo training | 0.625 [0.576, 0.671] | 0.552 | 0.509 | 0.434 | 0.569 | 0.323 | 0.424 |
<!-- /table:main -->

**Retention (perturbed / clean).**
<!-- table:retention -->
| system | typo10 | typo25 | typo50 | reorder | truncate50 | dropout30 |
|---|---|---|---|---|---|---|
| BM25 | 0.96 [0.95, 0.98] | 0.94 [0.92, 0.96] | 0.88 [0.84, 0.91] | 1.00 [1.00, 1.00] | 0.76 [0.70, 0.82] | 0.87 [0.85, 0.90] |
| MiniLM dense | 0.94 [0.92, 0.96] | 0.91 [0.88, 0.93] | 0.81 [0.78, 0.85] | 0.99 [0.96, 1.01] | 0.64 [0.58, 0.70] | 0.78 [0.75, 0.81] |
| mini-GR | 0.80 [0.76, 0.83] | 0.68 [0.64, 0.72] | 0.51 [0.47, 0.56] | 0.89 [0.85, 0.93] | 0.47 [0.40, 0.54] | 0.63 [0.60, 0.67] |
| mini-GR + typo training | 0.88 [0.85, 0.91] | 0.81 [0.78, 0.85] | 0.69 [0.65, 0.74] | 0.91 [0.88, 0.94] | 0.52 [0.46, 0.58] | 0.68 [0.64, 0.71] |
<!-- /table:retention -->

**Paired differences in retention.**
<!-- table:paired -->
| difference in retained nDCG@10 | typo10 | typo25 | typo50 | reorder | truncate50 | dropout30 |
|---|---|---|---|---|---|---|
| dense - BM25 | -0.02 [-0.08, +0.04] | -0.04 [-0.09, +0.02] | -0.06 [-0.12, -0.00] | -0.01 [-0.07, +0.04] | -0.11 [-0.18, -0.05] | -0.09 [-0.15, -0.04] |
| mini-GR - dense | -0.14 [-0.22, -0.07] | -0.23 [-0.30, -0.16] | -0.30 [-0.37, -0.23] | -0.10 [-0.19, -0.01] | -0.17 [-0.25, -0.10] | -0.15 [-0.21, -0.08] |
| typo-trained GR - mini-GR | +0.08 [+0.04, +0.13] | +0.13 [+0.09, +0.18] | +0.18 [+0.14, +0.22] | +0.02 [-0.03, +0.08] | +0.05 [-0.01, +0.10] | +0.04 [+0.01, +0.08] |
<!-- /table:paired -->

**Top-10 overlap with the clean ranking.**
<!-- table:stability -->
| system | typo10 | typo25 | typo50 | reorder | truncate50 | dropout30 |
|---|---|---|---|---|---|---|
| BM25 | 0.80 | 0.73 | 0.59 | 1.00 | 0.43 | 0.62 |
| MiniLM dense | 0.75 | 0.69 | 0.57 | 0.79 | 0.45 | 0.60 |
| mini-GR | 0.50 | 0.42 | 0.29 | 0.55 | 0.23 | 0.39 |
| mini-GR + typo training | 0.50 | 0.43 | 0.31 | 0.53 | 0.23 | 0.37 |
<!-- /table:stability -->

**Applied perturbation strength.**
<!-- table:calib -->
| condition | wordpiece ratio | query OOV rate | MiniLM cosine to clean | seeds |
|---|---|---|---|---|
| clean | 1.00 | 0.024 | 1.000 | - |
| typo10 | 1.10 | 0.122 | 0.914 | 5 |
| typo25 | 1.16 | 0.172 | 0.876 | 5 |
| typo50 | 1.29 | 0.292 | 0.789 | 5 |
| reorder | 1.00 | 0.024 | 0.942 | 5 |
| truncate50 | 0.51 | 0.025 | 0.775 | 1 |
| dropout30 | 0.71 | 0.024 | 0.841 | 5 |
<!-- /table:calib -->

**Fragmentation check.**
<!-- table:dose -->
| system | Spearman rho (wordpiece ratio vs per-query nDCG drop) | n query-perturbations |
|---|---|---|
| BM25 | 0.106 [0.076, 0.140] | 4500 |
| MiniLM dense | 0.115 [0.086, 0.144] | 4500 |
| mini-GR | 0.229 [0.200, 0.256] | 4500 |
| mini-GR + typo training | 0.162 [0.133, 0.191] | 4500 |
<!-- /table:dose -->

**Beam size and prefix survival (mini-GR, seed 1).**
<!-- table:beam -->
| beam | nDCG@10 clean | nDCG@10 typo25 | prefix survival clean (d1/d2/d3/d4) | prefix survival typo25 |
|---|---|---|---|---|
| 1 | 0.377 | 0.235 | 0.62 / 0.48 / 0.42 / 0.40 | 0.61 / 0.38 / 0.28 / 0.25 |
| 4 | 0.520 | 0.349 | 0.91 / 0.72 / 0.63 / 0.60 | 0.87 / 0.62 / 0.46 / 0.43 |
| 16 | 0.556 | 0.384 | 1.00 / 0.90 / 0.78 / 0.74 | 1.00 / 0.81 / 0.63 / 0.57 |
| 32 | 0.556 | 0.384 | 1.00 / 0.94 / 0.83 / 0.78 | 1.00 / 0.87 / 0.69 / 0.64 |
| 64 | 0.556 | 0.384 | 1.00 / 0.97 / 0.87 / 0.83 | 1.00 / 0.93 / 0.75 / 0.72 |
<!-- /table:beam -->

Observations. mini-GR retains less than the dense model under every perturbation, including reordering; the paired intervals for typo10, typo25 and typo50 exclude zero. BM25 and dense are not separable on typos at 10 and 25 %; dense is worse than BM25 on truncation and word dropout. Typo-trained mini-GR retains more than plain mini-GR on all three typo rates but its top-10 overlap with the clean ranking is unchanged. Prefix survival at the last depth is lower for typo queries than for clean queries at every beam size, while nDCG@10 stops changing from beam 16.

## 5. Limitations
mini-GR is a small DSI-style model with a frozen encoder, trained once (seed 0) on pseudo-queries from the test corpus; it says nothing quantitative about PAG or MS MARCO-scale systems. SciFact has 300 test queries. Perturbations are rule-based and not validated for intent preservation; no synonym or paraphrase condition was run. The fragmentation analysis is correlational and cannot separate wordpiece inflation from embedding shift. The typo-trained model sees more training pairs; I did not run a size-matched control. BM25 uses no stemming.

## 6. Conclusion
On this small benchmark, trie-constrained generative decoding is markedly less robust to query noise than BM25 or the dense model with the same encoder, typo-augmented training recovers part of the gap, and wordpiece inflation explains only a small share of the per-query loss. The study is a calibrated measurement template, not evidence about PAG.

## References
[1] K. Amde Mekonnen, Y. Li, Y. Tang, S. Lupart, M. de Rijke. Lost in Decoding? Reproducing and Stress-Testing the Look-Ahead Prior in Generative Retrieval. SIGIR 2026. doi:10.1145/3805712.3808567, arXiv:2604.23396.
[2] SciFact, BEIR format, `mteb/scifact` on the Hugging Face hub: https://huggingface.co/datasets/mteb/scifact
[3] `sentence-transformers/all-MiniLM-L6-v2`: https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
