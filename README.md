# MLB S26 Hackathon (Group 2)

Zero-shot protein fitness prediction using ESM-2 log-likelihood ratios (LLR), ESM-IF inverse-folding scores, and Ridge calibration on a single round of queried labels.

**Best public leaderboard Spearman ρ = 0.41266**

## Quick Start (Colab)

The fastest way to reproduce our best result is the self-contained Colab notebook
[`notebook.ipynb`](notebook.ipynb).

1. Open https://colab.research.google.com/github/ivy3h/mlb-hackathon-s26/blob/main/notebook.ipynb
   (or upload `notebook.ipynb` directly to Colab).
2. **Runtime → Run all**.
3. Sections 1–3 install dependencies, clone this repo (which already includes all
   `Hackathon_data/` CSVs and the `esm_cache/` `.npy` features), and reproduce the
   best leaderboard submission. **No GPU required** for the best-result reproduction.
4. The submission file `predictions_q1cal_w20.csv` (Spearman ρ ≈ 0.41266) is
   written to the working directory at the end of Section 3.

Optional sections in the notebook:
- **Section 4** — re-compute ESM-2 masked-marginal features from scratch (GPU recommended).
- **Section 5** — wider ablation from `predict_v3.py` (Nelder–Mead weights, Ridge, isotonic, 5-fold CV).
- **Section 6** — MSA-Transformer blend from `blend_msa.py`.

## Method Overview

1. **ESM-2 LLR** — Compute masked marginal log-likelihood ratios from 4 ESM-2 models (8M, 35M, 150M, 650M) and take the median.
2. **ESM-IF LLR** — Structure-conditioned inverse-folding log-likelihood ratios from ESM-IF1.
3. **Entropy weighting** — Down-weight evolutionarily variable positions using ESM-2 per-position entropy.
4. **Ridge calibration** — Fit a Ridge regression on 100 queried labels (Round 1) using base score, pLDDT, and model disagreement as features, then blend with the uncalibrated baseline.

## Requirements

```bash
pip install fair-esm torch numpy pandas scikit-learn scipy
```

## Quick Reproduce

```bash
bash reproduce_best.sh
```

## Reproduce from Scratch

```bash
bash reproduce_from_scratch.sh
```
