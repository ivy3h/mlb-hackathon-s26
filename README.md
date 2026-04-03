# MLB S26 Hackathon — Protein Fitness Prediction (Group 2)

Zero-shot protein fitness prediction using ESM-2 log-likelihood ratios (LLR), ESM-IF inverse-folding scores, and Ridge calibration on a single round of queried labels.

**Best public leaderboard Spearman ρ = 0.41266**

## Method Overview

1. **ESM-2 LLR** — Compute masked marginal log-likelihood ratios from 4 ESM-2 models (8M, 35M, 150M, 650M) and take the median.
2. **ESM-IF LLR** — Structure-conditioned inverse-folding log-likelihood ratios from ESM-IF1.
3. **Entropy weighting** — Down-weight evolutionarily variable positions using ESM-2 per-position entropy.
4. **Ridge calibration** — Fit a Ridge regression on 100 queried labels (Round 1) using base score, pLDDT, and model disagreement as features, then blend with the uncalibrated baseline.

## Requirements

```bash
pip install fair-esm torch numpy pandas scikit-learn scipy
```

## Quick Reproduce (Best Submission)

All precomputed caches (`esm_cache/`) and data files are included in this repo.

```bash
bash reproduce_best.sh
```

This runs `calibrate_q1.py` and outputs predictions to `results/kaggle/`.

## Reproduce from Scratch

Requires a GPU and working ESM-IF dependencies (`torch_scatter` + GLIBC ≥ 2.32).

```bash
bash reproduce_from_scratch.sh
```

This recomputes all ESM logprobs, entropy, and ESM-IF scores from the raw sequence before running calibration.

## Files

| File | Description |
|------|-------------|
| `compute_esm_scores.py` | Compute ESM-2 masked marginal logprobs for all model sizes |
| `compute_entropy.py` | Compute per-position entropy from ESM-2 650M |
| `compute_esmif_scores.py` | Compute ESM-IF inverse-folding logprobs |
| `calibrate_q1.py` | Ridge calibration on queried labels + blend with baseline |
| `predict_v3.py` | Standalone ESM-2 avg4 LLR predictor (no calibration) |
| `blend_msa.py` | MSA Transformer blending utilities |
| `esm_cache/` | Precomputed logprobs, entropy, and pLDDT arrays |
| `Hackathon_data/` | Train/test splits and queried labels |
