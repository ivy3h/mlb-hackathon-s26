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

## Quick Reproduce

```bash
bash reproduce_best.sh
```

## Reproduce from Scratch

```bash
bash reproduce_from_scratch.sh
```
