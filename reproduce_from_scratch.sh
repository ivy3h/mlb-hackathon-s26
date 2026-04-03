#!/usr/bin/env bash
set -euo pipefail

# Full reproduction: compute ESM-2 logprobs + entropy + ESM-IF logprobs, then run calibration.
# Requires GPU and working ESM-IF dependencies (torch_scatter + GLIBC>=2.32).

python compute_esm_scores.py
python compute_entropy.py
python compute_esmif_scores.py

python calibrate_q1.py
