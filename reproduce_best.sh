#!/usr/bin/env bash
set -euo pipefail

# Reproduce best public score (0.41266)
# Requires cache files in esm_cache/ and queried_round1.csv in Hackathon_data/

python calibrate_q1.py
