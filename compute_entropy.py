import numpy as np
import os

CACHE_DIR = "esm_cache"
MODELS = [
    "esm2_t6_8M_UR50D",
    "esm2_t12_35M_UR50D",
    "esm2_t30_150M_UR50D",
    "esm2_t33_650M_UR50D",
]

out_path = os.path.join(CACHE_DIR, "esm2_entropy.npy")

entropies = []
for m in MODELS:
    path = os.path.join(CACHE_DIR, f"{m}_logprobs.npy")
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    logp = np.load(path)
    # logp already log-probs
    p = np.exp(logp)
    ent = -(p * logp).sum(axis=1)
    entropies.append(ent)

entropy = np.mean(np.vstack(entropies), axis=0)
np.save(out_path, entropy.astype(np.float32))
print(f"Saved {out_path} (shape={entropy.shape})")
