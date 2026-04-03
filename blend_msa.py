import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import re
import esm
import sys, types

# Monkey-patch to avoid GLIBC error on login nodes
sys.modules['esm.inverse_folding'] = types.ModuleType('esm.inverse_folding')

DATA_DIR = "Hackathon_data"
CACHE_DIR = "esm_cache"

MODELS = [
    "esm2_t6_8M_UR50D",
    "esm2_t12_35M_UR50D",
    "esm2_t30_150M_UR50D",
    "esm2_t33_650M_UR50D",
]


def get_aa_to_tok():
    _, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
    aa_list = "ACDEFGHIKLMNPQRSTVWY"
    return {aa: alphabet.get_idx(aa) for aa in aa_list}


def parse_mutants(mutants):
    wt = []
    pos = []
    mut = []
    for code in mutants:
        m = re.match(r'^([A-Z])(\d+)([A-Z])$', code)
        wt.append(m.group(1))
        pos.append(int(m.group(2)))
        mut.append(m.group(3))
    return wt, np.array(pos, dtype=np.int64), mut


def llr_from_logprobs(logprobs, pos, wt_tok, mut_tok):
    # pos is 1-based in dataset; existing pipeline uses it directly
    return logprobs[pos, mut_tok] - logprobs[pos, wt_tok]


def compute_base_preds(mutants, aa_to_tok, logprobs_dict, esmif_lp, entropy):
    wt, pos, mut = parse_mutants(mutants)
    wt_tok = np.array([aa_to_tok[a] for a in wt], dtype=np.int64)
    mut_tok = np.array([aa_to_tok[a] for a in mut], dtype=np.int64)

    llrs = []
    for m in MODELS:
        lp = logprobs_dict[m]
        llrs.append(llr_from_logprobs(lp, pos, wt_tok, mut_tok))
    llrs = np.vstack(llrs).T  # (n,4)

    if_llr = llr_from_logprobs(esmif_lp, pos, wt_tok, mut_tok)
    ew = 1.0 / (1.0 + 0.3 * entropy[pos])

    med4 = np.median(llrs, axis=1)
    base = ew * (0.85 * med4 + 0.15 * if_llr)
    return base


def compute_msa_llr(mutants, aa_to_tok, msa_lp):
    wt, pos, mut = parse_mutants(mutants)
    wt_tok = np.array([aa_to_tok[a] for a in wt], dtype=np.int64)
    mut_tok = np.array([aa_to_tok[a] for a in mut], dtype=np.int64)
    return llr_from_logprobs(msa_lp, pos, wt_tok, mut_tok)


def main():
    train_df = pd.read_csv(f"{DATA_DIR}/train.csv")
    test_df = pd.read_csv(f"{DATA_DIR}/test.csv")

    aa_to_tok = get_aa_to_tok()

    logprobs_dict = {m: np.load(f"{CACHE_DIR}/{m}_logprobs.npy") for m in MODELS}
    esmif_lp = np.load(f"{CACHE_DIR}/esmif_logprobs.npy")
    entropy = np.load(f"{CACHE_DIR}/esm2_entropy.npy")
    msa_lp = np.load(f"{CACHE_DIR}/esm_msa1b_logprobs.npy")

    train_base = compute_base_preds(train_df['mutant'].tolist(), aa_to_tok, logprobs_dict, esmif_lp, entropy)
    test_base = compute_base_preds(test_df['mutant'].tolist(), aa_to_tok, logprobs_dict, esmif_lp, entropy)

    train_msa = compute_msa_llr(train_df['mutant'].tolist(), aa_to_tok, msa_lp)
    test_msa = compute_msa_llr(test_df['mutant'].tolist(), aa_to_tok, msa_lp)

    y = train_df['DMS_score'].values

    print(f"Train rho base: {spearmanr(y, train_base).statistic:.4f}")
    print(f"Train rho msa:  {spearmanr(y, train_msa).statistic:.4f}")

    weights = [0.02, 0.05, 0.10, 0.15]
    for w in weights:
        train_blend = (1 - w) * train_base + w * train_msa
        test_blend = (1 - w) * test_base + w * test_msa
        rho = spearmanr(y, train_blend).statistic
        print(f"Train rho blend w={w:.2f}: {rho:.4f}")

        out_kaggle = pd.DataFrame({"id": np.arange(len(test_df)), "DMS_score": test_blend})
        out_gs = pd.DataFrame({"mutant": test_df['mutant'], "DMS_score_predicted": test_blend})
        kaggle_path = f"results/kaggle/kaggle_msa_blend_w{int(w*100):02d}.csv"
        gs_path = f"predictions_msa_blend_w{int(w*100):02d}.csv"
        out_kaggle.to_csv(kaggle_path, index=False)
        out_gs.to_csv(gs_path, index=False)
        print(f"  wrote {kaggle_path} and {gs_path}")


if __name__ == "__main__":
    main()
