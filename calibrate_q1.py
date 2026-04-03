import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import re
import esm
import sys, types
from sklearn.linear_model import Ridge

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
    return logprobs[pos, mut_tok] - logprobs[pos, wt_tok]


def compute_base_and_disp(mutants, aa_to_tok, logprobs_dict, esmif_lp, entropy):
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

    disp = llrs.std(axis=1)
    return base, disp, pos


def main():
    train_df = pd.read_csv(f"{DATA_DIR}/train.csv")
    test_df = pd.read_csv(f"{DATA_DIR}/test.csv")
    q1_df = pd.read_csv(f"{DATA_DIR}/queried_round1.csv")

    aa_to_tok = get_aa_to_tok()

    logprobs_dict = {m: np.load(f"{CACHE_DIR}/{m}_logprobs.npy") for m in MODELS}
    esmif_lp = np.load(f"{CACHE_DIR}/esmif_logprobs.npy")
    entropy = np.load(f"{CACHE_DIR}/esm2_entropy.npy")
    plddt = np.load(f"{CACHE_DIR}/esmfold_plddt.npy")

    train_base, train_disp, train_pos = compute_base_and_disp(train_df['mutant'].tolist(), aa_to_tok, logprobs_dict, esmif_lp, entropy)
    test_base, test_disp, test_pos = compute_base_and_disp(test_df['mutant'].tolist(), aa_to_tok, logprobs_dict, esmif_lp, entropy)
    q1_base, q1_disp, q1_pos = compute_base_and_disp(q1_df['mutant'].tolist(), aa_to_tok, logprobs_dict, esmif_lp, entropy)

    # Features for calibration: base, low_pLDDT, disagreement
    def build_X(base, disp, pos):
        low_plddt = 1.0 - plddt[pos] / 100.0
        return np.vstack([base, low_plddt, disp]).T

    X_q1 = build_X(q1_base, q1_disp, q1_pos)
    y_q1 = q1_df['DMS_score'].values

    # Regularized linear calibration on query1 labels
    mdl = Ridge(alpha=1.0)
    mdl.fit(X_q1, y_q1)

    X_train = build_X(train_base, train_disp, train_pos)
    X_test = build_X(test_base, test_disp, test_pos)

    train_cal = mdl.predict(X_train)
    test_cal = mdl.predict(X_test)

    print(f"Train rho base: {spearmanr(train_df['DMS_score'].values, train_base).statistic:.4f}")
    print(f"Train rho cal : {spearmanr(train_df['DMS_score'].values, train_cal).statistic:.4f}")

    # Blends with baseline to reduce overfit
    weights = [0.1, 0.2, 0.3]
    for w in weights:
        train_blend = (1 - w) * train_base + w * train_cal
        test_blend = (1 - w) * test_base + w * test_cal
        rho = spearmanr(train_df['DMS_score'].values, train_blend).statistic
        tag = f"q1cal_w{int(w*100):02d}"

        kaggle_path = f"results/kaggle/kaggle_{tag}.csv"
        gs_path = f"predictions_{tag}.csv"

        pd.DataFrame({"id": np.arange(len(test_df)), "DMS_score": test_blend}).to_csv(kaggle_path, index=False)
        pd.DataFrame({"mutant": test_df['mutant'], "DMS_score_predicted": test_blend}).to_csv(gs_path, index=False)

        print(f"w={w:.2f} -> train rho {rho:.4f} | {kaggle_path}")


if __name__ == "__main__":
    main()
