
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import KFold
from scipy.stats import spearmanr
from scipy.optimize import minimize
import esm
import os
import re

DATA_DIR = "Hackathon_data"
CACHE_DIR = "esm_cache"

ALL_MODELS = [
    "esm2_t6_8M_UR50D",
    "esm2_t12_35M_UR50D",
    "esm2_t30_150M_UR50D",
    "esm2_t33_650M_UR50D",
    "esm2_t36_3B_UR50D",
    "esm2_t48_15B_UR50D",
]

def load_wt_sequence(fasta_path):
    lines = []
    with open(fasta_path) as f:
        for line in f:
            if not line.startswith(">"):
                lines.append(line.strip())
    return "".join(lines)

def parse_mutant(code):
    m = re.match(r'^([A-Z])(\d+)([A-Z])$', code)
    return m.group(1), int(m.group(2)), m.group(3)

def get_esm_alphabet_mapping():
    _, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
    aa_list = "ACDEFGHIKLMNPQRSTVWY"
    aa_to_tok = {}
    for aa in aa_list:
        aa_to_tok[aa] = alphabet.get_idx(aa)
    return aa_to_tok

def compute_llr_matrix(mutant_codes, wt_seq, aa_to_tok):
    available = []
    logprobs_dict = {}
    for mname in ALL_MODELS:
        lp_file = os.path.join(CACHE_DIR, f"{mname}_logprobs.npy")
        if os.path.exists(lp_file):
            logprobs_dict[mname] = np.load(lp_file)
            available.append(mname)
            print(f"  Loaded {mname}: {logprobs_dict[mname].shape}")

    if not available:
        raise RuntimeError("No ESM features found. Run compute_esm_parallel.py first.")

    n = len(mutant_codes)
    n_models = len(available)
    llr_matrix = np.zeros((n, n_models), dtype=np.float32)

    for i, code in enumerate(mutant_codes):
        wt_aa, pos, mut_aa = parse_mutant(code)
        wt_tok = aa_to_tok[wt_aa]
        mut_tok = aa_to_tok[mut_aa]
        for j, mname in enumerate(available):
            lp = logprobs_dict[mname]
            llr_matrix[i, j] = lp[pos, mut_tok] - lp[pos, wt_tok]

    return llr_matrix, available

def main():
    wt_seq = load_wt_sequence(f"{DATA_DIR}/sequence.fasta")
    print(f"WT sequence length: {len(wt_seq)}")

    train_df = pd.read_csv(f"{DATA_DIR}/train.csv")
    test_df = pd.read_csv(f"{DATA_DIR}/test.csv")
    print(f"Train: {len(train_df)}, Test: {len(test_df)}")

    aa_to_tok = get_esm_alphabet_mapping()
    train_mutants = train_df['mutant'].tolist()
    test_mutants = test_df['mutant'].tolist()

    print("\nLoading ESM logprobs...")
    train_llr, available = compute_llr_matrix(train_mutants, wt_seq, aa_to_tok)
    test_llr, _ = compute_llr_matrix(test_mutants, wt_seq, aa_to_tok)
    y_train = train_df['DMS_score'].values

    n_models = len(available)
    print(f"\n{n_models} models available: {available}")

    train_avg = train_llr.mean(axis=1)
    test_avg = test_llr.mean(axis=1)
    rho_avg = spearmanr(y_train, train_avg).statistic
    print(f"\n[1] Equal avg LLR (train rho): {rho_avg:.4f}")

    print("\nPer-model train Spearman:")
    for j, mname in enumerate(available):
        rho = spearmanr(y_train, train_llr[:, j]).statistic
        print(f"  {mname}: {rho:.4f}")

    def neg_spearman(w):
        w = np.abs(w)
        w = w / w.sum()
        pred = train_llr @ w
        return -spearmanr(y_train, pred).statistic

    best_result = None
    best_score = -1
    for seed in range(50):
        rng = np.random.RandomState(seed)
        w0 = rng.dirichlet(np.ones(n_models))
        result = minimize(neg_spearman, w0, method='Nelder-Mead',
                          options={'maxiter': 5000, 'xatol': 1e-6})
        if -result.fun > best_score:
            best_score = -result.fun
            best_result = result

    w_opt = np.abs(best_result.x)
    w_opt = w_opt / w_opt.sum()
    train_weighted = train_llr @ w_opt
    test_weighted = test_llr @ w_opt
    rho_weighted = spearmanr(y_train, train_weighted).statistic
    print(f"\n[3] Optimized weights (train rho): {rho_weighted:.4f}")
    print(f"  Weights: {dict(zip(available, [f'{w:.3f}' for w in w_opt]))}")

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    rhos_ridge = []
    for tr, va in kf.split(train_llr):
        mdl = Ridge(alpha=1.0)
        mdl.fit(train_llr[tr], y_train[tr])
        rhos_ridge.append(spearmanr(y_train[va], mdl.predict(train_llr[va])).statistic)
    print(f"\n[4] Ridge on LLR (5-fold CV): {np.mean(rhos_ridge):.4f} +/- {np.std(rhos_ridge):.4f}")

    ridge_final = Ridge(alpha=1.0)
    ridge_final.fit(train_llr, y_train)
    test_ridge = ridge_final.predict(test_llr)
    print(f"  Ridge coeffs: {dict(zip(available, [f'{c:.4f}' for c in ridge_final.coef_]))}")

    iso = IsotonicRegression(out_of_bounds='clip')
    iso.fit(train_weighted, y_train)
    train_iso = iso.predict(train_weighted)
    test_iso = iso.predict(test_weighted)
    rho_iso = spearmanr(y_train, train_iso).statistic
    print(f"\n[5] Isotonic calibration of weighted LLR (train rho): {rho_iso:.4f}")

    for alpha in [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]:
        blend = alpha * test_ridge + (1 - alpha) * test_weighted
        blend_train = alpha * ridge_final.predict(train_llr) + (1 - alpha) * train_weighted
        rho_blend = spearmanr(y_train, blend_train).statistic
        print(f"  [6] Blend alpha={alpha:.1f} (train rho): {rho_blend:.4f}")

    print("\n" + "=" * 60)
    print("Generating submission files...")

    submissions = {
        "avg_llr_v3": test_avg,
        "weighted_llr_v3": test_weighted,
        "ridge_llr_v3": test_ridge,
        "isotonic_v3": test_iso,
    }

    for alpha in [0.2, 0.3]:
        blend = alpha * test_ridge + (1 - alpha) * test_weighted
        submissions[f"blend_{int(alpha*100)}ridge_v3"] = blend

    for name, preds in submissions.items():
        kaggle = pd.DataFrame({'id': range(len(test_df)), 'DMS_score': preds})
        kaggle.to_csv(f"kaggle_{name}.csv", index=False)

        gs = pd.DataFrame({'mutant': test_df['mutant'], 'DMS_score_predicted': preds})
        gs.to_csv(f"predictions_{name}.csv", index=False)

        print(f"  {name}: range=[{preds.min():.4f}, {preds.max():.4f}], "
              f"mean={preds.mean():.4f}, std={preds.std():.4f}")

    print("\nDone! Submit kaggle_weighted_llr_v3.csv as primary submission.")
    print("If that doesn't improve, try kaggle_ridge_llr_v3.csv or kaggle_blend_20ridge_v3.csv")

if __name__ == "__main__":
    main()
