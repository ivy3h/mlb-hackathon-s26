import torch
import esm
import numpy as np
import time
import os

DATA_DIR = "Hackathon_data"
CACHE_DIR = "esm_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def load_wt_sequence(fasta_path):
    lines = []
    with open(fasta_path) as f:
        for line in f:
            if not line.startswith(">"):
                lines.append(line.strip())
    return "".join(lines)

def compute_masked_marginals(wt_seq, model_name):
    cache_file = os.path.join(CACHE_DIR, f"{model_name}_logprobs.npy")
    emb_cache = os.path.join(CACHE_DIR, f"{model_name}_embeddings.npy")

    if os.path.exists(cache_file) and os.path.exists(emb_cache):
        print(f"[{model_name}] Loading cached results")
        return np.load(cache_file), np.load(emb_cache)

    print(f"Loading {model_name}...")
    model, alphabet = getattr(esm.pretrained, model_name)()
    batch_converter = alphabet.get_batch_converter()
    model.eval().to(device)

    mask_idx = alphabet.mask_idx
    seq_len = len(wt_seq)
    repr_layer = int(model_name.split("_t")[1].split("_")[0])

    print(f"[{model_name}] Computing WT embeddings...")
    wt_data = [("wt", wt_seq)]
    _, _, wt_tokens = batch_converter(wt_data)
    with torch.no_grad():
        wt_result = model(wt_tokens.to(device), repr_layers=[repr_layer])
    wt_embeddings = wt_result["representations"][repr_layer][0, 1:-1, :].cpu().numpy()
    print(f"  WT embedding shape: {wt_embeddings.shape}")

    vocab_size = len(alphabet)
    logprobs_matrix = np.zeros((seq_len, vocab_size), dtype=np.float32)

    print(f"[{model_name}] Masked marginals for {seq_len} positions...")
    t0 = time.time()

    batch_size = 32 if torch.cuda.is_available() else 8
    for start in range(0, seq_len, batch_size):
        end = min(start + batch_size, seq_len)
        batch_data = [(f"p{pos}", wt_seq) for pos in range(start, end)]
        _, _, tokens = batch_converter(batch_data)

        for i, pos in enumerate(range(start, end)):
            tokens[i, pos + 1] = mask_idx

        with torch.no_grad():
            result = model(tokens.to(device))
        log_probs = torch.log_softmax(result["logits"], dim=-1).cpu()

        for i, pos in enumerate(range(start, end)):
            logprobs_matrix[pos] = log_probs[i, pos + 1].numpy()

        if (start // batch_size) % 10 == 0:
            elapsed = time.time() - t0
            eta = elapsed / max(end, 1) * (seq_len - end)
            print(f"  {end}/{seq_len}  ({elapsed:.0f}s elapsed, ~{eta:.0f}s left)")

    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s")

    np.save(cache_file, logprobs_matrix)
    np.save(emb_cache, wt_embeddings)

    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    return logprobs_matrix, wt_embeddings

if __name__ == "__main__":
    wt_seq = load_wt_sequence(f"{DATA_DIR}/sequence.fasta")
    print(f"Sequence length: {len(wt_seq)}")

    for name in ["esm2_t6_8M_UR50D", "esm2_t12_35M_UR50D", "esm2_t30_150M_UR50D", "esm2_t33_650M_UR50D"]:
        try:
            lp, emb = compute_masked_marginals(wt_seq, name)
            print(f"  {name}: logprobs {lp.shape}, embeddings {emb.shape}\n")
        except Exception as e:
            print(f"  {name} failed: {e}\n")

    print("All done.")
