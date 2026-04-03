import argparse
import os
import numpy as np
import torch
import esm

from esm.inverse_folding.util import load_coords, CoordBatchConverter

CACHE_DIR = "esm_cache"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdb", default=os.path.join(CACHE_DIR, "alphafold_wt.pdb"))
    parser.add_argument("--chain", default=None, help="Chain ID (e.g., A). Default: all chains in PDB")
    parser.add_argument("--out", default=os.path.join(CACHE_DIR, "esmif_logprobs.npy"))
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model, alphabet = esm.pretrained.esm_if1_gvp4_t16_142M_UR50()
    model = model.eval().to(device)

    coords, seq = load_coords(args.pdb, args.chain)
    batch_converter = CoordBatchConverter(alphabet)
    batch = [(coords, None, seq)]
    coords, confidence, strs, tokens, padding_mask = batch_converter(batch)

    coords = coords.to(device)
    confidence = confidence.to(device)
    padding_mask = padding_mask.to(device)

    prev_output_tokens = tokens[:, :-1].to(device)
    logits, _ = model.forward(coords, padding_mask, confidence, prev_output_tokens)
    log_probs = torch.log_softmax(logits, dim=-1)

    # remove batch dimension
    log_probs = log_probs[0].detach().cpu().numpy()
    np.save(args.out, log_probs.astype(np.float32))
    print(f"Saved {args.out} with shape {log_probs.shape}")


if __name__ == "__main__":
    main()
