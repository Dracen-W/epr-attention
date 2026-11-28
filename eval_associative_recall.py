"""
Non-Verbatim Long-Range Task: Distant 2-Key Associative Disambiguation.

Evaluates TWO regimes across `dist` tokens of distractor haystack (200 trials each):
  1. Standard 2-Key Binding: `[K1 -> V1, K2 -> V2, hay, Q_key] -> V_correct vs V_wrong`
  2. Counterfactual Cross-Paired Binding (Prior-Free Strict Test!):
     Takes two natural phrases `(K1, V1_nat)` and `(K2, V2_nat)` and SWAPS their values
     in context: `[K1 -> V2_nat, K2 -> V1_nat, hay, Q_key]`.
     Here local 7-gram language priors actively pull the model toward the WRONG answer
     (`< 50%` if blind to distant context!), isolating pure content-addressable binding!

Usage:
  python3 eval_associative_recall.py out-6m-base out-6m-fox out-6m-epr
"""

import math
import os
import sys
import numpy as np
import torch
import torch.nn.functional as F

from run_sota_baselines import load_any_checkpoint


@torch.no_grad()
def eval_assoc_recall(
    model, data, distances=(64, 128, 256, 512, 1024, 1536),
    num_trials=200, k_len=3, v_len=4, cross_paired=False, device="cuda",
):
    results = []
    pair_len = k_len + v_len
    for dist in distances:
        rng = np.random.RandomState(2026 + dist)
        max_start = len(data) - dist - pair_len * 4 - 64
        wins = 0
        margins = []

        for t in range(num_trials):
            s1 = rng.randint(0, max_start)
            s2 = rng.randint(0, max_start)
            while abs(s2 - s1) < pair_len * 2 + 32:
                s2 = rng.randint(0, max_start)
            s_hay = rng.randint(0, max_start)
            while abs(s_hay - s1) < dist + 32 or abs(s_hay - s2) < dist + 32:
                s_hay = rng.randint(0, max_start)

            p1 = data[s1 : s1 + pair_len].astype(np.int64)
            p2 = data[s2 : s2 + pair_len].astype(np.int64)
            k1, v1_nat = p1[:k_len], p1[k_len:]
            k2, v2_nat = p2[:k_len], p2[k_len:]
            hay = data[s_hay : s_hay + dist].astype(np.int64)

            if cross_paired:
                # Counterfactual swap: context binds K1 -> V2_nat and K2 -> V1_nat!
                v1_bound, v2_bound = v2_nat, v1_nat
            else:
                v1_bound, v2_bound = v1_nat, v2_nat

            if t % 2 == 0:
                q_key, v_true, v_false = k1, v1_bound, v2_bound
            else:
                q_key, v_true, v_false = k2, v2_bound, v1_bound

            prefix = np.concatenate([k1, v1_bound, k2, v2_bound, hay, q_key])
            seq_true = np.concatenate([prefix, v_true])
            seq_false = np.concatenate([prefix, v_false])

            x_batch = torch.stack([
                torch.from_numpy(seq_true[:-1]),
                torch.from_numpy(seq_false[:-1]),
            ]).to(device)
            y_batch = torch.stack([
                torch.from_numpy(seq_true[1:]),
                torch.from_numpy(seq_false[1:]),
            ]).to(device)

            logits = model(x_batch, y_batch)
            nll_true = F.cross_entropy(logits[0, -v_len:], y_batch[0, -v_len:]).item()
            nll_false = F.cross_entropy(logits[1, -v_len:], y_batch[1, -v_len:]).item()

            if nll_true < nll_false:
                wins += 1
            margins.append(nll_false - nll_true)

        acc = 100.0 * wins / num_trials
        ci = 196.0 * math.sqrt((acc / 100.0) * (1.0 - acc / 100.0) / num_trials)
        margin = float(np.mean(margins))
        results.append(f"{acc:4.1f}±{ci:3.1f}% ({margin:+.2f})")
    return results


def main():
    ckpt_dirs = []
    distances = [64, 128, 256, 512, 1024, 1536]
    num_trials = 200
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

    for a in sys.argv[1:]:
        if a.startswith("--distances="):
            distances = [int(x) for x in a.split("=", 1)[1].split(",")]
        elif a.startswith("--trials="):
            num_trials = int(a.split("=", 1)[1])
        elif a.startswith("--device="):
            device = a.split("=", 1)[1]
        else:
            ckpt_dirs.append(a)

    if not ckpt_dirs:
        ckpt_dirs = ["out-6m-base", "out-6m-epr"]

    base_dir = os.path.dirname(os.path.abspath(__file__))
    loaded = []
    for path in ckpt_dirs:
        m, _, ckpt = load_any_checkpoint(path, device=device)
        val_bin = os.path.join(base_dir, "data", ckpt.get("config", {}).get("dataset", "wikitext2"), "val.bin")
        data = np.memmap(val_bin, dtype=np.uint16, mode="r")
        loaded.append((os.path.basename(path.rstrip("/")), m, data))

    for cross, title in [
        (False, f"=== [1] Standard 2-Key Associative Recall ({num_trials} trials, Random=50.0%) ==="),
        (True,  f"=== [2] Counterfactual Cross-Paired 2-Key Recall ({num_trials} trials, Anti-Prior Strict Test!) ==="),
    ]:
        print("\n" + title)
        header = f"{'Checkpoint':24s} | " + " | ".join(f"{f'Dist={D}':>18s}" for D in distances)
        print(header)
        print("-" * len(header))
        for label, m, data in loaded:
            cols = eval_assoc_recall(m, data, distances=distances, num_trials=num_trials, cross_paired=cross, device=device)
            print(f"{label:24s} | " + " | ".join(f"{c:>18s}" for c in cols))


if __name__ == "__main__":
    main()
