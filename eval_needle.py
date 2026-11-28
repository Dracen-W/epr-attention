"""
Zero-Shot Distant Needle-in-a-Haystack Recall Benchmark (200 Trials + Trial-Level 95% CI).

Places a unique 16-token validation span ("needle") at position 0 (vs length-matched
dummy distractor at position 0 so target positions stay strictly identical), inserts
a distractor haystack of length D, cues the first 4 tokens of the needle, and measures:
  - Top-1 Recall Accuracy (%) ± Trial-Level 95% Confidence Interval on the remaining 12 tokens
  - Recall NLL Gain (Delta NLL = NLL_without_needle - NLL_with_needle)

Usage:
  python3 eval_needle.py out-6m-base out-6m-fox out-6m-epr --distances=64,128,256,512,1024,1536
"""

import math
import os
import sys
import numpy as np
import torch
import torch.nn.functional as F

from run_sota_baselines import load_any_checkpoint


@torch.no_grad()
def evaluate_needle_single_dist(
    model, data, dist: int, needle_len: int = 16, prefix_cue: int = 4,
    num_trials: int = 200, seed: int = 2026, device: str = "cuda",
):
    """Canonical single-distance needle evaluator (shared identically with `eval_kv_eviction.py`)."""
    rng = np.random.RandomState(seed)
    nll_with, nll_without, acc_with = [], [], []
    max_start = len(data) - dist - 200
    for _ in range(num_trials):
        i_needle = rng.randint(0, max_start)
        i_hay = rng.randint(0, max_start)
        while abs(i_hay - i_needle) < dist + 64:
            i_hay = rng.randint(0, max_start)
        needle = data[i_needle : i_needle + needle_len].astype(np.int64)
        hay = data[i_hay : i_hay + dist].astype(np.int64)
        i_dummy = (i_hay + dist + 50) % max_start
        dummy = data[i_dummy : i_dummy + needle_len].astype(np.int64)

        seq_w = np.concatenate([needle, hay, needle])
        seq_wo = np.concatenate([dummy, hay, needle])
        x_batch = torch.stack([torch.from_numpy(seq_w[:-1]), torch.from_numpy(seq_wo[:-1])]).to(device)
        y_batch = torch.stack([torch.from_numpy(seq_w[1:]), torch.from_numpy(seq_wo[1:])]).to(device)

        logits = model(x_batch, targets=y_batch)
        tail_start = needle_len + dist + prefix_cue - 1
        tl_w, tt_w = logits[0, tail_start:, :], y_batch[0, tail_start:]
        tl_wo, tt_wo = logits[1, tail_start:, :], y_batch[1, tail_start:]

        nll_with.append(F.cross_entropy(tl_w, tt_w).item())
        nll_without.append(F.cross_entropy(tl_wo, tt_wo).item())
        acc_with.append((tl_w.argmax(dim=-1) == tt_w).float().mean().item() * 100.0)

    acc_pct = float(np.mean(acc_with))
    # Unbiased trial-level 95% CI (accounts for within-span token correlation!)
    se = float(np.std(acc_with, ddof=1)) / math.sqrt(max(1, num_trials))
    ci_95 = 1.96 * se
    dnll = float(np.mean(nll_without) - np.mean(nll_with))
    return acc_pct, ci_95, dnll


@torch.no_grad()
def evaluate_needle_recall(model, data, distances, needle_len=16, prefix_cue=4, num_trials=200, device="cuda"):
    cols = []
    for D in distances:
        acc_pct, ci_95, dnll = evaluate_needle_single_dist(
            model, data, D, needle_len=needle_len, prefix_cue=prefix_cue, num_trials=num_trials, seed=2026, device=device
        )
        cols.append(f"{acc_pct:4.1f}±{ci_95:3.1f}% ({dnll:+.2f})")
    return cols


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
        print("Usage: python3 eval_needle.py <ckpt_dir_1> [...] [--distances=64,128,256,512,1024,1536]")
        sys.exit(1)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    header = f"{'Checkpoint':26s} | " + " | ".join(f"{f'Dist={D}':>18s}" for D in distances)
    print(f"=== Distant Needle-in-a-Haystack Recall ({num_trials} trials): Acc±95%CI [Delta NLL] ===")
    print(header)
    print("-" * len(header))

    for path in ckpt_dirs:
        model, _, ckpt = load_any_checkpoint(path, device=device)
        val_bin = os.path.join(base_dir, "data", ckpt.get("config", {}).get("dataset", "wikitext2"), "val.bin")
        data = np.memmap(val_bin, dtype=np.uint16, mode="r")
        cols = evaluate_needle_recall(model, data, distances, num_trials=num_trials, device=device)
        label = os.path.basename(path.rstrip("/"))
        print(f"{label:26s} | " + " | ".join(f"{c:>18s}" for c in cols))


if __name__ == "__main__":
    main()
