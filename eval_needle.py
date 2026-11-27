"""
Zero-Shot Distant Needle-in-a-Haystack Recall Benchmark.

Places a unique 16-token validation span ("needle") at position 0, inserts a
distractor haystack of length D in {64, 128, 256, 512, 1024, 1536}, then cues
the first 4 tokens of the needle and measures:
  - Exact Top-1 Recall Accuracy (%) on the remaining 12 tokens
  - Recall NLL Gain (Delta NLL = NLL_without_needle - NLL_with_needle)

Usage:
  python3 eval_needle.py out-6m-base out-6m-v6-wormhole --distances=64,128,256,512,1024,1536
"""

import os
import sys
import numpy as np
import torch
import torch.nn.functional as F

from model import ModelArgs, Transformer


@torch.no_grad()
def evaluate_needle_recall(model, data, distances, needle_len=16, prefix_cue=4, num_trials=200, device="cuda"):
    cols = []
    for D in distances:
        rng = np.random.RandomState(2026)
        nll_with, nll_without, acc_with = [], [], []
        max_start = len(data) - D - 200
        for _ in range(num_trials):
            i_needle = rng.randint(0, max_start)
            i_hay = rng.randint(0, max_start)
            while abs(i_hay - i_needle) < D + 64:
                i_hay = rng.randint(0, max_start)
            needle = data[i_needle : i_needle + needle_len].astype(np.int64)
            hay = data[i_hay : i_hay + D].astype(np.int64)
            i_dummy = (i_hay + D + 50) % max_start
            dummy = data[i_dummy : i_dummy + needle_len].astype(np.int64)

            for seq, nll_list, acc_list in [
                (np.concatenate([needle, hay, needle]), nll_with, acc_with),
                (np.concatenate([dummy, hay, needle]), nll_without, None),
            ]:
                x = torch.from_numpy(seq[:-1]).unsqueeze(0).to(device)
                y = torch.from_numpy(seq[1:]).unsqueeze(0).to(device)
                logits = model(x, targets=y)
                tail_start = needle_len + D + prefix_cue - 1
                tl, tt = logits[0, tail_start:, :], y[0, tail_start:]
                nll_list.append(F.cross_entropy(tl, tt).item())
                if acc_list is not None:
                    acc_list.append((tl.argmax(dim=-1) == tt).float().mean().item())

        acc_pct = float(np.mean(acc_with)) * 100.0
        dnll = float(np.mean(nll_without) - np.mean(nll_with))
        cols.append(f"{acc_pct:5.1f}% ({dnll:+.2f})")
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
    header = f"{'Checkpoint':26s} | " + " | ".join(f"{f'Dist={D}':>14s}" for D in distances)
    print("=== Distant Needle-in-a-Haystack Recall: Top-1 Acc (%) [Delta NLL] ===")
    print(header)
    print("-" * len(header))

    valid_fields = set(ModelArgs.__dataclass_fields__.keys())
    for path in ckpt_dirs:
        ckpt_path = os.path.join(path, "ckpt.pt") if os.path.isdir(path) else path
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        margs = ModelArgs(**{k: v for k, v in ckpt["model_args"].items() if k in valid_fields})
        model = Transformer(margs)
        model.load_state_dict({k.replace("_orig_mod.", ""): v for k, v in ckpt["model"].items()}, strict=True)
        model.to(device).eval()

        val_bin = os.path.join(base_dir, "data", ckpt.get("config", {}).get("dataset", "wikitext2"), "val.bin")
        data = np.memmap(val_bin, dtype=np.uint16, mode="r")
        cols = evaluate_needle_recall(model, data, distances, num_trials=num_trials, device=device)
        label = os.path.basename(path.rstrip("/"))
        print(f"{label:26s} | " + " | ".join(f"{c:>14s}" for c in cols))


if __name__ == "__main__":
    main()
