"""
Non-Verbatim Long-Range Task: Distant 2-Key Associative Disambiguation.
Plants TWO distinct pairs [K1 -> V1] and [K2 -> V2] before `dist` tokens of haystack,
then queries either K1 or K2 (50/50) at the end and checks whether NLL(V_correct) < NLL(V_wrong).
  - Random chance / blind position-0 copying = 50.0%
  - Requires content-addressable Key->Value binding across `dist` tokens.
"""
import math, os, sys
import numpy as np
import torch
import torch.nn.functional as F
from model import ModelArgs, Transformer


@torch.no_grad()
def eval_assoc_recall(model, data, distances=(64, 128, 256, 512, 1024, 1536), num_trials=200, k_len=3, v_len=4, device="cuda"):
    results = []
    pair_len = k_len + v_len
    for dist in distances:
        rng = np.random.RandomState(2026 + dist)
        max_start = len(data) - dist - pair_len * 4 - 20
        wins = 0
        margins = []

        for t in range(num_trials):
            s1 = rng.randint(0, max_start)
            s2 = rng.randint(0, max_start)
            s_hay = rng.randint(0, max_start)

            p1 = data[s1 : s1 + pair_len].astype(np.int64)
            p2 = data[s2 : s2 + pair_len].astype(np.int64)
            k1, v1 = p1[:k_len], p1[k_len:]
            k2, v2 = p2[:k_len], p2[k_len:]
            hay = data[s_hay : s_hay + dist].astype(np.int64)

            if t % 2 == 0:
                q_key, v_true, v_false = k1, v1, v2
            else:
                q_key, v_true, v_false = k2, v2, v1

            prefix = np.concatenate([k1, v1, k2, v2, hay, q_key])
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
    ckpt_dirs = sys.argv[1:] if len(sys.argv) > 1 else ["out-6m-base", "out-6m-v6-wormhole"]
    distances = [64, 128, 256, 512, 1024, 1536]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=== Indirect 2-Key Associative Recall (200 trials, Random=50.0%): Acc±95%CI [NLL Margin] ===")
    header = f"{'Checkpoint':22s} | " + " | ".join(f"{f'Dist={D}':>18s}" for D in distances)
    print(header)
    print("-" * len(header))

    valid_fields = set(ModelArgs.__dataclass_fields__.keys())
    for path in ckpt_dirs:
        ckpt_path = os.path.join(path, "ckpt.pt") if os.path.isdir(path) else path
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        margs = ModelArgs(**{k: v for k, v in ckpt["model_args"].items() if k in valid_fields})
        margs.dropout = 0.0
        model = Transformer(margs)
        model.load_state_dict({k.replace("_orig_mod.", ""): v for k, v in ckpt["model"].items()}, strict=True)
        model.to(device).eval()

        val_bin = os.path.join("data", ckpt.get("config", {}).get("dataset", "wikitext2"), "val.bin")
        data = np.memmap(val_bin, dtype=np.uint16, mode="r")
        cols = eval_assoc_recall(model, data, distances=distances, num_trials=200, device=device)
        label = os.path.basename(path.rstrip("/"))
        print(f"{label:22s} | " + " | ".join(f"{c:>18s}" for c in cols))


if __name__ == "__main__":
    main()
