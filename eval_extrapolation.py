"""
Zero-Shot Long-Context Extrapolation Benchmark (supports Vanilla, Dynamic NTK, and YaRN RoPE).

Usage:
  python3 eval_extrapolation.py out-6m-base out-6m-v6-wormhole --lengths=256,512,1024,2048
  python3 eval_extrapolation.py out-6m-base --lengths=256,512,1024,2048 --rope_mode=ntk
"""

import math
import os
import sys
import numpy as np
import torch

from model import ModelArgs, Transformer, precompute_freqs_cis


def make_rope_table(head_dim: int, L: int, train_T: int, mode: str = "vanilla"):
    """Build RoPE cos/sin tables for Vanilla, Dynamic NTK-RoPE, or YaRN-RoPE."""
    if mode == "vanilla" or L <= train_T:
        return precompute_freqs_cis(head_dim, L, theta=10000.0)
    scale = L / float(train_T)
    if mode == "ntk":
        theta = 10000.0 * ((2.0 * scale - 1.0) ** (head_dim / (head_dim - 2.0)))
        return precompute_freqs_cis(head_dim, L, theta=theta)
    if mode == "yarn":
        base = 10000.0
        dim_idx = torch.arange(0, head_dim, 2).float()
        freqs_extrap = 1.0 / (base ** (dim_idx / head_dim))
        freqs_interp = freqs_extrap / scale
        low = max(0.0, (head_dim / (2 * math.pi)) * math.log(train_T / (2 * math.pi * 32)))
        high = min(head_dim / 2.0, (head_dim / (2 * math.pi)) * math.log(train_T / (2 * math.pi * 1)))
        ramp = ((dim_idx / 2.0 - low) / max(high - low, 1e-5)).clamp(0.0, 1.0)
        freqs = freqs_interp * ramp + freqs_extrap * (1.0 - ramp)
        angles = torch.outer(torch.arange(L).float(), freqs)
        mscale = 0.1 * math.log(scale) + 1.0
        return torch.cos(angles) * mscale, torch.sin(angles) * mscale
    raise ValueError(f"Unknown rope_mode: {mode}")


def main():
    ckpt_dirs = []
    lengths = [256, 512, 1024, 2048]
    eval_batches = 24
    batch_size = 16
    rope_mode = "vanilla"
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

    for a in sys.argv[1:]:
        if a.startswith("--lengths="):
            lengths = [int(x) for x in a.split("=", 1)[1].split(",")]
        elif a.startswith("--device="):
            device = a.split("=", 1)[1]
        elif a.startswith("--eval_batches="):
            eval_batches = int(a.split("=", 1)[1])
        elif a.startswith("--rope_mode="):
            rope_mode = a.split("=", 1)[1]
        else:
            ckpt_dirs.append(a)

    if not ckpt_dirs:
        print("Usage: python3 eval_extrapolation.py <ckpt_dir_1> [...] [--lengths=256,512,1024,2048] [--rope_mode=vanilla|ntk|yarn]")
        sys.exit(1)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    header = f"{'Checkpoint':32s} | {'Train T':>7s} | {'RoPE':>7s} | " + " | ".join(f"{f'PPL@{L}':>10s}" for L in lengths)
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
        head_dim = margs.dim // margs.n_heads

        ppls = []
        with torch.no_grad():
            for L in lengths:
                if rope_mode != "vanilla":
                    fc, fs = make_rope_table(head_dim, L, margs.max_seq_len, mode=rope_mode)
                    model.register_buffer("freqs_cos", fc.to(device), persistent=False)
                    model.register_buffer("freqs_sin", fs.to(device), persistent=False)
                rng = np.random.RandomState(1234)
                max_idx = len(data) - L - 1
                losses = []
                for _ in range(eval_batches):
                    ix = rng.randint(0, max_idx, size=(batch_size,))
                    x = torch.stack([torch.from_numpy(data[i : i + L].astype(np.int64)) for i in ix]).to(device)
                    y = torch.stack([torch.from_numpy(data[i + 1 : i + 1 + L].astype(np.int64)) for i in ix]).to(device)
                    model(x, y)
                    losses.append(model.last_loss.item())
                ppls.append(math.exp(min(20.0, float(np.mean(losses)))))

        label = os.path.basename(path.rstrip("/"))
        print(f"{label:32s} | {margs.max_seq_len:7d} | {rope_mode:>7s} | " + " | ".join(f"{p:10.2f}" for p in ppls))

        # B1 Fixed-Position Bucketed PPL on max(lengths) to eliminate window-dilution bias
        L_max = max(lengths)
        if L_max >= 512:
            buckets = [(0, 256), (256, 512)]
            if L_max >= 1024:
                buckets.append((512, 1024))
            if L_max >= 2048:
                buckets.append((1024, 2048))
            bucket_losses = {b: [] for b in buckets}
            with torch.no_grad():
                if rope_mode != "vanilla":
                    fc, fs = make_rope_table(head_dim, L_max, margs.max_seq_len, mode=rope_mode)
                    model.register_buffer("freqs_cos", fc.to(device), persistent=False)
                    model.register_buffer("freqs_sin", fs.to(device), persistent=False)
                rng = np.random.RandomState(1234)
                max_idx = len(data) - L_max - 1
                for _ in range(eval_batches):
                    ix = rng.randint(0, max_idx, size=(batch_size,))
                    x = torch.stack([torch.from_numpy(data[i : i + L_max].astype(np.int64)) for i in ix]).to(device)
                    y = torch.stack([torch.from_numpy(data[i + 1 : i + 1 + L_max].astype(np.int64)) for i in ix]).to(device)
                    logits = model(x, y)
                    tok_loss = torch.nn.functional.cross_entropy(
                        logits.view(-1, logits.size(-1)), y.view(-1), reduction="none"
                    ).view(batch_size, L_max)
                    for b0, b1 in buckets:
                        bucket_losses[(b0, b1)].append(tok_loss[:, b0:b1].mean().item())
            b_str = " | ".join(f"[{b0}:{b1}]: {math.exp(min(20.0, float(np.mean(bucket_losses[(b0, b1)])))):6.2f}" for b0, b1 in buckets)
            print(f"  └─ Bucketed PPL (no dilution)  | {b_str}")


if __name__ == "__main__":
    main()
