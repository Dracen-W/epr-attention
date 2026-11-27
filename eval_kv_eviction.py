"""
Evaluation: Quality Cost of Asymmetric EPRKVCache Sliding-Window Eviction.
Measures exact tradeoff between KV-Cache memory savings and:
  1. Un-diluted Tail Extrapolation PPL ([1024:2048])
  2. 200-trial Distant Needle Recall (Dist=1536) with 95% CI
"""
import math, os, sys
import numpy as np
import torch
import torch.nn.functional as F
from model import ModelArgs, Transformer


def set_damped_window(model, window_size):
    """Apply exact EPRKVCache sliding-window mask strictly to damped heads (0..H-2), keeping Wormhole head (H-1) full-context."""
    for layer in model.layers:
        attn = layer.attention
        if not hasattr(attn, "_orig_compute"):
            attn._orig_compute = attn._compute_fields_and_qkv
        attn.use_flex_attention = False

        def make_wrapped(orig_fn, W, n_heads, use_wh, use_vac):
            def wrapped(x, fc, fs):
                q_aug, k_aug, v_aug, attn_mask, dtau, rho = orig_fn(x, fc, fs)
                if W is not None:
                    T = x.size(1)
                    n_damped = (n_heads - 1) if use_wh else n_heads
                    q_pos = torch.arange(T, device=x.device).unsqueeze(1)
                    k_pos = torch.arange(T, device=x.device).unsqueeze(0)
                    evict = (q_pos - k_pos) >= W
                    if use_vac:
                        evict_pad = F.pad(evict, (1, 0), value=False).view(1, 1, T, T + 1)
                    else:
                        evict_pad = evict.view(1, 1, T, T)
                    attn_mask = attn_mask.clone()
                    attn_mask[:, :n_damped] = attn_mask[:, :n_damped].masked_fill(evict_pad, float("-inf"))
                return q_aug, k_aug, v_aug, attn_mask, dtau, rho
            return wrapped

        attn._compute_fields_and_qkv = make_wrapped(
            attn._orig_compute, window_size, attn.n_heads, attn.use_wormhole, attn.use_attn_vacuum
        )


@torch.no_grad()
def eval_tail_ppl(model, data, L=2048, b0=1024, b1=2048, eval_batches=16, batch_size=2, device="cuda"):
    rng = np.random.RandomState(1234)
    max_idx = len(data) - L - 1
    losses = []
    for _ in range(eval_batches):
        ix = rng.randint(0, max_idx, size=(batch_size,))
        x = torch.stack([torch.from_numpy(data[i : i + L].astype(np.int64)) for i in ix]).to(device)
        y = torch.stack([torch.from_numpy(data[i + 1 : i + 1 + L].astype(np.int64)) for i in ix]).to(device)
        logits = model(x, y)
        tok_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), reduction="none").view(batch_size, L)
        losses.append(tok_loss[:, b0:b1].mean().item())
    return math.exp(min(20.0, float(np.mean(losses))))


@torch.no_grad()
def eval_needle_single(model, data, dist=1536, num_trials=200, cue_len=4, val_len=12, device="cuda"):
    rng = np.random.RandomState(42)
    span = cue_len + val_len
    correct, total = 0, 0
    nll_with, nll_without = [], []
    max_start = len(data) - dist - span * 3 - 10

    for _ in range(num_trials):
        s_needle = rng.randint(0, max_start)
        s_hay = rng.randint(0, max_start)
        needle = data[s_needle : s_needle + span].astype(np.int64)
        hay = data[s_hay : s_hay + dist].astype(np.int64)

        seq_with = np.concatenate([needle, hay, needle])
        seq_no = np.concatenate([hay, needle])

        for seq, is_with in [(seq_with, True), (seq_no, False)]:
            x = torch.from_numpy(seq[:-1]).unsqueeze(0).to(device)
            y = torch.from_numpy(seq[1:]).unsqueeze(0).to(device)
            logits = model(x, y)
            pred_logits = logits[0, -val_len:, :]
            targets = y[0, -val_len:]
            loss = F.cross_entropy(pred_logits, targets).item()
            if is_with:
                nll_with.append(loss)
                correct += (pred_logits.argmax(dim=-1) == targets).sum().item()
                total += val_len
            else:
                nll_without.append(loss)

    acc = 100.0 * correct / max(1, total)
    ci = 196.0 * math.sqrt((acc / 100.0) * (1.0 - acc / 100.0) / max(1, total))
    dnll = float(np.mean(nll_without) - np.mean(nll_with))
    return acc, ci, dnll


def main():
    ckpt_dir = sys.argv[1] if len(sys.argv) > 1 else "out-6m-v6-wormhole"
    ckpt_path = os.path.join(ckpt_dir, "ckpt.pt") if os.path.isdir(ckpt_dir) else ckpt_dir
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    valid_fields = set(ModelArgs.__dataclass_fields__.keys())
    margs = ModelArgs(**{k: v for k, v in ckpt["model_args"].items() if k in valid_fields})
    margs.dropout = 0.0
    model = Transformer(margs)
    model.load_state_dict({k.replace("_orig_mod.", ""): v for k, v in ckpt["model"].items()}, strict=True)
    model.to(device).eval()

    val_bin = os.path.join("data", ckpt.get("config", {}).get("dataset", "wikitext2"), "val.bin")
    data = np.memmap(val_bin, dtype=np.uint16, mode="r")

    H = margs.n_heads
    n_damped = H - 1 if margs.use_wormhole else H
    print(f"=== Asymmetric EPRKVCache Eviction Cost ({ckpt_dir}, H={H}, Wormhole={margs.use_wormhole}) ===")
    print(f"{'Damped Window W':>16s} | {'KV Saved @2K':>12s} | {'KV Saved @8K':>12s} | {'Tail PPL [1K:2K]':>16s} | {'Needle@1536 (200 trials)':>26s}")
    print("-" * 96)

    for W in [None, 512, 256, 128, 64]:
        set_damped_window(model, W)
        ppl_tail = eval_tail_ppl(model, data, device=device)
        acc, ci, dnll = eval_needle_single(model, data, dist=1536, num_trials=200, device=device)
        if W is None:
            s2k, s8k, w_lbl = 0.0, 0.0, "Full (∞)"
        else:
            s2k = (1.0 - (n_damped * min(W, 2048) + (H - n_damped) * 2048) / (H * 2048)) * 100.0
            s8k = (1.0 - (n_damped * min(W, 8192) + (H - n_damped) * 8192) / (H * 8192)) * 100.0
            w_lbl = f"W = {W}"
        print(f"{w_lbl:>16s} | {s2k:11.1f}% | {s8k:11.1f}% | {ppl_tail:16.2f} | {acc:5.1f}±{ci:3.1f}% (ΔNLL {dnll:+.2f})")


if __name__ == "__main__":
    main()
