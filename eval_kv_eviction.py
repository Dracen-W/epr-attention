"""
Evaluation: Quality Cost & Decode Speed of Asymmetric `EPRKVCache` Eviction.

Strictly shares the EXACT 384-sequence Tail PPL setup (`eval_extrapolation.py`) and
EXACT 200-trial dummy-aligned Needle setup (`eval_needle.py`) so `Full (∞)` matches
the main benchmark tables to the second decimal place! Also verifies stateful
`EPRKVCache` ring-buffer parity and autoregressive decoding throughput (`tok/s`).

Usage:
  python3 eval_kv_eviction.py out-6m-epr
"""

import os
import sys
import time
import numpy as np
import torch
import torch.nn.functional as F

from eval_extrapolation import evaluate_bucketed_ppl
from eval_needle import evaluate_needle_single_dist
from run_sota_baselines import load_any_checkpoint


def apply_damped_sliding_window_hook(model, window_size):
    """
    Apply exact `EPRKVCache` ring-buffer eviction mask to damped heads (`0..n_damped-1`)
    during parallel evaluation, keeping Wormhole head(s) full-context.
    """
    for layer in model.layers:
        attn = layer.attention
        if not hasattr(attn, "_orig_forward"):
            attn._orig_forward = attn.forward

        if window_size is None:
            attn.forward = attn._orig_forward
            continue

        def make_evicted_forward(a, W):
            def evicted_forward(x, freqs_cos, freqs_sin, kv_cache=None, layer_idx=0):
                if kv_cache is not None:
                    return a._orig_forward(x, freqs_cos, freqs_sin, kv_cache=kv_cache, layer_idx=layer_idx)
                B, T, _ = x.shape
                q_aug, k_vac, k_real, xv, _, tau = a._compute_step_tensors(x, freqs_cos, freqs_sin, tau_offset=None)
                if a.use_attn_vacuum:
                    k_aug = torch.cat([k_vac, k_real], dim=-2)
                    v_aug = F.pad(xv, (0, 0, 1, 0), value=0.0)
                else:
                    k_aug, v_aug = k_real, xv
                if a.n_rep > 1:
                    k_aug = k_aug.repeat_interleave(a.n_rep, dim=1)
                    v_aug = v_aug.repeat_interleave(a.n_rep, dim=1)

                dt = (tau.unsqueeze(-1) - tau.unsqueeze(-2)).clamp(min=0.0)
                q_pos = torch.arange(T, device=x.device).unsqueeze(1)
                k_pos = torch.arange(T, device=x.device).unsqueeze(0)
                cm = torch.where(k_pos <= q_pos, 0.0, float("-inf")).view(1, 1, T, T)
                evict = ((q_pos - k_pos) >= W).view(1, 1, T, T)

                if a.use_resonant_tunneling:
                    cm_all = cm.expand(1, a.n_heads, T, T).clone()
                    if a.n_damped > 0:
                        cm_all[:, : a.n_damped] = cm_all[:, : a.n_damped].masked_fill(evict, float("-inf"))
                    out = a._attend_exact_tunneling(
                        q_aug, k_aug, v_aug, dt=dt, causal_mask=cm_all,
                        has_vacuum=a.use_attn_vacuum, head_slice=slice(0, a.n_heads),
                    )
                else:
                    if a.use_attn_vacuum:
                        mask = F.pad(-dt, (1, 0), value=0.0) + F.pad(cm, (1, 0), value=0.0)
                        evict_pad = F.pad(evict, (1, 0), value=False)
                        if a.n_damped > 0:
                            mask[:, : a.n_damped] = mask[:, : a.n_damped].masked_fill(evict_pad, float("-inf"))
                    else:
                        mask = -dt + cm
                        if a.n_damped > 0:
                            mask[:, : a.n_damped] = mask[:, : a.n_damped].masked_fill(evict, float("-inf"))
                    out = F.scaled_dot_product_attention(q_aug, k_aug, v_aug, attn_mask=mask.to(q_aug.dtype), is_causal=False, scale=1.0)
                return a.resid_dropout(a.wo(out.transpose(1, 2).contiguous().view(B, T, -1)))
            return evicted_forward

        attn.forward = make_evicted_forward(attn, window_size)


@torch.no_grad()
def bench_stateful_decode(model, window_size, prompt_len=512, gen_tokens=128, device="cuda"):
    """Measure real stateful `EPRKVCache` autoregressive decode speed (tok/s)."""
    x = torch.randint(0, model.vocab_size, (1, prompt_len), device=device)
    _ = model.generate(x[:, :32], max_new_tokens=4, temperature=0.0, use_kv_cache=True, damped_window=window_size)
    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    _ = model.generate(x, max_new_tokens=gen_tokens, temperature=0.0, use_kv_cache=True, damped_window=window_size)
    if device == "cuda":
        torch.cuda.synchronize()
    return gen_tokens / max(1e-6, (time.time() - t0))


def main():
    ckpt_dir = sys.argv[1] if len(sys.argv) > 1 else "out-6m-epr"
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

    model, margs, ckpt = load_any_checkpoint(ckpt_dir, device=device)
    if not margs.use_epr:
        raise SystemExit("eval_kv_eviction.py requires an EPR Attention checkpoint.")

    base_dir = os.path.dirname(os.path.abspath(__file__))
    val_bin = os.path.join(base_dir, "data", ckpt.get("config", {}).get("dataset", "wikitext2"), "val.bin")
    data = np.memmap(val_bin, dtype=np.uint16, mode="r")

    attn0 = model.layers[0].attention
    print(f"=== Asymmetric EPRKVCache Eviction Tradeoff ({ckpt_dir}, H={attn0.n_heads}, WH={attn0.n_wh}) ===")
    print(f"{'Damped Window W':>16s} | {'Saved @2K':>9s} | {'Saved @8K':>9s} | {'Tail PPL [1K:2K]':>16s} | {'Needle@1536 (200 trials)':>26s} | {'Decode tok/s':>12s}")
    print("-" * 106)

    for W in [None, 512, 256, 128, 64]:
        apply_damped_sliding_window_hook(model, W)
        bp = evaluate_bucketed_ppl(model, data, L_max=2048, buckets=[(1024, 2048)], eval_batches=24, batch_size=16, seed=1234, device=device)
        ppl_tail = bp[(1024, 2048)]
        acc, ci, dnll = evaluate_needle_single_dist(model, data, dist=1536, num_trials=200, seed=2026, device=device)

        apply_damped_sliding_window_hook(model, None)
        tok_s = bench_stateful_decode(model, W, prompt_len=512, gen_tokens=64, device=device)

        if W is None:
            s2k, s8k, w_lbl = 0.0, 0.0, "Full (∞)"
        else:
            c2k = model.build_kv_cache(batch_size=1, max_seq_len=2048, window_size=W, dtype=torch.bfloat16)
            c8k = model.build_kv_cache(batch_size=1, max_seq_len=8192, window_size=W, dtype=torch.bfloat16)
            s2k = c2k.savings_ratio(2048) * 100.0
            s8k = c8k.savings_ratio(8192) * 100.0
            w_lbl = f"W = {W}"
        print(f"{w_lbl:>16s} | {s2k:8.1f}% | {s8k:8.1f}% | {ppl_tail:16.2f} | {acc:5.1f}±{ci:3.1f}% (ΔNLL {dnll:+.2f}) | {tok_s:10.1f}/s")


if __name__ == "__main__":
    main()
