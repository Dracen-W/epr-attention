"""
Physics Diagnostics & Parameter Inspector for Unified Fiber-Gravity & ER=EPR Wormhole (v6).

Inspects per-layer, per-head learned physical quantities on live validation batches:
  - sigma_g:   Base spacetime Euclidean RBF width (tighter on Wormhole/Sun heads)
  - dtau:      Mean proper time velocity & half-life (strictly 0.000 / inf on Wormhole head)
  - alpha_dn:  Solar-system top-down nesting cascade coupling
  - v0 & sil%: Vacuum noise floor & live vacuum silence fraction (%)

Usage:
  python3 inspect_fields.py out-6m-v6-wormhole/ckpt.pt
"""

import math
import os
import sys
import numpy as np
import torch
import torch.nn.functional as F

from model import ModelArgs, Transformer

base_dir = os.path.dirname(os.path.abspath(__file__))
if len(sys.argv) > 1:
    ckpt_path = sys.argv[1]
    if os.path.isdir(ckpt_path):
        ckpt_path = os.path.join(ckpt_path, "ckpt.pt")
else:
    ckpt_path = os.path.join(base_dir, "out-6m-v6-wormhole", "ckpt.pt")

ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
valid_fields = set(ModelArgs.__dataclass_fields__.keys())
margs = {k: v for k, v in ckpt["model_args"].items() if k in valid_fields}
sd = {k.replace("_orig_mod.", ""): v for k, v in ckpt["model"].items()}
best_val = float(ckpt.get("best_val_loss", float("nan")))
step = ckpt.get("iter_num", -1)

print(f"=== Checkpoint Physics Inspection: {ckpt_path} ===")
print(f"Step: {step} | best_val_loss: {best_val:.4f} (PPL: {math.exp(min(20.0, best_val)):.2f})")

if not margs.get("use_unified_gravity", False):
    print("Pure Llama 2 baseline (no physics modules enabled).")
    sys.exit(0)

model = Transformer(ModelArgs(**margs))
model.load_state_dict(sd, strict=True)
model.eval()

n_layers, n_heads = margs["n_layers"], margs["n_heads"]
use_wh = margs.get("use_wormhole", True) and n_heads > 1
d_r, d_g = model.layers[0].attention.d_r, model.layers[0].attention.d_g

# Run live validation diagnostics if dataset is available
cfg = ckpt.get("config", {})
val_bin = os.path.join(base_dir, "data", cfg.get("dataset", "wikitext2"), "val.bin")
vac_rates, dtau_means, rho_means = None, None, None
if os.path.exists(val_bin):
    data = np.memmap(val_bin, dtype=np.uint16, mode="r")
    T = margs["max_seq_len"]
    rng = np.random.RandomState(42)
    ix = rng.randint(0, len(data) - T - 1, size=(16,))
    x_tok = torch.stack([torch.from_numpy(data[i : i + T].astype(np.int64)) for i in ix])
    with torch.no_grad():
        h = model.tok_embeddings(x_tok)
        f_cos, f_sin = model._get_rope_freqs(T, x_tok.device)
        vac_rates, dtau_means, rho_means = [], [], []
        for layer in model.layers:
            vr, dm, rm = layer.attention.measure_diagnostics(layer.attention_norm(h), f_cos, f_sin)
            vac_rates.append(vr.tolist() if vr is not None else None)
            dtau_means.append(dm.tolist())
            rho_means.append(rm.tolist() if rm is not None else None)
            h = layer(h, f_cos, f_sin)

print(f"\nFiber-Base Packing: d_r={d_r} fiber + d_g={d_g} base + 2 = {d_r + d_g + 2} | ER=EPR Wormhole: {use_wh}")
print("-" * 110)

if n_heads == 4:
    scale_names = ["Moon ", "Sat  ", "Earth", "Worm "] if use_wh else ["Moon ", "Sat  ", "Earth", "Sun  "]
else:
    scale_names = [f"H{h:<4d}" for h in range(n_heads)]
    if use_wh:
        scale_names[-1] = "Worm "

for i in range(n_layers):
    p = f"layers.{i}.attention."
    sigmas = torch.exp(-0.5 * sd[p + "log_inv_sigma2"]).tolist() if (p + "log_inv_sigma2") in sd else None
    alphas = F.softplus(sd[p + "raw_nest_alpha"]).tolist() if (p + "raw_nest_alpha") in sd else None
    v0s = sd[p + "v0"].tolist() if (p + "v0") in sd else None
    print(f"Layer {i}:")
    for h_idx in range(n_heads):
        cols = [f"  H{h_idx}({scale_names[h_idx]})"]
        if sigmas is not None:
            cols.append(f"σ_g:{sigmas[h_idx]:.2f}")
        if dtau_means is not None:
            dm = dtau_means[i][h_idx]
            hl_str = "   ∞  " if dm < 1e-6 else f"{math.log(2)/dm:5.1f}t"
            cols.append(f"δτ:{dm:.3f}({hl_str})")
        if alphas is not None:
            if use_wh and h_idx == n_heads - 1:
                a_str = "NoPE"
            elif h_idx < len(alphas):
                a_str = f"{alphas[h_idx]:.2f}"
            else:
                a_str = "root"
            cols.append(f"α↓:{a_str:>4s}")
        if v0s is not None:
            s = f"v0:{v0s[h_idx]:+.2f}"
            if vac_rates is not None and vac_rates[i] is not None:
                s += f" [沉默{vac_rates[i][h_idx]*100:4.1f}%]"
            cols.append(s)
        print(" | ".join(cols))
