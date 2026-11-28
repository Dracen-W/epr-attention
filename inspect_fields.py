"""
Physics Diagnostics & Parameter Inspector for Unified EPR Attention (v2.0).

Inspects per-layer, per-head learned physical quantities on live validation batches:
  - sigma_g:   Base spacetime Euclidean-RBF width (plus elastic alpha_el if enabled)
  - dtau:      Mean proper-time velocity & half-life (strictly 0.000 / inf on Wormhole head)
  - tunnel:    Pairwise resonance tunneling (theta, beta) if enabled
  - v0 & sil%: Vacuum noise floor & live vacuum silence fraction (%)

Usage:
  python3 inspect_fields.py out-6m-epr/ckpt.pt
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
    ckpt_path = os.path.join(base_dir, "out-6m-epr", "ckpt.pt")

ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
valid_fields = set(ModelArgs.__dataclass_fields__.keys())
margs_dict = {k: v for k, v in ckpt["model_args"].items() if k in valid_fields}
margs_dict["dropout"] = 0.0
margs = ModelArgs(**margs_dict)
sd = {k.replace("_orig_mod.", ""): v for k, v in ckpt["model"].items()}
best_val = float(ckpt.get("best_val_loss", float("nan")))
step = ckpt.get("iter_num", -1)

print(f"=== Checkpoint Physics Inspection: {ckpt_path} ===")
print(f"Step: {step} | best_val_loss: {best_val:.4f} (PPL: {math.exp(min(20.0, best_val)):.2f})")

if not margs.use_epr:
    print("Pure Llama 2 baseline (no EPR modules enabled).")
    sys.exit(0)

model = Transformer(margs)
model.load_state_dict(sd, strict=True)
model.eval()

attn0 = model.layers[0].attention
n_layers, n_heads, n_kv_heads = margs.n_layers, attn0.n_heads, attn0.n_kv_heads
n_rep = attn0.n_rep
use_wh = attn0.use_wormhole
d_r, d_g = attn0.d_r, attn0.d_g

cfg = ckpt.get("config", {})
val_bin = os.path.join(base_dir, "data", cfg.get("dataset", "wikitext2"), "val.bin")
vac_rates, dtau_means = None, None
if os.path.exists(val_bin):
    data = np.memmap(val_bin, dtype=np.uint16, mode="r")
    T = margs.max_seq_len
    rng = np.random.RandomState(42)
    ix = rng.randint(0, len(data) - T - 1, size=(16,))
    x_tok = torch.stack([torch.from_numpy(data[i : i + T].astype(np.int64)) for i in ix])
    with torch.no_grad():
        h = model.tok_embeddings(x_tok)
        f_cos, f_sin = model._get_rope_freqs(T, x_tok.device)
        vac_rates, dtau_means = [], []
        for layer in model.layers:
            vr, dm, _ = layer.attention.measure_diagnostics(layer.attention_norm(h), f_cos, f_sin)
            vac_rates.append(vr.tolist() if vr is not None else None)
            dtau_means.append(dm.tolist())
            h = layer(h, f_cos, f_sin)

print(
    f"\nFiber-Base Packing: d_r={d_r} fiber + d_g={d_g} base + 2 = {d_r + d_g + 2} | "
    f"Wormhole={use_wh} | ElasticRBF={attn0.use_elastic_rbf} | Tunnel={attn0.use_resonant_tunneling}"
)
print("-" * 114)

if n_heads == 4:
    scale_names = ["Moon ", "Sat  ", "Earth", "Worm "] if use_wh else ["Moon ", "Sat  ", "Earth", "Sun  "]
else:
    scale_names = [f"H{h:<4d}" for h in range(n_heads)]
    if use_wh:
        for w_idx in range(attn0.n_damped, n_heads):
            scale_names[w_idx] = "Worm "

for i in range(n_layers):
    attn = model.layers[i].attention
    sigmas = torch.exp(-0.5 * attn.log_inv_sigma2.detach()).tolist() if attn.use_base_rbf else None
    el_alphas = F.softplus(attn.raw_elastic_alpha.detach()).tolist() if attn.use_elastic_rbf else None
    v0s = attn.v0.detach().tolist() if attn.use_attn_vacuum else None
    t_thetas = attn.tunnel_theta.detach().tolist() if attn.use_resonant_tunneling else None
    t_betas = F.softplus(attn.tunnel_beta.detach()).tolist() if attn.use_resonant_tunneling else None

    print(f"Layer {i}:")
    for h_idx in range(n_heads):
        kv_idx = h_idx // n_rep
        cols = [f"  H{h_idx}({scale_names[h_idx]})"]
        if sigmas is not None:
            s_str = f"σ_0:{sigmas[kv_idx]:.2f}"
            if el_alphas is not None:
                s_str += f"(α_el:{el_alphas[kv_idx]:.2f})"
            cols.append(s_str)
        if dtau_means is not None:
            dm = dtau_means[i][h_idx]
            hl_str = "   ∞  " if dm < 1e-6 else f"{math.log(2)/dm:5.1f}t"
            cols.append(f"δτ:{dm:.3f}({hl_str})")
        if t_thetas is not None:
            cols.append(f"tun(θ:{t_thetas[h_idx]:+.2f},β:{t_betas[h_idx]:.2f})")
        if v0s is not None:
            s = f"v0:{v0s[kv_idx]:+.2f}"
            if vac_rates is not None and vac_rates[i] is not None:
                s += f" [sil:{vac_rates[i][h_idx]*100:4.1f}%]"
            cols.append(s)
        print(" | ".join(cols))
