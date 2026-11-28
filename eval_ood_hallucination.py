"""
OOD Uncertainty Calibration & Zero-Cost Vacuum Hallucination Probe Benchmark.

Tests the core bounded-RBF + Vacuum-Floor anti-hallucination hypothesis in two parts:
  1. Predictive Overconfidence & Entropy Shift on OOD / Scrambled Contexts (`T=1024`):
     Compares how Pure Llama 2 (unbounded RoPE dot-product), FoX, and EPR react when fed
     random / scrambled out-of-distribution token sequences vs clean validation text:
       - Lower OOD Max-Prob (`P_max ↓`) & Higher OOD Entropy (`H_ood ↑`) = less confident hallucination!
  2. Zero-Cost Unanswerable Context Detection via Wormhole Vacuum Silence (`AUROC`):
     Without any classifier training, reads the Wormhole head's vacuum probability `W(i, vac)`
     on query cue tokens when a distant needle is ABSENT (unanswerable) vs PRESENT (grounded),
     computing exact ROC-AUC for zero-cost hallucination abstention!

Usage:
  python3 eval_ood_hallucination.py out-6m-base out-6m-fox out-6m-epr
"""

import math
import os
import sys
import numpy as np
import torch
import torch.nn.functional as F

from epr_attention import EPRAttention
from run_sota_baselines import load_any_checkpoint


def compute_auroc(pos_scores, neg_scores) -> float:
    """Exact Mann-Whitney U ROC-AUC (`pos_scores` expected higher than `neg_scores`)."""
    pos = np.asarray(pos_scores, dtype=np.float64)
    neg = np.asarray(neg_scores, dtype=np.float64)
    diffs = pos[:, None] - neg[None, :]
    wins = np.sum(diffs > 0) + 0.5 * np.sum(diffs == 0)
    return float(wins / max(1, len(pos) * len(neg)))


@torch.no_grad()
def eval_ood_calibration(model, data, vocab_size: int, L: int = 1024, batches: int = 20, bs: int = 8, device: str = "cuda"):
    rng = np.random.RandomState(2026)
    max_idx = len(data) - L - 1
    id_pmax, id_ent = [], []
    ood_pmax, ood_ent = [], []

    for _ in range(batches):
        ix = rng.randint(0, max_idx, size=(bs,))
        x_id = torch.stack([torch.from_numpy(data[i : i + L].astype(np.int64)) for i in ix]).to(device)
        # OOD scrambled context: random uniform tokens in [0, vocab_size)
        x_ood = torch.from_numpy(rng.randint(0, vocab_size, size=(bs, L), dtype=np.int64)).to(device)

        for x_in, pm_list, ent_list in [(x_id, id_pmax, id_ent), (x_ood, ood_pmax, ood_ent)]:
            logits = model(x_in, targets=x_in)[:, L // 2 :, :].float()  # tail half (512..1024)
            probs = F.softmax(logits, dim=-1)
            log_probs = F.log_softmax(logits, dim=-1)
            ent = -(probs * log_probs).sum(dim=-1)
            pm = probs.max(dim=-1).values
            pm_list.append(pm.mean().item())
            ent_list.append(ent.mean().item())

    return (
        float(np.mean(id_pmax)) * 100.0,
        float(np.mean(id_ent)),
        float(np.mean(ood_pmax)) * 100.0,
        float(np.mean(ood_ent)),
    )


@torch.no_grad()
def eval_vacuum_abstention_probe(model, data, dist: int = 1024, num_trials: int = 150, device: str = "cuda"):
    """
    Zero-cost abstention probe on EPR models: checks whether Wormhole head vacuum silence
    `W_wh(cue, vac)` increases when the distant needle is absent vs present!
    """
    if not getattr(model.params, "use_epr", False):
        return None
    attn0 = model.layers[0].attention
    if not isinstance(attn0, EPRAttention) or not attn0.use_attn_vacuum:
        return None

    rng = np.random.RandomState(2026)
    needle_len, cue_len = 16, 4
    max_start = len(data) - dist - 200
    vac_absent, vac_present = [], []

    for _ in range(num_trials):
        i_n = rng.randint(0, max_start)
        i_h = rng.randint(0, max_start)
        while abs(i_h - i_n) < dist + 64:
            i_h = rng.randint(0, max_start)
        needle = data[i_n : i_n + needle_len].astype(np.int64)
        hay = data[i_h : i_h + dist].astype(np.int64)
        dummy = data[(i_h + dist + 50) % max_start : (i_h + dist + 50) % max_start + needle_len].astype(np.int64)

        seq_pres = np.concatenate([needle, hay, needle])[:-1]
        seq_abs = np.concatenate([dummy, hay, needle])[:-1]
        x_pair = torch.stack([torch.from_numpy(seq_pres), torch.from_numpy(seq_abs)]).to(device)

        h = model.tok_embeddings(x_pair)
        fc, fs = model._get_rope_freqs(x_pair.size(1), device)
        tail_start = needle_len + dist + cue_len - 1
        wh_vacs = []
        for layer in model.layers:
            _, _, tok_vac = layer.attention.measure_diagnostics(layer.attention_norm(h), fc, fs, return_token_vac=True)
            # Measure Wormhole head(s) if enabled, else slowest head
            if layer.attention.n_wh > 0:
                v_tail = tok_vac[:, layer.attention.n_damped :, tail_start:].mean(dim=(1, 2))
            else:
                v_tail = tok_vac[:, -1:, tail_start:].mean(dim=(1, 2))
            wh_vacs.append(v_tail)
            h = layer(h, fc, fs)

        mean_vac = torch.stack(wh_vacs, dim=0).mean(dim=0)  # [2]
        vac_present.append(mean_vac[0].item() * 100.0)
        vac_absent.append(mean_vac[1].item() * 100.0)

    auroc = compute_auroc(vac_absent, vac_present)
    return float(np.mean(vac_present)), float(np.mean(vac_absent)), auroc


def main():
    ckpt_dirs = [a for a in sys.argv[1:] if not a.startswith("--")]
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    for a in sys.argv[1:]:
        if a.startswith("--device="):
            device = a.split("=", 1)[1]

    if not ckpt_dirs:
        ckpt_dirs = ["out-6m-base", "out-6m-epr"]

    base_dir = os.path.dirname(os.path.abspath(__file__))
    print("=== [1] OOD Scrambled-Context Overconfidence & Entropy Calibration (T=1024, Tail [512:1024]) ===")
    print(f"{'Checkpoint':26s} | {'In-Dist P_max':>13s} | {'In-Dist Ent':>11s} | {'OOD P_max (↓)':>13s} | {'OOD Ent (↑)':>11s} | {'ΔEnt (OOD-ID)':>13s}")
    print("-" * 102)

    probe_rows = []
    for path in ckpt_dirs:
        model, margs, ckpt = load_any_checkpoint(path, device=device)
        val_bin = os.path.join(base_dir, "data", ckpt.get("config", {}).get("dataset", "wikitext2"), "val.bin")
        data = np.memmap(val_bin, dtype=np.uint16, mode="r")
        label = os.path.basename(path.rstrip("/"))

        id_pm, id_h, ood_pm, ood_h = eval_ood_calibration(model, data, margs.vocab_size, L=1024, device=device)
        print(f"{label:26s} | {id_pm:12.2f}% | {id_h:10.3f}  | {ood_pm:12.2f}% | {ood_h:10.3f}  | {ood_h - id_h:+12.3f}")

        probe = eval_vacuum_abstention_probe(model, data, dist=1024, num_trials=150, device=device)
        if probe is not None:
            probe_rows.append((label, *probe))

    if probe_rows:
        print("\n=== [2] Zero-Cost Unanswerable Context Detection via Wormhole Vacuum Silence (Dist=1024, 150 trials) ===")
        print(f"{'EPR Checkpoint':26s} | {'Grounded Vac%':>14s} | {'Unanswerable Vac%':>18s} | {'Vac Shift':>11s} | {'Abstention AUROC':>16s}")
        print("-" * 98)
        for label, vp, va, auc in probe_rows:
            print(f"{label:26s} | {vp:13.2f}% | {va:17.2f}% | {va - vp:+10.2f}% | {auc:16.4f}")


if __name__ == "__main__":
    main()
