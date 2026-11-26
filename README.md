<div align="center">

# Unified Fiber-Gravity & ER=EPR Wormhole Attention (v6)

**Edge-First, High-Density Long-Context Architecture on a Clean Llama 2 Backbone**

**English** | [简体中文](README_zh.md) · [📊 Full Benchmarks, Translation & Samples](docs/BENCHMARKS.md)

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Built with Meta Llama 2](https://img.shields.io/badge/Architecture-Meta%20Llama%202-blueviolet)](NOTICE)
[![Causality Audit](https://img.shields.io/badge/fp64%20Causality%20Audit-0.00e+00%20Leak-brightgreen)](audit_all.py)

</div>

> **Attribution:** Built with the **Meta Llama 2** architecture (Touvron et al., 2023) on top of Andrej Karpathy's [`llama2.c`](https://github.com/karpathy/llama2.c) (MIT License). See [`LICENSE`](LICENSE), [`NOTICE`](NOTICE), and [`CITATION.bib`](CITATION.bib) for full license and citation terms.

**Unified Fiber-Gravity & ER=EPR Wormhole Attention (`v6-wormhole`)** is an **edge-first, high-density long-context architecture** designed so smaller models can run on everyday consumer GPUs, laptops, and edge devices while punching well above their parameter weight class. By replacing redundant attention parameters with compact geometric priors, it resolves the classic long-context dilemma: **damped attention extrapolates cleanly but forgets distant details, whereas undamped attention recalls distant needles but explodes beyond training length.**

Using **strictly fewer parameters (`-1.4% to -1.6%`, up to `-372K` fewer weights at 28M)** than Pure Llama 2 and **verified zero future leakage (`0.00e+00` in `float64`)**, `v6-wormhole` wins across **6 benchmarks (`1.3M → 28M`)**:
- **Beats SOTA (WikiText-2, 6M)**: **-19.3%** lower 4× extrapolation PPL than **FoX (ICLR 2025)** (`34.72` vs `43.05`), while maintaining **`32.3% (+0.59 ΔNLL)`** 3× needle recall where ALiBi & FoX decay to `+0.00`.
- **Scales Stronger at 28M (RealWorld 16K)**: **-8.6%** in-domain PPL (`34.95` vs `38.24`), **-81.6%** at 4× extrapolation (`32.26` vs `175.29`), **4× accuracy / 11× ΔNLL** at `Dist=1536` (`41.1% / +1.34` vs `10.4% / +0.12`), and **wins at equal GPU wall-clock time** ([Iso-Wallclock PPL →](docs/BENCHMARKS.md#iso-wallclock)).
- **Zero-Shot Cross-Lingual Interlingua (WMT14)**: **+22.0% sacreBLEU** on `En→De` (`5.27` vs `4.32`) and **+41.1% / +42.1% sacreBLEU** on unseen **Zero-Shot `De↔Fr`** (`4.80` vs `3.40` / `4.34` vs `3.06`) ([Translation Benchmarks →](docs/BENCHMARKS.md#translation)).

---

## 1. Architecture Overview

![Unified Fiber-Gravity & ER=EPR Wormhole Attention Architecture](docs/figures/architecture_wormhole_attention.png)

- **Orthogonal Fiber–Base Packing (`d_r + d_g + 2 == head_dim`)**: Splits each head into a directional **Resonance Fiber** ($d_r$) and a low-dim **Base Spacetime RBF** ($d_g$), folding bilateral mass ($M_i M_j$) and vacuum floor ($e^{v_0}$) into **one single $D$-aligned SDPA call with zero $V$ padding**.
- **Dual-Regime Topology**: Damped **Solar-System Heads** (`Moon → Sat → Earth`) handle local/mid-range modeling with nested dynamic proper time; the **ER=EPR Wormhole Head** (`H-1`) uses **zero time decay ($\delta\tau \equiv 0$) + NoPE + high vacuum throat ($v_0 \approx +3.0$) + tight RBF lock** to stay silent on noise yet instantaneously retrieve exact distant needles.
- **Damped Pauli Repulsion Decoding**: Exponentially decaying degeneracy pressure eliminates local repetition loops while preserving long-range entity recall across paragraphs.

---

## 2. Headline Benchmark Summary

> 🔍 **Dive Deeper:** [Full 6-Benchmark Parameter Matrix](docs/BENCHMARKS.md#panoramic-summary) · [Iso-Wallclock PPL](docs/BENCHMARKS.md#iso-wallclock) · [Scaling Law & 28M Needle](docs/BENCHMARKS.md#scaling-law) · [WMT14 & Zero-Shot De↔Fr](docs/BENCHMARKS.md#translation) · [SOTA Showdown](docs/BENCHMARKS.md#sota-showdown) · [Samples](docs/BENCHMARKS.md#samples)

| Benchmark (`dim/L/H`, Vocab) | Model | Decayed Params | Train PPL (`T=512`) | 4× Extrap (`T=2048`) | `Dist=1536` Needle / BLEU |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **6.0M WikiText-2** (`256/6L/4H`, 4K) | Pure Llama 2 / FoX (ICLR'25) | `5.87M` / `5.87M` | `47.98` / `44.88` | `96.23` / `43.05` | `13.5% (+0.09)` / `23.7% (+0.00)` |
| **6.0M WikiText-2** (`256/6L/4H`, 4K) | **`v6-wormhole` (Ours)** | **`5.77M`** (`-94K`) | **`37.50`** | **`34.72`** (`-19.3% vs FoX`) | **`32.3% (+0.59)`** |
| **6.0M TinyStories** (`256/6L/4H`, 4K) | Pure Llama 2 vs **`v6-wormhole`** | `5.87M` vs **`5.77M`** | `5.50` vs **`5.20`** | `28.67` vs **`4.90`** (`-82.9%`) | `16.4%` vs **`32.6%`** |
| **6.8M FineWeb-Edu** (`256/6L/4H`, 8K) | Pure Llama 2 vs **`v6-wormhole`** | `6.91M` vs **`6.82M`** | `37.52` vs **`34.87`** | `161.27` vs **`32.29`** (`-80.0%`) | `9.6% (+0.13)` vs **`29.2% (+0.55)`** |
| **27.6M RealWorld** (`512/6L/8H`, 16K) | Pure Llama 2 vs **`v6-wormhole`** | `27.66M` vs **`27.28M`** | `38.24` vs **`34.95`** | `175.29` vs **`32.26`** (`-81.6%`) | `10.4% (+0.12)` vs **`41.1% (+1.34)`** |
| **6.8M WMT14 En→De** (`256/6L/4H`, 8K) | Pure Llama 2 vs **`v6-wormhole`** | `6.91M` vs **`6.82M`** | `36.63` vs **`32.69`** | — | **BLEU: `4.32` vs `5.27` (`+22.0%`)** |
| **7.9M Zero-Shot De↔Fr** (`256/6L/4H`, 12K) | Pure Llama 2 vs **`v6-wormhole`** | `7.96M` vs **`7.87M`** | `52.33` vs **`47.94`** | — | **De→Fr `+41.1%` / Fr→De `+42.1%`** |

---

## 3. Core Parameters & Hardware Channel Packing

| Head Dim $D$ | Fiber $d_r$ | Base $d_g$ | Augmented Slots | Total Packed Dim | `V` Padding | Decayed Weight Delta vs Base |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `32` (`1.3M, 4H`) | `24` | `6` | `+2` | **`32`** | `0` | **`-14,848`** |
| `64` (`6M–8M, 4H`) | `48` | `14` | `+2` | **`64`** | `0` | **`-93,696`** |
| `64` (`28M, 8H`) | `48` | `14` | `+2` | **`64`** | `0` | **`-371,712`** |
| `128` | `96` | `30` | `+2` | **`128`** | `0` | Strictly fewer than Base |

Key `ModelArgs` switches in `model.py`: `use_unified_gravity=True`, `use_wormhole=True`, `wormhole_v0_init=2.5`, `use_base_rbf=True`, `use_bilateral_mass=True`, `use_time_dilation=True`, `use_nested_time=True`, `use_attn_vacuum=True`, and `pauli_alpha` / `pauli_halflife` in `sample.py`.

---

## 4. Formula Origins: Physical & Academic Lineage

$$\Psi_h(i, j) = \underbrace{\frac{q_{\text{res},i}\cdot k_{\text{res},j}}{\sqrt{d_r}}}_{\text{Fiber Resonance}} + \underbrace{u_{\text{recv},i} + u_{\text{emit},j}}_{\text{Bilateral Mass } M_i M_j} - \underbrace{\frac{\|z_i - z_j\|_2^2}{2\sigma_h^2\sqrt{d_g}}}_{\text{Base Spacetime RBF}} - \underbrace{(\tau_{h,i} - \tau_{h,j})}_{\text{Nested Proper Time}}, \qquad W_h(i, j) = \frac{e^{\Psi_h(i, j)}}{e^{v_{0,h}} + \sum_{k \le i} e^{\Psi_h(i, k)}}$$

| Component | Theoretical Physics Origin | Deep Learning Lineage |
| :--- | :--- | :--- |
| **Fiber–Base Split ($d_r \oplus d_g$)** | Fiber Bundle / Gauge–Gravity Coupling | Retains RoPE (Su et al., 2021) while isolating RBF to low-dim $d_g$ |
| **Base Euclidean RBF** | Low-dim Gaussian Gravitational Potential / Heat Kernel | L2/RBF Attention (Kim et al., 2021), folded into standard SDPA |
| **Bilateral Mass ($M_i M_j$)** | Newtonian Two-Body Coupling ($F \propto m_1 m_2$) | Query reception mass dynamically lowers vacuum horizon |
| **Nested Proper Time ($\delta\tau$)** | General Relativity Time Dilation + Solar Hierarchy | Generalizes ALiBi (ICLR 2022) & FoX (ICLR 2025) with top-down cascade |
| **ER=EPR Wormhole Head** | ER = EPR Conjecture (Maldacena & Susskind, 2013) | Combines NoPE (NeurIPS 2023) + high-threshold Softmax1 + tight RBF |
| **Vacuum Floor ($e^{v_{0,h}}$)** | QFT Mass Gap / Vacuum Ground State | Softmax1 / Quiet Attention (Evan Miller, 2023) with learnable $v_{0,h}$ |
| **Damped Pauli Repulsion** | Pauli Degeneracy Pressure / Lennard-Jones Balance | Smooth exponential alternative to hard-window repetition penalty |

---

## 5. Repository Structure

```text
llama2-wormhole-gravity/
├── model.py                     # Core Transformer + UnifiedFiberGravityAttention (v6) + Pauli decoding
├── train.py                     # Multi-hardware training loop (L4/A100 bf16, T4 fp16, Apple MPS)
├── sample.py                    # Autoregressive sampling with Damped Pauli Repulsion CLI
├── run_all_tests.py / .sh       # One-click master runner (audit + train + extrap + needle + BLEU)
├── audit_all.py                 # 6-pillar float64 mathematical parity & zero-leakage causality audit
├── inspect_fields.py            # Per-layer, per-head physics diagnostics (σ_g, δτ, v0, silence %)
├── eval_extrapolation.py        # Zero-shot length extrapolation (Vanilla / Dynamic NTK / YaRN RoPE)
├── eval_needle.py               # Distant Needle-in-a-Haystack recall benchmark (64 -> 1536)
├── eval_bleu_wmt14.py           # Official newstest2014 WMT14 En->De sacreBLEU evaluator
├── eval_zeroshot_defr.py        # Supervised En->De/Fr + Zero-Shot De<->Fr sacreBLEU evaluator
├── run_sota_baselines.py        # Apples-to-apples ALiBi (ICLR 2022) & FoX (ICLR 2025) trainer
├── config/                      # All 7 benchmark configs (1.3M / 6M / 6.8M / 7.9M / 28M)
├── scripts/                     # Streaming HuggingFace dataset builders (5 corpora)
├── docs/
│   ├── BENCHMARKS.md            # Complete English benchmarks, Iso-Wallclock, Translation & samples
│   ├── BENCHMARKS_zh.md         # Complete Chinese benchmarks, Iso-Wallclock, Translation & samples
│   └── figures/                 # 6 publication-grade 300-DPI architecture & benchmark figures
├── CITATION.cff / CITATION.bib  # Official GitHub citation metadata & academic BibTeX
└── LICENSE / NOTICE             # MIT License (Andrej Karpathy + Dracen-W) & Meta Llama 2 notice
```

---

## 6. One-Click Quick Start & `float64` Audit

```bash
# 1. Run 6-pillar fp64 audit only (~3 seconds, zero future leak verified)
python3 run_all_tests.py --mode=audit

# 2. One-click full benchmark suites (auto-prepares data, trains Base vs v6, audits & evaluates!)
python3 run_all_tests.py --mode=wikitext2   # ~8 min on L4: 6M WikiText-2 + 2048 extrap + 1536 needle
python3 run_all_tests.py --mode=ende        # ~12 min on L4: 6.8M WMT14 En->De sacreBLEU
python3 run_all_tests.py --mode=zeroshot    # ~15 min on L4: 7.9M Trilingual Zero-Shot De<->Fr
python3 run_all_tests.py --mode=28m         # ~42 min on L4: 28M RealWorld 16K (8-head) showcase
```

---

## 7. Edge-First Positioning & Engineering Roadmap

Training is a one-time cost; **inference footprint (parameter count, RAM/VRAM budget, and KV-cache size)** determines whether an architecture can run on everyday consumer hardware. Because `v6-wormhole` achieves lower PPL, stronger focus, and 4× distant recall with **fewer parameters** than Pure Llama 2, it is naturally suited as a high-density backbone for resource-constrained devices:

- **Why Current PyTorch Training Is ~1.35× per Step (and Shrinks with Width)**: Passing a custom float `attn_mask` to `F.scaled_dot_product_attention` bypasses FlashAttention-2's fused SRAM path and falls back to unfused kernels. As model width increases (`dim=256 → 512`), FFN GEMMs dominate and the per-step ratio drops from `1.51×` to **`1.35×`**, already achieving **Iso-Wallclock victory at 1,368s**.
- **Roadmap 1 — Fused Triton / Flash-Wormhole Kernel**: Fusing cumulative proper-time subtraction $(\tau_i - \tau_j)$ and the vacuum slot directly inside online-softmax SRAM tiles (analogous to FoX's fused kernel) will eliminate the `attn_mask` fallback and bring training throughput to parity with Pure Llama 2.
- **Roadmap 2 — Asymmetric Edge KV-Cache Compression (`run.c`)**: Because damped heads (`H0..H{H-2}`) decay exponentially with distance (`half-life ~ 0.5t to 37t`), edge C/mobile inference engines can evict distant KV entries on damped heads via sliding windows while retaining full-context KV **only on the single ER=EPR Wormhole head (`H-1`)**, cutting long-context KV-cache memory by **75%–87.5%** (for 4-head / 8-head models).

---

## 8. License & Citation

- **License**: MIT License (preserving Andrej Karpathy's original `llama2.c` copyright; see [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE) for Meta Llama 2 compliance).
- **Citation**: Click **"Cite this repository"** in the right sidebar ([`CITATION.cff`](CITATION.cff)) or copy below:

```bibtex
@misc{dracenw2026wormhole,
  title        = {Unified Fiber-Gravity and ER=EPR Wormhole Attention},
  author       = {Dracen-W},
  year         = {2026},
  howpublished = {\url{https://github.com/Dracen-W/llama2-wormhole-gravity}},
  note         = {Built on karpathy/llama2.c and the Meta Llama 2 architecture}
}
```
