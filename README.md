<div align="center">

# Unified Fiber-Gravity & ER=EPR Wormhole Attention (v6)

**Physics-Grounded Long-Context Attention on a Clean Llama 2 Backbone**

**English** | [简体中文](README_zh.md) · [📊 Full Benchmarks & Samples](docs/BENCHMARKS.md)

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Built with Meta Llama 2](https://img.shields.io/badge/Architecture-Meta%20Llama%202-blueviolet)](NOTICE)
[![Causality Audit](https://img.shields.io/badge/fp64%20Causality%20Audit-0.00e+00%20Leak-brightgreen)](audit_all.py)

</div>

> **Attribution:** Built with the **Meta Llama 2** architecture (Touvron et al., 2023) on top of Andrej Karpathy's [`llama2.c`](https://github.com/karpathy/llama2.c) (MIT License). See [`LICENSE`](LICENSE), [`NOTICE`](NOTICE), and [`CITATION.bib`](CITATION.bib) for full license and citation terms.

**Unified Fiber-Gravity & ER=EPR Wormhole Attention (`v6-wormhole`)** replaces standard dot-product attention with a hardware-aligned, physics-grounded interaction kernel that resolves the long-context dilemma: **damped attention extrapolates cleanly but forgets distant details, whereas undamped attention recalls distant needles but explodes beyond training length.**

Using **strictly fewer parameters (`-1.4% to -1.6%`)** than Pure Llama 2 and **verified zero future leakage (`0.00e+00` in `float64`)**, `v6-wormhole` achieves across **4 corpora (`1.3M → 28M`)**:
- **Beats SOTA (WikiText-2, 6M)**: **-19.3%** lower 4× extrapolation PPL than **FoX (ICLR 2025)** (`34.72` vs `43.05`), while maintaining **`32.3% (+0.59 ΔNLL)`** 3× needle recall where ALiBi & FoX decay to `+0.00`.
- **Scales Stronger at 28M (RealWorld 16K)**: **-8.6%** in-domain PPL (`34.95` vs `38.24`), **-81.6%** at 4× extrapolation (`32.26` vs `175.29`), **4× accuracy / 11× ΔNLL** at `Dist=1536` (`41.1% / +1.34` vs `10.4% / +0.12`), and **wins at equal GPU wall-clock time** ([see Iso-Wallclock chart →](docs/BENCHMARKS.md#iso-wallclock)).

---

## 1. Architecture Overview

![Unified Fiber-Gravity & ER=EPR Wormhole Attention Architecture](docs/figures/architecture_wormhole_attention.png)

- **Orthogonal Fiber–Base Packing (`d_r + d_g + 2 == head_dim`)**: Splits each head into a directional **Resonance Fiber** ($d_r$) and a low-dim **Base Spacetime RBF** ($d_g$), folding bilateral mass ($M_i M_j$) and vacuum floor ($e^{v_0}$) into **one single $D$-aligned SDPA call with zero $V$ padding**.
- **Dual-Regime Topology**: Damped **Solar-System Heads** (`Moon → Sat → Earth`) handle local/mid-range modeling with nested dynamic proper time; the **ER=EPR Wormhole Head** (`H-1`) uses **zero time decay ($\delta\tau \equiv 0$) + NoPE + high vacuum throat ($v_0 \approx +3.0$) + tight RBF lock** to stay silent on noise yet instantaneously retrieve exact distant needles.
- **Damped Pauli Repulsion Decoding**: Exponentially decaying degeneracy pressure eliminates local repetition loops while preserving long-range entity recall across paragraphs.

---

## 2. Headline Benchmark Summary

> 🔍 **Dive Deeper:** [Full 4-Corpus Tables](docs/BENCHMARKS.md#panoramic-summary) · [Same-Step & Iso-Wallclock Charts](docs/BENCHMARKS.md#iso-wallclock) · [Scaling Law & 28M Needle Recall](docs/BENCHMARKS.md#scaling-law) · [SOTA Showdown](docs/BENCHMARKS.md#sota-showdown) · [Qualitative Samples](docs/BENCHMARKS.md#samples)

| Scale & Corpus | Model | Decayed Params | Train PPL (`T=512`) | 4× Extrap (`T=2048`) | `Dist=1536` Needle (3×) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **6.0M / WikiText-2** | Pure Llama 2 / FoX (ICLR'25) | `5.87M` / `5.87M` | `47.98` / `44.88` | `96.23` / `43.05` | `13.5% (+0.09)` / `23.7% (+0.00)` |
| **6.0M / WikiText-2** | **`v6-wormhole` (Ours)** | **`5.77M`** (`-94K`) | **`37.50`** | **`34.72`** (`-19.3% vs FoX`) | **`32.3% (+0.59)`** |
| **6.0M / TinyStories** | Pure Llama 2 vs **`v6-wormhole`** | `5.87M` vs **`5.77M`** | `5.50` vs **`5.20`** | `28.67` vs **`4.90`** (`-82.9%`) | `16.4%` vs **`32.6%`** |
| **6.8M / FineWeb-Edu** | Pure Llama 2 vs **`v6-wormhole`** | `6.91M` vs **`6.82M`** | `37.52` vs **`34.87`** | `161.27` vs **`32.29`** (`-80.0%`) | `9.6% (+0.13)` vs **`29.2% (+0.55)`** |
| **27.6M / RealWorld 16K** | Pure Llama 2 vs **`v6-wormhole`** | `27.66M` vs **`27.28M`** | `38.24` vs **`34.95`** | `175.29` vs **`32.26`** (`-81.6%`) | `10.4% (+0.12)` vs **`41.1% (+1.34)`** |

---

## 3. Core Parameters & Hardware Channel Packing

| Head Dim $D$ | Fiber $d_r$ | Base $d_g$ | Augmented Slots | Total Packed Dim | `V` Padding | Decayed Weight Delta vs Base |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `32` (`1.3M`) | `24` | `6` | `+2` | **`32`** | `0` | **`-14,848`** |
| `64` (`6.0M`) | `48` | `14` | `+2` | **`64`** | `0` | **`-93,696`** |
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
├── audit_all.py                 # 6-pillar float64 mathematical parity & zero-leakage causality audit
├── inspect_fields.py            # Per-layer, per-head physics diagnostics (σ_g, δτ, v0, silence %)
├── eval_extrapolation.py        # Zero-shot length extrapolation (Vanilla / Dynamic NTK / YaRN RoPE)
├── eval_needle.py               # Distant Needle-in-a-Haystack recall benchmark (64 -> 1536)
├── run_sota_baselines.py        # Apples-to-apples ALiBi (ICLR 2022) & FoX (ICLR 2025) trainer
├── config/                      # Ready-to-run configs (WikiText-2 1.3M/6M, TinyStories, FineWeb-Edu, 28M RealWorld)
├── scripts/                     # Streaming HuggingFace dataset builders (TinyStories, FineWeb-Edu, RealWorld 16K)
├── docs/
│   ├── BENCHMARKS.md            # Complete English benchmarks, Iso-Wallclock, Scaling Laws & samples
│   ├── BENCHMARKS_zh.md         # Complete Chinese benchmarks, Iso-Wallclock, Scaling Laws & samples
│   └── figures/                 # 5 publication-grade 300-DPI architecture & benchmark figures
├── CITATION.cff / CITATION.bib  # Official GitHub citation metadata & academic BibTeX
└── LICENSE / NOTICE             # MIT License (Andrej Karpathy + Dracen-W) & Meta Llama 2 notice
```

---

## 6. Quick Start & 6-Pillar `float64` Audit

```bash
# 1. Run 6-pillar audit (fp64 future perturbation=0.00e+00, future grad=0.00e+00, math parity <= 3.3e-16)
python3 audit_all.py

# 2. Train 6.0M / 512-context v6-Wormhole on WikiText-2 (Colab L4: ~4 min)
python3 train.py config/train_wikitext2_colab_scale.py \
  --use_unified_gravity=True --use_wormhole=True --out_dir=out-6m-v6-wormhole

# 3. Audit trained checkpoint, inspect physics, evaluate 4x extrapolation & 3x needle recall
python3 audit_all.py --ckpt=out-6m-v6-wormhole/ckpt.pt
python3 inspect_fields.py out-6m-v6-wormhole/ckpt.pt
python3 eval_extrapolation.py out-6m-v6-wormhole --lengths=256,512,1024,2048
python3 eval_needle.py out-6m-v6-wormhole --distances=64,128,256,512,1024,1536
```

---

## 7. License & Citation

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
