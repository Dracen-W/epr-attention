<div align="center">

# EPR Attention (Euclidean-RBF, Proper-time & Resonance-fiber)

**An Experimental Long-Context Attention Mechanism Built on `llama2.c`**

**English** | [简体中文](README_zh.md) · [📐 Math & Derivations](docs/MATH.md) · [📊 v1.0 Legacy Archive](docs/BENCHMARKS.md) · [🏷️ v1.0 Release Tag](https://github.com/Dracen-W/epr-attention/releases/tag/1.0)

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Built with Meta Llama 2](https://img.shields.io/badge/Architecture-Meta%20Llama%202-blueviolet)](NOTICE)
[![8-Pillar Audit](https://img.shields.io/badge/8--Pillar%20Audit-0.00e+00%20Leak-brightgreen)](audit_all.py)

</div>

> **Attribution & Scope:** Built on Andrej Karpathy's [`llama2.c`](https://github.com/karpathy/llama2.c) (MIT License) and the **Meta Llama 2** architecture (Touvron et al., 2023). Note that `export.py`, `run.c`, and `runq.c` are inherited from `llama2.c` and **do not yet support EPR Attention** (Python/PyTorch only). See [`LICENSE`](LICENSE), [`NOTICE`](NOTICE), and [`CITATION.bib`](CITATION.bib).

> **Physics Inspiration Disclaimer:** The dual-regime topology and scoring geometry of **EPR Attention** are **inspired by** ER=EPR spacetime folding and spatial gravitational field intuition—**not** derived from general relativity or quantum gravity equations. All components are implemented as standard, differentiable PyTorch primitives that fold algebraically into a single $D$-aligned attention kernel (see [`docs/MATH.md`](docs/MATH.md)).

**EPR Attention** (**E**uclidean-RBF, **P**roper-time & **R**esonance-fiber Attention) combines three verified core primitives (plus a standalone zero-dependency implementation in [`epr_attention.py`](epr_attention.py)):
1. **Euclidean-RBF + Resonance Fiber Packing (`d_r + d_g + 2 == head_dim`)**: Folds a low-dim latent Euclidean-RBF potential and directional dot-product into one hardware-aligned attention call with **zero $V$ padding** and strictly fewer parameters than standard Llama 2.
2. **Learnable Vacuum Floor ($e^{v_0}$)**: Equivalent to Softmax1 / Attention Sink, allowing heads to stay silent on irrelevant context.
3. **Dual-Regime Head Topology**: Damped heads apply RoPE + data-dependent cumulative decay ($\tau_i - \tau_j$) for stable extrapolation and $O(1)$ sliding-window KV eviction; **Wormhole head(s)** use **NoPE + zero decay ($\delta\tau \equiv 0$) + high vacuum threshold ($v_0 \approx +2.5$)** for distance-independent long-range retrieval.

---

## 1. Honest Tradeoffs: Compute, Speed & Memory

We explicitly report per-step overhead alongside Iso-PPL efficiency and KV-cache savings:

| Benchmark (NVIDIA L4, `bfloat16`) | Pure Llama 2 (FA2) | EPR Attention | Honest Takeaway |
| :--- | :---: | :---: | :--- |
| **6M `d0` Single-Step Time (`T=512, B=32`)** | `32.85 ms` (`499k tok/s`) | `53.70 ms` (`305k tok/s`) | **1.63× slower per step** at 6M (small-model kernel-launch bound; MFU `~4%`). |
| **6M `d0` Iso-PPL (`Val PPL ≤ 57.7`)** | `1,500` steps (`49.3 s`) | **`750` steps (`39.1 s`)** | **-50.0% steps/tokens**, **-20.7% pure GPU time**, **-50.8% est. FLOPs**. |
| **6M `d0` Capacity (`Step 3000 Train Loss`)** | `2.5766` (PPL `13.15`) | **`2.2618` (PPL `9.60`)** | **-27.0% lower Train PPL** with `-94K` fewer parameters. |
| **1.1B `FlexAttention` + Native GQA (`T=1024`)** | `3,504 tok/s` (Train)<br>`18,569 tok/s` (`Prefill@2k`) | **`3,399 tok/s` (`97.0%`)**<br>**`18,765 tok/s` (`101.1%`)** | Kernel-launch overhead amortizes at 1.1B (`-20.7M` params, `10.32` vs `10.51 GB` VRAM). |

> **Important Backend Caveat:** Fused `FlexAttention` activates automatically on CUDA `bf16/fp16` with `T >= 256` and **`dropout == 0.0`**. When `dropout > 0` (e.g., small WikiText-2 runs), training falls back to masked `SDPA` with an explicit $O(T^2)$ mask.

---

## 2. Verified 6M Benchmarks (Un-Diluted Extrapolation, 200-Trial Recall & KV Eviction)

*All results below are measured on `6.0M` WikiText-2 (`dim=256, 6L, 4H, T_train=512`, `5.78M` EPR vs `5.87M` Llama 2). Note: `ΔNLL` denotes the reduction in target-token Negative Log-Likelihood when the distant needle is present vs. absent.*

### 2.1 Un-Diluted Bucketed Extrapolation & 200-Trial Long-Range Recall
| Model | `[0:256]` | `[256:512]` | `[512:1024]` | `[1024:2048]` | Needle `Dist=1536` (200 trials) | 2-Key Assoc. Recall `Dist=1024 / 1536` (Random=50%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Pure Llama 2** | `49.51` | `46.80` | `84.77` | `146.90` | `14.6±1.4% (+0.11)` | `77.5±5.8% (+1.74)` / `73.5±6.1% (+1.42)` |
| **EPR Attention** | **`40.30`** | **`35.36`** | **`34.53`** | **`34.06`** | **`31.9±1.9% (+0.48)`** | **`88.0±4.5% (+2.84)` / `81.0±5.4% (+2.24)`** |

*(Legacy `v1.0` SOTA comparison on the same 6M WikiText-2 setup: **FoX (ICLR'25)** reached `PPL@512 = 44.88`, `PPL@2048 = 43.05`, `Dist=1536 Needle = 23.7% (+0.00)` vs EPR `37.50 / 34.72 / 32.3% (+0.59)`. See [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md).)*

### 2.2 Asymmetric `EPRKVCache` Eviction Quality Cost (`eval_kv_eviction.py`)
Evicting distant KV strictly on damped heads (`0..H-2`) into a fixed ring buffer of size `W`, while keeping full context on the Wormhole head (`H-1`):

| Damped Window `W` | KV Saved `@2K` | KV Saved `@8K` | Tail PPL `[1024:2048]` | Needle `Dist=1536` (200 trials) |
| :--- | :---: | :---: | :---: | :---: |
| `Full (∞)` | `0.0%` | `0.0%` | `36.06` | `30.2±1.8% (ΔNLL +0.44)` |
| `W = 256` | `65.6%` | `72.7%` | `36.07` (`+0.01`) | `30.2±1.8% (ΔNLL +0.44)` |
| **`W = 128`** | **`70.3%`** | **`73.8%`** | **`36.10` (`+0.04`)** | **`30.1±1.8% (ΔNLL +0.44)`** |
| `W = 64` | `72.7%` | `74.4%` | `36.30` (`+0.24`) | `30.5±1.8% (ΔNLL +0.44)` |

### 2.3 Full 6M Single-Switch Ablation Study
| Variant | `PPL@512` | Tail `[1024:2048]` | Needle `Dist=1536` | 2-Key Assoc. `Dist=1024` | Verdict |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Full EPR (`v6`)** | **`37.73`** | **`34.06`** | **`31.9% (+0.48)`** | `88.0% (+2.84)` | Baseline |
| `w/o Vacuum Floor` | `40.76` (`+3.03`) | `37.63` (`+3.57`) | `28.1% (+0.22)` | `88.5% (+2.73)` | **Essential (#1)** |
| `w/o Euclidean-RBF` | `39.54` (`+1.81`) | `36.55` (`+2.49`) | `28.4% (+0.28)` | `85.0% (+2.73)` | **Essential (#2)** |
| `w/o Wormhole Head` | `39.52` (`+1.79`) | `36.35` (`+2.29`) | `24.8% (+0.00)` | `85.0% (+2.65)` | **Essential (#3 — sole driver of long recall)** |
| `w/o Nested Time` | `37.82` (`+0.09`) | `34.14` (`+0.08`) | `31.9% (+0.49)` | `88.5% (+2.86)` | Redundant (prunable for speed) |
| `w/o Bilateral Mass` | `37.75` (`+0.02`) | `33.94` (`-0.12`) | `32.0% (+0.50)` | `89.0% (+2.88)` | Redundant (prunable for params/speed) |

---

## 3. Relation to Prior Work & Verification Status

**Relation to Existing Literature:**
- **Cumulative Data-Dependent Decay ($\tau$)**: Mathematically isomorphic to **FoX (Forgetting Transformer)** when the nested cascade is removed (`M = I`).
- **Vacuum Floor ($e^{v_0}$)**: Equivalent to **Softmax1 (Attention Is Off By One)** / learnable **Attention Sink**.
- **NoPE Wormhole Head**: Related to hybrid RoPE/NoPE schemes such as **iRoPE** (Llama 4), differentiated here by pairing NoPE + zero decay with a high vacuum threshold (`v0 ≈ +2.5`) and Euclidean-RBF key locking.
- **Euclidean-RBF Potential**: Related to L2 / Gaussian-bias attention, evaluated on learned low-dimensional coordinates $z$ and algebraically folded into $D$-aligned dot-product attention.

**Verified vs. Unverified Checklist:**
- ✅ **Verified**: 8-pillar audit (`audit_all.py`: `fp64` textbook parity `3.33e-16`, zero future leak `0.00e+00`, CUDA `bf16` FlexAttention parity `1.95e-03` & zero leak `0.00e+00`, stateful `EPRKVCache` parity `3.89e-16`), un-diluted bucketed extrapolation, 200-trial needle + 2-key associative recall, `W=128` KV-eviction zero-drop, 6M full single-switch ablation, 1.1B hardware throughput.
- ⏳ **Not Yet Verified**: Multi-seed variance across all runs; quality/perplexity at `≥100M` / `1.1B` scale (1.1B numbers above are hardware throughput only); C-runtime (`run.c` / `export.py`) support.

---

## 4. Quick Start

```bash
# 1. Run 8-pillar audit (fp64 causality + CUDA bf16 FlexAttention + stateful EPRKVCache)
python3 audit_all.py

# 2. Standalone zero-dependency core module
# See epr_attention.py (EPRConfig, EPRAttention, EPRKVCache)

# 3. Evaluate un-diluted extrapolation, 200-trial needle, 2-key recall & KV-eviction cost
python3 eval_extrapolation.py out-6m-base out-6m-v6-wormhole --lengths=256,512,1024,2048
python3 eval_needle.py out-6m-base out-6m-v6-wormhole --distances=64,128,256,512,1024,1536
python3 eval_associative_recall.py out-6m-base out-6m-v6-wormhole
python3 eval_kv_eviction.py out-6m-v6-wormhole
```

---

## 5. Disclosure, License & Citation

- **AI Assistance Disclosure**: AI assistants were used during this project for code implementation, auditing scripts, benchmark automation, and documentation drafting; all architectural hypotheses, experimental decisions, and hardware verification runs were directed and executed by the author.
- **License**: MIT License (preserving Andrej Karpathy's original `llama2.c` copyright; see [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE)).
- **Citation**: Use **"Cite this repository"** ([`CITATION.cff`](CITATION.cff)) or copy:

```bibtex
@misc{dracenw2026epr,
  title        = {EPR Attention: An Experimental Long-Context Attention Mechanism Built on llama2.c},
  author       = {Dracen-W},
  year         = {2026},
  howpublished = {\url{https://github.com/Dracen-W/epr-attention}},
  note         = {Built on karpathy/llama2.c and the Meta Llama 2 architecture}
}
```
