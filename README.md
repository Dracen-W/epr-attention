<div align="center">

# EPR Attention

**A Drop-In Transformer Attention Module That Uses Fewer Parameters, Less KV Memory, and Extrapolates Cleanly**

**English** | [简体中文](README_zh.md) · [🤗 HF 1.08B Model](https://huggingface.co/DracenW/TinyLlama-1.1B-EPR-Elastic) · [📐 Math Details](docs/MATH.md) · [🗂️ Raw v3.0 Data](benchmarks/raw_v3/README.md) · [🏷️ v3.0 Release](https://github.com/Dracen-W/epr-attention/releases/tag/3.0) · [🗂️ Raw v2.0 Data](benchmarks/raw_v2/README.md) · [🏷️ v2.0 Release](https://github.com/Dracen-W/epr-attention/releases/tag/2.0) · [📊 v1.0 Archive](docs/BENCHMARKS.md) · [🏷️ v1.0 Release](https://github.com/Dracen-W/epr-attention/releases/tag/1.0)

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Hugging Face Weights](https://img.shields.io/badge/%F0%9F%A4%97%20Weights-TinyLlama--1.1B--EPR--Elastic-ffb000)](https://huggingface.co/DracenW/TinyLlama-1.1B-EPR-Elastic)
[![9-Pillar Audit](https://img.shields.io/badge/9--Pillar%20Audit-0.00e+00%20Leak-brightgreen)](audit_all.py)
[![Release: v3.0](https://img.shields.io/badge/Release-v3.0%20(1.3M%20%E2%86%92%201.1B)-0284c7)](https://github.com/Dracen-W/epr-attention/releases/tag/3.0)

</div>

> 💡 **Quick naming confession**: Some variable names in the code (`wormhole`, `proper_time`, `vacuum`) are leftover 2 AM whiteboard nicknames—please don't take them literally. Zero physics mysticism inside; under the hood it is just distance-weighted attention, learned decay, an attention sink, and KV-page recycling.

## Core Highlights

- **Fewer Parameters, Faster Learning**: Trained from scratch on identical `FineWeb-Edu` data at `1.1B`, EPR uses **`22.17M` fewer parameters** (~`20%` smaller `Q/K` routers) yet reaches **`26.7%` lower validation perplexity** (`225.38` vs. `307.32`) at **`97.2%` of FlashAttention-2 speed**.
- **`72.1%` Smaller Serving KV Cache**: Includes a custom `@triton.jit` paged decoding kernel that recycles old KV pages on local heads while keeping global heads intact—shrinking `1.1B` continuous-batching KV pools from `1,357 MB` (Standard PagedAttention) to **`379 MB`**.
- **Faster `8K` Training & Smooth Extrapolation**: `FlexAttention` block-skipping speeds up `8K` Fwd+Bwd by **`1.63×`**, while preventing standard RoPE's perplexity cliff at `2×–4×` longer contexts.
- **47-Minute Retrofit for Existing Open Models**: Converts off-the-shelf `TinyLlama-1.1B` weights in place (`-22.2M` params; weights at [`DracenW/TinyLlama-1.1B-EPR-Elastic`](https://huggingface.co/DracenW/TinyLlama-1.1B-EPR-Elastic)); after just 47 minutes (`9.8M` tokens) on one L4 GPU, it retains **`93.9%`** of `3T` 5-benchmark accuracy and drops `4K` extrapolated perplexity from `13,399` down to **`14.69`**.
- **Standard Head Shapes & Built-In Anti-Cheat Audit**: Fits cleanly into standard `64` / `128` GPU head dimensions with zero `V` padding, verified by [`audit_all.py`](audit_all.py) (`0.00e+00` future leakage).

---

## Experimental Roadmap: Understanding the Three Evaluation Tracks

> [!IMPORTANT]
> **Read Before Comparing Tables — Different Tracks Answer Different Questions:**
> To evaluate EPR thoroughly under single-GPU (`NVIDIA L4 22 GiB`) constraints, we run **three distinct experimental tracks**. **Do not conflate Track A and Track B**, as their training histories differ by `305,000×`:
> - **Track A — Equal-Budget From-Scratch Pretraining (Fair Architectural Capacity Test)**: Both `Pure Llama` and `EPR` start from random initialization and train on the **exact same token stream and step count**. *Use Track A (and Section 2's 3-seed `6M` suite) to judge whether EPR learns better than standard RoPE / `FoX` under equal compute.*
> - **Track B — Post-Training Checkpoint Surgery + Ultra-Short Healing (Retrofit Feasibility Test)**: Starts from an off-the-shelf `TinyLlama-1.1B` trained on **`3,000B` (`3T`) tokens**. Control `base_1b_cpt` keeps 100% of those `3T` weights intact (`0` cut), whereas `epr_elastic_1b_cpt` **deletes `-22.17M` `3T`-trained `Q/K` weights (`25%` of channels)**, inserts randomly initialized EPR geometry/Wormhole parameters, and heals for only **`0.0098B` tokens (`47` min)**. *Use Track B to judge low-cost retrofit feasibility and `4K` extrapolation rescue—not equal-compute short-context ceiling.*
> - **Systems Track — Kernel Throughput & Physical Serving Memory**: Measures allocated physical KV-cache memory (`MB ↓`) and GPU kernel throughput (`tokens/s ↑`) on `1.1B GQA`.

![EPR v3.0 1.1B Pretraining, Triton PagedAttention, 8K Sparse BlockMask & TinyLlama-1.1B Surgery](docs/figures/v3_1b_and_paged_industrial.png)

**Overview Table Guide**: Summarizes headline outcomes across all tracks on NVIDIA L4 (`bfloat16`). Lower is better (`↓`) for Perplexity (`PPL`) and Memory (`MB`); higher is better (`↑`) for Accuracy (`%`), Throughput (`tok/s`), and Speedup (`×`).

| Track & Evaluation Scenario | Baseline (`FA2` / `PagedAttn` / `TinyLlama`) | **EPR + Mass-Elastic RBF (`v3.0`)** | Primary Metric Direction & Outcome |
| :--- | :---: | :---: | :--- |
| **Track A: `1.1B` From-Scratch `FineWeb-Edu`** *(Equal-token fair test)* | `Val PPL = 307.32` (`1,034.5M`, `924.3s`) | **`Val PPL = 225.38`** (`1,012.3M`, **`913.7s`**) | `PPL ↓`: **`-26.7%` lower PPL**, **`-22.17M` params**, overtakes 300-step `FA2` before Step 200 at **`97.2%` `FA2` speed** |
| **Track A: `1.1B` From-Scratch `4×` Extrap** *(`T_train=1024 → [2K:4K]`)* | `PPL = 358.08` | **`PPL = 236.07`** | `PPL ↓`: **`-34.1%` lower `4×` zero-shot extrapolation PPL** |
| **Systems: `1.1B` Continuous-Batch KV Pool** *(`B=16`, lens up to `8192`)* | `1,357.1 MB` *(Paged)* / `2,816.0 MB` *(Static)* | **`379.0 MB`** *(`@triton.jit`, `W=128+1 WH`)* | `Memory ↓`: **`-72.1%` vs. Standard PagedAttention** (`-86.5%` vs. Static), `0.00e+00` `bf16` error |
| **Systems: `1.1B` `8K` Sparse `BlockMask`** *(`T=8192`, Fwd+Bwd)* | `159,765 tok/s` *(Dense EPR Flex)* | **`261,076 tok/s`** *(`W=512 + Full WH`)* | `Speed ↑`: **`1.63×` sub-quadratic Fwd+Bwd speedup** at `8K` |
| **Controlled `6M` 3-Seed Suite** *(Equal-token fair test, `WikiText-2`)* | `Llama 2: 47.68` / `FoX: 44.52` | **`37.80 ± 0.07`** *(`4×` Extrap `34.21`)* | `PPL ↓ / Recall ↑`: **`-20.7%` vs. Llama 2**, **`-15.1%` vs. `FoX`**, **`+6.1%` `@1536` recall** |
| **Track B: `TinyLlama-1.1B` Retrofit Surgery** *(`-22.2M` cut + `0.00033%` tok)* | `4K PPL: 13,399` · `5-Task: 61.80%` *(3T)* | **`4K PPL: 14.69` (`14.71 @W=256`)** · **`58.05%`** | `Extrap PPL ↓ / Acc ↑`: **`912×` lower `4K` PPL** (`-71.9%` KV) & **`93.9%` of `3T` accuracy healed in 47 min** |

---

## How It Works (Plain Overview) & Parameter-Saving Formula

Instead of adding extra modules outside standard attention, **EPR Attention** ([`epr_attention.py`](epr_attention.py), [`epr_paged_attention.py`](epr_paged_attention.py), [`hf_epr.py`](hf_epr.py)) packs four simple mechanisms into a single standard `head_dim` (`D = 32, 64, 128`) dot product (full derivations in [`docs/MATH.md`](docs/MATH.md)):

1. **Shared Distance Subspace (`d_r + d_g + 2 = D`)**: Splits each head into directional channels (`d_r`), a shared low-dimensional L2/RBF distance subspace (`d_g`, with key-adaptive width), and `2` norm/bias slots—fitting exact power-of-2 GPU Tensor Core dimensions with zero `V` padding.
2. **Zero-Value Attention Sink**: Adds a learnable silent slot (zero output vector) so heads can stay quiet on irrelevant context—folded directly into Triton online-Softmax initial registers with zero extra memory reads.
3. **Local Decay Heads + `1` Global Head per Layer**: Most heads apply RoPE + content-driven decay (enabling smooth extrapolation, `BlockMask` tile-skipping, and $O(1)$ KV-page recycling), while one head group per layer uses `NoPE` + zero decay to preserve long-range retrieval.

### Exact Matrix Parameter-Saving Formula
Because Query reuses the shared distance coordinates `z` and the `+2` norm slots need no projection weights, EPR strictly reduces `Q/K` matrix parameters relative to standard MHA/GQA:

$$\Delta P_{\text{matrix}} = - L \cdot C \cdot \big[\, H(d_g + 1) + H_{\text{kv}} + H_{\text{wh}} \,\big]$$

*(Here $L$ = layers, $C$ = hidden size, $H$ = query heads, $H_{\text{kv}}$ = KV heads, $H_{\text{wh}}$ = global `NoPE` query heads, and $d_g$ = shared distance dim: `6` at `D=32`, `14` at `D=64`, `30` at `D=128`.)*

> [!IMPORTANT]
> **Verification Boundary (`1.3M → 1.1B` Empirical vs. `> 1.1B` Theoretical):**
> - **Empirically Verified (`1.3M` → `1.1B`)**: Verified to the exact integer across discrete tiers: **`1.3M` MHA** (`-16,896`), **`6.0M` MHA** (`-99,840`), **`28M` MHA** (`-396,288`), and **`1.1B` GQA** (`22L, C=2048, 32Q/4KV` $\implies$ **`-22,167,552`** matrix params; `-22,166,760` net delta after `+792` 1D scalars).
> - **Theoretical Scope (`> 1.1B`)**: Larger models (`7B+`) have **not yet been empirically trained**; however, because this equation is a direct linear-algebraic count of tensor shapes, **it holds strictly in theory at arbitrary scale**.

> **Attribution:** Built on Andrej Karpathy's [`llama2.c`](https://github.com/karpathy/llama2.c) (MIT) and Meta Llama 2 (milestones: [`v2.0` Release](https://github.com/Dracen-W/epr-attention/releases/tag/2.0) · [`v1.0` Release](https://github.com/Dracen-W/epr-attention/releases/tag/1.0); legacy `export.py / run.c` does not yet support EPR).

**Section Summary**: Packing shared distance scoring and attention sinks into `d_r + d_g + 2 = D` preserves standard GPU head dimensions while trimming ~`20%` of `Q/K` router weights. Empirical verification currently covers `1.3M` to `1.1B`; `7B+` training behavior awaits multi-GPU testing.

---

## 1. `v3.0` Billion-Scale (`1.1B`) & Industrial Systems Benchmarks (Raw Data in [`benchmarks/raw_v3/`](benchmarks/raw_v3/README.md))

### 1.1 Track A (Equal-Budget From-Scratch): `1.1B` Pretraining & Extrapolation on `FineWeb-Edu`
**What this table tests & how to read it**: A strictly fair, equal-compute comparison from random initialization on streamed `FineWeb-Edu` (`22L, dim=2048, 32Q/4KV`, `300 steps × 16,384 tokens/step = 4.92M tokens`, `T_train = 1024`, `torch.compile` on NVIDIA L4). **Lower is better (`↓`)** for Step Time, Wall-Clock Time, and Perplexity (`Val PPL` and `4K` Extrapolation buckets); **higher is better (`↑`)** for Throughput (`tok/s`).

| Model (`1.1B` From-Scratch) | Matrix Params (`↓`) | Step Time (`↓`) / Speed (`↑`) | Peak VRAM | Val PPL `@100` (`↓`) | Val PPL `@200` (`↓`) | Final Val PPL `@300` (`↓`) | `4K` Extrap `[0:1K] / [1K:2K] / [2K:4K]` (`↓`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`pure_llama_1b` (`FA2` Baseline)** | `1,034,420,224` | `2808.6 ms` (`5,833 tok/s`) | `10.90 GiB` | `607.46` (`357s`) | `364.09` (`641s`) | `307.32` (`924.3s`) | `287.34 / 315.58 / 358.08` |
| **`epr_elastic_1b` (Ours)** | **`1,012,252,672` (`-22.17M`)** | `2890.1 ms` (**`5,669 tok/s`, `97.2%` FA2**) | `11.03 GiB` | **`512.71`** (`330s`) | **`285.79` (`622s`)** | **`225.38` (`913.7s`, `-26.7%`)** | **`205.04 / 227.27 / 236.07` (`-34.1%`)** |

*Key reading*: At `step 200` (`622s`), `epr_elastic_1b` (`285.79`) has already surpassed `pure_llama_1b`'s `step 300` final perplexity (`307.32` at `924.3s`), using `33.3%` fewer tokens and `32.7%` less wall-clock time.

---

### 1.2 Track B (Post-Training Checkpoint Surgery): `TinyLlama-1.1B-3T` Retrofit, 5 Zero-Shot Benchmarks & `4K` Extrapolation
**What this table tests & how to read it**: Evaluates whether an existing `3T`-token pretrained checkpoint (`TinyLlama-1.1B-intermediate-step-1431k-3T`) can be surgically converted to EPR ([`upgrade_hf_llama_to_epr()`](hf_epr.py), cutting `-22.17M` weights) and healed with only **`300` CPT steps (`9.83M` `FineWeb-Edu` tokens = `0.00033%` of `3T`, 47 min on L4)**. **Higher is better (`↑`)** on the 5 short-context zero-shot accuracy benchmarks (`400` validation examples/task, `*` = standard length-normalized `acc_norm`); **lower is better (`↓`)** on `4K` zero-shot extrapolation Perplexity (`T_train = 2048 → L = 4096`).

| Model Variant | Total Params | `Q/K` Router Training History | `ARC-Easy` (`↑`) | `HellaSwag*` (`↑`) | `WinoGrande` (`↑`) | `PIQA*` (`↑`) | `BoolQ*` (`↑`) | **5-Task Avg (`↑`)** | `4K` Extrap `[2K:3K]` / `[3K:4K]` (`↓`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`TinyLlama-1.1B` (`3T` Anchor)** | `1,100.05M` | `100%` trained on `3,000B` tok *(intact)* | `62.25%` | `53.50%` | `55.25%` | `73.00%` | `65.00%` | `61.80%` | `1,444.35` / `12,829.90` |
| **`base_1b_cpt` (Intact Control)** | `1,100.05M` | `100%` intact (`0` cut), `3,000.01B` tok | `60.75%` | `52.75%` | `59.50%` | `74.00%` | `65.25%` | `62.45%` | `1,760.35` / `13,399.39` |
| **`epr_elastic_1b_cpt` (Surgery)** | **`1,077.88M` (`-22.2M`)** | `25%` `Q/K` cut; **new EPR only `0.01B` tok** | **`57.75%`** | **`47.00%`** | **`53.50%`** | **`70.75%`** | **`61.25%`** | **`58.05%` (`93.9%` of 3T)** | **`12.26` / `14.69` (`912×` lower!)** |
| **`epr_elastic_1b_cpt` (`W=256` Evict)** | **`1,077.88M` (`-22.2M`)** | Same checkpoint, **`71.9%` KV evicted `@4K`** | — | — | — | — | — | — | **`12.36` / `14.71` (`+0.02` vs. Full)** |

> [!NOTE]
> **Why `base_1b_cpt` is higher on short-context MC (`62.45%` vs. `58.05%`) while `epr_elastic_1b_cpt` is `912×` better on `4K` extrapolation:**
> 1. **Purpose of `base_1b_cpt` (Data Control)**: `base_1b_cpt` removes **zero weights** and trains on the exact same `300` `FineWeb-Edu` steps. Its `[3072:4096]` PPL still explodes (`13,399.39`), confirming that **`epr_elastic_1b_cpt`'s `14.69` `4K` extrapolation comes from the EPR architecture, not from the `FineWeb-Edu` CPT data**.
> 2. **Asymmetric Training History (`3,000.01B` vs. `0.01B` tokens on new attention organs)**: The 5 zero-shot benchmarks are short-context (`<300` tokens) tasks where `base_1b_cpt` uses 100% intact `3T`-trained `Q/K` matrices, whereas `epr_elastic_1b_cpt` discarded `22.17M` `3T`-trained weights and trained its newly inserted Euclidean-RBF (`z_base`), proper-time (`w_phys`), and NoPE Wormhole parameters on **only `0.0098B` tokens (`305,000×` fewer tokens, zero distillation)**. Retaining **`93.9%`** of `3T` accuracy in 47 minutes confirms rapid surgical healing; closing the remaining `~3.8%` gap—and waking up newly initialized NoPE Wormhole heads (`v0_init=+1.8`) on exact 5-digit verbatim `Passkey` copying (`0.0%` at `300` generic CPT steps vs. `100%/12.5%/0.0%` on `base_1b_cpt` `@1K/@2K/@3.5K`)—would require a larger CPT budget (`e.g., 1B+` tokens) or retrieval mixtures.

---

### 1.3 Systems Track: `@triton.jit` Asymmetric PagedAttention & `8K` Sub-Quadratic `BlockMask` (`1.1B GQA`)
**What these tables test & how to read them**: Left/top measures physical GPU memory (`MB ↓`) allocated by KV-cache pools during continuous batching (`B=16` concurrent variable-length sequences `260..8192`, avg `3944`, `page_size=16`) on `1.1B GQA` (`22L, 32Q/4KV, D=64`). Bottom measures compiled `FlexAttention` Fwd+Bwd attention throughput (`tok/s ↑`) and speedup (`× ↑`) when skipping distant tiles outside `W_damped=512`.

| KV-Cache Serving Strategy (`1.1B GQA`, `B=16`, up to `8192`) | Allocated Physical KV Memory (`MB ↓`) | Memory Saved vs. Static (`↑`) | Memory Saved vs. Std Paged (`↑`) |
| :--- | :---: | :---: | :---: |
| **Contiguous Static Pre-Allocation (`max=8192`)** | `2,816.00 MB` | `0.0%` | — |
| **Standard PagedAttention (`4 KV` heads, full history)** | `1,357.12 MB` | `51.8%` | `0.0%` |
| **EPR Asymmetric PagedAttention (`W=128` Damped + `1` Full WH)** | **`379.02 MB`** | **`86.5%`** | **`72.1%` (`3.58×` smaller)** |

*(22-layer fused `@triton.jit` `_epr_paged_decode_kernel` latency at `B=8`, constant `8` damped pages/seq: `@1K` `1.381 ms` / `5,793 tok/s` · `@2K` `2.497 ms` / `3,204 tok/s` · `@4K` `4.823 ms` / `1,659 tok/s` · `@8K` `9.462 ms` / `845 tok/s`.)*

| Sequence Length `T` (`1.1B GQA` Fwd+Bwd) | Dense EPR `FlexAttention` (`tok/s ↑`) | Sub-Quadratic `BlockMask` `W=512` (`tok/s ↑`) | Sparse Speedup (`× ↑`) |
| :--- | :---: | :---: | :---: |
| `T = 1024` (`B=8`) | `375,414 tok/s` | `376,158 tok/s` | `1.00×` |
| `T = 2048` (`B=4`) | `310,138 tok/s` | `341,473 tok/s` | `1.10×` |
| `T = 4096` (`B=2`) | `236,032 tok/s` | `309,781 tok/s` | `1.31×` |
| **`T = 8192` (`B=1`)** | `159,765 tok/s` | **`261,076 tok/s`** | **`1.63×`** |

**Section 1 Summary (`1.1B` & Industrial Systems)**: At `1.1B` scale, controlled equal-token training from scratch (Track A) indicates that EPR + Mass-Elastic RBF improves early-to-mid pretraining sample efficiency (`-26.7%` Val PPL) and `4×` extrapolation (`-34.1%` PPL) at near-native (`97.2%`) `FlashAttention-2` throughput while using `22.17M` fewer parameters—though full multi-hundred-billion-token `1.1B` pretraining runs remain future work. For post-training conversion of existing checkpoints (Track B), `47` minutes (`9.83M` tokens) of continued pretraining is sufficient to eliminate `4K` RoPE collapse (`14.69` vs. `13,399`) and recover `~94%` of `3T` short-context benchmark performance, while full parity on short-context benchmarks and activation of verbatim Wormhole copy circuits require longer continued pretraining. On the systems side, physical page recycling and `BlockMask` sparsity provide substantial KV memory (`-72.1%`) and `8K` training throughput (`1.63×`) reductions.

---

## 2. Controlled `v2.0` Multi-Seed & Ablation Suite (`6.0M`, Raw Data in [`benchmarks/raw_v2/`](benchmarks/raw_v2/README.md))

![EPR v2.0 6M Convergence, Iso-Wallclock, Extrapolation & Needle Recall](docs/figures/v2_6m_convergence_and_extrap.png)

### 2.1 Three-Seed Equal-Budget Statistical Summary (`seeds = 1337, 2026, 42`, `Mean ± Std`)
**What this table tests & how to read it**: Multi-seed equal-token from-scratch training on `WikiText-2` (`dim=256, 6L, 4H, T_train=512, 3000 steps = 49.15M tokens`), testing statistical reproducibility across random seeds. **Lower is better (`↓`)** for Parameter Count and all Perplexity (`PPL`) columns; **higher is better (`↑`)** for Passkey Needle retrieval accuracy (`%`, 200 trials/seed).

| Architecture (`Mean ± Sample Std`, 3 seeds) | Params (`↓`) | In-Domain `Val PPL@512` (`↓`) | Bucket `[256:512]` (`↓`) | `2×` Extrap `[512:1024]` (`↓`) | `4×` Extrap `[1024:2048]` (`↓`) | Needle `@512` (`↑`) | Needle `@1024` (`↑`) | Needle `@1536` (`↑`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Pure Llama 2 (RoPE)** | `5.87M` | `47.68 ± 0.51` | `46.44 ± 0.66` | `97.59 ± 11.10` | `168.41 ± 18.90` | `25.28 ± 0.46%` | `13.89 ± 0.86%` | `13.86 ± 0.44%` |
| **FoX (Forgetting Transformer, ICLR'25)** | `5.87M` | `44.52 ± 0.25` | `43.57 ± 0.42` | `42.93 ± 0.36` | `42.22 ± 0.33` | `26.14 ± 0.21%` | `26.28 ± 0.46%` | `25.82 ± 0.23%` |
| **EPR v2.0 Distilled (`bilat=F, nest=F`)** | **`5.77M`** | **`37.80 ± 0.07`** | **`35.58 ± 0.11`** | **`34.80 ± 0.14`** | **`34.21 ± 0.26`** | **`34.95 ± 0.96%`** | **`33.07 ± 0.59%`** | **`31.89 ± 0.39%`** |
| **EPR v2.0 + Mass-Elastic RBF** | **`5.77M`** | `37.95 ± 0.23` | `35.71 ± 0.32` | `34.96 ± 0.27` | `34.39 ± 0.35` | `34.65 ± 0.20%` | `32.90 ± 0.17%` | **`31.88 ± 0.86%`** *(peak `32.58%`)* |

*(Additional `seed=1337` baselines: **Dynamic NTK** `[1024:2048] PPL = 52.22`; **YaRN** `126.63`; **ALiBi** `42.60`, `Needle@1536 = 23.13%`. Full 8-variant ablations, OOD overconfidence probes (`ΔP_max = -1.55%` vs. `+8.06%` on Llama 2), and `W=128` KV eviction tables are archived in [`v2.0` Release](https://github.com/Dracen-W/epr-attention/releases/tag/2.0) and [`v2_6m_ood_and_kv_eviction.png`](docs/figures/v2_6m_ood_and_kv_eviction.png).)*

**Section 2 Summary (`6.0M` Controlled Suite)**: Across three random seeds at `6.0M` scale, EPR consistently lowers in-domain and `4×` extrapolated perplexity relative to standard RoPE and `FoX` with tight seed variance, while confirming that when trained from scratch, the intra-layer NoPE Wormhole head actively learns distance-invariant `Passkey` retrieval (`~31.9%` at `3×` training length vs. `25.8%` on `FoX` and `13.9%` on RoPE) and tolerates `W=128` damped-head KV eviction with negligible (`+0.05`) perplexity drift.

---

## 3. Convergent 2025–2026 Literature Positioning

- **Content-Driven Proper-Time Decay ($\tau_i - \tau_j$)**: Mathematically isomorphic to **FoX (Forgetting Transformer, ICLR 2025)** when $M = I$; EPR pairs $\tau$ with a shared Euclidean-RBF capture basin and an intra-layer NoPE Wormhole head (`-15.1%` lower `4×` PPL and `+6.1%` higher `@1536` recall than FoX at `6M`).
- **Zero-Value Vacuum Floor ($e^{v_0}, V_{\text{vac}} = \mathbf{0}_D$)**: Related to **Softmax1** (Miller, 2023), **OpenAI `gpt-oss`** learnable sinks (2025), and **ICML 2026** *SinkProbe* analyses, while folding into Triton online-Softmax registers with zero extra memory reads.
- **Intra-Layer Single-Kernel Dual-Regime vs. Layer-Interleaved Hybrids**: Unlike **Llama 4 (`iRoPE`)**, **Gemma 3/4**, **DuoAttention (ICLR 2025)**, or **Qwen3.5 / Kimi Linear**, EPR combines damped local heads and NoPE global heads **inside every layer within one $D$-aligned attention kernel**, avoiding dual RNN+Attention operator stacks while trimming `~20%` of `Q/K` router parameters.

**Section 3 Summary (Positioning)**: Several individual ingredients in EPR converge with concurrent 2025–2026 mechanisms (`FoX` forget gates, learnable attention sinks, local/global head specialization). EPR's distinct engineering and mathematical contribution is unifying **shared heteroscedastic Euclidean-RBF + intra-layer damped/Wormhole topology + zero-read vacuum PagedAttention** inside a single power-of-2 $D$-dimensional inner product that saves parameters rather than adding auxiliary router overhead.

---

## 4. Quick Start

```bash
# 1. Run 9-pillar mathematical, causality, stateful/paged KV-cache, HF wrapper & CUDA/Triton audit
python3 audit_all.py

# 2. v3.0 Industrialization & 1.1B Suite (CUDA GPU: Triton PagedAttention, 8K Sparse BlockMask & 1.1B Benchmarks)
python3 bench_paged_and_sparse.py
python3 eval_1b_industrial.py

# 3. Controlled 6M research & ablation suite (WikiText-2)
python3 train.py config/train_wikitext2_colab_scale.py --out_dir=out-6m-epr-elastic --use_epr=True --use_elastic_rbf=True
python3 eval_extrapolation.py out-6m-epr-elastic --lengths=256,512,1024,2048
python3 eval_kv_eviction.py out-6m-epr-elastic
```

---

## 5. Origin, Literature Attribution & AI Collaboration Disclosure

- **Independent Conceptual Origin**: All core conceptual hypotheses in this project—evolving from neurochemical dissipation and spatial potential fields to bounded Euclidean capture basins, mass-elastic capture radii, content-driven proper-time damping, zero-value physical vacuum floors, and intra-layer zero-decay wormhole heads—were **conceived and reasoned independently from first principles by the author prior to reading or consulting the concurrent/prior papers cited in Section 3** (documented in a chronological ideation archive).
- **Post-Hoc Literature Credit**: All mathematical equivalences and convergent design relationships with prior and 2025–2026 concurrent literature (`Softmax1`, `FoX`, `DuoAttention`, `iRoPE / RNoPE / Gemma 4`, `gpt-oss`, `ICLR/ICML 2026` sink & scaling analyses) were identified during subsequent AI-assisted literature auditing and are explicitly cited in [`CITATION.bib`](CITATION.bib).
- **Transparent AI Collaboration**: The human author originated the conceptual architecture, steered every pruning/refinement hypothesis, and directed experimental validation; AI assistants (`Gemini` / `Antigravity`) formalized the exact `d_r + d_g + 2 == D` algebraic folding and closed-form parameter law, implemented the PyTorch/`FlexAttention`/`@triton.jit` PagedAttention & HuggingFace surgery codebase and 9-pillar audit suite, conducted literature cross-checks, and executed automated GPU benchmarks.
- **License**: MIT License (preserving Andrej Karpathy's original `llama2.c` copyright; see [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE)).

```bibtex
@misc{dracenw2026epr,
  title        = {EPR Attention: A Geometry-Folded, Parameter-Saving Long-Context Attention Architecture},
  author       = {Dracen-W},
  year         = {2026},
  howpublished = {\url{https://github.com/Dracen-W/epr-attention}},
  note         = {Built on karpathy/llama2.c and the Meta Llama 2 architecture}
}
```
