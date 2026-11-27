<div align="center">

# EPR Attention（欧氏 RBF · 固有时间 · 共振纤维注意力机制）

**面向端侧与更多设备的轻量高密度长上下文架构**

[English](README.md) | **简体中文** · [📊 v1.0 旧版完整基准档案](docs/BENCHMARKS_zh.md) · [🏷️ v1.0 旧版存档 Tag](https://github.com/Dracen-W/llama2-wormhole-gravity/releases/tag/1.0)

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Built with Meta Llama 2](https://img.shields.io/badge/Architecture-Meta%20Llama%202-blueviolet)](NOTICE)
[![Causality Audit](https://img.shields.io/badge/fp64%20Causality%20Audit-0.00e+00%20Leak-brightgreen)](audit_all.py)

</div>

> **归属与协议声明：** 本项目基于 **Meta Llama 2** 架构（Touvron et al., 2023）与 Andrej Karpathy 的 [`llama2.c`](https://github.com/karpathy/llama2.c)（MIT License）构建。完整许可与学术引用见 [`LICENSE`](LICENSE)、[`NOTICE`](NOTICE) 与 [`CITATION.bib`](CITATION.bib)。

> **🚧 速度与内核优化进行中声明：** 下方 1.3M–28M 的全套测试数据均为 **[`v1.0` 旧版基线成绩](https://github.com/Dracen-W/llama2-wormhole-gravity/releases/tag/1.0)**（当时基于未融合 $O(T^2)$ 时间掩码的旧版 PyTorch `SDPA` 测得）。当前仓库正在进行**底层速度与端侧内存优化（`FlexAttention` 融合核 + 原生 GQA + 非对称 `EPRKVCache` 剪枝）**——最新 1.1B 规模 NVIDIA L4 实测中，`Prefill@2k` 推理吞吐已**反超原生 FlashAttention-2（`18,765` vs `18,569 tok/s`，`101.1%`）**，同时参数更少（`-2,072 万`）、峰值显存更低。

**EPR Attention**（**E**uclidean-RBF, **P**roper-time & **R**esonance-fiber Attention，原名统一纤维—引力与 ER=EPR 虫洞注意力）定位于**面向端侧与消费级设备的轻量高密度长上下文模型框架**——用紧凑的物理几何先验替代冗余注意力参数，一举打通长上下文大模型的经典死结：**带时间阻尼的架构外推极稳但容易忘掉超远细节；无阻尼的架构能捞回远处细节但超出训练长度就会外推崩盘。**

---

## 一、 架构原理图解

![EPR Attention 架构图](docs/figures/architecture_wormhole_attention.png)

- **正交纤维—底时空精确切分（`d_r + d_g + 2 == head_dim`）**：将每个头拆分为定向**共振纤维**（$d_r$）与低维**底时空欧氏 RBF**（$d_g$），把双边质量（$M_i M_j$）与真空底噪（$e^{v_0}$）代数折叠进**单次严格 $D$ 维对齐的注意力计算，`V` 通道零补零**。
- **双态时空拓扑**：阻尼**太阳系头**（`Moon → Sat → Earth`）以嵌套动态固有时间处理近中程语感；**ER=EPR 虫洞头**采用 **零时间衰减（$\delta\tau \equiv 0$）+ NoPE + 高真空喉道（$v_0 \approx +2.5$）+ 紧缩底时空 RBF 锁**，平时静默不污染外推，强共振时无视距离瞬间扯回远古“针”。
- **`FlexAttention` + 端侧非对称 KV-Cache（`EPRKVCache`）**：用 Triton `FlexAttention` 彻底消灭 $O(T^2)$ 时间掩码显存墙；推理时对阻尼头做固定滑窗剪枝、仅在虫洞头保留全长 KV，实测可削减 **75%–87.5%** 长上下文 KV-Cache 内存。

---

## 二、 速度优化阶段成果（1.1B 规模，NVIDIA L4 实测）

*基于严格 TinyLlama-1.1B 拓扑（`dim=2048, 22L, 32Q/4KV GQA, D=64, bfloat16`），对比原生 FlashAttention-2、`v1.0` 旧版 SDPA 与新版 `FlexAttention + 原生 GQA`：*

| 1.1B 架构对比 (NVIDIA L4) | 总参数量 | 训练吞吐 (`T=1024`) | `Prefill@2k` 推理 | 峰值显存 | 相对原生 FA2 速度 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **TinyLlama-1.1B (Base, FA2)** | `1100.05M` | `3,504 tok/s` | `18,569 tok/s` | `10.51 GB` | `100.0%` / `100.0%` |
| **EPR-1.1B (`v1.0` 旧版 SDPA)** | `1079.33M` (`-20.7M`) | `1,247 tok/s` | `2,462 tok/s` | `10.37 GB` | `35.6%` / `13.3%` |
| **EPR-1.1B (`FlexAttention` + GQA)** | **`1079.33M` (`-20.7M`)** | **`3,399 tok/s`** (`2.73×`) | **`18,765 tok/s`** (`7.62×`) | **`10.32 GB`** | **`97.0%` / `101.1%` (反超)** |

---

## 三、 [`v1.0` 旧版](https://github.com/Dracen-W/llama2-wormhole-gravity/releases/tag/1.0) 核心基准总览（`1.3M → 28M`）

> 📌 **说明：** 以下为 **[`v1.0` 旧版](https://github.com/Dracen-W/llama2-wormhole-gravity/releases/tag/1.0)** 留档精度成绩（优化前 SDPA 后端，已通过 `float64` 零未来泄露审计）。完整参数矩阵、同耗时 PPL 与样本对比归档于 [`docs/BENCHMARKS_zh.md`](docs/BENCHMARKS_zh.md)。

| 基准任务 (`dim/L/H`, 词表) | 模型 | 衰减权重数 | 训内 PPL (`T=512`) | 4× 外推 (`T=2048`) | `Dist=1536` 捞针 / BLEU |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **6.0M WikiText-2** (`256/6L/4H`, 4K) | Pure Llama 2 / FoX (ICLR'25) | `5.87M` / `5.87M` | `47.98` / `44.88` | `96.23` / `43.05` | `13.5% (+0.09)` / `23.7% (+0.00)` |
| **6.0M WikiText-2** (`256/6L/4H`, 4K) | **EPR (`v1.0` 旧版)** | **`5.77M`** (`-94K`) | **`37.50`** | **`34.72`** (`比 FoX 低 19.3%`) | **`32.3% (+0.59)`** |
| **6.0M TinyStories** (`256/6L/4H`, 4K) | Pure Llama 2 vs **EPR (`v1.0`)** | `5.87M` vs **`5.77M`** | `5.50` vs **`5.20`** | `28.67` vs **`4.90`** (`-82.9%`) | `16.4%` vs **`32.6%`** |
| **6.8M FineWeb-Edu** (`256/6L/4H`, 8K) | Pure Llama 2 vs **EPR (`v1.0`)** | `6.91M` vs **`6.82M`** | `37.52` vs **`34.87`** | `161.27` vs **`32.29`** (`-80.0%`) | `9.6% (+0.13)` vs **`29.2% (+0.55)`** |
| **27.6M RealWorld** (`512/6L/8H`, 16K) | Pure Llama 2 vs **EPR (`v1.0`)** | `27.66M` vs **`27.28M`** | `38.24` vs **`34.95`** | `175.29` vs **`32.26`** (`-81.6%`) | `10.4% (+0.12)` vs **`41.1% (+1.34)`** |
| **6.8M WMT14 英译德** (`256/6L/4H`, 8K) | Pure Llama 2 vs **EPR (`v1.0`)** | `6.91M` vs **`6.82M`** | `36.63` vs **`32.69`** | — | **BLEU: `4.32` vs `5.27` (`+22.0%`)** |
| **7.9M 零样本德法** (`256/6L/4H`, 12K) | Pure Llama 2 vs **EPR (`v1.0`)** | `7.96M` vs **`7.87M`** | `52.33` vs **`47.94`** | — | **De→Fr `+41.1%` / Fr→De `+42.1%`** |

---

## 四、 核心公式与快速上手

$$\Psi_h(i, j) = \underbrace{\frac{q_{\text{res},i}\cdot k_{\text{res},j}}{\sqrt{d_r}}}_{\text{纤维定向共振 (R)}} + \underbrace{u_{\text{recv},i} + u_{\text{emit},j}}_{\text{双边引力质量 } M_i M_j} - \underbrace{\frac{\|z_i - z_j\|_2^2}{2\sigma_h^2\sqrt{d_g}}}_{\text{欧氏 RBF (E)}} - \underbrace{(\tau_{h,i} - \tau_{h,j})}_{\text{固有时间 (P)}}, \qquad W_h(i, j) = \frac{e^{\Psi_h(i, j)}}{e^{v_{0,h}} + \sum_{k \le i} e^{\Psi_h(i, k)}}$$

```bash
# 1. 运行 6 维 float64 防作弊因果与等价性审计（约 3 秒）
python3 run_all_tests.py --mode=audit

# 2. 实测 FlexAttention 加速比与非对称 EPRKVCache 显存压缩（省 75%–87.5%）
python3 bench_epr_speed.py

# 3. 一键运行任意基准（WikiText-2 / WMT14 / 零样本德法 / 28M RealWorld）
python3 run_all_tests.py --mode=wikitext2
```

---

## 五、 开源协议与引用规范

- **代码协议**：MIT License（完整保留 Andrej Karpathy 原始 `llama2.c` 版权声明，详见 [`LICENSE`](LICENSE) 与 [`NOTICE`](NOTICE)）。
- **引用本项目**：可直接点击 GitHub 仓库右侧的 **"Cite this repository"**（[`CITATION.cff`](CITATION.cff)），或复制下方 BibTeX：

```bibtex
@misc{dracenw2026epr,
  title        = {EPR Attention: Euclidean-RBF, Proper-time and Resonance-fiber Attention},
  author       = {Dracen-W},
  year         = {2026},
  howpublished = {\url{https://github.com/Dracen-W/llama2-wormhole-gravity}},
  note         = {Built on karpathy/llama2.c and the Meta Llama 2 architecture}
}
```
