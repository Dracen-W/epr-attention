<div align="center">

# EPR Attention（欧氏 RBF · 固有时间 · 共振纤维注意力机制）

**基于 `llama2.c` 构建的实验性长上下文轻量注意力架构**

[English](README.md) | **简体中文** · [📐 数学推导与结构](docs/MATH.md) · [📊 v1.0 旧版基准档案](docs/BENCHMARKS_zh.md) · [🏷️ v1.0 存档 Tag](https://github.com/Dracen-W/epr-attention/releases/tag/1.0)

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Built with Meta Llama 2](https://img.shields.io/badge/Architecture-Meta%20Llama%202-blueviolet)](NOTICE)
[![8-Pillar Audit](https://img.shields.io/badge/8--Pillar%20Audit-0.00e+00%20Leak-brightgreen)](audit_all.py)

</div>

> **归属与适用范围声明：** 本项目基于 Andrej Karpathy 的 [`llama2.c`](https://github.com/karpathy/llama2.c)（MIT License）与 **Meta Llama 2** 架构（Touvron et al., 2023）构建。仓库中的 `export.py`、`run.c`、`runq.c` 继承自原版 `llama2.c`，**暂不支持 EPR 架构导出与 C 端推理**（当前仅支持 PyTorch）。完整许可见 [`LICENSE`](LICENSE) 与 [`NOTICE`](NOTICE)。

> **物理启发声明：** EPR Attention 的双态头设计与几何打分结构**仅受 ER=EPR 时空折叠与空间引力场的直觉启发（inspired by）**，并非从广义相对论或量子引力方程推导。所有模块均已抽象为标准可微 PyTorch 原语，并精确代数折叠进单次 $D$ 维对齐的注意力核（完整证明见 [`docs/MATH.md`](docs/MATH.md)）。

**EPR Attention**（独立零依赖核心见 [`epr_attention.py`](epr_attention.py)）由三个经 6M 消融证实的核心原语构成：
1. **欧氏 RBF + 共振纤维正交打包（`d_r + d_g + 2 == head_dim`）**：将低维潜在坐标的欧氏 RBF 势与定向点积代数折叠进单次硬件对齐计算，**`V` 通道零补零**，且总参数严格少于原生 Llama 2。
2. **可学习真空槽（$e^{v_0}$）**：等价于 Softmax1 / Attention Sink，允许注意力头在无关上下文上保持静默。
3. **双态头拓扑**：阻尼头采用 RoPE + 数据依赖累积衰减（$\tau_i - \tau_j$），保证稳定外推与 $O(1)$ 滑窗 KV 驱逐；**虫洞头**采用 **NoPE + 零衰减（$\delta\tau \equiv 0$）+ 高真空阈值（$v_0 \approx +2.5$）**，负责跨序列距离的远古信息召回。

---

## 一、 坦诚的算力账：单步开销、同 PPL 效率与显存

我们如实汇报小模型上的单步耗时劣势，并同时给出同 PPL 实际算力账与大模型吞吐：

| 对比维度（NVIDIA L4, `bfloat16`） | Pure Llama 2 (FA2) | EPR Attention | 实测结论 |
| :--- | :---: | :---: | :--- |
| **6M `d0` 单步稳态耗时 (`T=512, B=32`)** | `32.85 ms` (`499k tok/s`) | `53.70 ms` (`305k tok/s`) | 6M 下**单步慢 `1.63×`**（小模型受限于核启动延迟，MFU 约 `4%`） |
| **6M `d0` 同 PPL 算力账 (`Val PPL ≤ 57.7`)** | `1,500` 步 (`49.3 s`) | **`750` 步 (`39.1 s`)** | **步数/语料省 `50.0%`**，**纯 GPU 训练时间少 `20.7%`**，理论 FLOPs 省 **`50.8%`** |
| **6M `d0` 语料记忆极限 (`Step 3000 Train Loss`)** | `2.5766` (PPL `13.15`) | **`2.2618` (PPL `9.60`)** | 少 `9.4 万` 参数，Train PPL 反而低 **`27.0%`** |
| **1.1B `FlexAttention` + 原生 GQA (`T=1024`)** | `3,504 tok/s` (训练)<br>`18,569 tok/s` (`Prefill@2k`) | **`3,399 tok/s` (`97.0%`)**<br>**`18,765 tok/s` (`101.1%`)** | 1.1B 规模下小算子延迟被大矩阵乘淹没（少 `2,072 万` 参数，显存 `10.32` vs `10.51 GB`） |

> **后端触发限制说明：** `FlexAttention` 融合核仅在 CUDA `bf16/fp16`、`T >= 256` 且 **`dropout == 0.0`** 时自动启用；开启 `dropout > 0`（如小数据集防过拟合训练）时会回退至带 $O(T^2)$ 掩码的 `SDPA`。

---

## 二、 6M 核心实测（无稀释分段外推、200 次召回与 KV 驱逐代价）

*以下均在 `6.0M` WikiText-2（`dim=256, 6L, 4H, T_train=512`，EPR `5.78M` vs Llama 2 `5.87M`）上测得。括号内 `ΔNLL` 表示“存在远古针 vs 无针干扰上下文”时目标 token 的负对数似然改善量。*

### 2.1 消除窗口稀释后的分段外推 PPL 与 200 次长程召回
| 模型 | `[0:256]` | `[256:512]` | `[512:1024]` | `[1024:2048]` | 捞针 `Dist=1536` (200次) | 双键关联消歧 `Dist=1024 / 1536` (盲猜=50%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Pure Llama 2** | `49.51` | `46.80` | `84.77` | `146.90` | `14.6±1.4% (+0.11)` | `77.5±5.8% (+1.74)` / `73.5±6.1% (+1.42)` |
| **EPR Attention** | **`40.30`** | **`35.36`** | **`34.53`** | **`34.06`** | **`31.9±1.9% (+0.48)`** | **`88.0±4.5% (+2.84)` / `81.0±5.4% (+2.24)`** |

*（同配置 `v1.0` SOTA 对照：**FoX (ICLR'25)** 为 `PPL@512 = 44.88`、`PPL@2048 = 43.05`、`Dist=1536` 捞针 `23.7% (+0.00)`，详见 [`docs/BENCHMARKS_zh.md`](docs/BENCHMARKS_zh.md)。）*

### 2.2 非对称 `EPRKVCache` 滑窗驱逐的质量代价（`eval_kv_eviction.py`）
仅对阻尼头（`0..H-2`）做固定窗口 `W` 驱逐、虫洞头（`H-1`）保留全长上下文：

| 阻尼头窗口 `W` | `@2K` 省 KV | `@8K` 省 KV | 纯尾部外推 `[1024:2048]` | 捞针 `Dist=1536` (200次) |
| :--- | :---: | :---: | :---: | :---: |
| `Full (∞)` | `0.0%` | `0.0%` | `36.06` | `30.2±1.8% (ΔNLL +0.44)` |
| `W = 256` | `65.6%` | `72.7%` | `36.07` (`+0.01`) | `30.2±1.8% (ΔNLL +0.44)` |
| **`W = 128`** | **`70.3%`** | **`73.8%`** | **`36.10` (`+0.04`)** | **`30.1±1.8% (ΔNLL +0.44)`** |
| `W = 64` | `72.7%` | `74.4%` | `36.30` (`+0.24`) | `30.5±1.8% (ΔNLL +0.44)` |

### 2.3 6M 完整单开关消融总表
| 消融变体 | `PPL@512` | 尾部 `[1024:2048]` | 捞针 `Dist=1536` | 双键消歧 `Dist=1024` | 裁定 |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Full EPR (`v6`)** | **`37.73`** | **`34.06`** | **`31.9% (+0.48)`** | `88.0% (+2.84)` | 完整版 |
| `w/o 真空槽` (`-vac`) | `40.76` (`+3.03`) | `37.63` (`+3.57`) | `28.1% (+0.22)` | `88.5% (+2.73)` | **核心支柱 #1（必留）** |
| `w/o 欧氏 RBF` (`-rbf`) | `39.54` (`+1.81`) | `36.55` (`+2.49`) | `28.4% (+0.28)` | `85.0% (+2.73)` | **核心支柱 #2（必留）** |
| `w/o 虫洞头` (`-wh`) | `39.52` (`+1.79`) | `36.35` (`+2.29`) | `24.8% (+0.00)` | `85.0% (+2.65)` | **核心支柱 #3（远距召回唯一命脉）** |
| `w/o 嵌套时间` (`-nest`) | `37.82` (`+0.09`) | `34.14` (`+0.08`) | `31.9% (+0.49)` | `88.5% (+2.86)` | 冗余（可删，免费提速） |
| `w/o 双边质量` (`-bilat`) | `37.75` (`+0.02`) | `33.94` (`-0.12`) | `32.0% (+0.50)` | `89.0% (+2.88)` | 冗余（可删，省参又提速） |

---

## 三、 与已有工作的关系 & 已验证/未验证清单

**与已有文献的关系对照：**
- **数据依赖累积衰减（$\tau$）**：去掉嵌套级联（$M=I$）后与 **FoX（Forgetting Transformer）** 数学同构。
- **物理真空槽（$e^{v_0}$）**：等价于 **Softmax1（Attention Is Off By One）** / 可学习 **Attention Sink**。
- **NoPE 虫洞头**：与 **iRoPE（Llama 4）** 等部分 NoPE 方案相关，区别在于将 NoPE + 零衰减与高真空静默阈值（$v_0 \approx +2.5$）及欧氏 RBF 锁结合。
- **欧氏 RBF 势**：与 L2 / 高斯偏置注意力相关，本实现作用于可学习的低维潜在坐标 $z$ 并精确折叠进单次点积核。

**已验证 / 未验证清单：**
- ✅ **已验证**：8 维审计（`audit_all.py`：`fp64` 教科书等价 `3.33e-16`、零未来泄露 `0.00e+00`、CUDA `bf16` FlexAttention 等价 `1.95e-03` 与零泄露 `0.00e+00`、有状态 `EPRKVCache` 误差 `3.89e-16`）；无稀释分段外推；200 次捞针与双键关联消歧；`W=128` 驱逐零掉点实测；6M 完整消融；1.1B 硬件吞吐。
- ⏳ **尚未验证**：多随机种子方差统计；`≥100M` / `1.1B` 规模的真实精度与下游评测（上方 1.1B 仅为硬件吞吐）；C 端（`run.c` / `export.py`）适配。

---

## 四、 快速上手

```bash
# 1. 运行 8 维全栈审计（fp64 因果 + CUDA bf16 FlexAttention + 有状态 EPRKVCache）
python3 audit_all.py

# 2. 独立零依赖核心模块：直接引用 epr_attention.py（EPRConfig, EPRAttention, EPRKVCache）

# 3. 运行无稀释分段外推、200 次捞针、双键消歧与 KV 驱逐代价实测
python3 eval_extrapolation.py out-6m-base out-6m-v6-wormhole --lengths=256,512,1024,2048
python3 eval_needle.py out-6m-base out-6m-v6-wormhole --distances=64,128,256,512,1024,1536
python3 eval_associative_recall.py out-6m-base out-6m-v6-wormhole
python3 eval_kv_eviction.py out-6m-v6-wormhole
```

---

## 五、 AI 辅助声明、开源协议与引用

- **AI 辅助披露**：本项目在代码工程实现、审计脚本编写、自动化评测与文档整理过程中使用了 AI 辅助；所有架构构想、实验设计与真机验证均由作者主导完成。
- **代码协议**：MIT License（保留 Andrej Karpathy 原始 `llama2.c` 版权，详见 [`LICENSE`](LICENSE) 与 [`NOTICE`](NOTICE)）。
- **引用本项目**：可直接点击仓库右侧 **"Cite this repository"**（[`CITATION.cff`](CITATION.cff)），或复制下方 BibTeX：

```bibtex
@misc{dracenw2026epr,
  title        = {EPR Attention: An Experimental Long-Context Attention Mechanism Built on llama2.c},
  author       = {Dracen-W},
  year         = {2026},
  howpublished = {\url{https://github.com/Dracen-W/epr-attention}},
  note         = {Built on karpathy/llama2.c and the Meta Llama 2 architecture}
}
```
