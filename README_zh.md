<div align="center">

# 统一纤维—引力与 ER=EPR 虫洞注意力机制 (v6)

**基于纯正 Llama 2 底座的第一性原理物理长上下文架构**

[English](README.md) | **简体中文** · [📊 完整基准测试与样本档案](docs/BENCHMARKS_zh.md)

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Built with Meta Llama 2](https://img.shields.io/badge/Architecture-Meta%20Llama%202-blueviolet)](NOTICE)
[![Causality Audit](https://img.shields.io/badge/fp64%20Causality%20Audit-0.00e+00%20Leak-brightgreen)](audit_all.py)

</div>

> **归属与协议声明：** 本项目基于 **Meta Llama 2** 架构（Touvron et al., 2023）与 Andrej Karpathy 的 [`llama2.c`](https://github.com/karpathy/llama2.c)（MIT License）构建。完整许可与学术引用见 [`LICENSE`](LICENSE)、[`NOTICE`](NOTICE) 与 [`CITATION.bib`](CITATION.bib)。

**统一纤维—引力与 ER=EPR 虫洞注意力机制（`v6-wormhole`）** 以理论物理中的纤维丛规范—引力耦合、太阳系嵌套时间膨胀与 ER=EPR 虫洞为第一性原理，在**单次硬件对齐 SDPA** 中彻底打通了长上下文大模型的经典死结：**带时间阻尼的架构外推极稳但容易忘掉超远细节；无阻尼的架构能捞回远处细节但超出训练长度就会外推崩盘。**

在**衰减权重比纯正 Llama 2 还少 `1.4% ~ 1.6%`**、且通过 **`float64` 四层防作弊因果审计（未来泄露严格为 `0.00e+00`）** 的前提下，`v6-wormhole` 在 **4 大语料库（`1.3M → 28M`）** 上全面胜出：
- **6M SOTA 对撞（WikiText-2）**：`4×` 外推 PPL 比 **FoX (ICLR 2025)** 低 **19.3%**（`34.72` vs `43.05`），同时在 `Dist=1536` 保持 **`32.3% (+0.59 ΔNLL)`** 超远捞针（ALiBi 与 FoX 全部归零 `+0.00`）。
- **28M 扩参更强（RealWorld 16K）**：训内 PPL 低 **8.6%**（`34.95` vs `38.24`），`4×` 外推低 **81.6%**（`32.26` vs `175.29`），`Dist=1536` 捞针达到 Base 的 **4 倍准确率 / 11 倍 ΔNLL**（`41.1% / +1.34` vs `10.4% / +0.12`），且在**相同 GPU 墙钟时间下完成反超**（[查看同耗时曲线 →](docs/BENCHMARKS_zh.md#iso-wallclock)）。

---

## 一、 架构原理图解

![统一纤维-引力与 ER=EPR 虫洞注意力架构图](docs/figures/architecture_wormhole_attention.png)

- **正交纤维—底时空精确切分（`d_r + d_g + 2 == head_dim`）**：将每个头拆分为定向**共振纤维**（$d_r$）与低维**底时空欧氏 RBF**（$d_g$），把双边质量（$M_i M_j$）与真空底噪（$e^{v_0}$）代数折叠进**单次严格 $D$ 维对齐的 SDPA，`V` 通道零补零**。
- **双态时空拓扑**：阻尼**太阳系头**（`Moon → Sat → Earth`）以嵌套动态固有时间处理近中程语感；**ER=EPR 虫洞头**（`H-1`）采用 **零时间衰减（$\delta\tau \equiv 0$）+ NoPE + 高真空喉道（$v_0 \approx +3.0$）+ 紧缩底时空 RBF 锁**，平时静默不污染外推，强共振时无视距离瞬间扯回远古“针”。
- **阻尼泡利排斥解码**：指数衰减的短程简并压彻底掐死局部复读死循环，同时保留跨段合法实体回扣。

---

## 二、 核心基准测试总览

> 🔍 **深入查看完整数据：** [四大语料全景总表](docs/BENCHMARKS_zh.md#panoramic-summary) · [同步数与同耗时曲线](docs/BENCHMARKS_zh.md#iso-wallclock) · [Scaling Law 与 28M 捞针](docs/BENCHMARKS_zh.md#scaling-law) · [SOTA 强基线对撞](docs/BENCHMARKS_zh.md#sota-showdown) · [真实生成样本盲测](docs/BENCHMARKS_zh.md#samples)

| 规模与语料 | 模型 | 衰减权重数 | 训内 PPL (`T=512`) | 4× 外推 (`T=2048`) | `Dist=1536` 捞针 (3×) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **6.0M / WikiText-2** | Pure Llama 2 / FoX (ICLR'25) | `5.87M` / `5.87M` | `47.98` / `44.88` | `96.23` / `43.05` | `13.5% (+0.09)` / `23.7% (+0.00)` |
| **6.0M / WikiText-2** | **`v6-wormhole` (Ours)** | **`5.77M`** (`-94K`) | **`37.50`** | **`34.72`** (`比 FoX 低 19.3%`) | **`32.3% (+0.59)`** |
| **6.0M / TinyStories** | Pure Llama 2 vs **`v6-wormhole`** | `5.87M` vs **`5.77M`** | `5.50` vs **`5.20`** | `28.67` vs **`4.90`** (`-82.9%`) | `16.4%` vs **`32.6%`** |
| **6.8M / FineWeb-Edu** | Pure Llama 2 vs **`v6-wormhole`** | `6.91M` vs **`6.82M`** | `37.52` vs **`34.87`** | `161.27` vs **`32.29`** (`-80.0%`) | `9.6% (+0.13)` vs **`29.2% (+0.55)`** |
| **27.6M / RealWorld 16K** | Pure Llama 2 vs **`v6-wormhole`** | `27.66M` vs **`27.28M`** | `38.24` vs **`34.95`** | `175.29` vs **`32.26`** (`-81.6%`) | `10.4% (+0.12)` vs **`41.1% (+1.34)`** |

---

## 三、 核心参数规格与硬件对齐

| 头宽 $D$ | 纤维 $d_r$ | 底时空 $d_g$ | 增广槽 | 打包总维度 | `V` 补零税 | 相对纯正 Llama 2 权重差 |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `32` (`1.3M`) | `24` | `6` | `+2` | **`32`** | `0` | **少 `14,848`** |
| `64` (`6.0M`) | `48` | `14` | `+2` | **`64`** | `0` | **少 `93,696`** |
| `64` (`28M, 8头`) | `48` | `14` | `+2` | **`64`** | `0` | **少 `371,712`** |
| `128` | `96` | `30` | `+2` | **`128`** | `0` | 严格少于 Base |

`model.py` 核心开关：`use_unified_gravity=True`、`use_wormhole=True`、`wormhole_v0_init=2.5`、`use_base_rbf=True`、`use_bilateral_mass=True`、`use_time_dilation=True`、`use_nested_time=True`、`use_attn_vacuum=True`，以及 `sample.py` 中的 `pauli_alpha` / `pauli_halflife`。

---

## 四、 核心公式来源：物理学与学术谱系

$$\Psi_h(i, j) = \underbrace{\frac{q_{\text{res},i}\cdot k_{\text{res},j}}{\sqrt{d_r}}}_{\text{纤维定向共振}} + \underbrace{u_{\text{recv},i} + u_{\text{emit},j}}_{\text{双边引力质量 } M_i M_j} - \underbrace{\frac{\|z_i - z_j\|_2^2}{2\sigma_h^2\sqrt{d_g}}}_{\text{底时空欧氏 RBF}} - \underbrace{(\tau_{h,i} - \tau_{h,j})}_{\text{太阳系嵌套固有时间}}, \qquad W_h(i, j) = \frac{e^{\Psi_h(i, j)}}{e^{v_{0,h}} + \sum_{k \le i} e^{\Psi_h(i, k)}}$$

| 核心组件 | 理论物理来源 | 深度学习学术脉络 |
| :--- | :--- | :--- |
| **纤维—底时空切分 ($d_r \oplus d_g$)** | 纤维丛 / 规范—引力耦合 | 保留 RoPE (Su et al., 2021) 并隔离低维 RBF |
| **底时空欧氏 RBF** | 低维高斯引力势 / 热核传播子 | L2/RBF Attention (Kim et al., 2021) 折叠进标准 SDPA |
| **双边质量 ($M_i M_j$)** | 牛顿万有引力双体耦合 ($F \propto m_1 m_2$) | 接收端需求质量动态压低真空门槛 |
| **太阳系嵌套固有时间 ($\delta\tau$)** | 广义相对论时间膨胀 + 层级势阱 | 推广 ALiBi (ICLR 2022) 与 FoX (ICLR 2025) |
| **ER=EPR 虫洞头** | ER = EPR 猜想 (Maldacena & Susskind, 2013) | 结合 NoPE (NeurIPS 2023) + 高门槛 Softmax1 + 紧缩 RBF |
| **物理真空底噪 ($e^{v_{0,h}}$)** | 量子场论质量间隙 / 真空基态 | Softmax1 / Quiet Attention (Evan Miller, 2023) |
| **阻尼泡利排斥解码** | 泡利简并压 / 兰纳-琼斯势平衡 | 指数平滑衰减替代硬窗口 `repetition_penalty` |

---

## 五、 仓库结构规划

```text
llama2-wormhole-gravity/
├── model.py                     # 核心 Transformer + UnifiedFiberGravityAttention (v6) + 泡利排斥解码
├── train.py                     # 多硬件训练脚本（L4/A100 bf16、T4 fp16、Apple MPS）
├── sample.py                    # 支持阻尼泡利排斥的自回归采样脚本
├── audit_all.py                 # 6 维 float64 数学等价与四层零泄露防作弊审计套件
├── inspect_fields.py            # 逐层逐头物理参数开盒体检（σ_g、δτ、v0、沉默率）
├── eval_extrapolation.py        # 零样本长度外推评测（支持 Vanilla / Dynamic NTK / YaRN RoPE）
├── eval_needle.py               # 远距大海捞针精准回忆评测（64 -> 1536）
├── run_sota_baselines.py        # 同条件 ALiBi (ICLR 2022) 与 FoX (ICLR 2025) 强基线训练脚本
├── config/                      # 全套训练配置（WikiText-2 1.3M/6M、TinyStories、FineWeb-Edu、28M RealWorld）
├── scripts/                     # HuggingFace 流式语料构建脚本（TinyStories、FineWeb-Edu、RealWorld 16K）
├── docs/
│   ├── BENCHMARKS.md            # 完整英文基准档案（全景表、同耗时图、Scaling Law 与样本）
│   ├── BENCHMARKS_zh.md         # 完整中文基准档案（全景表、同耗时图、Scaling Law 与样本）
│   └── figures/                 # 5 张 300 DPI 论文级架构图与数据对比图
├── CITATION.cff / CITATION.bib  # GitHub 官方引用元数据与标准学术 BibTeX
└── LICENSE / NOTICE             # MIT 协议（Andrej Karpathy + Dracen-W）与 Meta Llama 2 合规声明
```

---

## 六、 快速开始与 6 维 `float64` 审计

```bash
# 1. 运行 6 维超强审计（fp64 未来扰动=0.00e+00、未来梯度=0.00e+00、数学等价 <= 3.3e-16）
python3 audit_all.py

# 2. 在 WikiText-2 上训练 6.0M / 512 上下文 v6-Wormhole（Colab L4 约 4 分钟）
python3 train.py config/train_wikitext2_colab_scale.py \
  --use_unified_gravity=True --use_wormhole=True --out_dir=out-6m-v6-wormhole

# 3. 权重防作弊审计 + 物理体检 + 2048 外推 + 1536 大海捞针
python3 audit_all.py --ckpt=out-6m-v6-wormhole/ckpt.pt
python3 inspect_fields.py out-6m-v6-wormhole/ckpt.pt
python3 eval_extrapolation.py out-6m-v6-wormhole --lengths=256,512,1024,2048
python3 eval_needle.py out-6m-v6-wormhole --distances=64,128,256,512,1024,1536
```

---

## 七、 开源协议与引用规范

- **代码协议**：MIT License（完整保留 Andrej Karpathy 原始 `llama2.c` 版权声明，详见 [`LICENSE`](LICENSE) 与 [`NOTICE`](NOTICE)）。
- **引用本项目**：可直接点击 GitHub 仓库右侧的 **"Cite this repository"**（由 [`CITATION.cff`](CITATION.cff) 自动生成），或复制下方 BibTeX：

```bibtex
@misc{dracenw2026wormhole,
  title        = {Unified Fiber-Gravity and ER=EPR Wormhole Attention},
  author       = {Dracen-W},
  year         = {2026},
  howpublished = {\url{https://github.com/Dracen-W/llama2-wormhole-gravity}},
  note         = {Built on karpathy/llama2.c and the Meta Llama 2 architecture}
}
```
