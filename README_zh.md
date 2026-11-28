<div align="center">

# EPR Attention（欧氏 RBF · 固有时间 · 共振纤维注意力机制）

**几何折叠省参 · 原生层内双态长上下文 · 工业级 Triton PagedAttention 注意力架构**

[English](README.md) | **简体中文** · [📐 数学推导与精确折叠](docs/MATH.md) · [🗂️ v3.0 (1.1B) 原始实测数据](benchmarks/raw_v3/README.md) · [🗂️ v2.0 (6M) 原始实测数据](benchmarks/raw_v2/README.md) · [🏷️ v2.0 存档 Tag](https://github.com/Dracen-W/epr-attention/releases/tag/2.0) · [📊 v1.0 旧版归档](docs/BENCHMARKS_zh.md) · [🏷️ v1.0 存档 Tag](https://github.com/Dracen-W/epr-attention/releases/tag/1.0)

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Built with Meta Llama 2](https://img.shields.io/badge/Architecture-Meta%20Llama%202-blueviolet)](NOTICE)
[![9-Pillar Audit](https://img.shields.io/badge/9--Pillar%20Audit-0.00e+00%20Leak-brightgreen)](audit_all.py)
[![Scale Verified: 1.3M to 1.1B](https://img.shields.io/badge/实测验证规模-1.3M%20%E2%86%92%201.1B%20GQA-0284c7)](benchmarks/raw_v3/README.md)

</div>

---

## 为什么选择 EPR Attention？—— 五大核心卖点

1. **同起跑线更少参数、更强收敛（公平从零训练全面反杀）**：在严格相同的数据流（`FineWeb-Edu`）、模型层宽与 `token` 预算下，**`1.1B` 级 EPR 少用 `-2,217 万` 参数，验证困惑度反而比标准 `Llama-1.1B` 直降 `-26.7%`（`225.38` vs `307.32`）**，单步训练达原生 `FlashAttention-2` **`97.2%`** 吞吐，**提前 `33.3%` 训练量 / 省时 `32.7%` 即击败 `FA2` 终点**！（`6.0M` 三随机种子同起跑线同样比 `Llama 2` 低 **`-20.7%`**、比 `ICLR'25 FoX` 低 **`-15.1%`**。）
2. **硬件 $2^k$ 对齐的代数折叠与结构性净省参数**：通过 `d_r + d_g + 2 == head_dim` 免平移双槽恒等式（`V` 通道零填充，100% 契合 Tensor Core），将语义纤维共振、异方差欧氏捕获盆地与零值真空槽合入单次标准内积，严格按闭式公式削减约 `20%` 的 `Q/K` 寻址参数（*已在 `1.3M → 1.1B` 逐档实测验证，更大规模属理论严格成立*）。
3. **工业级 `@triton.jit` PagedAttention 再省 `-72.1%` 物理 KV 显存**：层内阻尼头对滑窗外物理页执行 $O(1)$ 自动回收，NoPE 虫洞头保留全局记忆——在 `1.1B GQA` 连续批处理（`B=16`, 变长至 `8K`）下物理 KV 显存从标准 PagedAttention 的 `1,357.1 MB` 压至 **`379.0 MB`**（比静态预分配 `2,816.0 MB` 省 **`-86.5%`**），且真空槽直接折叠进在线 Softmax 寄存器实现零显存读取（`bf16` 核误差 `0.00e+00`）。
4. **`8K` 亚平方训练提速 `1.63×` + 免崩溃零样本长文外推**：非对称 `FlexAttention BlockMask` 在阻尼头上硬件级跳过远距 `128×128` 块，**`8K` 前反向吞吐提速 `1.63×`**；越界 `2×–4×` 零样本外推彻底告别标准 RoPE 崩溃（从零 `1.1B` `4×` 外推低 `-34.1%`；现成 `TinyLlama` 手术后 `[3072:4096]` PPL 从崩溃的 `13,399` 直降至 **`14.69`**，驱逐 `71.9%` KV 后仍为 **`14.71`**）。
5. **现成开源大模型 47 分钟极低成本热移植手术**：无需重砸 `3T tokens` 从头预训练——对现成 HuggingFace `Llama` 权重执行 [`upgrade_hf_llama_to_epr()`](hf_epr.py) 保频裁参手术（砍掉 `-22.17M` 参数），单张 L4 **仅热身 47 分钟（仅喂原 `3T` 预训练量百万分之三点三 `0.00033%` 的 `tokens`、零教师蒸馏）**，即可快速愈合恢复 **`93.9%`** 的 5 大零样本榜单能力，并立即获得 `912×` 外推修复与 `-72%` KV 压缩！

---

## 核心实测战报总览（`v3.0` 工业化与 `1.1B` 全套评测，NVIDIA L4 `bfloat16`）

![EPR v3.0 1.1B 从零预训练、Triton PagedAttention、8K 稀疏跳块与 TinyLlama-1.1B 手术实测总图](docs/figures/v3_1b_and_paged_industrial.png)

| 评测维度 | 对照基线 (`FA2` / 标准 `PagedAttn` / `TinyLlama-1.1B`) | **EPR + Mass-Elastic RBF (`v3.0`)** | 硬核实测结论 |
| :--- | :---: | :---: | :--- |
| **`1.1B` 从零预训练 (`FineWeb-Edu`)** *(公平同起跑线对比, `300 steps`)* | `Val PPL = 307.32` (`1,034.5M`, `924.3s`) | **`Val PPL = 225.38`** (`1,012.3M`, **`913.7s`**) | **同 `token` 预算 PPL 直降 `-26.7%`**（`-0.310` NLL），**净省 `-22.17M` 参数**，单步达原生 `FA2` **`97.2%` 吞吐**，**提前 `33.3%` `token` / 省时 `32.7%` 超越 `FA2` 终点** |
| **`1.1B` 从零 `4×` 零样本外推** *(`T_train=1024 → [2048:4096]`)* | `PPL = 358.08` | **`PPL = 236.07`** | **`4×` 越界外推困惑度降低 `-34.1%`** |
| **`1.1B` 连续批处理物理 KV 显存** *(`B=16`, 变长至 `8192`)* | `1,357.1 MB` *(标准 Paged)* / `2,816.0 MB` *(静态)* | **`379.0 MB`** *(`@triton.jit`, `W=128+1 WH`)* | **比标准 PagedAttention 再省 `-72.1%` 物理页显存**（比静态缓存省 `-86.5%`），阻尼页 $O(1)$ 自动回收，内核误差 `0.00e+00` |
| **`1.1B` `8K` 亚平方稀疏训练吞吐** *(`T=8192`, Fwd+Bwd)* | `159,765 tok/s` *(稠密 EPR Flex)* | **`261,076 tok/s`** *(`W=512 + 全长虫洞`)* | `24/32` 阻尼头硬件级跳过远距 `128×128` 块，**`8K` 前反向总吞吐提速 `1.63×`** |
| **`6.0M` 三随机种子统计对撞** *(公平同起跑线对比, `seeds=1337,2026,42`)* | `Llama 2: 47.68` / `FoX (ICLR'25): 44.52` | **`37.80 ± 0.07`** *(`4×` 外推 `34.21`)* | **比 Llama 2 低 `-20.7%`**、**比 `FoX` 低 `-15.1%`**，`@1536` 远距捞针高 `+6.1%`，OOD 乱码主动降置信（`-1.55%`） |
| **`TinyLlama-1.1B` 存量热移植验证** *(砍 `-22.2M` + **仅喂 `3T` 的 `0.00033%` 语料**)* | `4K 外推: 13,399` · `5 榜: 61.80%` *(3T)* | **`4K 外推: 14.69` (`14.71 @W=256`)** · **`58.05%`** | **`4K` 越界 PPL 压降 `912×`**（驱逐 `71.9%` KV 仅差 `+0.02`），新换器官仅见 **`9.83M` vs `3T` `tokens`（相差 30 万倍训练量）** 即愈合 **`93.9%` 基座能力**！ |

---

## 精确闭式省参公式（`1.3M → 1.1B` 离散分档实测验证）

由于 Query 直接复用对称欧氏基底坐标 $z \in \mathbb{R}^{d_g}$（省去 $H \cdot d_g$ 列线性投影），且末尾 `+2` 代数折叠槽位（$-\frac{1}{2}\|z\|_2^2$ 与常数 `1`）无需任何投影矩阵权重（省去 $2(H + H_{\text{kv}})$ 列投影），**EPR Attention 在严格保持 `head_dim = D` 硬件 $2^k$ 对齐（`d_r + d_g + 2 == D`、`V` 通道零填充）的同时，在结构上严格净省矩阵参数**：

$$\Delta P_{\text{matrix}} = - L \cdot C \cdot \big[\, H(d_g + 1) + H_{\text{kv}} + H_{\text{wh}} \,\big]$$

其中 $L$ 为层数，$C$ 为模型隐藏维，$H$ 为 Query 头数，$H_{\text{kv}}$ 为 KV 头数，$H_{\text{wh}}$ 为虫洞 Query 头数，$d_g$ 为跨头共享基底维数（`D=32` 时 `d_g=6`；`D=64` 时 `d_g=14`；`D=128` 时 `d_g=30`）。

> [!IMPORTANT]
> **严谨实证范围声明（`M` 级至 `1.1B` 级实测校验 vs. 更大规模理论成立）：**
> - **已完成实测验证的规模区间（`1.3M` → `1.1B`）**：该公式目前**仅在 `Million` 级至 `1.1B` 级的离散模型档位上完成了逐级真机实测校验**，且每一档实测参数差值均精确到个位数：
>   - **`1.3M` MHA (`4L, C=128, 4H, d_g=6`)**：公式预测矩阵减参 **`-16,896`**（算上 1D 标量净减 `-16,836`）——**实测逐位吻合**。
>   - **`6.0M` MHA (`6L, C=256, 4H, d_g=14`)**：公式预测矩阵减参 **`-99,840`**（总参 `5,769,050` vs `5,868,800`）——**实测逐位吻合**。
>   - **`28M` MHA (`6L, C=512, 8H, d_g=14`)**：公式预测矩阵减参 **`-396,288`**（`28.45M` vs `28.84M`）——**实测逐位吻合**。
>   - **`1.1B` GQA (`22L, C=2048, 32Q/4KV, H_wh=8, d_g=14`)**：公式预测 $-22 \times 2048 \times [32(15) + 4 + 8] =$ **`-22,167,552`** 矩阵参数；Track A 从零训练衰减矩阵实测 `1,012,252,672 - 1,034,420,224 = -22,167,552`，Track B `TinyLlama-1.1B` 手术实测净差 `-22,166,760`（即 `-22,167,552` 矩阵参数 `+ 792` 个 1D 物理标量）——**一字不差精确吻合**。
> - **更大规模边界说明（`> 1.1B` 属理论严格成立）**：受限于单卡算力资源，**`1.1B` 以上的更大规模（如 `7B`、`70B`，其中 `70B` 8:1 GQA 代入公式可减参 `-1.31B` / 削减 `21.7%` `Q/K` 路由权重）尚未进行实际训练与评测**；但由于上式完全由线性层输入输出张量的代数维数恒等决定，**该省参公式本身在理论上对任意更大规模继续严格成立**。

> **归属与阶段存档声明：** 本项目基于 Andrej Karpathy 的 [`llama2.c`](https://github.com/karpathy/llama2.c)（MIT License）与 **Meta Llama 2** 架构（Touvron et al., 2023）构建。历史里程碑存档：[`v2.0` Release](https://github.com/Dracen-W/epr-attention/releases/tag/2.0) · [`v1.0` Release](https://github.com/Dracen-W/epr-attention/releases/tag/1.0)。仓库中的 `export.py`、`run.c`、`runq.c` 继承自 `llama2.c`，暂不支持 EPR 架构导出（当前完整支持 PyTorch、`FlexAttention`、Triton `PagedAttention` 与 HuggingFace `Transformers`）。
>
> **物理与神经启发声明：** 文中的“固有时间”、“真空槽”、“虫洞头”、“质量弹性捕获盆地”等术语源于神经化学耗散与物理场的直觉启发（**inspired by**），并非从广义相对论或量子引力方程推导。在数学实现上，所有部件均为标准可微 PyTorch / Triton 原语，并无损折叠进单次硬件对齐的 $D$ 维注意力核中（完整证明见 [`docs/MATH.md`](docs/MATH.md)）。

---

## 核心架构速览

**EPR Attention**（[`epr_attention.py`](epr_attention.py)、[`epr_paged_attention.py`](epr_paged_attention.py)、[`hf_epr.py`](hf_epr.py)）将三大几何原语与**质量弹性异方差 RBF** 无损折叠进 `d_r + d_g + 2 == head_dim` 的单次标准内积核中：

1. **免平移双槽代数折叠（`d_r + d_g + 2 == D`）**：将定向语义纤维共振（$q_{\text{res},i} \cdot k_{\text{res},j} / \sqrt{d_r}$）与跨头共享的低维欧氏 RBF 捕获势（$-\frac{\gamma_{h,j}}{2}\|z_i - z_j\|_2^2$）直接打包进 $2^k$ 对齐维度（`D=32 → 24+6+2`；`D=64 → 48+14+2`；`D=128 → 96+30+2`），无需 Softmax 平移抵消。
2. **零值可学习真空槽（$e^{v_{0,h}}, V_{\text{vac}} = \mathbf{0}_D$）**：允许注意力头在无关或无答案上下文上干净静默，避免拿 `<BOS> token` 充当隐式垃圾桶引发高范数方差污染；在 `@triton.jit` PagedAttention 中更可直接折叠为片上在线 Softmax 寄存器初值（`m_i = v0, l_i = 1.0, acc = 0`），实现**零显存读取、零额外页表开销**。
3. **层内单核双态头拓扑**：在**每一层内部**，$H - H_{\text{wh}}$ 个**阻尼头**采用 RoPE + 内容驱动固有时间衰减（$\tau_i - \tau_j \ge 0$），负责局部建模、稳健外推、亚平方 `BlockMask` 跳块与 $O(1)$ 物理页回收；$H_{\text{wh}}$ 个**虫洞头**采用 **NoPE + 零时间衰减（$\delta\tau \equiv 0$）+ 高真空门控**，负责跨任意距离的全局检索。
4. **质量弹性异方差 RBF（`--use_elastic_rbf=True`，默认推荐主力）**：利用键势 $u_{h,j}$ 动态伸缩每个 `token` 的捕获方差 $\gamma_{h,j} = \gamma_{0,h} / (1 + \alpha_h \sigma(u_{h,j}))$，高显著锚点 `token` 自动撑大捕获盆地，且直接塞进同一双槽核（`1.1B` 规模实测跑出原生 `FlashAttention-2` **`97.2%`** 的单步速度）。

---

## 一、 `v3.0` 十亿参数级（`1.1B`）与工业级系统实测（全量原始数据见 [`benchmarks/raw_v3/`](benchmarks/raw_v3/README.md)）

### 1.1 Track A：`1.1B` 从零预训练收敛与外推对撞（`FineWeb-Edu`, `22L, dim=2048, 32Q/4KV`, NVIDIA L4）
*严格控制相同流式 `FineWeb-Edu` 数据序列、`fused AdamW` 优化器与 `token` 预算（`300 steps × 16,384 tokens/step = 4.92M tokens`, `T_train = 1024`, `torch.compile`）：*

| 模型（`1.1B` 从零预训练） | 衰减矩阵参数量 | 单步耗时 / 吞吐 | 峰值显存 | `step 100` PPL | `step 200` PPL | 终点 `step 300` PPL（总挂钟） | `4K` 外推 `[0:1K] / [1K:2K] / [2K:4K]` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`pure_llama_1b` (`FA2` baseline)** | `1,034,420,224` | `2808.6 ms` (`5,833 tok/s`) | `10.90 GiB` | `607.46` (`357s`) | `364.09` (`641s`) | `307.32` (`924.3s`) | `287.34 / 315.58 / 358.08` |
| **`epr_elastic_1b` (Ours)** | **`1,012,252,672` (`-22.17M`)** | `2890.1 ms` (**`5,669 tok/s`, 达 `FA2` `97.2%`**) | `11.03 GiB` | **`512.71`** (`330s`) | **`285.79` (`622s`)** | **`225.38` (`913.7s`, `-26.7%`)** | **`205.04 / 227.27 / 236.07` (`-34.1%`)** |

### 1.2 Track B：`TinyLlama-1.1B-3T` 保频减参架构手术、5 大下游能力评测与 `4K` 零样本外推
*通过 [`upgrade_hf_llama_to_epr()`](hf_epr.py) 对预训练 `TinyLlama-1.1B-intermediate-step-1431k-3T` 原地执行子空间手术（100% 保留 `MLP / W_V / W_O` 及前 `48/64` 维高频/中频 RoPE 旋转频率，净裁减 `-22,166,760` 参数），随后与原版对照组 `base_1b_cpt` 在完全相同的 `T=2048` `FineWeb-Edu` 批次上续训 `300 steps`（`9.83M tokens` = 原 `3T` 预训练量的 **`0.00033%`**），在 5 大主流零样本能力榜单（每榜 `400` 条验证样本）与 `4K` 长文外推上同台实测：*

| 模型变体 | 总参数量 | `ARC-Easy` (`acc`) | `HellaSwag` (`acc_norm`) | `WinoGrande` (`acc`) | `PIQA` (`acc_norm`) | `BoolQ` (`acc_norm`) | **5 榜均值** | `2×` 外推 PPL `[2048:3072]` | `2×` 外推 PPL `[3072:4096]` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`TinyLlama-1.1B` (`3T` 原版锚点)** | `1,100.05M` | `62.25%` | `53.50%` | `55.25%` | `73.00%` | `65.00%` | `61.80%` | `1,444.35` | `12,829.90` |
| **`base_1b_cpt` (`300` 步续训对照)** | `1,100.05M` | `60.75%` | `52.75%` | `59.50%` | `74.00%` | `65.25%` | `62.45%` | `1,760.35` | `13,399.39` |
| **`epr_elastic_1b_cpt` (手术 + `300` 步续训)** | **`1,077.88M` (`-22.17M`)** | **`57.75%`** | **`47.00%`** | **`53.50%`** | **`70.75%`** | **`61.25%`** | **`58.05%`（保留 3T 的 `93.9%`！）** | **`12.26`** | **`14.69`（低 `912×`！）** |
| **`epr_elastic_1b_cpt` + `W=256` 驱逐 (`@4K` 省 `71.9%` KV)** | **`1,077.88M` (`-22.17M`)** | — | — | — | — | — | — | **`12.36`** | **`14.71`（比全长仅差 `+0.02`！）** |

*（客观局限说明：在仅投入 `9.83M` 通用 `FineWeb-Edu` `tokens`（`3T` 的百万分之三点三）且不含合成捞针语料的极短续训下，手术模型已快速恢复 **`93.9%`** 的零样本常识/阅读推理能力，并将 `[3072:4096]` 外推困惑度从 `13,399.39` 彻底抚平至 `14.69`；但新初始化的 NoPE 虫洞头因带有高真空静默势垒 `v0_init=+1.8`，在未经更长续训或合成检索混合语料激活前，对 5 位纯数字精确逐字复述 `Passkey` 仍保持静默 `0.0%`（`base_1b_cpt` 在 `@1K/@2K/@3.5K` 为 `100%/12.5%/0.0%`），而从零训练时虫洞精确检索可自然充分激活，见第二节 3 种子捞针实测。）*

### 1.3 工业级 `@triton.jit` 非对称 PagedAttention 与亚平方 `BlockMask` 实测（`1.1B GQA`）
- **连续批处理物理 KV 显存池对比（`B=16` 并发变长请求 `260..8192`，均值 `3944`，`page_size=16`）**：
  - 静态连续预分配（`max=8192`）：`2,816.00 MB` · 标准 PagedAttention（`4 KV`）：`1,357.12 MB` · **EPR 非对称分页池（`W=128 + 1 WH`）**：**`379.02 MB`**（**比标准 PagedAttention 再省 `-72.1%`**，比静态缓存省 **`-86.5%`**）。
  - 22 层融合 `_epr_paged_decode_kernel` 解码延迟（`B=8`，阻尼头恒定仅占 `8` 个物理页/请求）：`@1K` `1.381 ms`（`5,793 tok/s`）· `@2K` `2.497 ms`（`3,204 tok/s`）· `@4K` `4.823 ms`（`1,659 tok/s`）· `@8K` `9.462 ms`（`845 tok/s`）。
- **亚平方 `FlexAttention BlockMask` 前反向训练吞吐（`W_damped=512`, `1.1B GQA`）**：
  - `T=1024`: `376.2K` vs `375.4K tok/s` (`1.00×`) · `T=2048`: `341.5K` vs `310.1K` (`1.10×`) · `T=4096`: `309.8K` vs `236.0K` (`1.31×`) · **`T=8192`: `261.1K` vs `159.8K tok/s` (`1.63×` 加速)**。

---

## 二、 `v2.0` 受控多种子统计与消融基准（`6.0M`，全量原始数据见 [`benchmarks/raw_v2/`](benchmarks/raw_v2/README.md)）

![EPR v2.0 6M 同Step收敛、同挂钟耗时、无稀释外推与200次捞针总图](docs/figures/v2_6m_convergence_and_extrap.png)

### 2.1 三随机种子统计总表（`seeds = 1337, 2026, 42`，`Mean ± Std`）
*配置：`6.0M` WikiText-2（`dim=256, 6L, 4H, T_train=512, 3000 steps = 49.15M tokens`），384 序列无稀释分段外推 PPL + 每种子 200 次对齐 `Passkey` 捞针：*

| 架构（3 种子均值 ± 样本标准差） | 参数量 | 域内 `Val PPL@512` | `[256:512]` | `2×` 外推 `[512:1024]` | `4×` 外推 `[1024:2048]` | 捞针 `@512` | 捞针 `@1024` | 捞针 `@1536` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Pure Llama 2 (RoPE)** | `5.87M` | `47.68 ± 0.51` | `46.44 ± 0.66` | `97.59 ± 11.10` | `168.41 ± 18.90` | `25.28 ± 0.46%` | `13.89 ± 0.86%` | `13.86 ± 0.44%` |
| **FoX (Forgetting Transformer, ICLR'25)** | `5.87M` | `44.52 ± 0.25` | `43.57 ± 0.42` | `42.93 ± 0.36` | `42.22 ± 0.33` | `26.14 ± 0.21%` | `26.28 ± 0.46%` | `25.82 ± 0.23%` |
| **EPR v2.0 精简基座 (`bilat=F, nest=F`)** | **`5.77M`** | **`37.80 ± 0.07`** | **`35.58 ± 0.11`** | **`34.80 ± 0.14`** | **`34.21 ± 0.26`** | **`34.95 ± 0.96%`** | **`33.07 ± 0.59%`** | **`31.89 ± 0.39%`** |
| **EPR v2.0 + Mass-Elastic RBF** | **`5.77M`** | `37.95 ± 0.23` | `35.71 ± 0.32` | `34.96 ± 0.27` | `34.39 ± 0.35` | `34.65 ± 0.20%` | `32.90 ± 0.17%` | **`31.88 ± 0.86%`** *(单次最高 `32.58%`)* |

*（同配置 `seed=1337` 其他外推基线：**Dynamic NTK** `[1024:2048] = 52.22`；**YaRN** `126.63`；**ALiBi** `42.60`、`Needle@1536 = 23.13%`。八变体消融、OOD 乱码抗过自信与 `W=128` 零损驱逐详表完整保留于 [`v2.0` Release](https://github.com/Dracen-W/epr-attention/releases/tag/2.0) 及 [`v2_6m_ood_and_kv_eviction.png`](docs/figures/v2_6m_ood_and_kv_eviction.png)。）*

---

## 三、 与 2025–2026 前沿文献的后验对照

- **内容驱动固有时间（$\tau_i - \tau_j$）**：去掉级联矩阵（$M=I$）后与 **FoX（Forgetting Transformer, ICLR 2025）** 的累积遗忘门数学同构；EPR 进一步耦合了共享欧氏流形与层内 NoPE 虫洞头，实测比 FoX 额外降低 `15.1%` 4× 外推 PPL、提升 `+6.1%` 远距捞针。
- **零值物理真空槽（$e^{v_0}, V_{\text{vac}} = \mathbf{0}_D$）**：与 **Softmax1**（Miller, 2023）、**OpenAI `gpt-oss`** 可学习 Sink（2025）以及 **ICML 2026** *SinkProbe* 理论相呼应，并在 Triton PagedAttention 中化简为片上寄存器初值，实现零显存读取。
- **层内单核双态 vs. 跨层异构混合**：区别于 **Llama 4 (`iRoPE`)**、**Gemma 3/4**、**DuoAttention (ICLR 2025)** 以及 **Qwen3.5 / Kimi Linear**，EPR 在**每一层同一个 $D$ 对齐核内**同时实现阻尼遗忘与全长虫洞检索，并结构性裁减约 `20%` 的 `Q/K` 投影参数。

---

## 四、 快速上手

```bash
# 1. 运行 9 维全栈审计（fp64 教科书等价 + 因果零泄漏 + 环形/分页 PagedKVCache + HF 封装 + CUDA/Triton 审计）
python3 audit_all.py

# 2. v3.0 工业化与 1.1B 标准评测套件（需 CUDA GPU：Triton PagedAttention、8K 稀疏跳块与 1.1B Track A/B）
python3 bench_paged_and_sparse.py
python3 eval_1b_industrial.py

# 3. 6M 受控预训练与消融评测套件（WikiText-2）
python3 train.py config/train_wikitext2_colab_scale.py --out_dir=out-6m-epr-elastic --use_epr=True --use_elastic_rbf=True
python3 eval_extrapolation.py out-6m-epr-elastic --lengths=256,512,1024,2048
python3 eval_kv_eviction.py out-6m-epr-elastic
```

---

## 五、 独立构想声明、后验文献归属与 AI 协作披露

- **独立第一性原理构想声明**：本项目的全部核心构想——从早期模拟人脑神经递质局部耗散与空间势场出发，逐步独立推演出有界欧氏捕获势、质量弹性异方差捕获半径、内容驱动固有时间阻尼、零值物理真空底噪槽以及层内零衰减虫洞头——均由作者在**事先未阅读、未参考第三节所列同期/前人文献的前提下独立思考提出**（完整保留有按时间戳记录的原始推演档案）。
- **后验文献审计与诚实归属**：与 `Softmax1`、`FoX`、`DuoAttention`、`iRoPE / RNoPE / Gemma 4`、`gpt-oss` 以及 `ICLR/ICML 2026` 相关工作的数学同构性与趋同演进关系，均是在架构成型后的后验文献查重阶段识别，并已在正文及 [`CITATION.bib`](CITATION.bib) 中完整引用致谢。
- **AI 协作分工透明披露**：人类作者负责提出全部原创物理/几何假设、架构取舍决策与实验审定；AI 助手（`Gemini` / `Antigravity`）负责将作者构想形式化为精确的 `d_r + d_g + 2 == D` 代数折叠恒等式与闭式省参公式、完成 PyTorch/`FlexAttention`/`@triton.jit` PagedAttention 与 HuggingFace 手术代码库及 9 维审计套件开发、执行后验文献检索并自动化运行 GPU 评测。
- **开源协议**：MIT License（完整保留 Andrej Karpathy 原始 `llama2.c` 版权声明，详见 [`LICENSE`](LICENSE) 与 [`NOTICE`](NOTICE)）。

```bibtex
@misc{dracenw2026epr,
  title        = {EPR Attention: A Geometry-Folded, Parameter-Saving Long-Context Attention Architecture},
  author       = {Dracen-W},
  year         = {2026},
  howpublished = {\url{https://github.com/Dracen-W/epr-attention}},
  note         = {Built on karpathy/llama2.c and the Meta Llama 2 architecture}
}
```
