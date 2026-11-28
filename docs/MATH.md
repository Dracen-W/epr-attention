# Mathematical Structure of EPR Attention (`v2.0` / `v3.0`)

**EPR Attention** (**E**uclidean-RBF, **P**roper-time & **R**esonance-fiber Attention) combines three standard differentiable ML primitives—plus an optional key-scaled heteroscedastic `L2-RBF` extension—that fold algebraically into a single hardware-aligned $D$-dimensional inner-product kernel with **zero $V$-channel padding**.

> **Naming Translation Note (Zero Physics Pretension):** Source-code identifiers such as `proper_time`, `attn_vacuum`, `wormhole_heads`, and `gravity_dim` are playful historical leftovers from the author's initial whiteboard sketches—not claims of relativistic or quantum physics. In plain ML terms: *"Euclidean/gravity basin"* = head-shared `L2-RBF` kernel; *"proper time"* = cumulative content-adaptive log-decay; *"vacuum floor"* = zero-value `Softmax1` attention sink; *"wormhole head"* = intra-layer `NoPE` global head. Below is the exact linear-algebraic formulation implemented in [`epr_attention.py`](../epr_attention.py) and [`epr_paged_attention.py`](../epr_paged_attention.py), verified to `~2.2e-16` (`float64`) in [`audit_all.py`](../audit_all.py).

---

## 1. Core Attention Score & Learnable Vacuum Floor (v2.0 Distilled)

For head $h \in \{0, \dots, H-1\}$, query position $i$, and causal key position $j \le i$:

$$\Psi_h(i, j) = \underbrace{\frac{q_{\text{res},h,i} \cdot k_{\text{res},h,j}}{\sqrt{d_r}}}_{\text{(R) Resonance Fiber}} + \underbrace{u_{h,j}}_{\text{Key Potential}} - \underbrace{\frac{\gamma_{h,j}}{2}\|z_i - z_j\|_2^2}_{\text{(E) Euclidean-RBF}} - \underbrace{(\tau_{h,i} - \tau_{h,j})}_{\text{(P) Proper-Time Decay}}$$

where:
- $q_{\text{res},h,i}, k_{\text{res},h,j} \in \mathbb{R}^{d_r}$ carry RoPE on damped heads ($h < H_{\text{damped}}$) and **NoPE** on Wormhole heads ($h \ge H_{\text{damped}}$).
- $z_i, z_j \in \mathbb{R}^{d_g}$ are head-shared low-dimensional base coordinates ($d_g \ll D$, e.g. $d_r = 48, d_g = 14$ for $D = 64$, so $d_r + d_g + 2 = 64$).
- In **v2.0 Distilled** (`use_elastic_rbf=False`), each head has a learnable inverse squared radius $\gamma_{h,j} \equiv \gamma_h = \frac{1}{\sigma_h^2 \sqrt{d_g}} > 0$.
- Normalized with a learnable **vacuum floor** $e^{v_{0,h}}$ (equivalent to Softmax1 / Attention Sink):

$$W_h(i, j) = \frac{\exp(\Psi_h(i, j))}{\exp(v_{0,h}) + \sum_{k \le i} \exp(\Psi_h(i, k))}, \qquad W_h(i, \text{vac}) = \frac{\exp(v_{0,h})}{\exp(v_{0,h}) + \sum_{k \le i} \exp(\Psi_h(i, k))}$$

Because $V_{\text{vac}} = \mathbf{0}_D$, probability mass routed to $W_h(i, \text{vac})$ cleanly silences the head on uninformative or unanswerable context rather than forcing attention onto noise tokens.

*(Note on v1.0 → v2.0 distillation: 6M single-switch ablations showed that receiver mass $u_{\text{recv},h,i}$ and the upper-triangular nested time cascade $M$ contributed `< 0.1` PPL difference while adding extra projections and Softmax shift terms. Both default to `False` in v2.0 and remain available via `use_bilateral_mass` / `use_nested_time`.)*

---

## 2. Un-Shifted Direct 2-Slot Algebraic Folding (`d_r + d_g + 2 == D`)

Removing redundant $u_{\text{recv},h,i}$ frees both auxiliary channels (`+2`) to fold **both constant-radius and heteroscedastic key-dependent RBF** directly into a single $D$-dimensional inner product **with zero Softmax shift**:

Expand the general heteroscedastic Euclidean-RBF term with key-specific stiffness $\gamma_{h,j} > 0$:

$$-\frac{1}{2}\gamma_{h,j}\|z_i - z_j\|_2^2 = z_i \cdot (\gamma_{h,j} z_j) + \Big(-\frac{1}{2}\|z_i\|_2^2\Big)\gamma_{h,j} - \frac{1}{2}\gamma_{h,j}\|z_j\|_2^2$$

Pack the augmented query, real key, and vacuum key into exact dimension $D = d_r + d_g + 2$:

$$Q_{\text{aug},h,i} = \Big[\, \tfrac{q_{\text{res},h,i}}{\sqrt{d_r}},\;\; z_i,\;\; -\tfrac{1}{2}\|z_i\|_2^2,\;\; 1 \,\Big] \in \mathbb{R}^D$$

$$K_{\text{real},h,j} = \Big[\, k_{\text{res},h,j},\;\; \gamma_{h,j} z_j,\;\; \gamma_{h,j},\;\; u_{h,j} - \tfrac{1}{2}\gamma_{h,j}\|z_j\|_2^2 \,\Big] \in \mathbb{R}^D$$

$$K_{\text{vac},h} = \Big[\, \mathbf{0}_{d_r},\;\; \mathbf{0}_{d_g},\;\; 0,\;\; v_{0,h} \,\Big] \in \mathbb{R}^D$$

**Proof of Direct Exact Identity (No Shift Needed):**
1. **Real key inner product:**
   $$Q_{\text{aug},h,i} \cdot K_{\text{real},h,j} = \frac{q_{\text{res},h,i} \cdot k_{\text{res},h,j}}{\sqrt{d_r}} + \gamma_{h,j}(z_i \cdot z_j) - \frac{1}{2}\gamma_{h,j}\|z_i\|_2^2 + u_{h,j} - \frac{1}{2}\gamma_{h,j}\|z_j\|_2^2 = \frac{q_{\text{res},h,i} \cdot k_{\text{res},h,j}}{\sqrt{d_r}} + u_{h,j} - \frac{\gamma_{h,j}}{2}\|z_i - z_j\|_2^2$$
2. **Vacuum key inner product:**
   $$Q_{\text{aug},h,i} \cdot K_{\text{vac},h} = 0 + 0 + 0 + 1 \cdot v_{0,h} = v_{0,h}$$

Subtracting causal proper-time decay $(\tau_{h,i} - \tau_{h,j})$ on real keys ($j \ge 1$ when $K_{\text{vac}}$ is prepended at index $0$) inside `FlexAttention`'s `score_mod` (or masked SDPA) yields $\Psi_h(i, j)$ and $v_{0,h}$ **directly** in hardware-aligned $D \in \{16, 32, 48, 64, 96, 128\}$ with `fp64` error $\le 3.33 \times 10^{-16}$.

---

## 3. Optional Extension 1: Mass-Elastic Heteroscedastic RBF (`use_elastic_rbf=True`)

In constant-radius RBF ($\gamma_{h,j} \equiv \gamma_h$), every token casts the same Gaussian capture basin in $\mathbb{R}^{d_g}$. With **Mass-Elastic Heteroscedastic RBF**, a token's learned key potential $u_{h,j}$ dynamically dilates its capture variance $\sigma_{h,j}^2$:

$$\sigma_{h,j}^2 = \sigma_{0,h}^2 \big(1 + \alpha_h \cdot \sigma(u_{h,j})\big) \quad \Longleftrightarrow \quad \gamma_{h,j} = \frac{\gamma_{0,h}}{1 + \alpha_h \cdot \sigma(u_{h,j})}, \qquad \alpha_h = \text{softplus}(\tilde{\alpha}_h) > 0$$

- **High-salience anchor tokens** ($u_{h,j} \gg 0 \implies \sigma(u_{h,j}) \to 1$) acquire a **wider capture basin** ($\gamma_{h,j} = \frac{\gamma_{0,h}}{1 + \alpha_h}$), allowing semantically related queries across paraphrases or distant positions to lock onto them more easily.
- **Low-salience filler tokens** ($u_{h,j} \ll 0 \implies \sigma(u_{h,j}) \to 0$) contract to a **tight local basin** ($\gamma_{h,j} \to \gamma_{0,h}$), suppressing background interference.
- Because $\gamma_{h,j}$ depends solely on key index $j$, the 2-slot identity in Section 2 holds **with zero extra channels** (`48 + 14 + 2 == 64`), adding only $H$ scalars (`+24` parameters across a 6-layer model) and retaining **>90%** of fused `FlexAttention` throughput.

---

## 4. Optional Extension 2: Pairwise Resonant Tunneling (`use_resonant_tunneling=True`)

Instead of relying solely on a static dedicated Wormhole head ($\delta\tau \equiv 0$), **Pairwise Resonant Tunneling** lets strong pairwise pre-decay affinity $S_{h,i,j} = Q_{\text{aug},h,i} \cdot K_{\text{real},h,j}$ dynamically suppress proper-time decay $\Delta\tau_{h,i,j} = \tau_{h,i} - \tau_{h,j} \ge 0$:

$$\Psi_h(i, j) = S_{h,i,j} - \Delta\tau_{h,i,j} \cdot \Big(1 - \sigma\big(\beta_h (S_{h,i,j} - \theta_h)\big)\Big), \qquad \beta_h = \text{softplus}(\tilde{\beta}_h) > 0$$

- When $S_{h,i,j} \ll \theta_h$, $1 - \sigma(\cdot) \to 1$, recovering standard exponential proper-time damping $S_{h,i,j} - \Delta\tau_{h,i,j}$.
- When a distant key strongly resonates with the query ($S_{h,i,j} > \theta_h$), $1 - \sigma(\cdot) \to 0$, dynamically opening a pairwise "tunnel" across arbitrary sequence distance $\Delta\tau_{h,i,j}$.
- **Hardware Tradeoff**: Because $\sigma(\beta_h(S_{h,i,j} - \theta_h))$ is non-linear in the pairwise dot-product `score`, `FlexAttention` backward must recompute pairwise score derivatives (~6× slower backward than rank-1 additive `score_mod`), making static Wormhole heads + Mass-Elastic RBF preferable for large-scale training throughput.

---

## 5. Dual-Regime Topology, Causality & Stateful `EPRKVCache`

For damped heads $h \in \{0, \dots, H_{\text{damped}}-1\}$:

$$\delta\tau_{h,t} = \text{softplus}\big((W_{\tau} x_t)_h + b_{\tau,h}\big) > 0, \qquad \tau_{h,t} = \sum_{s=0}^{t} \delta\tau_{h,s}$$

- **Strict Causality**: $\tau_{h,t}$ depends only on $x_{\le t}$, with $\tau_{h,i} - \tau_{h,j} = \sum_{s=j+1}^{i} \delta\tau_{h,s} \ge 0$ for all $j \le i$. This is mathematically isomorphic to the cumulative log-forget-gate of **FoX (Forgetting Transformer, ICLR 2025)**.
- **Damped Heads ($h < H_{\text{damped}}$)**: RoPE on $d_r$ + positive $\delta\tau_{h,t} > 0$. Exponential decay bounds attention entropy during zero-shot length extrapolation and allows distant keys outside a fixed ring buffer $W$ to be evicted with $O(1)$ memory.
- **Wormhole Heads ($h \ge H_{\text{damped}}$)**: **NoPE** + **zero decay ($\delta\tau \equiv 0$)** + **high vacuum threshold ($v_0 \approx +2.5$)**. They remain silent ($W_{\text{vac}} \to 1$) until a distant key achieves strong fiber + RBF resonance. Under GQA, head ordering is aligned cleanly to KV groups (`n_wh` is a multiple of `n_rep`).

**Incremental Update in `EPRKVCache`:**
At autoregressive step $t$ with cached running proper-time state $\tau^{\text{state}}_{h, t-1}$:

$$\tau_{h,t} = \tau^{\text{state}}_{h, t-1} + \delta\tau_{h,t}, \qquad \tau^{\text{state}}_{h, t} \leftarrow \tau_{h,t}$$

Caching `(K_real, V, tau_k)` in a fixed ring buffer of size $W$ for damped KV heads and `(K_real, V)` in full context for Wormhole KV heads achieves **`4.44e-16` (`float64`) parity** with full parallel evaluation prior to eviction, while cutting KV-cache memory by **70.2%–86.1%** at $W = 128$.

---

## 6. Industrial `v3.0` Kernels: Zero-Read Online-Softmax Vacuum PagedAttention & Sub-Quadratic `BlockMask`

### 6.1 Zero-Read Online-Softmax Vacuum Folding in `@triton.jit` PagedAttention ([`epr_paged_attention.py`](../epr_paged_attention.py))
In Flash-Decoding / PagedAttention over non-contiguous physical pages $p \in \mathcal{P}_b$, each threadblock maintains running online-Softmax scalars $(m, \ell)$ and output accumulator $\mathbf{o} \in \mathbb{R}^D$. Because $V_{\text{vac}} = \mathbf{0}_D$, initializing SRAM registers directly at:

$$m^{(0)} = v_{0,h_{\text{kv}}}, \qquad \ell^{(0)} = 1, \qquad \mathbf{o}^{(0)} = \mathbf{0}_D$$

and applying the standard blockwise update $m^{(n)} = \max(m^{(n-1)}, \max_{k \in B_n} s_k)$, $\ell^{(n)} = \ell^{(n-1)} e^{m^{(n-1)} - m^{(n)}} + \sum_{k \in B_n} e^{s_k - m^{(n)}}$, $\mathbf{o}^{(n)} = \mathbf{o}^{(n-1)} e^{m^{(n-1)} - m^{(n)}} + \sum_{k \in B_n} e^{s_k - m^{(n)}} V_k$ yields at termination:

$$\frac{\mathbf{o}^{(N)}}{\ell^{(N)}} = \frac{\sum_{k} \exp(s_k - m^{(N)}) V_k}{\exp(v_{0,h_{\text{kv}}} - m^{(N)}) + \sum_{k} \exp(s_k - m^{(N)})} = \frac{\sum_{k} \exp(s_k) V_k}{\exp(v_{0,h_{\text{kv}}}) + \sum_{k} \exp(s_k)}$$

which is **algebraically identical to prepending a dummy vacuum token (`2.78e-16` `fp64` parity)** with **zero global-memory reads and zero extra page-table entries**. Simultaneously, expired physical pages on damped heads ($t_{\text{page\_end}} < t_{\text{cur}} + 1 - W$) are popped from the per-sequence block table and returned to `free_d_pages` in $O(1)$.

### 6.2 Asymmetric Sub-Quadratic `FlexAttention` `BlockMask` (`flex_damped_window > 0`)
For long-context pretraining/prefill ($T \gg W$), setting `flex_damped_window = W` constructs a head-asymmetric `BlockMask` over $128 \times 128$ hardware tiles:
- **Damped Heads ($h < H_{\text{damped}}$)**: evaluate only tiles intersecting $(j = 0) \lor (0 \le i - j < W)$, skipping all $O(T^2)$ distant tiles in both forward and backward passes.
- **Wormhole Heads ($h \ge H_{\text{damped}}$)**: evaluate full causal tiles $j \le i$, preserving exact arbitrary-distance retrieval in $O\!\big(T \cdot (\frac{H_{\text{damped}}}{H} W + \frac{H_{\text{wh}}}{H} T)\big)$ total work (`1.63×` Fwd+Bwd speedup at `T=8192` on `1.1B GQA`).

### 6.3 Closed-Form Matrix Parameter-Saving Law & Verification Scope
Comparing a standard MHA/GQA layer ($W_Q \in \mathbb{R}^{(H \cdot D) \times C}$, $W_K, W_V \in \mathbb{R}^{(H_{\text{kv}} \cdot D) \times C}$, $W_O \in \mathbb{R}^{C \times (H \cdot D)}$) against EPR (`d_r + d_g + 2 = D`, $W_{\text{proj}} \in \mathbb{R}^{(H d_r + H_{\text{kv}} d_r + H_{\text{kv}} d_g + H_{\text{kv}} D) \times C}$, $W_{\text{phys}} \in \mathbb{R}^{(H_{\text{kv}} + H - H_{\text{wh}}) \times C}$, $W_O$ unchanged):

$$\Delta P_{\text{matrix}} = L \cdot C \cdot \Big[\big(H d_r + H_{\text{kv}}(d_r + d_g) + H_{\text{kv}} + H - H_{\text{wh}}\big) - \big(H D + H_{\text{kv}} D\big)\Big]$$

Substituting $d_r - D = -(d_g + 2)$ and $(d_r + d_g) - D = -2$ simplifies directly to:

$$\Delta P_{\text{matrix}} = - L \cdot C \cdot \big[\, H(d_g + 1) + H_{\text{kv}} + H_{\text{wh}} \,\big] < 0$$

- **Empirical Verification (`1.3M` → `1.1B`)**: Verified to the exact single integer across discrete tiers from `1.3M` (`-16,896`), `6.0M` (`-99,840`), `28M` (`-396,288`) up to `1.1B GQA` (`22L, C=2048, 32Q/4KV, H_wh=8, d_g=14` $\implies$ `-22,167,552` matrix parameters).
- **Theoretical Scope (`> 1.1B`)**: Models beyond `1.1B` have not yet been empirically trained; however, as an exact algebraic identity of the projection tensor dimensions, $\Delta P_{\text{matrix}}$ holds strictly in theory for arbitrary larger scales.
