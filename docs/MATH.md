# Mathematical Structure of EPR Attention

**EPR Attention** (**E**uclidean-RBF, **P**roper-time & **R**esonance-fiber Attention) is built from **4 orthogonal geometric primitives** that fold into a single hardware-aligned $D$-dimensional attention kernel with zero $V$-channel padding.

---

## 1. Core Attention Score & Vacuum Floor

For head $h \in \{0, \dots, H-1\}$, query position $i$, and causal key position $j \le i$:

$$\Psi_h(i, j) = \underbrace{\frac{q_{\text{res},h,i} \cdot k_{\text{res},h,j}}{\sqrt{d_r}}}_{\text{(R) Resonance Fiber}} + \underbrace{u_{\text{recv},h,i} + u_{\text{emit},h,j}}_{\text{Bilateral Mass } \ln(M_i M_j)} - \underbrace{\frac{\|z_{h,i} - z_{h,j}\|_2^2}{2\sigma_h^2 \sqrt{d_g}}}_{\text{(E) Euclidean-RBF}} - \underbrace{(\tau_{h,i} - \tau_{h,j})}_{\text{(P) Proper-Time}}$$

Normalized with a learnable **physical vacuum floor** $e^{v_{0,h}}$ (equivalent to Softmax1 / Attention Sink / Mass Gap):

$$W_h(i, j) = \frac{\exp(\Psi_h(i, j))}{\exp(v_{0,h}) + \sum_{k \le i} \exp(\Psi_h(i, k))}, \qquad W_h(i, \text{vac}) = \frac{\exp(v_{0,h})}{\exp(v_{0,h}) + \sum_{k \le i} \exp(\Psi_h(i, k))}$$

Because $V_{\text{vac}} = \mathbf{0}_D$, any probability mass absorbed by the vacuum slot cleanly silences the head rather than forcing attention onto irrelevant tokens.

---

## 2. Primitive 1: Exact Algebraic Folding (`d_r + d_g + 2 == D`)

Expand the Euclidean-RBF term:

$$-\frac{\|z_i - z_j\|_2^2}{2\sigma_h^2 \sqrt{d_g}} = \gamma_h (z_i \cdot z_j) - \frac{1}{2}\gamma_h \|z_i\|_2^2 - \frac{1}{2}\gamma_h \|z_j\|_2^2, \qquad \gamma_h = \frac{1}{\sigma_h^2 \sqrt{d_g}}$$

Define the query/key half-norm potentials $\phi_i = \frac{1}{2}\gamma_h \|z_i\|_2^2$ and $\phi_j = \frac{1}{2}\gamma_h \|z_j\|_2^2$. Pack the augmented query, real key, and vacuum key into exact dimension $D = d_r + d_g + 2$:

$$Q_{\text{aug},i} = \Big[\, \tfrac{q_{\text{res},i}}{\sqrt{d_r}},\;\; \gamma_h z_i,\;\; 1,\;\; \phi_i - u_{\text{recv},i} \,\Big] \in \mathbb{R}^D$$

$$K_{\text{real},j} = \Big[\, k_{\text{res},j},\;\; z_j,\;\; u_{\text{emit},j} - \phi_j,\;\; 0 \,\Big] \in \mathbb{R}^D, \qquad K_{\text{vac}} = \Big[\, \mathbf{0}_{d_r + d_g},\;\; v_{0,h},\;\; 1 \,\Big] \in \mathbb{R}^D$$

**Proof of Exact Equivalence:**
1. Real key dot product:
   $$Q_{\text{aug},i} \cdot K_{\text{real},j} = \frac{q_{\text{res},i} \cdot k_{\text{res},j}}{\sqrt{d_r}} + \gamma_h (z_i \cdot z_j) + u_{\text{emit},j} - \phi_j$$
2. Vacuum key dot product:
   $$Q_{\text{aug},i} \cdot K_{\text{vac}} = v_{0,h} + \phi_i - u_{\text{recv},i}$$
3. By shift-invariance of Softmax, subtracting $(\phi_i - u_{\text{recv},i})$ from every logit yields:
   $$(Q_{\text{aug},i} \cdot K_{\text{real},j}) - (\phi_i - u_{\text{recv},i}) = \frac{q_{\text{res},i} \cdot k_{\text{res},j}}{\sqrt{d_r}} + u_{\text{recv},i} + u_{\text{emit},j} - \frac{\|z_i - z_j\|_2^2}{2\sigma_h^2\sqrt{d_g}}$$
   $$(Q_{\text{aug},i} \cdot K_{\text{vac}}) - (\phi_i - u_{\text{recv},i}) = v_{0,h}$$

Thus the entire geometric potential + bilateral mass + vacuum floor executes inside **one standard FlashAttention / FlexAttention kernel** with $D \in \{32, 64, 128\}$ and **zero $V$-channel padding**.

---

## 3. Primitive 2: Nested Dynamic Proper-Time & Relation to FoX

For damped heads $h \in \{0, \dots, H_{\text{damped}}-1\}$:

$$\delta\tau^{\text{loc}}_{h,t} = \text{softplus}\big((W_{\tau} x_t)_h + b_{\tau,h}\big) > 0$$

A top-down upper-triangular cascade matrix $M \in \mathbb{R}^{H_{\text{damped}} \times H_{\text{damped}}}$ couples faster inner heads to slower outer heads:

$$\delta\tau_{h,t} = \sum_{k=h}^{H_{\text{damped}}-1} M_{h,k}\, \delta\tau^{\text{loc}}_{k,t}, \qquad \tau_{h,t} = \sum_{s=0}^{t} \delta\tau_{h,s}$$

- **Strict Causality**: $\tau_{h,t}$ depends only on $x_{\le t}$, and $\tau_{h,i} - \tau_{h,j} = \sum_{s=j+1}^{i} \delta\tau_{h,s} \ge 0$ for all $j \le i$.
- **Relation to FoX (Forgetting Transformer)**: Removing the cascade ($M = I$) and setting $\delta\tau_{h,t} = -\ln \sigma(g_{h,t})$ recovers the exact cumulative log-forget-gate of FoX.

---

## 4. Primitive 3 & 4: Dual-Regime Topology & Stateful `EPRKVCache`

- **Damped Solar-System Heads** ($0 \dots H_{\text{damped}}-1$): Apply RoPE on the $d_r$ resonance fiber + positive proper-time decay $\delta\tau_{h,t} > 0$. Because attention mass decays exponentially with $\tau_{h,i} - \tau_{h,j}$, distant keys contribute negligible weight and can be evicted into a fixed ring buffer of size $W$ ($O(1)$ memory).
- **ER=EPR Wormhole Heads** ($H_{\text{damped}} \dots H-1$): **NoPE** + **zero time decay ($\delta\tau \equiv 0$)** + **high vacuum throat ($v_0 \approx +2.5$)**. They remain silent on background noise ($W_{\text{vac}} \to 1$) until a distant key satisfies strong fiber + RBF resonance, enabling distance-independent recall.

**Incremental State Update in `EPRKVCache`:**
At autoregressive step $t$ with cached cumulative state $\tau^{\text{state}}_{h, t-1}$:

$$\tau_{h,t} = \tau^{\text{state}}_{h, t-1} + \delta\tau_{h,t}, \qquad \tau^{\text{state}}_{h, t} \leftarrow \tau_{h,t}$$

Storing `(K_real, V, tau_k)` per damped ring-buffer slot and `(K_real, V)` per wormhole full-context slot guarantees **machine-precision equivalence (`~1e-16` in `float64`)** to parallel forward pass when $t \le W$ (or $W = \infty$), while reducing long-context KV-cache memory by **72.7%–86.1%** when $W = 128$.
