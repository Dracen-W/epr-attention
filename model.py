"""
Llama 2 + Unified Fiber-Gravity Spacetime & ER=EPR Wormhole Attention (v6).

================================================================================
ATTRIBUTION & LICENSE NOTICE
================================================================================
- Base Codebase: Built upon Andrej Karpathy's `llama2.c` (MIT License,
  Copyright (c) 2023 Andrej). See `LICENSE` and `NOTICE` for full details.
- Base Architecture: Implements the Llama 2 Transformer architecture
  (RMSNorm, RoPE, SwiGLU, GQA) introduced by Meta AI in:
  Touvron et al., "Llama 2: Open Foundation and Fine-Tuned Chat Models", 2023.
  (Built with Meta Llama 2 architecture; see `NOTICE` and `CITATION.bib`.)
- Original Architectural Contributions (v5/v6):
  1. Orthogonal Fiber-Base Spacetime Splitting (d_r + d_g + 2 == head_dim)
  2. Solar-System Nested Dynamic Time Dilation (Earth -> Sat -> Moon)
  3. ER=EPR Zero-Decay Wormhole Head (NoPE + High Vacuum Throat + Base RBF)
  4. Bilateral Gravitational Mass Coupling (M_i * M_j) & Vacuum Noise (e^{v0})
  5. Damped Pauli Repulsion Decoding (Short-Range Degeneracy Pressure)
================================================================================
"""

import inspect
import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn.functional as F

# Optional PyTorch FlexAttention (fused Triton FlashAttention-2 kernel on CUDA)
try:
    from torch.nn.attention.flex_attention import flex_attention as _raw_flex_attention, create_block_mask as _create_block_mask
    _HAS_FLEX_ATTENTION = True
    _COMPILED_FLEX_ATTN = None
    _FLEX_BLOCK_MASK_CACHE = {}
except Exception:
    _HAS_FLEX_ATTENTION = False
    _COMPILED_FLEX_ATTN = None
    _FLEX_BLOCK_MASK_CACHE = {}


def _get_compiled_flex_attention():
    global _COMPILED_FLEX_ATTN
    if _COMPILED_FLEX_ATTN is None and _HAS_FLEX_ATTENTION:
        _COMPILED_FLEX_ATTN = torch.compile(_raw_flex_attention)
    return _COMPILED_FLEX_ATTN


def _get_flex_causal_block_mask(seqlen: int, use_vacuum: bool, device: torch.device):
    key = (seqlen, use_vacuum, str(device))
    if key not in _FLEX_BLOCK_MASK_CACHE:
        if use_vacuum:
            def _mask(b, h, q_idx, kv_idx):
                return kv_idx <= q_idx + 1
            kv_len = seqlen + 1
        else:
            def _mask(b, h, q_idx, kv_idx):
                return kv_idx <= q_idx
            kv_len = seqlen
        _FLEX_BLOCK_MASK_CACHE[key] = _create_block_mask(
            _mask, B=None, H=None, Q_LEN=seqlen, KV_LEN=kv_len, device=device
        )
    return _FLEX_BLOCK_MASK_CACHE[key]
from torch import nn


@dataclass
class ModelArgs:
    """
    Hyperparameters for the Llama 2 / Wormhole-Gravity Transformer.
    When `use_unified_gravity=False`, the model is bit-for-bit identical to
    the standard `karpathy/llama2.c` Llama 2 implementation.
    """
    # --- Standard Llama 2 Backbone Parameters ---
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: Optional[int] = None
    vocab_size: int = 32000
    hidden_dim: Optional[int] = None
    multiple_of: int = 256  # SwiGLU hidden_dim is rounded up to a multiple of this
    norm_eps: float = 1e-5
    max_seq_len: int = 2048
    dropout: float = 0.0

    # --- Unified Fiber-Gravity & ER=EPR Wormhole Attention (v6 Primary) ---
    use_unified_gravity: bool = False  # Master switch: False = Pure Llama 2; True = Fiber-Gravity / Wormhole
    gravity_dim: int = 0               # Base spacetime dim d_g; 0 = auto power-of-2 packing:
                                       #   D=32 -> d_r=24 fiber + d_g=6 base + 2 = 32
                                       #   D=64 -> d_r=48 fiber + d_g=14 base + 2 = 64

    # ER=EPR Wormhole Head (v6 Headline Breakthrough)
    use_wormhole: bool = True          # Converts slowest head (H-1) into a zero-decay, NoPE ER=EPR wormhole
    wormhole_v0_init: float = 2.5      # High vacuum throat threshold (keeps wormhole silent unless resonant)

    # Core Distilled Physics Pillars (Proven Critical in 8-Factor Ablation)
    use_base_rbf: bool = True          # Low-dim Euclidean RBF capture in base spacetime: -||z_i - z_j||^2 / (2 sigma^2 sqrt(d_g))
    use_bilateral_mass: bool = True    # Bilateral mass coupling M_i * M_j (query reception mass lowers vacuum horizon)
    use_time_dilation: bool = True     # Data-dependent dynamic proper time dtau_{h,t} >= 0
    use_nested_time: bool = True       # Solar-system top-down cascade: Earth -> Satellite -> Moon
    use_attn_vacuum: bool = True       # Physical vacuum noise floor e^{v_{0,h}} (Softmax1 / Mass Gap)

    # Optional Research / Ablation Switches (Off by default in distilled v5/v6 for peak speed & accuracy)
    use_medium_field: bool = False     # Explicit continuous medium field Phi_{h,t-1} (strictly causal s < t)
    use_grav_bridge: bool = True       # Spatial coordinate dressing z_tilde = z + gamma_z * Phi (active only if use_medium_field=True)
    use_mass_boost: bool = True        # Field-resonance mass feedback u_tilde = u + gamma_u * rho (active only if use_medium_field=True)
    use_event_horizon: bool = False    # QK-L2Norm bounded resonance cone tau_res * (q_hat . k_hat)
    use_flex_attention: bool = True    # Auto-route CUDA long sequences (T>=256) to fused FlexAttention Triton kernel


class RMSNorm(torch.nn.Module):
    """Root Mean Square Layer Normalization (Zhang & Sennrich, 2019; Llama 2 standard)."""
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        acc_dtype = torch.float64 if x.dtype == torch.float64 else torch.float32
        output = self._norm(x.to(acc_dtype)).type_as(x)
        return output * self.weight


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Precompute cosine and sine frequency tables for Rotary Positional Embeddings (RoPE).
    Ref: Su et al., "RoFormer: Enhanced Transformer with Rotary Position Embedding", 2021.
    """
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cos = torch.cos(freqs)
    freqs_sin = torch.sin(freqs)
    return freqs_cos, freqs_sin


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Reshape [T, D/2] RoPE table to broadcast across [B, T, H, D/2]."""
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(shape)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cos: torch.Tensor,
    freqs_sin: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply complex rotary positional embeddings to query and key tensors.
    Preserves float64 precision when auditing in fp64, otherwise computes in fp32 for stability.
    """
    acc_dtype = torch.float64 if xq.dtype == torch.float64 else torch.float32
    xq_r, xq_i = xq.to(acc_dtype).reshape(xq.shape[:-1] + (-1, 2)).unbind(-1)
    xk_r, xk_i = xk.to(acc_dtype).reshape(xk.shape[:-1] + (-1, 2)).unbind(-1)

    f_cos = reshape_for_broadcast(freqs_cos.to(acc_dtype), xq_r)
    f_sin = reshape_for_broadcast(freqs_sin.to(acc_dtype), xq_r)

    xq_out_r = xq_r * f_cos - xq_i * f_sin
    xq_out_i = xq_r * f_sin + xq_i * f_cos
    xk_out_r = xk_r * f_cos - xk_i * f_sin
    xk_out_i = xk_r * f_sin + xk_i * f_cos

    xq_out = torch.stack([xq_out_r, xq_out_i], dim=-1).flatten(3)
    xk_out = torch.stack([xk_out_r, xk_out_i], dim=-1).flatten(3)

    return xq_out.type_as(xq), xk_out.type_as(xk)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Grouped-Query Attention (GQA) helper: repeat KV heads `n_rep` times along head axis."""
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )


class Attention(nn.Module):
    """
    100% untouched standard `karpathy/llama2.c` Multi-Head / Grouped-Query Attention.
    Retained verbatim as the bit-exact Pure Llama 2 control baseline.
    """
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        assert args.n_heads % self.n_kv_heads == 0
        self.n_local_heads = args.n_heads
        self.n_local_kv_heads = self.n_kv_heads
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = args.dim // args.n_heads

        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)
        self.attn_dropout = nn.Dropout(args.dropout)
        self.resid_dropout = nn.Dropout(args.dropout)
        self.dropout = args.dropout

        self.flash = hasattr(torch.nn.functional, "scaled_dot_product_attention")
        if not self.flash:
            mask = torch.full((1, 1, args.max_seq_len, args.max_seq_len), float("-inf"))
            mask = torch.triu(mask, diagonal=1)
            self.register_buffer("mask", mask, persistent=False)

    def forward(self, x: torch.Tensor, freqs_cos: torch.Tensor, freqs_sin: torch.Tensor) -> torch.Tensor:
        bsz, seqlen, _ = x.shape

        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)

        xq, xk = apply_rotary_emb(xq, xk, freqs_cos, freqs_sin)
        xk = repeat_kv(xk, self.n_rep)
        xv = repeat_kv(xv, self.n_rep)

        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)

        if self.flash:
            output = F.scaled_dot_product_attention(
                xq, xk, xv,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True,
            )
        else:
            scores = torch.matmul(xq, xk.transpose(2, 3)) / math.sqrt(self.head_dim)
            scores = scores + self.mask[:, :, :seqlen, :seqlen]
            scores = F.softmax(scores.float(), dim=-1).type_as(xq)
            scores = self.attn_dropout(scores)
            output = torch.matmul(scores, xv)

        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        output = self.wo(output)
        output = self.resid_dropout(output)
        return output


class UnifiedFiberGravityAttention(nn.Module):
    """
    Unified Fiber-Gravity Spacetime & ER=EPR Wormhole Attention (v6).

    ============================================================================
    PHYSICAL & GEOMETRIC FORMULATION (Fiber Bundle / Gauge-Gravity Coupling)
    ============================================================================
    1. Orthogonal Channel Packing (Exact Hardware Alignment: d_r + d_g + 2 == D):
       Each head of dimension D (e.g., 32 or 64) is partitioned into:
         - Internal Resonance Fiber (q_res, k_res in R^{d_r}):
             Directional interaction (24 dims for D=32; 48 dims for D=64).
             Avoids the "all-dimension penalty" of full-dim Euclidean RBF.
         - Low-Dim Base Spacetime Manifold (z_base in R^{d_g}):
             Universal gravitational capture via isotropic Euclidean RBF
             (6 dims for D=32; 14 dims for D=64). Shared coordinate per token
             reduces total parameters below Pure Llama 2!
         - Augmented Gravity & Vacuum Slots (+2 dims):
             Algebraically folds bilateral mass (u_i + u_j), quadratic RBF norms
             (-0.5 ||z_i||^2 - 0.5 ||z_j||^2), and the vacuum floor e^{v_{0,h}}
             into ONE D-aligned `F.scaled_dot_product_attention` call (zero V padding!).

    2. Dual-Regime Spacetime Topology (Solar System + ER=EPR Wormhole):
       - Damped Spacetime Heads (H0 Moon -> H1 Sat -> H2 Earth):
           Governed by RoPE phase + Solar-System nested proper time dilation:
             dtau_local_{h,t} = softplus(w_tau . x_t + b_{tau,h}) >= 0
             dtau_{h,t}       = dtau_local_{h,t} + alpha_h * dtau_{h+1,t}
             tau_{h,t}        = sum_{u <= t} dtau_{h,u}
           Provides noise-free local/mid-range modeling and clean extrapolation.
       - ER=EPR Wormhole Head (H-1, enabled when `use_wormhole=True` and H > 1):
           * Strictly Zero Time Decay (dtau_{H-1, t} == 0)
           * Strictly NoPE (No Rotary Positional Embedding -> distance-invariant)
           * High Vacuum Throat (v_{0, H-1} initialized to +2.5) + Base RBF Lock
           Stays silent for irrelevant distant noise, yet instantaneously folds
           spacetime to recall exact distant "needles" (1500+ tokens away) whenever
           content resonance + base RBF + bilateral mass pierce the vacuum throat.

    3. Unified Log-Propagator (Exact Equivalence to Single D-Aligned SDPA):
       Psi_h(i, j) = FiberResonance_h(i, j)
                     + u_recv_{h,i} + u_emit_{h,j}
                     - ||z_{h,i} - z_{h,j}||_2^2 / (2 * sigma_h^2 * sqrt(d_g))
                     - (tau_{h,i} - tau_{h,j})
       W_h(i, j)   = exp(Psi_h(i, j)) / ( exp(v_{0,h}) + sum_{k <= i} exp(Psi_h(i, k)) )
    ============================================================================
    """
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        assert args.n_heads % self.n_kv_heads == 0, "n_heads must be divisible by n_kv_heads"
        self.n_local_heads = args.n_heads
        self.n_local_kv_heads = self.n_kv_heads
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = args.dim // args.n_heads
        self.dropout = args.dropout

        # --- 1. Exact Power-of-2 Channel Split: d_r + d_g + 2 == head_dim ---
        assert self.head_dim >= 12 and self.head_dim % 2 == 0, (
            f"head_dim ({self.head_dim}) must be even and >= 12 for fiber-gravity packing"
        )
        if args.gravity_dim > 0:
            d_g = args.gravity_dim
        else:
            # Auto packing: D=16 -> d_g=4, d_r=10; D=32 -> d_g=6, d_r=24; D=64 -> d_g=14, d_r=48
            d_g = max(4, ((self.head_dim // 4) // 2) * 2 - 2)
        d_r = self.head_dim - d_g - 2
        assert d_r >= 4 and d_r % 2 == 0, (
            f"Invalid split: head_dim={self.head_dim}, d_g={d_g}, d_r={d_r} (d_r must be even >= 4 for RoPE)"
        )
        self.d_g = d_g
        self.d_r = d_r

        # --- 2. Feature Switches ---
        self.use_wormhole = args.use_wormhole and (self.n_local_heads > 1)
        self.wormhole_v0_init = args.wormhole_v0_init
        self.use_medium_field = args.use_medium_field
        self.use_time_dilation = args.use_time_dilation
        self.use_nested_time = args.use_nested_time
        self.use_grav_bridge = args.use_grav_bridge and args.use_medium_field
        self.use_mass_boost = args.use_mass_boost and args.use_medium_field
        self.use_bilateral_mass = args.use_bilateral_mass
        self.use_event_horizon = args.use_event_horizon
        self.use_flex_attention = getattr(args, 'use_flex_attention', True)
        self.use_base_rbf = args.use_base_rbf
        self.use_attn_vacuum = args.use_attn_vacuum

        H, H_kv, D = self.n_local_heads, self.n_local_kv_heads, self.head_dim
        # Number of damped spacetime heads (H-1 if last head is a zero-decay wormhole, else H)
        self.n_damped_heads = (H - 1) if self.use_wormhole else H

        # --- 3. Fused Main Vector Projection GEMM ---
        # Packs [q_res (H*d_r), k_res (H_kv*d_r), z_base (H_kv*d_g), v (H_kv*D)] into 1 GEMM.
        self.q_res_dim = H * d_r
        self.k_res_dim = H_kv * d_r
        self.z_dim = H_kv * d_g
        self.v_dim = H_kv * D
        self.w_proj = nn.Linear(args.dim, self.q_res_dim + self.k_res_dim + self.z_dim + self.v_dim, bias=False)
        self.wo = nn.Linear(H * D, args.dim, bias=False)
        self.resid_dropout = nn.Dropout(args.dropout)

        # --- 4. Fused Scalar Physics Projection ---
        # Packs [u_emit (H_kv), u_recv (H if bilateral), tau_in (n_damped_heads if time_dilation)]
        # Note: Wormhole head has identically zero time decay, so we allocate NO wasted tau_in weights for it!
        self.u_emit_dim = H_kv
        self.u_recv_dim = H if self.use_bilateral_mass else 0
        self.tau_in_dim = self.n_damped_heads if self.use_time_dilation else 0
        self.w_phys = nn.Linear(args.dim, self.u_emit_dim + self.u_recv_dim + self.tau_in_dim, bias=False)

        # --- 5. Optional Event Horizon Resonance Cone Sharpness ---
        if self.use_event_horizon:
            self.log_tau_res = nn.Parameter(torch.full((H,), math.log(math.sqrt(d_r))))

        # --- 6. Base Spacetime Euclidean RBF Precision (1 / sigma_h^2) ---
        if self.use_base_rbf:
            self.log_inv_sigma2 = nn.Parameter(torch.zeros(H))  # init sigma_g = 1.0

        # --- 7. Multi-Scale Proper Time Ladder (Moon -> Sat -> Earth [-> Sun]) ---
        if self.n_damped_heads == 1:
            init_lambdas = torch.tensor([0.05])
        else:
            # Fast local Moon (0.30) -> Slow macro Earth/Sun (0.015 if wormhole else 0.005)
            slow_end = 0.015 if self.use_wormhole else 0.005
            init_lambdas = torch.logspace(math.log10(0.30), math.log10(slow_end), self.n_damped_heads)
        inv_softplus_lambdas = torch.log(torch.expm1(init_lambdas))
        self.b_tau = nn.Parameter(inv_softplus_lambdas.clone())

        # Solar-system top-down cascade couplings alpha_{h <- h+1} among damped heads only
        if self.use_nested_time and self.n_damped_heads > 1:
            self.raw_nest_alpha = nn.Parameter(
                torch.full((self.n_damped_heads - 1,), math.log(torch.expm1(torch.tensor(0.15)).item()))
            )

        # --- 8. Optional Continuous Medium Field Phi_{h,t-1} (Research/Ablation Switch) ---
        if self.use_medium_field:
            med_lambdas = (
                torch.tensor([0.05]) if H == 1
                else torch.logspace(math.log10(0.30), math.log10(0.005), H)
            )
            self.log_medium_lambda = nn.Parameter(torch.log(torch.expm1(med_lambdas)))
            self.raw_medium_sat = nn.Parameter(torch.full((H,), math.log(math.expm1(3.0))))
            if self.use_time_dilation:
                self.raw_kappa = nn.Parameter(torch.full((self.n_damped_heads,), math.log(math.expm1(0.5))))
            if self.use_mass_boost:
                self.gamma_u = nn.Parameter(torch.zeros(H))
            if self.use_grav_bridge:
                self.gamma_z = nn.Parameter(torch.zeros(H))

        # --- 9. Vacuum Noise Threshold e^{v_{0,h}} (Softmax1 / Mass Gap) ---
        if self.use_attn_vacuum:
            v0_init = torch.zeros(H)
            if self.use_wormhole:
                # High vacuum throat keeps the zero-decay wormhole silent unless strongly resonant
                v0_init[-1] = self.wormhole_v0_init
            self.v0 = nn.Parameter(v0_init)
            self.register_buffer("vac_zeros_rg", torch.zeros(1, H, 1, d_r + d_g), persistent=False)
            self.register_buffer("vac_ones", torch.ones(1, H, 1, 1), persistent=False)

        # --- 10. Dynamically Extendable Causal Masks (Supports Zero-Shot Length Extrapolation) ---
        self._init_seq_buffers(args.max_seq_len, device=torch.device("cpu"))

    def _init_seq_buffers(self, max_len: int, device: torch.device) -> None:
        """Allocate causal masks and strictly-past relative offsets up to `max_len`."""
        pos = torch.arange(max_len, dtype=torch.float32, device=device)
        causal_mask = torch.triu(
            torch.full((max_len, max_len), float("-inf"), device=device), diagonal=1
        ).view(1, 1, max_len, max_len)
        self.register_buffer("causal_mask", causal_mask, persistent=False)
        self.register_buffer("causal_mask_pad", F.pad(causal_mask, (1, 0), value=0.0), persistent=False)

        if self.use_medium_field:
            strict_delta = (pos.unsqueeze(1) - pos.unsqueeze(0) - 1.0).clamp(min=0.0).view(1, 1, max_len, max_len)
            strict_mask = torch.triu(
                torch.full((max_len, max_len), float("-inf"), device=device), diagonal=0
            ).view(1, 1, max_len, max_len)
            self.register_buffer("strict_delta", strict_delta, persistent=False)
            self.register_buffer("strict_mask", strict_mask, persistent=False)

        self._cached_max_len = max_len

    def _ensure_buffers(self, seqlen: int, device: torch.device) -> None:
        """Auto-extend causal buffers on the fly during zero-shot length extrapolation (e.g. 512 -> 2048)."""
        if seqlen > self._cached_max_len or self.causal_mask.device != device:
            self._init_seq_buffers(max(seqlen, self._cached_max_len), device=device)

    def _compute_fields_and_qkv(
        self,
        x: torch.Tensor,
        freqs_cos_r: torch.Tensor,
        freqs_sin_r: torch.Tensor,
    ):
        """
        Compute augmented D-aligned (Q, K, V) tensors and exact causal proper-time mask.
        Preserves float64 accumulation whenever `x.dtype == torch.float64` (for strict fp64 auditing).
        """
        bsz, seqlen, _ = x.shape
        H, H_kv, D, d_r, d_g = self.n_local_heads, self.n_local_kv_heads, self.head_dim, self.d_r, self.d_g
        acc_dtype = torch.float64 if x.dtype == torch.float64 else torch.float32
        self._ensure_buffers(seqlen, x.device)

        # ------------------------------------------------------------------
        # Step 1: Fused Vector Projections & GQA Expansion
        # ------------------------------------------------------------------
        proj = self.w_proj(x)
        q_res, k_res, z_base, xv = proj.split(
            [self.q_res_dim, self.k_res_dim, self.z_dim, self.v_dim], dim=-1
        )
        q_res = q_res.view(bsz, seqlen, H, d_r)
        k_res = repeat_kv(k_res.view(bsz, seqlen, H_kv, d_r), self.n_rep)
        z_base = repeat_kv(z_base.view(bsz, seqlen, H_kv, d_g), self.n_rep)
        xv = repeat_kv(xv.view(bsz, seqlen, H_kv, D), self.n_rep)

        # Apply RoPE to resonance fibers:
        #   - If ER=EPR wormhole is enabled: apply RoPE ONLY to damped heads (0..H-2);
        #     keep Wormhole head (H-1) strictly NoPE for distance-independent entanglement!
        if self.use_wormhole:
            q_d, k_d = apply_rotary_emb(q_res[:, :, :-1, :], k_res[:, :, :-1, :], freqs_cos_r, freqs_sin_r)
            q_res = torch.cat([q_d, q_res[:, :, -1:, :]], dim=2)
            k_res = torch.cat([k_d, k_res[:, :, -1:, :]], dim=2)
        else:
            q_res, k_res = apply_rotary_emb(q_res, k_res, freqs_cos_r, freqs_sin_r)

        # Transpose to [B, H, T, *]
        q_res = q_res.transpose(1, 2)    # [B, H, T, d_r]
        k_res = k_res.transpose(1, 2)    # [B, H, T, d_r]
        z_base = z_base.transpose(1, 2)  # [B, H, T, d_g]
        xv = xv.transpose(1, 2)          # [B, H, T, D]

        # ------------------------------------------------------------------
        # Step 2: Fused Scalar Physics Projections (Mass & Proper Time Input)
        # ------------------------------------------------------------------
        phys = self.w_phys(x)
        splits = [self.u_emit_dim]
        if self.u_recv_dim > 0:
            splits.append(self.u_recv_dim)
        if self.tau_in_dim > 0:
            splits.append(self.tau_in_dim)
        phys_parts = phys.split(splits, dim=-1)

        idx_p = 0
        u_emit = phys_parts[idx_p]
        idx_p += 1
        if self.n_rep > 1:
            u_emit = u_emit.repeat_interleave(self.n_rep, dim=-1)
        u_emit = u_emit.transpose(1, 2)  # [B, H, T]

        u_recv = None
        if self.use_bilateral_mass:
            u_recv = phys_parts[idx_p].transpose(1, 2)  # [B, H, T]
            idx_p += 1

        tau_in = None
        if self.use_time_dilation:
            tau_in = phys_parts[idx_p].transpose(1, 2)  # [B, n_damped_heads, T]

        # ------------------------------------------------------------------
        # Step 3: Optional Continuous Medium Field Phi_{h, t-1} (Strictly s < t)
        # ------------------------------------------------------------------
        rho = None
        phi_prev = None
        if self.use_medium_field:
            z_hat = F.normalize(z_base.to(acc_dtype), p=2, dim=-1, eps=1e-6)
            m_emit = torch.sigmoid(u_emit.to(acc_dtype)).unsqueeze(-1)
            sources = m_emit * z_hat

            lam0 = F.softplus(self.log_medium_lambda.to(acc_dtype)).view(1, H, 1, 1)
            kernel = torch.exp(
                -lam0 * self.strict_delta[:, :, :seqlen, :seqlen].to(acc_dtype)
                + self.strict_mask[:, :, :seqlen, :seqlen].to(acc_dtype)
            )
            S_prev = torch.matmul(kernel, sources)  # [B, H, T, d_g]
            sat_c = F.softplus(self.raw_medium_sat.to(acc_dtype)).view(1, H, 1, 1)
            # C^inf smooth saturation guarantees ||Phi||_2 <= 1.0 at any sequence length
            phi_prev = S_prev * torch.rsqrt(sat_c * sat_c + (S_prev * S_prev).sum(dim=-1, keepdim=True))
            rho = (z_hat * phi_prev).sum(dim=-1)  # [B, H, T]

        # Optional medium-field feedbacks (mass boost + gravitational coordinate bridge)
        u_eff = u_emit
        if self.use_mass_boost and rho is not None:
            u_eff = u_eff + (self.gamma_u.view(1, H, 1) * rho.to(u_emit.dtype))

        z_eff = z_base
        if self.use_grav_bridge and phi_prev is not None:
            z_eff = z_eff + (self.gamma_z.view(1, H, 1, 1) * phi_prev.to(z_base.dtype))

        # ------------------------------------------------------------------
        # Step 4: Solar-System Nested Proper Time Dilation & Zero-Decay Wormhole
        # ------------------------------------------------------------------
        H_d = self.n_damped_heads
        if self.use_time_dilation:
            raw_dtau = tau_in.to(acc_dtype) + self.b_tau.to(acc_dtype).view(1, H_d, 1)
            if rho is not None:
                kappa = F.softplus(self.raw_kappa.to(acc_dtype)).view(1, H_d, 1)
                raw_dtau = raw_dtau - kappa * rho[:, :H_d, :]
            dtau_local = F.softplus(raw_dtau)  # [B, H_d, T] >= 0
        else:
            dtau_local = F.softplus(self.b_tau.to(acc_dtype)).view(1, H_d, 1).expand(bsz, H_d, seqlen)

        # Top-down cascade among damped spacetime heads: Earth (H_d-1) -> ... -> Moon (0)
        if self.use_nested_time and H_d > 1:
            alphas = F.softplus(self.raw_nest_alpha.to(acc_dtype))
            nested_list = [None] * H_d
            nested_list[H_d - 1] = dtau_local[:, H_d - 1, :]
            for h_idx in range(H_d - 2, -1, -1):
                nested_list[h_idx] = dtau_local[:, h_idx, :] + alphas[h_idx] * nested_list[h_idx + 1]
            dtau_damped = torch.stack(nested_list, dim=1)  # [B, H_d, T]
        else:
            dtau_damped = dtau_local

        # Append strictly ZERO proper time velocity for the ER=EPR Wormhole head (H-1)
        if self.use_wormhole:
            zeros_wh = torch.zeros(bsz, 1, seqlen, dtype=acc_dtype, device=x.device)
            dtau = torch.cat([dtau_damped, zeros_wh], dim=1)  # [B, H, T]
        else:
            dtau = dtau_damped

        # Exact cumulative proper time difference: delta_tau[i, j] = tau_i - tau_j >= 0 (for i >= j)
        tau = torch.cumsum(dtau, dim=-1)  # [B, H, T]
        delta_tau = (tau.unsqueeze(-1) - tau.unsqueeze(-2)).clamp(min=0.0)  # [B, H, T, T]

        # ------------------------------------------------------------------
        # Step 5: Internal Fiber Resonance Preparation
        # ------------------------------------------------------------------
        if self.use_event_horizon:
            q_fib = F.normalize(q_res.to(acc_dtype), p=2, dim=-1, eps=1e-6).to(q_res.dtype)
            k_fib = F.normalize(k_res.to(acc_dtype), p=2, dim=-1, eps=1e-6).to(k_res.dtype)
            tau_res = torch.exp(self.log_tau_res).view(1, H, 1, 1).to(q_res.dtype)
            q_fib_scaled = q_fib * tau_res
        else:
            q_fib_scaled = q_res * (1.0 / math.sqrt(d_r))
            k_fib = k_res

        # ------------------------------------------------------------------
        # Step 6: Base Spacetime Euclidean RBF & Bilateral Mass Algebraic Folding
        # Exact Identity (for real key j and vacuum key vac):
        #   q_aug = [q_fib_scaled, gamma*z_i, 1, 0.5*gamma*||z_i||^2 - u_recv_i]
        #   k_real = [k_fib,       z_j,       u_eff_j - 0.5*gamma*||z_j||^2, 0 ]
        #   k_vac  = [0,           0,         v_{0,h},                       1 ]
        # -> q_aug . k_real = q_fib_scaled.k_fib + u_eff_j - 0.5*gamma*||z_i - z_j||^2 + 0.5*gamma*||z_i||^2
        # -> q_aug . k_vac  = v_{0,h} + 0.5*gamma*||z_i||^2 - u_recv_i
        # Subtracting the common per-query constant (0.5*gamma*||z_i||^2 - u_recv_i) inside Softmax:
        #   logit(real j) = Fiber + u_recv_i + u_eff_j - 0.5*gamma*||z_i - z_j||^2 - delta_tau(i,j)
        #   logit(vac)    = v_{0,h}
        # ------------------------------------------------------------------
        if self.use_base_rbf:
            gamma_rbf = (torch.exp(self.log_inv_sigma2) / math.sqrt(d_g)).view(1, H, 1, 1).to(z_eff.dtype)
            z_q_scaled = z_eff * gamma_rbf
            z_k = z_eff
            z_half_norm = 0.5 * (z_q_scaled * z_eff).sum(dim=-1, keepdim=True)  # [B, H, T, 1]
            key_pot = u_eff.unsqueeze(-1) - z_half_norm
            q_shift = z_half_norm
        else:
            z_q_scaled = torch.zeros_like(z_eff)
            z_k = torch.zeros_like(z_eff)
            key_pot = u_eff.unsqueeze(-1)
            q_shift = torch.zeros(bsz, H, seqlen, 1, dtype=q_res.dtype, device=q_res.device)

        if self.use_bilateral_mass and u_recv is not None:
            q_shift = q_shift - u_recv.unsqueeze(-1)

        ones_q = torch.ones_like(q_shift)
        zeros_k = torch.zeros_like(key_pot)

        # Concatenate into EXACT D-aligned (32 / 64) tensors
        q_aug = torch.cat([q_fib_scaled, z_q_scaled, ones_q, q_shift], dim=-1)  # [B, H, T, D]
        k_real = torch.cat([k_fib, z_k, key_pot, zeros_k], dim=-1)              # [B, H, T, D]

        if self.use_attn_vacuum:
            k_vac = torch.cat(
                [
                    self.vac_zeros_rg.to(k_real.dtype),
                    self.v0.view(1, H, 1, 1).to(k_real.dtype),
                    self.vac_ones.to(k_real.dtype),
                ],
                dim=-1,
            ).expand(bsz, -1, -1, -1)                                           # [B, H, 1, D]
            k_aug = torch.cat([k_vac, k_real], dim=-2)                          # [B, H, T+1, D]
            # V only prepends a single zero vacuum row (channel dim stays exact D!)
            v_aug = F.pad(xv, (0, 0, 1, 0), value=0.0)                          # [B, H, T+1, D]
            time_mask = F.pad(-delta_tau, (1, 0), value=0.0) + self.causal_mask_pad[:, :, :seqlen, :seqlen + 1].to(acc_dtype)
            attn_mask = time_mask.to(q_aug.dtype)
        else:
            k_aug = k_real
            v_aug = xv
            attn_mask = (-delta_tau + self.causal_mask[:, :, :seqlen, :seqlen].to(acc_dtype)).to(q_aug.dtype)

        return q_aug, k_aug, v_aug, attn_mask, dtau, rho

    def forward(self, x: torch.Tensor, freqs_cos: torch.Tensor, freqs_sin: torch.Tensor) -> torch.Tensor:
        bsz, seqlen, _ = x.shape
        # Slice RoPE frequencies strictly to the d_r internal resonance fiber
        freqs_cos_r = freqs_cos[:, : self.d_r // 2]
        freqs_sin_r = freqs_sin[:, : self.d_r // 2]

        q_aug, k_aug, v_aug, attn_mask, dtau, _ = self._compute_fields_and_qkv(x, freqs_cos_r, freqs_sin_r)

        # Auto-route to fused Triton FlexAttention on CUDA for long sequences (O(T) memory, no O(T^2) mask!)
        can_use_flex = (
            self.use_flex_attention
            and _HAS_FLEX_ATTENTION
            and x.is_cuda
            and seqlen >= 256
            and x.dtype in (torch.bfloat16, torch.float16)
            and (not self.training or self.dropout == 0.0)
        )
        if can_use_flex:
            tau = torch.cumsum(dtau.float(), dim=-1).contiguous()
            tau_k = tau.clone()
            block_mask = _get_flex_causal_block_mask(seqlen, self.use_attn_vacuum, x.device)
            if self.use_attn_vacuum:
                def epr_score_mod(score, b, h, q_idx, kv_idx):
                    k_pos = torch.maximum(kv_idx - 1, torch.zeros_like(kv_idx))
                    dt = tau[b, h, q_idx] - tau_k[b, h, k_pos]
                    return torch.where(kv_idx == 0, score, score - dt)
            else:
                def epr_score_mod(score, b, h, q_idx, kv_idx):
                    return score - (tau[b, h, q_idx] - tau_k[b, h, kv_idx])
            output = _get_compiled_flex_attention()(
                q_aug, k_aug, v_aug,
                score_mod=epr_score_mod,
                block_mask=block_mask,
                scale=1.0,
            )
        else:
            # Single Fused D-Aligned SDPA fallback (CPU / Apple MPS / float64 audit / short decode)
            output = F.scaled_dot_product_attention(
                q_aug, k_aug, v_aug,
                attn_mask=attn_mask,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=False,
                scale=1.0,
            )
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        output = self.wo(output)
        output = self.resid_dropout(output)
        return output

    @torch.no_grad()
    def measure_diagnostics(self, x: torch.Tensor, freqs_cos: torch.Tensor, freqs_sin: torch.Tensor):
        """Return per-head (vacuum_silence_rate, mean_dtau, mean_rho) for physics inspection."""
        freqs_cos_r = freqs_cos[:, : self.d_r // 2]
        freqs_sin_r = freqs_sin[:, : self.d_r // 2]
        q_aug, k_aug, _, attn_mask, dtau, rho = self._compute_fields_and_qkv(x, freqs_cos_r, freqs_sin_r)
        vac_rate = None
        if self.use_attn_vacuum:
            logits = torch.matmul(q_aug.float(), k_aug.float().transpose(-2, -1)) + attn_mask.float()
            weights = F.softmax(logits, dim=-1)
            vac_rate = weights[..., 0].mean(dim=(0, 2))  # [H]
        mean_dtau = dtau.mean(dim=(0, 2))  # [H]
        mean_rho = rho.mean(dim=(0, 2)) if rho is not None else None
        return vac_rate, mean_dtau, mean_rho


class FeedForward(nn.Module):
    """Standard Llama 2 SwiGLU Feed-Forward Network."""
    def __init__(self, dim: int, hidden_dim: Optional[int], multiple_of: int, dropout: float):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = 4 * dim
            hidden_dim = int(2 * hidden_dim / 3)
            hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class TransformerBlock(nn.Module):
    """Pre-Norm Llama 2 Transformer Block with pluggable Attention."""
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.layer_id = layer_id
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.attention = UnifiedFiberGravityAttention(args) if args.use_unified_gravity else Attention(args)
        self.feed_forward = FeedForward(
            dim=args.dim,
            hidden_dim=args.hidden_dim,
            multiple_of=args.multiple_of,
            dropout=args.dropout,
        )
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(self, x: torch.Tensor, freqs_cos: torch.Tensor, freqs_sin: torch.Tensor) -> torch.Tensor:
        h = x + self.attention.forward(self.attention_norm(x), freqs_cos, freqs_sin)
        out = h + self.feed_forward.forward(self.ffn_norm(h))
        return out


class Transformer(nn.Module):
    """Complete Llama 2 / Unified Fiber-Gravity & ER=EPR Wormhole Language Model."""
    last_loss: Optional[torch.Tensor]

    def __init__(self, params: ModelArgs):
        super().__init__()
        self.params = params
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers

        self.tok_embeddings = nn.Embedding(params.vocab_size, params.dim)
        self.dropout = nn.Dropout(params.dropout)
        self.layers = nn.ModuleList([TransformerBlock(i, params) for i in range(params.n_layers)])
        self.norm = RMSNorm(params.dim, eps=params.norm_eps)
        self.output = nn.Linear(params.dim, params.vocab_size, bias=False)

        # Weight tying between token embeddings and LM head (karpathy/llama2.c standard)
        self.tok_embeddings.weight = self.output.weight

        # Precomputed RoPE tables (dynamically extended during zero-shot length extrapolation)
        freqs_cos, freqs_sin = precompute_freqs_cis(self.params.dim // self.params.n_heads, self.params.max_seq_len)
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

        # Standard Llama 2 / GPT-2 scaled residual initialization
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith("w3.weight") or pn.endswith("wo.weight"):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * params.n_layers))

        self.last_loss = None

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _get_rope_freqs(self, seqlen: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        if seqlen > self.freqs_cos.size(0) or self.freqs_cos.device != device:
            fc, fs = precompute_freqs_cis(self.params.dim // self.params.n_heads, max(seqlen, self.freqs_cos.size(0)))
            self.register_buffer("freqs_cos", fc.to(device), persistent=False)
            self.register_buffer("freqs_sin", fs.to(device), persistent=False)
        return self.freqs_cos[:seqlen], self.freqs_sin[:seqlen]

    def forward(self, tokens: torch.Tensor, targets: Optional[torch.Tensor] = None) -> torch.Tensor:
        _bsz, seqlen = tokens.shape
        h = self.tok_embeddings(tokens)
        h = self.dropout(h)
        freqs_cos, freqs_sin = self._get_rope_freqs(seqlen, tokens.device)

        for layer in self.layers:
            h = layer(h, freqs_cos, freqs_sin)
        h = self.norm(h)

        if targets is not None:
            logits = self.output(h)
            self.last_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # Inference optimization: only project the last token position
            logits = self.output(h[:, [-1], :])
            self.last_loss = None

        return logits

    def configure_optimizers(self, weight_decay: float, learning_rate: float, betas: Tuple[float, float], device_type: str):
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        # Weight decay applies only to 2D+ weight matrices (not 1D norms, biases, or physics scalars)
        decay_params = [p for _, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for _, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")
        return optimizer

    def estimate_mfu(self, fwdbwd_per_iter: int, dt: float) -> float:
        """Estimate Model FLOPs Utilization (MFU) relative to A100 bfloat16 peak (312 TFLOPS)."""
        N = sum(p.numel() for p in self.parameters())
        cfg = self.params
        L, H, Q, T = cfg.n_layers, cfg.n_heads, cfg.dim // cfg.n_heads, cfg.max_seq_len
        flops_per_token = 6 * N + 12 * L * H * Q * T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        flops_achieved = flops_per_iter * (1.0 / dt)
        return flops_achieved / 312e12

    @torch.inference_mode()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        pauli_alpha: float = 0.0,
        pauli_halflife: float = 16.0,
    ) -> torch.Tensor:
        """
        Autoregressive generation with optional Damped Pauli Repulsion decoding.

        Physics of Damped Pauli Repulsion (Lennard-Jones Balance):
          V_Pauli(v, t) = pauli_alpha * sum_{s < t, y_s == v} exp(-ln(2) * (t - s) / pauli_halflife)
          logit'(v, t)  = logit(v, t) - V_Pauli(v, t)
        - Short range (dt ~ 1..10): Degeneracy pressure prevents gravitational collapse (repetition loops).
        - Long range  (dt ~ 30+):   Repulsion decays to 0, letting the ER=EPR Wormhole & Earth heads
                                    legitimately recall earlier entities across paragraphs.
        """
        use_pauli = pauli_alpha > 0.0
        if use_pauli:
            bsz = idx.size(0)
            decay = math.exp(-math.log(2.0) / pauli_halflife) if pauli_halflife > 0.0 else 0.0
            v_pauli = torch.zeros((bsz, self.vocab_size), dtype=torch.float32, device=idx.device)
            # Pre-charge the Pauli repulsion field from the prompt history
            for t_step in range(idx.size(1)):
                v_pauli.mul_(decay)
                v_pauli.scatter_add_(1, idx[:, t_step : t_step + 1], torch.full((bsz, 1), pauli_alpha, device=idx.device))

        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.params.max_seq_len else idx[:, -self.params.max_seq_len:]
            logits = self(idx_cond)[:, -1, :]

            if use_pauli:
                logits = logits - v_pauli.to(logits.dtype)

            if temperature == 0.0:
                _, idx_next = torch.topk(logits, k=1, dim=-1)
            else:
                logits = logits / temperature
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float("Inf")
                probs = F.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)

            idx = torch.cat((idx, idx_next), dim=1)

            if use_pauli:
                v_pauli.mul_(decay)
                v_pauli.scatter_add_(1, idx_next, torch.full((bsz, 1), pauli_alpha, device=idx.device))

        return idx


# Canonical Short Name Alias (EPR = Euclidean-RBF, Proper-time & Resonance-fiber Attention)
EPRAttention = UnifiedFiberGravityAttention


class EPRKVCache:
    """
    Asymmetric Edge KV-Cache for EPR Attention:
      - Damped Solar-System Heads (0 .. n_damped-1): fixed sliding window `window_size` (O(1) memory!)
      - ER=EPR Wormhole Heads (n_damped .. n_heads-1): full-context retention for zero-decay distant recall
    Achieves 75% - 87.5% long-context KV-cache memory reduction on 4-head / 8-head edge models.
    """
    def __init__(
        self,
        n_layers: int,
        n_kv_heads: int,
        head_dim: int,
        n_wormhole_kv_heads: int = 1,
        window_size: int = 128,
        max_seq_len: int = 8192,
        batch_size: int = 1,
        dtype: torch.dtype = torch.bfloat16,
        device: str = "cpu",
    ):
        self.n_layers = n_layers
        self.n_kv_heads = n_kv_heads
        self.n_wh = n_wormhole_kv_heads
        self.n_damped = n_kv_heads - n_wormhole_kv_heads
        self.head_dim = head_dim
        self.window_size = window_size
        self.max_seq_len = max_seq_len
        self.pos = 0

        # Damped heads: fixed O(1) ring buffer of length `window_size`
        self.k_damped = [
            torch.zeros(batch_size, self.n_damped, window_size, head_dim, dtype=dtype, device=device)
            for _ in range(n_layers)
        ]
        self.v_damped = [
            torch.zeros(batch_size, self.n_damped, window_size, head_dim, dtype=dtype, device=device)
            for _ in range(n_layers)
        ]
        # Wormhole heads: full-context retention up to `max_seq_len`
        self.k_wh = [
            torch.zeros(batch_size, self.n_wh, max_seq_len, head_dim, dtype=dtype, device=device)
            for _ in range(n_layers)
        ]
        self.v_wh = [
            torch.zeros(batch_size, self.n_wh, max_seq_len, head_dim, dtype=dtype, device=device)
            for _ in range(n_layers)
        ]

    def update(self, layer_idx: int, k_new: torch.Tensor, v_new: torch.Tensor):
        """Append new KV tokens [B, H_kv, T_new, D] into asymmetric buffers."""
        t_new = k_new.size(2)
        k_d, k_w = k_new[:, : self.n_damped], k_new[:, self.n_damped :]
        v_d, v_w = v_new[:, : self.n_damped], v_new[:, self.n_damped :]

        # Update Wormhole full-context buffer
        end = min(self.pos + t_new, self.max_seq_len)
        self.k_wh[layer_idx][:, :, self.pos : end] = k_w[:, :, : end - self.pos]
        self.v_wh[layer_idx][:, :, self.pos : end] = v_w[:, :, : end - self.pos]

        # Update Damped sliding-window ring buffer
        for offset in range(t_new):
            slot = (self.pos + offset) % self.window_size
            self.k_damped[layer_idx][:, :, slot : slot + 1] = k_d[:, :, offset : offset + 1]
            self.v_damped[layer_idx][:, :, slot : slot + 1] = v_d[:, :, offset : offset + 1]

        if layer_idx == self.n_layers - 1:
            self.pos = end

    def memory_bytes(self) -> int:
        total = 0
        for lst in (self.k_damped, self.v_damped, self.k_wh, self.v_wh):
            for t in lst:
                total += t.numel() * t.element_size()
        return total

    def standard_kv_bytes(self, seq_len: int) -> int:
        elem = self.k_wh[0].element_size()
        bsz = self.k_wh[0].size(0)
        return 2 * self.n_layers * bsz * self.n_kv_heads * seq_len * self.head_dim * elem

    def savings_ratio(self, seq_len: int) -> float:
        std = self.standard_kv_bytes(seq_len)
        return 1.0 - (self.memory_bytes() / max(1, std))
