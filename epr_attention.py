"""
epr_attention.py — Unified, Zero-Dependency EPR Attention Core (v2.0)
(Euclidean-RBF, Proper-time & Resonance-fiber Attention)

Inspired by ER=EPR spacetime folding and spatial gravitational field intuition.
Single Source of Truth across standalone usage, `model.py`, training, and evaluation:
  1. EPRConfig      — Pure attention hyperparameters (decoupled from any backbone),
                      defaulting to the 3 ablation-proven core pillars while supporting
                      Mass-Elastic Heteroscedastic RBF (`use_elastic_rbf`), Pairwise
                      Resonant Tunneling (`use_resonant_tunneling`), and legacy switches.
  2. EPRKVCache     — Stateful autoregressive KV-cache tracking cumulative proper time
                      `tau_state` with O(1) damped-head ring-buffer eviction.
  3. EPRAttention   — Drop-in nn.Module supporting interleaved (llama2.c) & half_split
                      (HuggingFace) RoPE, native GQA FlexAttention, exact SDPA fallback,
                      and lossless / sliding-window cached autoregressive decoding.
"""

from dataclasses import dataclass
import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch.nn.attention.flex_attention import (
        flex_attention as _raw_flex_attention,
        create_block_mask as _create_block_mask,
    )
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


def _get_flex_causal_block_mask(
    seqlen: int,
    use_vacuum: bool,
    device: torch.device,
    n_heads: int = 0,
    n_damped: int = 0,
    damped_window: int = 0,
):
    """
    Build & cache a PyTorch `FlexAttention` `BlockMask`:
      - `damped_window <= 0`: Full causal mask (`H=None` broadcast).
      - `damped_window > 0`: Hardware-native Asymmetric Sub-Quadratic `BlockMask` (`H=n_heads`)!
        Damped heads (`h < n_damped`) skip all distant tiles outside `damped_window` (keeping
        index 0 when `use_vacuum=True`), while Wormhole heads (`h >= n_damped`) attend full causal!
    """
    eff_win = damped_window if (0 < damped_window < seqlen and n_damped > 0) else 0
    key = (seqlen, use_vacuum, n_heads if eff_win > 0 else 0, n_damped if eff_win > 0 else 0, eff_win, str(device))
    if key not in _FLEX_BLOCK_MASK_CACHE:
        if eff_win == 0:
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
        else:
            if use_vacuum:
                def _asym_mask(b, h, q_idx, kv_idx):
                    causal = kv_idx <= (q_idx + 1)
                    in_win = (kv_idx == 0) | ((q_idx + 1 - kv_idx) < eff_win)
                    return causal & ((h >= n_damped) | in_win)
                kv_len = seqlen + 1
            else:
                def _asym_mask(b, h, q_idx, kv_idx):
                    causal = kv_idx <= q_idx
                    in_win = (q_idx - kv_idx) < eff_win
                    return causal & ((h >= n_damped) | in_win)
                kv_len = seqlen
            _FLEX_BLOCK_MASK_CACHE[key] = _create_block_mask(
                _asym_mask, B=None, H=n_heads, Q_LEN=seqlen, KV_LEN=kv_len, device=device
            )
    return _FLEX_BLOCK_MASK_CACHE[key]


def _apply_rope_interleaved(
    q: torch.Tensor, k: torch.Tensor, freqs_cos: torch.Tensor, freqs_sin: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """llama2.c interleaved RoPE. q: [B, H_d, T, d_r], k: [B, H_kv_d, T, d_r]; freqs: [T, d_r//2]."""
    acc_dtype = torch.float64 if q.dtype == torch.float64 else torch.float32
    q_r, q_i = q.to(acc_dtype).reshape(*q.shape[:-1], -1, 2).unbind(-1)
    k_r, k_i = k.to(acc_dtype).reshape(*k.shape[:-1], -1, 2).unbind(-1)
    fc = freqs_cos.to(acc_dtype).unsqueeze(0).unsqueeze(0)
    fs = freqs_sin.to(acc_dtype).unsqueeze(0).unsqueeze(0)
    q_out = torch.stack([q_r * fc - q_i * fs, q_r * fs + q_i * fc], dim=-1).flatten(-2)
    k_out = torch.stack([k_r * fc - k_i * fs, k_r * fs + k_i * fc], dim=-1).flatten(-2)
    return q_out.type_as(q), k_out.type_as(k)


def _apply_rope_half_split(
    q: torch.Tensor, k: torch.Tensor, freqs_cos: torch.Tensor, freqs_sin: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """HuggingFace half-split RoPE. q, k: [B, H, T, d_r]; freqs: [T, d_r] or [T, d_r//2]."""
    acc_dtype = torch.float64 if q.dtype == torch.float64 else torch.float32
    if freqs_cos.shape[-1] == q.shape[-1] // 2:
        freqs_cos = torch.cat([freqs_cos, freqs_cos], dim=-1)
        freqs_sin = torch.cat([freqs_sin, freqs_sin], dim=-1)
    fc = freqs_cos.to(acc_dtype).unsqueeze(0).unsqueeze(0)
    fs = freqs_sin.to(acc_dtype).unsqueeze(0).unsqueeze(0)
    q_f, k_f = q.to(acc_dtype), k.to(acc_dtype)
    half = q.shape[-1] // 2
    q_rot = torch.cat([-q_f[..., half:], q_f[..., :half]], dim=-1)
    k_rot = torch.cat([-k_f[..., half:], k_f[..., :half]], dim=-1)
    return ((q_f * fc) + (q_rot * fs)).type_as(q), ((k_f * fc) + (k_rot * fs)).type_as(k)


@dataclass
class EPRConfig:
    """
    Pure EPR Attention hyperparameters, decoupled from any specific Transformer backbone.
    Defaults implement the Occam's-razor distilled 3-pillar architecture (v2.0):
      1. Orthogonal Fiber-Base Packing (`d_r + d_g + 2 == head_dim`, zero V padding)
      2. Learnable Vacuum Floor (`e^{v_0}`, Softmax1 / Attention Sink)
      3. Dual-Regime Topology (Damped RoPE + proper-time heads & Zero-decay NoPE Wormhole head)
    """
    dim: int = 256
    n_heads: int = 4
    n_kv_heads: Optional[int] = None
    gravity_dim: int = 0                  # 0 = unified power-of-2 packing (D=32->24+6+2; D=64->48+14+2)

    # Core Pillar Switches
    use_wormhole: bool = True             # Convert last KV head group into zero-decay NoPE Wormhole
    n_wormhole_heads: int = 0             # Explicit Q wormhole count (0 = auto: n_rep if use_wormhole else 0)
    wormhole_v0_init: float = 2.5         # High initial vacuum threshold keeping Wormhole silent on noise
    use_base_rbf: bool = True             # Shared low-dim Euclidean-RBF capture well
    use_time_dilation: bool = True        # Data-dependent proper-time decay dtau >= 0 on damped heads
    use_attn_vacuum: bool = True          # Learnable Softmax1 vacuum slot e^{v_0}

    # Polished Potential Upgrades (Zero extra head_dim channels!)
    use_elastic_rbf: bool = False         # Heteroscedastic mass-elastic capture radius: gamma_j = gamma_0 / (1 + alpha * sigmoid(u_j))
    elastic_alpha_init: float = 1.0       # Initial elasticity coupling softplus(raw_elastic_alpha)
    use_resonant_tunneling: bool = False  # Pairwise resonance wormhole gate: dt * (1 - sigmoid(beta * (S - theta)))
    tunnel_theta_init: float = 2.5        # Resonance score threshold theta_h to trigger pairwise tunneling
    tunnel_beta_init: float = 2.0         # Tunneling gate steepness beta_h

    # Legacy / Single-Switch Ablation Compatibility (Default False — proven redundant at 6M)
    use_bilateral_mass: bool = False      # Query-side reception mass u_recv_i
    use_nested_time: bool = False         # Top-down solar-system cascade matrix M

    dropout: float = 0.0
    rope_style: str = "interleaved"       # "interleaved" (llama2.c) or "half_split" (HuggingFace)
    use_flex_attention: bool = True       # Auto-route CUDA bf16/fp16 (T>=256, dropout==0) to fused FlexAttention
    flex_damped_window: int = 0           # >0 enables sub-quadratic BlockMask tile-skipping on damped heads!


class EPRKVCache:
    """
    Stateful Asymmetric Edge KV-Cache for EPR Attention:
      - Tracks exact cumulative proper time `tau_state` [B, H, 1] per layer.
      - Damped heads (0 .. n_damped_kv - 1): fixed ring buffer of size `window_size` (O(1) memory).
        If `window_size is None` or `>= max_seq_len`, runs in lossless full-context mode.
      - Wormhole heads (n_damped_kv .. n_kv_heads - 1): full-context retention up to `max_seq_len`.
    """
    def __init__(
        self,
        n_layers: int,
        n_heads: int,
        n_kv_heads: int,
        head_dim: int,
        n_wormhole_kv_heads: int = 1,
        window_size: Optional[int] = 128,
        max_seq_len: int = 8192,
        batch_size: int = 1,
        dtype: torch.dtype = torch.bfloat16,
        device: str = "cpu",
    ):
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.n_rep = n_heads // n_kv_heads
        self.n_wh_kv = n_wormhole_kv_heads
        self.n_damped_kv = n_kv_heads - n_wormhole_kv_heads
        self.n_wh_q = n_wormhole_kv_heads * self.n_rep
        self.n_damped_q = n_heads - self.n_wh_q
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.window_size = max_seq_len if (window_size is None or window_size >= max_seq_len) else window_size
        self.batch_size = batch_size
        self.dtype = dtype
        self.device = device
        self.seq_pos = 0

        tau_dtype = torch.float64 if dtype == torch.float64 else torch.float32
        self.tau_dtype = tau_dtype
        self.tau_state = [
            torch.zeros(batch_size, n_heads, 1, dtype=tau_dtype, device=device)
            for _ in range(n_layers)
        ]

        self.k_damped = [
            torch.zeros(batch_size, self.n_damped_kv, self.window_size, head_dim, dtype=dtype, device=device)
            for _ in range(n_layers)
        ]
        self.v_damped = [
            torch.zeros(batch_size, self.n_damped_kv, self.window_size, head_dim, dtype=dtype, device=device)
            for _ in range(n_layers)
        ]
        self.tau_damped = [
            torch.zeros(batch_size, self.n_damped_q, self.window_size, dtype=tau_dtype, device=device)
            for _ in range(n_layers)
        ]

        self.k_wh = [
            torch.zeros(batch_size, self.n_wh_kv, max_seq_len, head_dim, dtype=dtype, device=device)
            for _ in range(n_layers)
        ]
        self.v_wh = [
            torch.zeros(batch_size, self.n_wh_kv, max_seq_len, head_dim, dtype=dtype, device=device)
            for _ in range(n_layers)
        ]

    def reset(self):
        self.seq_pos = 0
        for i in range(self.n_layers):
            self.tau_state[i].zero_()
            self.k_damped[i].zero_()
            self.v_damped[i].zero_()
            self.tau_damped[i].zero_()
            self.k_wh[i].zero_()
            self.v_wh[i].zero_()

    def update(
        self,
        layer_idx: int,
        k_real: torch.Tensor,
        v_real: torch.Tensor,
        tau_step: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        t_new = k_real.size(2)
        start = self.seq_pos
        end = start + t_new
        if end > self.max_seq_len:
            raise ValueError(f"EPRKVCache overflow: {end} > max_seq_len={self.max_seq_len}")

        k_d, k_w = k_real[:, : self.n_damped_kv], k_real[:, self.n_damped_kv :]
        v_d, v_w = v_real[:, : self.n_damped_kv], v_real[:, self.n_damped_kv :]
        tau_d = tau_step[:, : self.n_damped_q]

        if self.n_wh_kv > 0:
            self.k_wh[layer_idx][:, :, start:end] = k_w
            self.v_wh[layer_idx][:, :, start:end] = v_w

        if self.n_damped_kv > 0:
            if start == 0 and t_new <= self.window_size:
                self.k_damped[layer_idx][:, :, :t_new] = k_d
                self.v_damped[layer_idx][:, :, :t_new] = v_d
                self.tau_damped[layer_idx][:, :, :t_new] = tau_d
            else:
                for offset in range(t_new):
                    slot = (start + offset) % self.window_size
                    self.k_damped[layer_idx][:, :, slot : slot + 1] = k_d[:, :, offset : offset + 1]
                    self.v_damped[layer_idx][:, :, slot : slot + 1] = v_d[:, :, offset : offset + 1]
                    self.tau_damped[layer_idx][:, :, slot : slot + 1] = tau_d[:, :, offset : offset + 1]

        self.tau_state[layer_idx] = tau_step[:, :, -1:].to(self.tau_dtype)

        if end <= self.window_size:
            k_d_act = self.k_damped[layer_idx][:, :, :end]
            v_d_act = self.v_damped[layer_idx][:, :, :end]
            tau_d_act = self.tau_damped[layer_idx][:, :, :end]
        else:
            head_ptr = end % self.window_size
            if head_ptr == 0:
                k_d_act = self.k_damped[layer_idx]
                v_d_act = self.v_damped[layer_idx]
                tau_d_act = self.tau_damped[layer_idx]
            else:
                k_d_act = torch.cat([self.k_damped[layer_idx][:, :, head_ptr:], self.k_damped[layer_idx][:, :, :head_ptr]], dim=2)
                v_d_act = torch.cat([self.v_damped[layer_idx][:, :, head_ptr:], self.v_damped[layer_idx][:, :, :head_ptr]], dim=2)
                tau_d_act = torch.cat([self.tau_damped[layer_idx][:, :, head_ptr:], self.tau_damped[layer_idx][:, :, :head_ptr]], dim=2)

        k_w_act = self.k_wh[layer_idx][:, :, :end]
        v_w_act = self.v_wh[layer_idx][:, :, :end]

        if layer_idx == self.n_layers - 1:
            self.seq_pos = end

        return k_d_act, v_d_act, tau_d_act, k_w_act, v_w_act

    def memory_bytes(self, active_seq_len: Optional[int] = None) -> int:
        """
        Compute KV-cache footprint at `active_seq_len` (or full `max_seq_len` allocation if None),
        including `tau_damped` and `tau_state` buffers for honest accounting.
        """
        if active_seq_len is None:
            total = 0
            for lst in (self.k_damped, self.v_damped, self.tau_damped, self.k_wh, self.v_wh, self.tau_state):
                for t in lst:
                    total += t.numel() * t.element_size()
            return total
        elem_kv = self.k_wh[0].element_size() if self.n_wh_kv > 0 else self.k_damped[0].element_size()
        elem_tau = self.tau_state[0].element_size()
        d_len = min(active_seq_len, self.window_size)
        kv_d = 2 * self.n_layers * self.batch_size * self.n_damped_kv * d_len * self.head_dim * elem_kv
        tau_d = self.n_layers * self.batch_size * self.n_damped_q * d_len * elem_tau
        kv_w = 2 * self.n_layers * self.batch_size * self.n_wh_kv * active_seq_len * self.head_dim * elem_kv
        st_tau = self.n_layers * self.batch_size * self.n_heads * elem_tau
        return kv_d + tau_d + kv_w + st_tau

    def standard_kv_bytes(self, seq_len: int) -> int:
        elem = self.k_wh[0].element_size() if self.n_wh_kv > 0 else self.k_damped[0].element_size()
        return 2 * self.n_layers * self.batch_size * self.n_kv_heads * seq_len * self.head_dim * elem

    def savings_ratio(self, seq_len: int) -> float:
        std = self.standard_kv_bytes(seq_len)
        return 1.0 - (self.memory_bytes(active_seq_len=seq_len) / max(1, std))


class EPRAttention(nn.Module):
    """
    Unified EPR Attention (Euclidean-RBF, Proper-time & Resonance-fiber).
    Exact D-aligned algebraic folding (`d_r + d_g + 2 == head_dim`, zero V channel padding):
      - Direct un-shifted packing (`use_bilateral_mass=False`, default):
          Q_aug,i  = [ q_res,i / sqrt(d_r),   z_i,            -0.5 ||z_i||^2,   1                      ]
          K_real,j = [ k_res,j,               gamma_j * z_j,  gamma_j,          u_j - 0.5 gamma_j||z_j||^2 ]
          K_vac    = [ 0_{d_r + d_g},                         0,                v_{0,h}                ]
        Directly yields exact un-shifted logits for both constant gamma_h and heteroscedastic
        mass-elastic gamma_j = gamma_{0,h} / (1 + alpha_h * sigmoid(u_j))!
    """
    def __init__(self, cfg: EPRConfig):
        super().__init__()
        self.cfg = cfg
        self.n_heads = cfg.n_heads
        self.n_kv_heads = cfg.n_heads if cfg.n_kv_heads is None else cfg.n_kv_heads
        assert self.n_heads % self.n_kv_heads == 0, "n_heads must be divisible by n_kv_heads"
        self.n_rep = self.n_heads // self.n_kv_heads
        self.head_dim = cfg.dim // self.n_heads
        self.dropout = cfg.dropout

        # 1. Unified Power-of-2 Channel Split: d_r + d_g + 2 == head_dim
        assert self.head_dim >= 12 and self.head_dim % 2 == 0, (
            f"head_dim ({self.head_dim}) must be even and >= 12 for fiber-base packing"
        )
        if cfg.gravity_dim > 0:
            d_g = cfg.gravity_dim
        else:
            # D=16 -> 10+4+2; D=32 -> 24+6+2; D=48 -> 36+10+2; D=64 -> 48+14+2; D=128 -> 96+30+2
            d_g = max(4, ((self.head_dim // 4) // 2) * 2 - 2)
        d_r = self.head_dim - d_g - 2
        assert d_r >= 4 and d_r % 2 == 0 and d_g >= 2, (
            f"Invalid split: head_dim={self.head_dim}, d_r={d_r}, d_g={d_g}"
        )
        self.d_r = d_r
        self.d_g = d_g

        # 2. Dual-Regime Head Topology (Damped + ER=EPR Wormhole)
        if cfg.n_wormhole_heads > 0:
            assert cfg.n_wormhole_heads % self.n_rep == 0, "n_wormhole_heads must be divisible by n_rep"
            assert cfg.n_wormhole_heads <= self.n_heads
            self.n_wh = cfg.n_wormhole_heads
        elif cfg.use_wormhole and self.n_kv_heads > 1:
            self.n_wh = self.n_rep
        else:
            self.n_wh = 0

        self.use_wormhole = self.n_wh > 0
        self.n_wh_kv = self.n_wh // self.n_rep
        self.n_damped = self.n_heads - self.n_wh
        self.n_damped_kv = self.n_kv_heads - self.n_wh_kv

        self.use_base_rbf = cfg.use_base_rbf
        self.use_elastic_rbf = cfg.use_elastic_rbf and cfg.use_base_rbf
        self.use_time_dilation = cfg.use_time_dilation
        self.use_attn_vacuum = cfg.use_attn_vacuum
        self.use_resonant_tunneling = cfg.use_resonant_tunneling
        self.use_bilateral_mass = cfg.use_bilateral_mass
        self.use_nested_time = cfg.use_nested_time
        self.use_flex_attention = cfg.use_flex_attention
        self.flex_damped_window = cfg.flex_damped_window

        if self.use_elastic_rbf and self.use_bilateral_mass:
            raise ValueError("use_elastic_rbf requires use_bilateral_mass=False (uses direct un-shifted 2-slot packing).")
        if self.use_resonant_tunneling and self.use_bilateral_mass:
            raise ValueError("use_resonant_tunneling requires use_bilateral_mass=False (requires un-shifted pre-decay scores).")

        # 3. Fused Main Vector Projection GEMM
        self.q_dim = self.n_heads * self.d_r
        self.k_dim = self.n_kv_heads * self.d_r
        self.z_dim = self.n_kv_heads * self.d_g
        self.v_dim = self.n_kv_heads * self.head_dim
        self.w_proj = nn.Linear(cfg.dim, self.q_dim + self.k_dim + self.z_dim + self.v_dim, bias=False)
        self.wo = nn.Linear(self.n_heads * self.head_dim, cfg.dim, bias=False)

        # 4. Fused Scalar Physics Projection GEMM
        self.u_emit_dim = self.n_kv_heads
        self.u_recv_dim = self.n_heads if self.use_bilateral_mass else 0
        self.tau_in_dim = self.n_damped if (self.use_time_dilation and self.n_damped > 0) else 0
        self.w_phys = nn.Linear(cfg.dim, self.u_emit_dim + self.u_recv_dim + self.tau_in_dim, bias=False)

        # 5. Base Spacetime Euclidean-RBF Precision & Optional Mass-Elasticity
        if self.use_base_rbf:
            self.log_inv_sigma2 = nn.Parameter(torch.zeros(self.n_kv_heads))
            if self.use_elastic_rbf:
                init_el = math.log(math.expm1(max(cfg.elastic_alpha_init, 1e-4)))
                self.raw_elastic_alpha = nn.Parameter(torch.full((self.n_kv_heads,), init_el))

        # 6. Proper-Time Ladder on Damped Heads
        if self.n_damped > 0:
            if self.n_damped == 1:
                init_lambdas = torch.tensor([0.05])
            else:
                slow_end = 0.015 if self.use_wormhole else 0.005
                init_lambdas = torch.logspace(math.log10(0.30), math.log10(slow_end), self.n_damped)
            self.b_tau = nn.Parameter(torch.log(torch.expm1(init_lambdas)))
            if self.use_nested_time and self.n_damped > 1:
                self.raw_nest_alpha = nn.Parameter(
                    torch.full((self.n_damped - 1,), math.log(math.expm1(0.15)))
                )
                self.register_buffer(
                    "nest_triu_mask",
                    torch.triu(torch.ones(self.n_damped, self.n_damped, dtype=torch.bool)),
                    persistent=False,
                )

        # 7. Optional Pairwise Resonant Tunneling Gate
        if self.use_resonant_tunneling:
            self.tunnel_theta = nn.Parameter(torch.full((self.n_heads,), cfg.tunnel_theta_init))
            init_beta = math.log(math.expm1(max(cfg.tunnel_beta_init, 1e-4)))
            self.tunnel_beta = nn.Parameter(torch.full((self.n_heads,), init_beta))

        # 8. Learnable Vacuum Floor (Softmax1 / Mass Gap)
        if self.use_attn_vacuum:
            v0_init = torch.zeros(self.n_kv_heads)
            if self.n_wh_kv > 0:
                v0_init[-self.n_wh_kv :] = cfg.wormhole_v0_init
            self.v0 = nn.Parameter(v0_init)
            self.register_buffer("vac_zeros_rg", torch.zeros(1, self.n_kv_heads, 1, self.d_r + self.d_g), persistent=False)
            self.register_buffer("vac_zeros_1", torch.zeros(1, self.n_kv_heads, 1, 1), persistent=False)
            self.register_buffer("vac_ones_1", torch.ones(1, self.n_kv_heads, 1, 1), persistent=False)

        self.attn_dropout = nn.Dropout(cfg.dropout)
        self.resid_dropout = nn.Dropout(cfg.dropout)

    def _compute_step_tensors(
        self,
        x: torch.Tensor,
        freqs_cos: torch.Tensor,
        freqs_sin: torch.Tensor,
        tau_offset: Optional[torch.Tensor] = None,
    ):
        B, T, _ = x.shape
        H, H_kv, D, d_r, d_g = self.n_heads, self.n_kv_heads, self.head_dim, self.d_r, self.d_g
        acc_dtype = torch.float64 if x.dtype == torch.float64 else torch.float32

        q_res, k_res, z_base, xv = self.w_proj(x).split([self.q_dim, self.k_dim, self.z_dim, self.v_dim], dim=-1)
        q_res = q_res.view(B, T, H, d_r).transpose(1, 2)
        k_res = k_res.view(B, T, H_kv, d_r).transpose(1, 2)
        z_base = z_base.view(B, T, H_kv, d_g).transpose(1, 2)
        xv = xv.view(B, T, H_kv, D).transpose(1, 2)

        # Apply RoPE strictly to damped heads on the d_r resonance fiber
        if self.n_damped > 0:
            fc_r = freqs_cos[:, : d_r // 2] if self.cfg.rope_style == "interleaved" else freqs_cos[:, :d_r] if freqs_cos.shape[-1] >= d_r else freqs_cos[:, : d_r // 2]
            fs_r = freqs_sin[:, : d_r // 2] if self.cfg.rope_style == "interleaved" else freqs_sin[:, :d_r] if freqs_sin.shape[-1] >= d_r else freqs_sin[:, : d_r // 2]
            rope_fn = _apply_rope_interleaved if self.cfg.rope_style == "interleaved" else _apply_rope_half_split
            qd, kd = rope_fn(q_res[:, : self.n_damped], k_res[:, : self.n_damped_kv], fc_r, fs_r)
            q_res = torch.cat([qd, q_res[:, self.n_damped :]], dim=1) if self.n_wh > 0 else qd
            k_res = torch.cat([kd, k_res[:, self.n_damped_kv :]], dim=1) if self.n_wh_kv > 0 else kd

        # Scalar physics projections
        splits = [self.u_emit_dim]
        if self.u_recv_dim > 0:
            splits.append(self.u_recv_dim)
        if self.tau_in_dim > 0:
            splits.append(self.tau_in_dim)
        phys_parts = self.w_phys(x).split(splits, dim=-1)

        idx_p = 0
        u_emit = phys_parts[idx_p].transpose(1, 2)  # [B, H_kv, T]
        idx_p += 1
        u_recv = None
        if self.u_recv_dim > 0:
            u_recv = phys_parts[idx_p].transpose(1, 2)  # [B, H, T]
            idx_p += 1
        tau_in = None
        if self.tau_in_dim > 0:
            tau_in = phys_parts[idx_p].transpose(1, 2)  # [B, n_damped, T]

        # Proper-time velocity dtau >= 0 on damped heads, strictly 0 on wormhole heads
        if self.n_damped > 0:
            if self.use_time_dilation and tau_in is not None:
                dtau_loc = F.softplus(tau_in.to(acc_dtype) + self.b_tau.to(acc_dtype).view(1, self.n_damped, 1))
            else:
                dtau_loc = F.softplus(self.b_tau.to(acc_dtype)).view(1, self.n_damped, 1).expand(B, self.n_damped, T)

            if self.use_nested_time and self.n_damped > 1:
                alphas = F.softplus(self.raw_nest_alpha.to(acc_dtype)).clamp(min=1e-6)
                log_pref = F.pad(torch.cumsum(torch.log(alphas), dim=0), (1, 0), value=0.0)
                log_M = log_pref.unsqueeze(0) - log_pref.unsqueeze(1)
                M = torch.where(self.nest_triu_mask, torch.exp(log_M.clamp(max=20.0)), torch.zeros_like(log_M))
                dtau_damped = torch.einsum("ij,bjt->bit", M, dtau_loc)
            else:
                dtau_damped = dtau_loc

            if self.n_wh > 0:
                dtau = torch.cat([dtau_damped, torch.zeros(B, self.n_wh, T, dtype=acc_dtype, device=x.device)], dim=1)
            else:
                dtau = dtau_damped
        else:
            dtau = torch.zeros(B, H, T, dtype=acc_dtype, device=x.device)

        tau = torch.cumsum(dtau, dim=-1)
        if tau_offset is not None:
            tau = tau + tau_offset.to(acc_dtype)

        # Orthogonal Fiber-Base Algebraic Folding (d_r + d_g + 2 == D)
        q_fib_scaled = q_res * (1.0 / math.sqrt(d_r))

        if not self.use_bilateral_mass:
            # --- Regime 1 (Default v2.0 & Elastic-RBF): Exact Un-Shifted Direct Packing ---
            if self.use_base_rbf:
                gamma_0 = (torch.exp(self.log_inv_sigma2.to(acc_dtype)) / math.sqrt(d_g)).view(1, H_kv, 1, 1)
                if self.use_elastic_rbf:
                    m_emit = torch.sigmoid(u_emit.to(acc_dtype)).unsqueeze(-1)
                    el_alpha = F.softplus(self.raw_elastic_alpha.to(acc_dtype)).view(1, H_kv, 1, 1)
                    gamma_kv = (gamma_0 / (1.0 + el_alpha * m_emit)).to(z_base.dtype)  # [B, H_kv, T, 1]
                else:
                    gamma_kv = gamma_0.to(z_base.dtype).expand(B, H_kv, T, 1)

                z_k_scaled = z_base * gamma_kv
                z_norm2_kv = (z_base * z_base).sum(dim=-1, keepdim=True)
                k_slot3 = gamma_kv
                k_slot4 = u_emit.unsqueeze(-1) - 0.5 * gamma_kv * z_norm2_kv

                z_q = z_base.repeat_interleave(self.n_rep, dim=1) if self.n_rep > 1 else z_base
                q_slot3 = -0.5 * (z_q * z_q).sum(dim=-1, keepdim=True)
                q_slot4 = torch.ones_like(q_slot3)
            else:
                z_k_scaled = torch.zeros_like(z_base)
                k_slot3 = torch.zeros(B, H_kv, T, 1, dtype=z_base.dtype, device=z_base.device)
                k_slot4 = u_emit.unsqueeze(-1)
                z_q = torch.zeros(B, H, T, d_g, dtype=q_res.dtype, device=q_res.device)
                q_slot3 = torch.zeros(B, H, T, 1, dtype=q_res.dtype, device=q_res.device)
                q_slot4 = torch.ones_like(q_slot3)

            q_aug = torch.cat([q_fib_scaled, z_q, q_slot3, q_slot4], dim=-1)
            k_real = torch.cat([k_res, z_k_scaled, k_slot3, k_slot4], dim=-1)

            k_vac = None
            if self.use_attn_vacuum:
                k_vac = torch.cat(
                    [
                        self.vac_zeros_rg.to(k_real.dtype),
                        self.vac_zeros_1.to(k_real.dtype),
                        self.v0.view(1, H_kv, 1, 1).to(k_real.dtype),
                    ],
                    dim=-1,
                ).expand(B, -1, -1, -1)
        else:
            # --- Regime 2 (Legacy Bilateral-Mass Shift-Invariant Packing for Ablation Repro) ---
            if self.use_base_rbf:
                gamma_kv = (torch.exp(self.log_inv_sigma2.to(acc_dtype)) / math.sqrt(d_g)).view(1, H_kv, 1, 1).to(z_base.dtype)
                z_q_kv = z_base * gamma_kv
                z_half_kv = 0.5 * (z_q_kv * z_base).sum(dim=-1, keepdim=True)
                key_pot = u_emit.unsqueeze(-1) - z_half_kv
                z_q = z_q_kv.repeat_interleave(self.n_rep, dim=1) if self.n_rep > 1 else z_q_kv
                z_half_q = z_half_kv.repeat_interleave(self.n_rep, dim=1) if self.n_rep > 1 else z_half_kv
                q_shift = z_half_q - u_recv.unsqueeze(-1)
                k_z = z_base
            else:
                z_q = torch.zeros(B, H, T, d_g, dtype=q_res.dtype, device=q_res.device)
                k_z = torch.zeros_like(z_base)
                key_pot = u_emit.unsqueeze(-1)
                q_shift = -u_recv.unsqueeze(-1)

            q_aug = torch.cat([q_fib_scaled, z_q, torch.ones_like(q_shift), q_shift], dim=-1)
            k_real = torch.cat([k_res, k_z, key_pot, torch.zeros_like(key_pot)], dim=-1)

            k_vac = None
            if self.use_attn_vacuum:
                k_vac = torch.cat(
                    [
                        self.vac_zeros_rg.to(k_real.dtype),
                        self.v0.view(1, H_kv, 1, 1).to(k_real.dtype),
                        self.vac_ones_1.to(k_real.dtype),
                    ],
                    dim=-1,
                ).expand(B, -1, -1, -1)

        return q_aug, k_vac, k_real, xv, dtau, tau

    def _attend_exact_tunneling(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        dt: Optional[torch.Tensor],
        causal_mask: Optional[torch.Tensor],
        has_vacuum: bool,
        head_slice: slice,
    ) -> torch.Tensor:
        """Exact non-Flex reference / decode path for Pairwise Resonant Tunneling."""
        acc_dtype = torch.float64 if q.dtype == torch.float64 else torch.float32
        scores = torch.matmul(q.to(acc_dtype), k.to(acc_dtype).transpose(-2, -1))
        beta = F.softplus(self.tunnel_beta[head_slice].to(acc_dtype)).view(1, -1, 1, 1)
        theta = self.tunnel_theta[head_slice].to(acc_dtype).view(1, -1, 1, 1)

        if has_vacuum:
            s_vac = scores[..., :1]
            s_real = scores[..., 1:]
            if dt is not None:
                gate = torch.sigmoid(beta * (s_real - theta))
                s_real = s_real - dt.to(acc_dtype) * (1.0 - gate)
            if causal_mask is not None:
                s_real = s_real + causal_mask.to(acc_dtype)
            scores = torch.cat([s_vac, s_real], dim=-1)
        else:
            if dt is not None:
                gate = torch.sigmoid(beta * (scores - theta))
                scores = scores - dt.to(acc_dtype) * (1.0 - gate)
            if causal_mask is not None:
                scores = scores + causal_mask.to(acc_dtype)

        weights = F.softmax(scores, dim=-1)
        if self.training and self.dropout > 0.0:
            weights = self.attn_dropout(weights)
        return torch.matmul(weights.to(v.dtype), v)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cos: torch.Tensor,
        freqs_sin: torch.Tensor,
        kv_cache: Optional[EPRKVCache] = None,
        layer_idx: int = 0,
    ) -> torch.Tensor:
        B, T, _ = x.shape

        # --- Path 1: Stateful Autoregressive Decoding with EPRPagedKVCache or EPRKVCache ---
        if kv_cache is not None:
            if hasattr(kv_cache, "write_and_attend"):
                tau_off = kv_cache.tau_state[layer_idx][:B]
                q_aug, _, k_real, xv, _, tau = self._compute_step_tensors(x, freqs_cos, freqs_sin, tau_offset=tau_off)
                out = kv_cache.write_and_attend(
                    layer_idx, q_aug, k_real, xv, tau, self.v0,
                    use_vacuum=self.use_attn_vacuum, use_time_decay=self.use_time_dilation,
                )
                return self.resid_dropout(self.wo(out.transpose(1, 2).contiguous().view(B, T, -1)))

            tau_off = kv_cache.tau_state[layer_idx]
            q_aug, k_vac, k_real, xv, _, tau = self._compute_step_tensors(x, freqs_cos, freqs_sin, tau_offset=tau_off)
            k_d_act, v_d_act, tau_d_act, k_w_act, v_w_act = kv_cache.update(layer_idx, k_real, xv, tau)

            outs = []
            # 1a. Damped heads (sliding-window or full-context)
            if self.n_damped > 0:
                q_d = q_aug[:, : self.n_damped]
                if self.use_attn_vacuum:
                    k_d = torch.cat([k_vac[:, : self.n_damped_kv], k_d_act], dim=-2)
                    v_d = F.pad(v_d_act, (0, 0, 1, 0), value=0.0)
                else:
                    k_d, v_d = k_d_act, v_d_act
                if self.n_rep > 1:
                    k_d = k_d.repeat_interleave(self.n_rep, dim=1)
                    v_d = v_d.repeat_interleave(self.n_rep, dim=1)

                dt = (tau[:, : self.n_damped].unsqueeze(-1) - tau_d_act.unsqueeze(-2)).clamp(min=0.0)
                t_kv = tau_d_act.size(-1)
                cm = None
                if T > 1:
                    q_pos = torch.arange(t_kv - T, t_kv, device=x.device).unsqueeze(-1)
                    k_pos = torch.arange(t_kv, device=x.device).unsqueeze(0)
                    cm = torch.where(k_pos <= q_pos, 0.0, float("-inf")).view(1, 1, T, t_kv)

                if self.use_resonant_tunneling:
                    out_d = self._attend_exact_tunneling(
                        q_d, k_d, v_d, dt=dt, causal_mask=cm,
                        has_vacuum=self.use_attn_vacuum, head_slice=slice(0, self.n_damped),
                    )
                else:
                    if self.use_attn_vacuum:
                        mask_d = F.pad(-dt, (1, 0), value=0.0)
                        if cm is not None:
                            mask_d = mask_d + F.pad(cm, (1, 0), value=0.0)
                    else:
                        mask_d = -dt if cm is None else (-dt + cm)
                    out_d = F.scaled_dot_product_attention(
                        q_d, k_d, v_d, attn_mask=mask_d.to(q_d.dtype), is_causal=False, scale=1.0
                    )
                outs.append(out_d)

            # 1b. Wormhole heads (zero decay, full context)
            if self.n_wh > 0:
                q_w = q_aug[:, self.n_damped :]
                if self.use_attn_vacuum:
                    k_w = torch.cat([k_vac[:, self.n_damped_kv :], k_w_act], dim=-2)
                    v_w = F.pad(v_w_act, (0, 0, 1, 0), value=0.0)
                else:
                    k_w, v_w = k_w_act, v_w_act
                if self.n_rep > 1:
                    k_w = k_w.repeat_interleave(self.n_rep, dim=1)
                    v_w = v_w.repeat_interleave(self.n_rep, dim=1)

                t_kv_w = k_w_act.size(-2)
                if T > 1:
                    q_pos = torch.arange(t_kv_w - T, t_kv_w, device=x.device).unsqueeze(-1)
                    k_pos = torch.arange(t_kv_w, device=x.device).unsqueeze(0)
                    cm = torch.where(k_pos <= q_pos, 0.0, float("-inf")).view(1, 1, T, t_kv_w)
                    mask_w = F.pad(cm, (1, 0), value=0.0).to(q_w.dtype) if self.use_attn_vacuum else cm.to(q_w.dtype)
                    out_w = F.scaled_dot_product_attention(q_w, k_w, v_w, attn_mask=mask_w, is_causal=False, scale=1.0)
                else:
                    out_w = F.scaled_dot_product_attention(q_w, k_w, v_w, is_causal=False, scale=1.0)
                outs.append(out_w)

            out = torch.cat(outs, dim=1) if len(outs) > 1 else outs[0]
            return self.resid_dropout(self.wo(out.transpose(1, 2).contiguous().view(B, T, -1)))

        # --- Path 2: Parallel Training / Prefill ---
        q_aug, k_vac, k_real, xv, _, tau = self._compute_step_tensors(x, freqs_cos, freqs_sin, tau_offset=None)
        if self.use_attn_vacuum:
            k_aug = torch.cat([k_vac, k_real], dim=-2)
            v_aug = F.pad(xv, (0, 0, 1, 0), value=0.0)
        else:
            k_aug, v_aug = k_real, xv

        can_use_flex = (
            self.use_flex_attention
            and _HAS_FLEX_ATTENTION
            and x.is_cuda
            and T >= 256
            and x.dtype in (torch.bfloat16, torch.float16)
            and (not self.training or self.dropout == 0.0)
        )
        if can_use_flex:
            tau_f = tau.float().contiguous()
            tau_k = tau_f.clone()
            block_mask = _get_flex_causal_block_mask(
                T, self.use_attn_vacuum, x.device,
                n_heads=self.n_heads, n_damped=self.n_damped, damped_window=self.flex_damped_window,
            )

            if self.use_resonant_tunneling:
                t_beta = F.softplus(self.tunnel_beta.float()).contiguous()
                t_theta = self.tunnel_theta.float().contiguous()
                if self.use_attn_vacuum:
                    def epr_score_mod(score, b, h, q_idx, kv_idx):
                        k_pos = torch.maximum(kv_idx - 1, torch.zeros_like(kv_idx))
                        dt = tau_f[b, h, q_idx] - tau_k[b, h, k_pos]
                        gate = torch.sigmoid(t_beta[h] * (score - t_theta[h]))
                        return torch.where(kv_idx == 0, score, score - dt * (1.0 - gate))
                else:
                    def epr_score_mod(score, b, h, q_idx, kv_idx):
                        dt = tau_f[b, h, q_idx] - tau_k[b, h, kv_idx]
                        gate = torch.sigmoid(t_beta[h] * (score - t_theta[h]))
                        return score - dt * (1.0 - gate)
            else:
                if self.use_attn_vacuum:
                    def epr_score_mod(score, b, h, q_idx, kv_idx):
                        k_pos = torch.maximum(kv_idx - 1, torch.zeros_like(kv_idx))
                        dt = tau_f[b, h, q_idx] - tau_k[b, h, k_pos]
                        return torch.where(kv_idx == 0, score, score - dt)
                else:
                    def epr_score_mod(score, b, h, q_idx, kv_idx):
                        return score - (tau_f[b, h, q_idx] - tau_k[b, h, kv_idx])

            out = _get_compiled_flex_attention()(
                q_aug, k_aug, v_aug,
                score_mod=epr_score_mod,
                block_mask=block_mask,
                scale=1.0,
                enable_gqa=(self.n_rep > 1),
            )
        else:
            if self.n_rep > 1:
                k_aug = k_aug.repeat_interleave(self.n_rep, dim=1)
                v_aug = v_aug.repeat_interleave(self.n_rep, dim=1)
            dt = (tau.unsqueeze(-1) - tau.unsqueeze(-2)).clamp(min=0.0)
            cm = torch.triu(torch.full((T, T), float("-inf"), dtype=tau.dtype, device=x.device), diagonal=1).view(1, 1, T, T)

            if self.use_resonant_tunneling:
                out = self._attend_exact_tunneling(
                    q_aug, k_aug, v_aug, dt=dt, causal_mask=cm,
                    has_vacuum=self.use_attn_vacuum, head_slice=slice(0, self.n_heads),
                )
            else:
                if self.use_attn_vacuum:
                    mask = (F.pad(-dt, (1, 0), value=0.0) + F.pad(cm, (1, 0), value=0.0)).to(q_aug.dtype)
                else:
                    mask = (-dt + cm).to(q_aug.dtype)
                out = F.scaled_dot_product_attention(
                    q_aug, k_aug, v_aug,
                    attn_mask=mask,
                    dropout_p=self.dropout if self.training else 0.0,
                    is_causal=False,
                    scale=1.0,
                )

        return self.resid_dropout(self.wo(out.transpose(1, 2).contiguous().view(B, T, -1)))

    @torch.no_grad()
    def measure_diagnostics(
        self, x: torch.Tensor, freqs_cos: torch.Tensor, freqs_sin: torch.Tensor, return_token_vac: bool = False
    ):
        """
        Return per-head `(vac_rate [H], mean_dtau [H], token_vac_probs [B, H, T] or None)`
        for live physics inspection and zero-cost OOD / hallucination probing.
        """
        B, T, _ = x.shape
        q_aug, k_vac, k_real, _, dtau, tau = self._compute_step_tensors(x, freqs_cos, freqs_sin, tau_offset=None)
        mean_dtau = dtau.mean(dim=(0, 2))
        if not self.use_attn_vacuum:
            return None, mean_dtau, None

        k_aug = torch.cat([k_vac, k_real], dim=-2)
        if self.n_rep > 1:
            k_aug = k_aug.repeat_interleave(self.n_rep, dim=1)
        scores = torch.matmul(q_aug.float(), k_aug.float().transpose(-2, -1))
        dt = (tau.float().unsqueeze(-1) - tau.float().unsqueeze(-2)).clamp(min=0.0)
        cm = torch.triu(torch.full((T, T), float("-inf"), device=x.device), diagonal=1).view(1, 1, T, T)

        if self.use_resonant_tunneling:
            beta = F.softplus(self.tunnel_beta.float()).view(1, -1, 1, 1)
            theta = self.tunnel_theta.float().view(1, -1, 1, 1)
            s_real = scores[..., 1:]
            gate = torch.sigmoid(beta * (s_real - theta))
            scores = torch.cat([scores[..., :1], s_real - dt * (1.0 - gate) + cm], dim=-1)
        else:
            scores = scores + F.pad(-dt, (1, 0), value=0.0) + F.pad(cm, (1, 0), value=0.0)

        vac_probs = F.softmax(scores, dim=-1)[..., 0]  # [B, H, T]
        vac_rate = vac_probs.mean(dim=(0, 2))          # [H]
        return vac_rate, mean_dtau, (vac_probs if return_token_vac else None)

    def extract_manifold_coords(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return `(z_base [B, H_kv, T, d_g], k_res [B, H_kv, T, d_r])` for cross-lingual manifold geometry analysis."""
        B, T, _ = x.shape
        _, k_res, z_base, _ = self.w_proj(x).split([self.q_dim, self.k_dim, self.z_dim, self.v_dim], dim=-1)
        k_res = k_res.view(B, T, self.n_kv_heads, self.d_r).transpose(1, 2)
        z_base = z_base.view(B, T, self.n_kv_heads, self.d_g).transpose(1, 2)
        return z_base, k_res
