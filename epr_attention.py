"""
epr_attention.py — Standalone, Zero-Dependency EPR Attention Core
(Euclidean-RBF, Proper-time & Resonance-fiber Attention)

Inspired by ER=EPR spacetime folding and spatial gravitational field coupling.
Pure PyTorch implementation with:
  1. EPRConfig      — Pure attention hyperparameters (decoupled from any backbone)
  2. EPRKVCache     — Real stateful autoregressive KV-cache with cumulative tau tracking
                      and asymmetric damped-head sliding-window eviction (O(1) memory)
  3. EPRAttention   — Drop-in nn.Module supporting both interleaved (llama2.c) and
                      half_split (HuggingFace Llama/Qwen) RoPE, native GQA FlexAttention,
                      and exact SDPA / cached autoregressive decoding.
"""

from dataclasses import dataclass
import math
from typing import Optional, Tuple

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


def _get_flex_causal_block_mask(seqlen: int, device: torch.device):
    key = (seqlen, str(device))
    if key not in _FLEX_BLOCK_MASK_CACHE:
        def _mask(b, h, q_idx, kv_idx):
            return kv_idx <= q_idx + 1
        _FLEX_BLOCK_MASK_CACHE[key] = _create_block_mask(
            _mask, B=None, H=None, Q_LEN=seqlen, KV_LEN=seqlen + 1, device=device
        )
    return _FLEX_BLOCK_MASK_CACHE[key]


def _apply_rope_interleaved(
    q: torch.Tensor, k: torch.Tensor, freqs_cos: torch.Tensor, freqs_sin: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """llama2.c interleaved RoPE. q, k: [B, H, T, d_r]; freqs: [T, d_r//2]."""
    q_r, q_i = q.float().reshape(*q.shape[:-1], -1, 2).unbind(-1)
    k_r, k_i = k.float().reshape(*k.shape[:-1], -1, 2).unbind(-1)
    fc = freqs_cos.float().unsqueeze(0).unsqueeze(0)
    fs = freqs_sin.float().unsqueeze(0).unsqueeze(0)
    q_out = torch.stack([q_r * fc - q_i * fs, q_r * fs + q_i * fc], dim=-1).flatten(-2)
    k_out = torch.stack([k_r * fc - k_i * fs, k_r * fs + k_i * fc], dim=-1).flatten(-2)
    return q_out.type_as(q), k_out.type_as(k)


def _apply_rope_half_split(
    q: torch.Tensor, k: torch.Tensor, freqs_cos: torch.Tensor, freqs_sin: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """HuggingFace half-split RoPE. q, k: [B, H, T, d_r]; freqs: [T, d_r] or [T, d_r//2]."""
    if freqs_cos.shape[-1] == q.shape[-1] // 2:
        freqs_cos = torch.cat([freqs_cos, freqs_cos], dim=-1)
        freqs_sin = torch.cat([freqs_sin, freqs_sin], dim=-1)
    fc = freqs_cos.unsqueeze(0).unsqueeze(0).to(q.dtype)
    fs = freqs_sin.unsqueeze(0).unsqueeze(0).to(q.dtype)
    half = q.shape[-1] // 2
    q_rot = torch.cat([-q[..., half:], q[..., :half]], dim=-1)
    k_rot = torch.cat([-k[..., half:], k[..., :half]], dim=-1)
    return (q * fc) + (q_rot * fs), (k * fc) + (k_rot * fs)


@dataclass
class EPRConfig:
    """Pure attention hyperparameters, decoupled from any specific Transformer backbone."""
    dim: int = 256
    n_heads: int = 4
    n_kv_heads: Optional[int] = None
    gravity_dim: int = 0               # 0 = auto power-of-2 packing: d_r + d_g + 2 == head_dim
    n_wormhole_heads: int = 1          # Number of zero-decay, NoPE ER=EPR wormhole Q-heads (must be multiple of n_rep)
    wormhole_v0_init: float = 2.5      # High vacuum throat threshold for wormhole heads
    dropout: float = 0.0
    rope_style: str = "interleaved"    # "interleaved" (llama2.c) or "half_split" (HuggingFace)
    use_flex_attention: bool = True    # Auto-route CUDA long sequences (T>=256, bf16/fp16) to FlexAttention


class EPRKVCache:
    """
    Stateful Asymmetric Edge KV-Cache for EPR Attention.
      - Maintains exact cumulative proper time `tau_state` [B, H, 1] per layer.
      - Damped heads (0 .. n_damped_kv - 1): fixed sliding window `window_size` (O(1) memory).
        If `window_size >= max_seq_len` (or None), operates in lossless full-context mode.
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

        # Cumulative proper-time state per layer: [B, H, 1] (always float64 if dtype==float64 else float32)
        tau_dtype = torch.float64 if dtype == torch.float64 else torch.float32
        self.tau_dtype = tau_dtype
        self.tau_state = [
            torch.zeros(batch_size, n_heads, 1, dtype=tau_dtype, device=device)
            for _ in range(n_layers)
        ]

        # Damped KV + per-slot tau_k (ring buffer of size window_size)
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

        # Wormhole KV (full-context retention; tau is identically 0 so no tau buffer needed!)
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
        """
        Update layer cache with new tokens and return active buffers:
          k_real, v_real: [B, H_kv, T_new, D]
          tau_step:       [B, H, T_new] (cumulative proper time at these positions)
        Returns:
          (k_d_active, v_d_active, tau_d_active, k_w_active, v_w_active)
        """
        t_new = k_real.size(2)
        start = self.seq_pos
        end = start + t_new
        if end > self.max_seq_len:
            raise ValueError(f"EPRKVCache overflow: {end} > max_seq_len={self.max_seq_len}")

        k_d, k_w = k_real[:, : self.n_damped_kv], k_real[:, self.n_damped_kv :]
        v_d, v_w = v_real[:, : self.n_damped_kv], v_real[:, self.n_damped_kv :]
        tau_d = tau_step[:, : self.n_damped_q]

        # 1. Update Wormhole full-context buffer
        if self.n_wh_kv > 0:
            self.k_wh[layer_idx][:, :, start:end] = k_w
            self.v_wh[layer_idx][:, :, start:end] = v_w

        # 2. Update Damped ring buffer (contiguous slice if no wrap, else circular write)
        if self.n_damped_kv > 0:
            for offset in range(t_new):
                slot = (start + offset) % self.window_size
                self.k_damped[layer_idx][:, :, slot : slot + 1] = k_d[:, :, offset : offset + 1]
                self.v_damped[layer_idx][:, :, slot : slot + 1] = v_d[:, :, offset : offset + 1]
                self.tau_damped[layer_idx][:, :, slot : slot + 1] = tau_d[:, :, offset : offset + 1]

        # 3. Update cumulative tau state
        self.tau_state[layer_idx] = tau_step[:, :, -1:].to(self.tau_dtype)

        # 4. Read active slices (chronological order for damped heads)
        if end <= self.window_size:
            k_d_act = self.k_damped[layer_idx][:, :, :end]
            v_d_act = self.v_damped[layer_idx][:, :, :end]
            tau_d_act = self.tau_damped[layer_idx][:, :, :end]
        else:
            # Reorder ring buffer so oldest active token is first, newest is last
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

    def memory_bytes(self) -> int:
        total = 0
        for lst in (self.k_damped, self.v_damped, self.tau_damped, self.k_wh, self.v_wh, self.tau_state):
            for t in lst:
                total += t.numel() * t.element_size()
        return total

    def standard_kv_bytes(self, seq_len: int) -> int:
        elem = self.k_wh[0].element_size()
        return 2 * self.n_layers * self.batch_size * self.n_kv_heads * seq_len * self.head_dim * elem

    def savings_ratio(self, seq_len: int) -> float:
        std = self.standard_kv_bytes(seq_len)
        return 1.0 - (self.memory_bytes() / max(1, std))


class EPRAttention(nn.Module):
    """
    Standalone EPR Attention (Euclidean-RBF, Proper-time & Resonance-fiber).
    Implements the 4 distilled geometric primitives:
      1. Orthogonal Fiber-Base Packing (d_r + d_g + 2 == head_dim, zero V padding)
      2. Physical Vacuum Floor (Softmax1 / Mass Gap e^{v_0})
      3. Nested Dynamic Proper-Time (Solar-system damped heads)
      4. Dual-Regime Topology (Damped RoPE heads + Zero-decay NoPE ER=EPR Wormhole heads)
    """
    def __init__(self, cfg: EPRConfig):
        super().__init__()
        self.cfg = cfg
        self.n_heads = cfg.n_heads
        self.n_kv_heads = cfg.n_heads if cfg.n_kv_heads is None else cfg.n_kv_heads
        assert self.n_heads % self.n_kv_heads == 0, "n_heads must be divisible by n_kv_heads"
        self.n_rep = self.n_heads // self.n_kv_heads
        self.head_dim = cfg.dim // self.n_heads

        # Orthogonal Fiber-Base Packing: d_r + d_g + 2 == head_dim
        if cfg.gravity_dim > 0:
            self.d_g = cfg.gravity_dim
            self.d_r = self.head_dim - self.d_g - 2
        else:
            self.d_r = (3 * self.head_dim) // 4
            if self.d_r % 2 != 0:
                self.d_r -= 1
            self.d_g = self.head_dim - self.d_r - 2
        assert self.d_r >= 2 and self.d_r % 2 == 0 and self.d_g >= 2

        assert cfg.n_wormhole_heads % self.n_rep == 0, "n_wormhole_heads must be a multiple of n_rep"
        self.n_wh = cfg.n_wormhole_heads
        self.n_damped = self.n_heads - self.n_wh
        self.n_wh_kv = self.n_wh // self.n_rep
        self.n_damped_kv = self.n_kv_heads - self.n_wh_kv

        self.q_dim = self.n_heads * self.d_r
        self.k_dim = self.n_kv_heads * self.d_r
        self.z_dim = self.n_kv_heads * self.d_g
        self.v_dim = self.n_kv_heads * self.head_dim

        self.w_proj = nn.Linear(cfg.dim, self.q_dim + self.k_dim + self.z_dim + self.v_dim, bias=False)
        self.wo = nn.Linear(self.n_heads * self.head_dim, cfg.dim, bias=False)
        self.w_phys = nn.Linear(cfg.dim, self.n_kv_heads + self.n_heads + self.n_damped, bias=False)

        self.log_inv_sigma2 = nn.Parameter(torch.zeros(self.n_kv_heads))
        if self.n_damped > 0:
            init_lambdas = torch.logspace(math.log10(0.30), math.log10(0.015), self.n_damped)
            self.b_tau = nn.Parameter(torch.log(torch.expm1(init_lambdas)))
            if self.n_damped > 1:
                self.raw_nest_alpha = nn.Parameter(torch.full((self.n_damped - 1,), math.log(math.expm1(0.15))))
            else:
                self.register_parameter("raw_nest_alpha", None)
            self.register_buffer("nest_triu_mask", torch.triu(torch.ones(self.n_damped, self.n_damped, dtype=torch.bool)), persistent=False)

        v0_init = torch.zeros(self.n_kv_heads)
        if self.n_wh_kv > 0:
            v0_init[-self.n_wh_kv:] = cfg.wormhole_v0_init
        self.v0 = nn.Parameter(v0_init)

        self.register_buffer("vac_zeros_rg", torch.zeros(1, self.n_kv_heads, 1, self.d_r + self.d_g), persistent=False)
        self.register_buffer("vac_ones", torch.ones(1, self.n_kv_heads, 1, 1), persistent=False)
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
            fc_r = freqs_cos[:, : d_r // 2] if self.cfg.rope_style == "interleaved" else freqs_cos
            fs_r = freqs_sin[:, : d_r // 2] if self.cfg.rope_style == "interleaved" else freqs_sin
            rope_fn = _apply_rope_interleaved if self.cfg.rope_style == "interleaved" else _apply_rope_half_split
            qd, kd = rope_fn(q_res[:, : self.n_damped], k_res[:, : self.n_damped_kv], fc_r, fs_r)
            q_res = torch.cat([qd, q_res[:, self.n_damped :]], dim=1) if self.n_wh > 0 else qd
            k_res = torch.cat([kd, k_res[:, self.n_damped_kv :]], dim=1) if self.n_wh_kv > 0 else kd

        u_emit, u_recv, tau_in = self.w_phys(x).split([H_kv, H, self.n_damped], dim=-1)
        u_emit = u_emit.transpose(1, 2)
        u_recv = u_recv.transpose(1, 2)

        # Nested dynamic proper time
        if self.n_damped > 0:
            tau_in = tau_in.transpose(1, 2).to(acc_dtype)
            dtau_loc = F.softplus(tau_in + self.b_tau.to(acc_dtype).view(1, self.n_damped, 1))
            if self.n_damped > 1:
                alphas = F.softplus(self.raw_nest_alpha.to(acc_dtype)).clamp(min=1e-6)
                log_pref = F.pad(torch.cumsum(torch.log(alphas), dim=0), (1, 0), value=0.0)
                log_M = log_pref.unsqueeze(0) - log_pref.unsqueeze(1)
                M = torch.where(self.nest_triu_mask, torch.exp(log_M.clamp(max=20.0)), torch.zeros_like(log_M))
                dtau_damped = torch.einsum("ij,bjt->bit", M, dtau_loc)
            else:
                dtau_damped = dtau_loc
            dtau = torch.cat([dtau_damped, torch.zeros(B, self.n_wh, T, dtype=acc_dtype, device=x.device)], dim=1) if self.n_wh > 0 else dtau_damped
        else:
            dtau = torch.zeros(B, H, T, dtype=acc_dtype, device=x.device)

        tau = torch.cumsum(dtau, dim=-1)
        if tau_offset is not None:
            tau = tau + tau_offset.to(acc_dtype)

        # Orthogonal Fiber-Base Packing
        gamma_kv = (torch.exp(self.log_inv_sigma2.to(acc_dtype)) / math.sqrt(d_g)).view(1, H_kv, 1, 1).to(z_base.dtype)
        z_q_kv = z_base * gamma_kv
        z_half_kv = 0.5 * (z_q_kv * z_base).sum(dim=-1, keepdim=True)

        key_pot = u_emit.unsqueeze(-1) - z_half_kv
        k_real = torch.cat([k_res, z_base, key_pot, torch.zeros_like(key_pot)], dim=-1)
        k_vac = torch.cat(
            [
                self.vac_zeros_rg.to(k_real.dtype),
                self.v0.view(1, H_kv, 1, 1).to(k_real.dtype),
                self.vac_ones.to(k_real.dtype),
            ],
            dim=-1,
        ).expand(B, -1, -1, -1)

        z_q = z_q_kv.repeat_interleave(self.n_rep, dim=1) if self.n_rep > 1 else z_q_kv
        z_half_q = z_half_kv.repeat_interleave(self.n_rep, dim=1) if self.n_rep > 1 else z_half_kv
        q_shift = z_half_q - u_recv.unsqueeze(-1)
        q_aug = torch.cat([q_res * (1.0 / math.sqrt(d_r)), z_q, torch.ones_like(q_shift), q_shift], dim=-1)

        return q_aug, k_vac, k_real, xv, tau

    def forward(
        self,
        x: torch.Tensor,
        freqs_cos: torch.Tensor,
        freqs_sin: torch.Tensor,
        kv_cache: Optional[EPRKVCache] = None,
        layer_idx: int = 0,
    ) -> torch.Tensor:
        B, T, _ = x.shape

        # --- Path 1: Stateful Autoregressive Decoding with Asymmetric EPRKVCache ---
        if kv_cache is not None:
            tau_off = kv_cache.tau_state[layer_idx]
            q_aug, k_vac, k_real, xv, tau = self._compute_step_tensors(x, freqs_cos, freqs_sin, tau_offset=tau_off)
            k_d_act, v_d_act, tau_d_act, k_w_act, v_w_act = kv_cache.update(layer_idx, k_real, xv, tau)

            outs = []
            # 1a. Damped heads (sliding-window or full-context)
            if self.n_damped > 0:
                q_d = q_aug[:, : self.n_damped]
                k_d = torch.cat([k_vac[:, : self.n_damped_kv], k_d_act], dim=-2)
                v_d = F.pad(v_d_act, (0, 0, 1, 0), value=0.0)
                if self.n_rep > 1:
                    k_d = k_d.repeat_interleave(self.n_rep, dim=1)
                    v_d = v_d.repeat_interleave(self.n_rep, dim=1)

                dt = (tau[:, : self.n_damped].unsqueeze(-1) - tau_d_act.unsqueeze(-2))
                # Causal mask for multi-token prefill into cache (T > 1)
                t_kv = tau_d_act.size(-1)
                if T > 1:
                    q_pos = torch.arange(t_kv - T, t_kv, device=x.device).unsqueeze(-1)
                    k_pos = torch.arange(t_kv, device=x.device).unsqueeze(0)
                    cm = torch.where(k_pos <= q_pos, 0.0, float("-inf")).view(1, 1, T, t_kv)
                    dt = dt.clamp(min=0.0)
                    mask_d = (F.pad(-dt, (1, 0), value=0.0) + F.pad(cm, (1, 0), value=0.0)).to(q_d.dtype)
                else:
                    mask_d = F.pad(-dt.clamp(min=0.0), (1, 0), value=0.0).to(q_d.dtype)

                out_d = F.scaled_dot_product_attention(q_d, k_d, v_d, attn_mask=mask_d, is_causal=False, scale=1.0)
                outs.append(out_d)

            # 1b. Wormhole heads (zero decay, full context)
            if self.n_wh > 0:
                q_w = q_aug[:, self.n_damped :]
                k_w = torch.cat([k_vac[:, self.n_damped_kv :], k_w_act], dim=-2)
                v_w = F.pad(v_w_act, (0, 0, 1, 0), value=0.0)
                if self.n_rep > 1:
                    k_w = k_w.repeat_interleave(self.n_rep, dim=1)
                    v_w = v_w.repeat_interleave(self.n_rep, dim=1)

                t_kv_w = k_w_act.size(-2)
                if T > 1:
                    q_pos = torch.arange(t_kv_w - T, t_kv_w, device=x.device).unsqueeze(-1)
                    k_pos = torch.arange(t_kv_w, device=x.device).unsqueeze(0)
                    cm = torch.where(k_pos <= q_pos, 0.0, float("-inf")).view(1, 1, T, t_kv_w)
                    mask_w = F.pad(cm, (1, 0), value=0.0).to(q_w.dtype)
                    out_w = F.scaled_dot_product_attention(q_w, k_w, v_w, attn_mask=mask_w, is_causal=False, scale=1.0)
                else:
                    out_w = F.scaled_dot_product_attention(q_w, k_w, v_w, is_causal=False, scale=1.0)
                outs.append(out_w)

            out = torch.cat(outs, dim=1)
            return self.resid_dropout(self.wo(out.transpose(1, 2).contiguous().view(B, T, -1)))

        # --- Path 2: Parallel Training / Prefill ---
        q_aug, k_vac, k_real, xv, tau = self._compute_step_tensors(x, freqs_cos, freqs_sin, tau_offset=None)
        k_aug = torch.cat([k_vac, k_real], dim=-2)
        v_aug = F.pad(xv, (0, 0, 1, 0), value=0.0)

        can_use_flex = (
            self.cfg.use_flex_attention
            and _HAS_FLEX_ATTENTION
            and x.is_cuda
            and T >= 256
            and x.dtype in (torch.bfloat16, torch.float16)
            and (not self.training or self.cfg.dropout == 0.0)
        )
        if can_use_flex:
            tau_f = tau.float().contiguous()
            tau_k = tau_f.clone()
            block_mask = _get_flex_causal_block_mask(T, x.device)

            def epr_score_mod(score, b, h, q_idx, kv_idx):
                k_pos = torch.maximum(kv_idx - 1, torch.zeros_like(kv_idx))
                dt = tau_f[b, h, q_idx] - tau_k[b, h, k_pos]
                return torch.where(kv_idx == 0, score, score - dt)

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
            mask = (F.pad(-dt, (1, 0), value=0.0) + F.pad(cm, (1, 0), value=0.0)).to(q_aug.dtype)
            out = F.scaled_dot_product_attention(
                q_aug, k_aug, v_aug,
                attn_mask=mask,
                dropout_p=self.cfg.dropout if self.training else 0.0,
                is_causal=False,
                scale=1.0,
            )

        return self.resid_dropout(self.wo(out.transpose(1, 2).contiguous().view(B, T, -1)))
