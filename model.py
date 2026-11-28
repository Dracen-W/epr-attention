"""
model.py — Llama 2 Backbone with Pluggable EPR Attention (v2.0 Unified)

================================================================================
ATTRIBUTION & LICENSE NOTICE
================================================================================
- Base Codebase: Built upon Andrej Karpathy's `llama2.c` (MIT License,
  Copyright (c) 2023 Andrej). See `LICENSE` and `NOTICE` for full details.
- Base Architecture: Implements the Llama 2 Transformer architecture
  (RMSNorm, RoPE, SwiGLU, GQA) introduced by Meta AI in:
  Touvron et al., "Llama 2: Open Foundation and Fine-Tuned Chat Models", 2023.
- EPR Attention Core: Imported directly from standalone `epr_attention.py`
  (zero duplicate attention / KV-cache code). When `use_epr=False` (or
  `use_unified_gravity=False`), routes bit-for-bit to untouched `llama2.c` Attention.
================================================================================
"""

import inspect
import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from epr_attention import EPRConfig, EPRAttention, EPRKVCache
from epr_paged_attention import EPRPagedKVCache


@dataclass
class ModelArgs:
    """
    Hyperparameters for Pure Llama 2 (`use_epr=False`) & EPR Transformer (`use_epr=True`).
    Maintains seamless alias compatibility with `use_unified_gravity` for all configs/checkpoints.
    """
    # --- Standard Llama 2 Backbone Parameters ---
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: Optional[int] = None
    vocab_size: int = 32000
    hidden_dim: Optional[int] = None
    multiple_of: int = 256
    norm_eps: float = 1e-5
    max_seq_len: int = 2048
    dropout: float = 0.0

    # --- EPR Attention Switches (v2.0 Distilled Defaults + Potential Upgrades) ---
    use_epr: bool = False
    use_unified_gravity: bool = False     # Alias synchronized with `use_epr`
    gravity_dim: int = 0                  # 0 = auto packing (D=32 -> 24+6+2; D=64 -> 48+14+2)

    use_wormhole: bool = True
    n_wormhole_heads: int = 0
    wormhole_v0_init: float = 2.5
    use_base_rbf: bool = True
    use_time_dilation: bool = True
    use_attn_vacuum: bool = True

    # Polished Potential Upgrades
    use_elastic_rbf: bool = False         # Mass-elastic heteroscedastic RBF radius (zero extra channels!)
    elastic_alpha_init: float = 1.0
    use_resonant_tunneling: bool = False  # Pairwise resonance wormhole tunneling gate
    tunnel_theta_init: float = 2.5
    tunnel_beta_init: float = 2.0

    # Legacy / Single-Switch Ablation Switches (Default False — proven redundant at 6M)
    use_bilateral_mass: bool = False
    use_nested_time: bool = False

    rope_style: str = "interleaved"
    use_flex_attention: bool = True
    flex_damped_window: int = 0

    def __post_init__(self):
        if self.use_unified_gravity or self.use_epr:
            self.use_epr = True
            self.use_unified_gravity = True

    def to_epr_config(self) -> EPRConfig:
        return EPRConfig(
            dim=self.dim,
            n_heads=self.n_heads,
            n_kv_heads=self.n_kv_heads,
            gravity_dim=self.gravity_dim,
            use_wormhole=self.use_wormhole,
            n_wormhole_heads=self.n_wormhole_heads,
            wormhole_v0_init=self.wormhole_v0_init,
            use_base_rbf=self.use_base_rbf,
            use_time_dilation=self.use_time_dilation,
            use_attn_vacuum=self.use_attn_vacuum,
            use_elastic_rbf=self.use_elastic_rbf,
            elastic_alpha_init=self.elastic_alpha_init,
            use_resonant_tunneling=self.use_resonant_tunneling,
            tunnel_theta_init=self.tunnel_theta_init,
            tunnel_beta_init=self.tunnel_beta_init,
            use_bilateral_mass=self.use_bilateral_mass,
            use_nested_time=self.use_nested_time,
            dropout=self.dropout,
            rope_style=self.rope_style,
            use_flex_attention=self.use_flex_attention,
            flex_damped_window=self.flex_damped_window,
        )


class RMSNorm(nn.Module):
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
    """Precompute cosine and sine frequency tables for Rotary Positional Embeddings (RoPE)."""
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    return torch.cos(freqs), torch.sin(freqs)


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
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
    """Apply complex rotary positional embeddings to query and key tensors."""
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
        return self.resid_dropout(output)


# Backward-compatible alias pointing directly to standalone EPRAttention
UnifiedFiberGravityAttention = EPRAttention


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
    """Pre-Norm Llama 2 Transformer Block routing cleanly to `EPRAttention` or `Attention`."""
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.layer_id = layer_id
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.attention = EPRAttention(args.to_epr_config()) if args.use_epr else Attention(args)
        self.feed_forward = FeedForward(
            dim=args.dim,
            hidden_dim=args.hidden_dim,
            multiple_of=args.multiple_of,
            dropout=args.dropout,
        )
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cos: torch.Tensor,
        freqs_sin: torch.Tensor,
        kv_cache: Optional[EPRKVCache] = None,
    ) -> torch.Tensor:
        if kv_cache is not None and isinstance(self.attention, EPRAttention):
            h = x + self.attention(self.attention_norm(x), freqs_cos, freqs_sin, kv_cache=kv_cache, layer_idx=self.layer_id)
        else:
            h = x + self.attention(self.attention_norm(x), freqs_cos, freqs_sin)
        return h + self.feed_forward(self.ffn_norm(h))


class Transformer(nn.Module):
    """Complete Llama 2 / EPR Attention Language Model with Stateful `EPRKVCache` Decoding."""
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

        # Precompute RoPE tables:
        #   - EPR Attention rotates only the `d_r` resonance fiber (`10000^(2k / d_r)` or `head_dim` slice).
        #   - Note: using `d_r` for EPR matches standalone `EPRAttention` and maximizes frequency resolution!
        rope_dim = self.layers[0].attention.d_r if params.use_epr else (params.dim // params.n_heads)
        self._rope_dim = rope_dim
        freqs_cos, freqs_sin = precompute_freqs_cis(rope_dim, params.max_seq_len)
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

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
            fc, fs = precompute_freqs_cis(self._rope_dim, max(seqlen, self.freqs_cos.size(0)))
            self.register_buffer("freqs_cos", fc.to(device), persistent=False)
            self.register_buffer("freqs_sin", fs.to(device), persistent=False)
        return self.freqs_cos[:seqlen], self.freqs_sin[:seqlen]

    def build_kv_cache(
        self,
        batch_size: int = 1,
        max_seq_len: Optional[int] = None,
        window_size: Optional[int] = 128,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ) -> EPRKVCache:
        if not self.params.use_epr:
            raise ValueError("EPRKVCache requires an EPR Attention model (`use_epr=True`).")
        attn0: EPRAttention = self.layers[0].attention
        p0 = next(self.parameters())
        return EPRKVCache(
            n_layers=self.n_layers,
            n_heads=attn0.n_heads,
            n_kv_heads=attn0.n_kv_heads,
            head_dim=attn0.head_dim,
            n_wormhole_kv_heads=attn0.n_wh_kv,
            window_size=window_size,
            max_seq_len=max_seq_len or self.params.max_seq_len,
            batch_size=batch_size,
            dtype=dtype or p0.dtype,
            device=str(device or p0.device),
        )

    def build_paged_kv_cache(
        self,
        max_batch_size: int = 1,
        max_seq_len: Optional[int] = None,
        window_size: Optional[int] = 128,
        page_size: int = 16,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        use_triton: bool = True,
    ) -> EPRPagedKVCache:
        if not self.params.use_epr:
            raise ValueError("EPRPagedKVCache requires an EPR Attention model (`use_epr=True`).")
        attn0: EPRAttention = self.layers[0].attention
        p0 = next(self.parameters())
        return EPRPagedKVCache(
            n_layers=self.n_layers,
            n_heads=attn0.n_heads,
            n_kv_heads=attn0.n_kv_heads,
            head_dim=attn0.head_dim,
            n_wormhole_kv_heads=attn0.n_wh_kv,
            window_size=window_size,
            max_seq_len=max_seq_len or self.params.max_seq_len,
            max_batch_size=max_batch_size,
            page_size=page_size,
            dtype=dtype or p0.dtype,
            device=device or p0.device,
            use_triton=use_triton,
        )

    def forward(
        self,
        tokens: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        kv_cache: Optional[EPRKVCache] = None,
    ) -> torch.Tensor:
        _bsz, seqlen = tokens.shape
        h = self.tok_embeddings(tokens)
        h = self.dropout(h)

        if kv_cache is not None:
            start = kv_cache.seq_pos
            fc_all, fs_all = self._get_rope_freqs(start + seqlen, tokens.device)
            freqs_cos, freqs_sin = fc_all[start : start + seqlen], fs_all[start : start + seqlen]
        else:
            freqs_cos, freqs_sin = self._get_rope_freqs(seqlen, tokens.device)

        for layer in self.layers:
            h = layer(h, freqs_cos, freqs_sin, kv_cache=kv_cache)
        h = self.norm(h)

        if targets is not None:
            logits = self.output(h)
            self.last_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            logits = self.output(h[:, [-1], :])
            self.last_loss = None

        return logits

    def configure_optimizers(self, weight_decay: float, learning_rate: float, betas: Tuple[float, float], device_type: str):
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
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
        return (flops_per_iter / dt) / 312e12

    @torch.inference_mode()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        pauli_alpha: float = 0.0,
        pauli_halflife: float = 16.0,
        use_kv_cache: bool = True,
        damped_window: Optional[int] = None,
        paged_kv: bool = False,
        page_size: int = 16,
    ) -> torch.Tensor:
        """
        Autoregressive generation with stateful `EPRKVCache` or `EPRPagedKVCache` (`paged_kv=True`)
        and optional Damped Pauli Repulsion decoding (`pauli_alpha > 0`).
        """
        bsz = idx.size(0)
        use_pauli = pauli_alpha > 0.0
        if use_pauli:
            decay = math.exp(-math.log(2.0) / pauli_halflife) if pauli_halflife > 0.0 else 0.0
            v_pauli = torch.zeros((bsz, self.vocab_size), dtype=torch.float32, device=idx.device)
            for t_step in range(idx.size(1)):
                v_pauli.mul_(decay)
                v_pauli.scatter_add_(1, idx[:, t_step : t_step + 1], torch.full((bsz, 1), pauli_alpha, device=idx.device))

        kv_cache = None
        if self.params.use_epr and use_kv_cache and max_new_tokens > 0:
            total_cap = idx.size(1) + max_new_tokens + 16
            if paged_kv:
                kv_cache = self.build_paged_kv_cache(
                    max_batch_size=bsz,
                    max_seq_len=total_cap,
                    window_size=damped_window,
                    page_size=page_size,
                    dtype=next(self.parameters()).dtype,
                    device=idx.device,
                )
            else:
                kv_cache = self.build_kv_cache(
                    batch_size=bsz,
                    max_seq_len=total_cap,
                    window_size=damped_window,
                    dtype=next(self.parameters()).dtype,
                    device=idx.device,
                )

        for step_idx in range(max_new_tokens):
            if kv_cache is not None:
                tok_in = idx if step_idx == 0 else idx[:, -1:]
                logits = self(tok_in, kv_cache=kv_cache)[:, -1, :]
            else:
                idx_cond = idx if idx.size(1) <= self.params.max_seq_len else idx[:, -self.params.max_seq_len :]
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
