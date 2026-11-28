"""
hf_epr.py — HuggingFace `transformers` Integration & Zero-Padding Pretrained Llama Surgery (v3.0)

Provides two industrial capabilities:
  1. Standalone HuggingFace-compatible `EPRLlamaConfig` & `EPRLlamaForCausalLM` (`PretrainedConfig` /
     `PreTrainedModel` when `transformers` is installed), enabling standard `save_pretrained()`,
     `from_pretrained()`, HuggingFace `Trainer`, and `lm-evaluation-harness` evaluation.
  2. `upgrade_hf_llama_to_epr(hf_model, ...)`: Drop-in architectural surgery converting any pretrained
     HuggingFace `LlamaForCausalLM` (e.g., `TinyLlama-1.1B`, `Llama-3.2-1B`, `SmolLM2`) into an
     `EPR-Attention` model with `-22.17M` fewer parameters (at 1.1B), preserving 100% of `MLP`,
     `W_V`, `W_O`, and the exact pretrained angular frequencies of the top `d_r // 2` RoPE pairs!
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from epr_attention import EPRConfig, EPRAttention, EPRKVCache
from epr_paged_attention import EPRPagedKVCache
from model import ModelArgs, Transformer

try:
    from transformers import PretrainedConfig, PreTrainedModel
    from transformers.modeling_outputs import CausalLMOutputWithPast
    _HAS_TRANSFORMERS = True
except Exception:
    _HAS_TRANSFORMERS = False
    PretrainedConfig = object
    PreTrainedModel = nn.Module
    CausalLMOutputWithPast = None


# =============================================================================
# 1. HuggingFace `PretrainedConfig` & `PreTrainedModel` Wrapper
# =============================================================================

class EPRLlamaConfig(PretrainedConfig):
    model_type = "epr_llama"

    def __init__(
        self,
        vocab_size: int = 32000,
        dim: int = 2048,
        n_layers: int = 22,
        n_heads: int = 32,
        n_kv_heads: int = 4,
        hidden_dim: Optional[int] = 5632,
        multiple_of: int = 256,
        norm_eps: float = 1e-5,
        max_seq_len: int = 2048,
        dropout: float = 0.0,
        use_epr: bool = True,
        gravity_dim: int = 0,
        use_wormhole: bool = True,
        n_wormhole_heads: int = 0,
        wormhole_v0_init: float = 2.5,
        use_base_rbf: bool = True,
        use_time_dilation: bool = True,
        use_attn_vacuum: bool = True,
        use_elastic_rbf: bool = True,
        elastic_alpha_init: float = 1.0,
        use_resonant_tunneling: bool = False,
        rope_style: str = "half_split",
        use_flex_attention: bool = True,
        flex_damped_window: int = 0,
        tie_word_embeddings: bool = True,
        **kwargs,
    ):
        if _HAS_TRANSFORMERS:
            # `Transformer` ties `tok_embeddings.weight = output.weight` directly; keep HF base class neutral
            # across both transformers 4.x (list `_tied_weights_keys`) and 5.x (dict `_tied_weights_keys`).
            kwargs.pop("tie_word_embeddings", None)
            super().__init__(tie_word_embeddings=False, **kwargs)
        self.vocab_size = vocab_size
        self.dim = dim
        self.hidden_size = dim
        self.n_layers = n_layers
        self.num_hidden_layers = n_layers
        self.n_heads = n_heads
        self.num_attention_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.num_key_value_heads = n_kv_heads
        self.hidden_dim = hidden_dim
        self.multiple_of = multiple_of
        self.norm_eps = norm_eps
        self.max_seq_len = max_seq_len
        self.max_position_embeddings = max_seq_len
        self.dropout = dropout
        self.use_epr = use_epr
        self.gravity_dim = gravity_dim
        self.use_wormhole = use_wormhole
        self.n_wormhole_heads = n_wormhole_heads
        self.wormhole_v0_init = wormhole_v0_init
        self.use_base_rbf = use_base_rbf
        self.use_time_dilation = use_time_dilation
        self.use_attn_vacuum = use_attn_vacuum
        self.use_elastic_rbf = use_elastic_rbf
        self.elastic_alpha_init = elastic_alpha_init
        self.use_resonant_tunneling = use_resonant_tunneling
        self.rope_style = rope_style
        self.use_flex_attention = use_flex_attention
        self.flex_damped_window = flex_damped_window

    def to_model_args(self) -> ModelArgs:
        return ModelArgs(
            dim=self.dim,
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            n_kv_heads=self.n_kv_heads,
            vocab_size=self.vocab_size,
            hidden_dim=self.hidden_dim,
            multiple_of=self.multiple_of,
            norm_eps=self.norm_eps,
            max_seq_len=self.max_seq_len,
            dropout=self.dropout,
            use_epr=self.use_epr,
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
            rope_style=self.rope_style,
            use_flex_attention=self.use_flex_attention,
            flex_damped_window=self.flex_damped_window,
        )


class EPRLlamaForCausalLM(PreTrainedModel):
    config_class = EPRLlamaConfig

    def __init__(self, config: EPRLlamaConfig):
        if _HAS_TRANSFORMERS:
            super().__init__(config)
        else:
            super().__init__()
            self.config = config
        self.backbone = Transformer(config.to_model_args())
        if _HAS_TRANSFORMERS:
            self.post_init()

    def get_input_embeddings(self):
        return self.backbone.tok_embeddings

    def set_input_embeddings(self, value):
        self.backbone.tok_embeddings = value
        self.backbone.output.weight = value.weight

    def get_output_embeddings(self):
        return self.backbone.output

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        kv_cache: Optional[EPRKVCache] = None,
        return_dict: bool = True,
        **kwargs,
    ):
        B, T = input_ids.shape
        h = self.backbone.tok_embeddings(input_ids)
        if kv_cache is not None:
            start = kv_cache.seq_pos
            fc_all, fs_all = self.backbone._get_rope_freqs(start + T, input_ids.device)
            fc, fs = fc_all[start : start + T], fs_all[start : start + T]
        else:
            fc, fs = self.backbone._get_rope_freqs(T, input_ids.device)

        for layer in self.backbone.layers:
            h = layer(h, fc, fs, kv_cache=kv_cache)
        h = self.backbone.norm(h)
        logits = self.backbone.output(h)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1), ignore_index=-100)

        if not return_dict or CausalLMOutputWithPast is None:
            return (loss, logits) if loss is not None else (logits,)
        return CausalLMOutputWithPast(loss=loss, logits=logits)


# =============================================================================
# 2. In-Place Pretrained HuggingFace Llama -> EPR Attention Surgery Adapter
# =============================================================================

class HFLlamaEPRAttentionAdapter(nn.Module):
    """
    Drop-in replacement for `transformers.models.llama.modeling_llama.LlamaAttention` inside any
    pretrained HuggingFace `LlamaForCausalLM` (compatible with `transformers >= 4.38` through `5.x`).
    Preserves the exact pretrained `inv_freq[:d_r//2]` angular frequencies on the retained fiber!
    """

    def __init__(
        self,
        epr_attn: EPRAttention,
        layer_idx: int,
        orig_head_dim: int,
        rope_theta: float = 10000.0,
    ):
        super().__init__()
        self.epr = epr_attn
        self.layer_idx = layer_idx
        self.orig_head_dim = orig_head_dim
        self.rope_theta = rope_theta
        d_r = epr_attn.d_r
        # Exact frequency preservation: keep the first `d_r // 2` frequencies from `orig_head_dim`!
        inv_freq = 1.0 / (rope_theta ** (torch.arange(0, orig_head_dim, 2)[: (d_r // 2)].float() / orig_head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._cached_len = 0
        self.register_buffer("_cos_cache", torch.empty(0), persistent=False)
        self.register_buffer("_sin_cache", torch.empty(0), persistent=False)
        self.external_kv_cache = None  # Optional attached EPRKVCache / EPRPagedKVCache
        self.return_3_tuple = False    # Auto-configured by `upgrade_hf_llama_to_epr` from DecoderLayer source

    def _get_cos_sin(self, seq_len: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        if seq_len > self._cached_len or self._cos_cache.device != device:
            cap = max(seq_len, 2048)
            t = torch.arange(cap, device=device, dtype=torch.float32)
            freqs = torch.outer(t, self.inv_freq.to(device))
            emb = torch.cat([freqs, freqs], dim=-1)  # [cap, d_r] for half_split RoPE
            self._cos_cache = emb.cos()
            self._sin_cache = emb.sin()
            self._cached_len = cap
        return self._cos_cache[:seq_len], self._sin_cache[:seq_len]

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value=None,
        past_key_values=None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ):
        B, T, _ = hidden_states.shape
        if self.external_kv_cache is not None:
            start = self.external_kv_cache.seq_pos
            cos_all, sin_all = self._get_cos_sin(start + T, hidden_states.device)
            fc, fs = cos_all[start : start + T], sin_all[start : start + T]
            out = self.epr(hidden_states, fc, fs, kv_cache=self.external_kv_cache, layer_idx=self.layer_idx)
        else:
            cos_all, sin_all = self._get_cos_sin(T, hidden_states.device)
            out = self.epr(hidden_states, cos_all, sin_all, kv_cache=None, layer_idx=self.layer_idx)

        if self.return_3_tuple:
            return out, None, past_key_value
        return out, None


@torch.no_grad()
def upgrade_hf_llama_to_epr(
    hf_model: nn.Module,
    use_elastic_rbf: bool = True,
    use_wormhole: bool = True,
    wormhole_v0_init: float = 1.5,
    damped_v0_init: float = -3.5,
    dtau_bias_init: float = -4.2,
    flex_damped_window: int = 0,
) -> Tuple[nn.Module, int, int]:
    """
    Perform exact Subspace-Preserving Architectural Surgery on a pretrained HuggingFace `LlamaForCausalLM`:
      - Keeps `embed_tokens`, `RMSNorm`, `MLP` (`gate/up/down_proj`), `v_proj`, and `o_proj` 100% intact.
      - Transfers the top `d_r // 2` high/mid-frequency RoPE pairs (`48 / 64` dims) from `q_proj` and
        `k_proj`, rescaled by `sqrt(d_r / D)` so step-0 attention logits preserve pretrained inner products!
      - Initializes `z_base` from the pruned low-frequency `k_proj` subspace (scaled gently so Step-0
        RBF perturbation is near zero) and enables `Mass-Elastic RBF` + `Wormhole` + `FlexAttention`.
    Returns `(upgraded_hf_model, orig_params, new_params)`.
    """
    orig_params = sum(p.numel() for p in hf_model.parameters())
    cfg = hf_model.config
    dim = cfg.hidden_size
    n_heads = cfg.num_attention_heads
    n_kv_heads = getattr(cfg, "num_key_value_heads", n_heads)
    head_dim = getattr(cfg, "head_dim", dim // n_heads)
    rope_theta = float(getattr(cfg, "rope_theta", 10000.0))

    p0 = next(hf_model.parameters())
    target_dtype, target_device = p0.dtype, p0.device

    import inspect as _inspect
    layer0_src = _inspect.getsource(hf_model.model.layers[0].forward)
    needs_3_tuple = ("present_key_value" in layer0_src and "= self.self_attn(" in layer0_src and layer0_src.split("= self.self_attn(")[0].splitlines()[-1].count(",") >= 2)

    for layer_idx, layer in enumerate(hf_model.model.layers):
        old_attn = layer.self_attn
        has_o_bias = getattr(old_attn.o_proj, "bias", None) is not None

        epr_cfg = EPRConfig(
            dim=dim,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            use_wormhole=use_wormhole,
            wormhole_v0_init=wormhole_v0_init,
            use_base_rbf=True,
            use_time_dilation=True,
            use_attn_vacuum=True,
            use_elastic_rbf=use_elastic_rbf,
            use_resonant_tunneling=False,
            rope_style="half_split",
            use_flex_attention=True,
            flex_damped_window=flex_damped_window,
        )
        epr = EPRAttention(epr_cfg).to(device=target_device, dtype=target_dtype)
        d_r, d_g = epr.d_r, epr.d_g
        half_orig = head_dim // 2
        half_dr = d_r // 2

        # Rescale factor so (q_res . k_res) / sqrt(d_r) == (q_sub . k_sub) / sqrt(head_dim) at Step 0!
        logit_scale = math.sqrt(d_r / float(head_dim))

        # Slice half_split RoPE indices `[0 : half_dr]` and `[half_orig : half_orig + half_dr]`
        keep_idx = torch.cat([
            torch.arange(0, half_dr, device=target_device),
            torch.arange(half_orig, half_orig + half_dr, device=target_device),
        ])
        rem_pairs = half_orig - half_dr
        rem_idx = torch.cat([
            torch.arange(half_dr, half_orig, device=target_device),
            torch.arange(half_orig + half_dr, head_dim, device=target_device),
        ])  # length `2 * rem_pairs == d_g + 2`!

        wq_old = old_attn.q_proj.weight.data.view(n_heads, head_dim, dim)
        wk_old = old_attn.k_proj.weight.data.view(n_kv_heads, head_dim, dim)
        wv_old = old_attn.v_proj.weight.data  # [n_kv_heads * head_dim, dim]
        wo_old = old_attn.o_proj.weight.data  # [dim, n_heads * head_dim]

        wq_res = (wq_old[:, keep_idx, :] * math.sqrt(logit_scale)).reshape(n_heads * d_r, dim)
        wk_res = (wk_old[:, keep_idx, :] * math.sqrt(logit_scale)).reshape(n_kv_heads * d_r, dim)
        # Initialize z_base gently from the pruned low-frequency k_proj channels (`rem_idx[:d_g]`)
        wz_init = (wk_old[:, rem_idx[:d_g], :] * 0.05).reshape(n_kv_heads * d_g, dim)

        epr.w_proj.weight.data.copy_(torch.cat([wq_res, wk_res, wz_init, wv_old], dim=0))
        epr.wo.weight.data.copy_(wo_old)
        if has_o_bias:
            epr.wo.bias = nn.Parameter(old_attn.o_proj.bias.data.clone())

        # Gentle Step-0 initialization for scalar fields so pretrained capabilities survive immediately
        epr.w_phys.weight.data.normal_(mean=0.0, std=0.002)
        if hasattr(epr, "v0") and epr.v0 is not None:
            epr.v0.data.fill_(damped_v0_init)
            if epr.n_wh_kv > 0:
                epr.v0.data[-epr.n_wh_kv :] = wormhole_v0_init
        if hasattr(epr, "b_tau") and epr.b_tau is not None:
            # Softplus(-4.2) ≈ 0.015 per token -> gentle ~67-token e-folding horizon on slowest/fastest mix
            rates = torch.linspace(dtau_bias_init + 0.8, dtau_bias_init - 0.8, epr.n_damped, device=target_device, dtype=target_dtype)
            epr.b_tau.data.copy_(rates)

        adapter = HFLlamaEPRAttentionAdapter(epr, layer_idx=layer_idx, orig_head_dim=head_dim, rope_theta=rope_theta)
        adapter.return_3_tuple = needs_3_tuple
        layer.self_attn = adapter

    new_params = sum(p.numel() for p in hf_model.parameters())
    return hf_model, orig_params, new_params
