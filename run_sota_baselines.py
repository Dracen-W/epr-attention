"""
SOTA Long-Context Baseline Trainer & Unified Checkpoint Loader:
  - ALiBi (Press et al., ICLR 2022)
  - FoX / Forgetting Transformer (Lin et al., ICLR 2025)

Saves standard `model_args` + `baseline_kind` metadata inside `ckpt.pt` and provides
`load_any_checkpoint()` so all evaluation scripts (`eval_extrapolation.py`,
`eval_needle.py`, `eval_associative_recall.py`, `eval_ood_hallucination.py`)
can load Pure Llama 2, ALiBi, FoX, and EPR checkpoints seamlessly!
"""

import math
import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from model import ModelArgs, Transformer, apply_rotary_emb, repeat_kv


class ALiBiAttention(nn.Module):
    """ALiBi (Press et al., ICLR 2022): No RoPE, fixed geometric linear slopes -m_h * (i - j)."""
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        self.n_rep = self.n_heads // self.n_kv_heads
        self.head_dim = args.dim // args.n_heads
        self.dropout = args.dropout
        self.wq = nn.Linear(args.dim, args.dim, bias=False)
        self.wk = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(args.dim, args.dim, bias=False)
        self.resid_dropout = nn.Dropout(args.dropout)
        slopes = torch.tensor([2.0 ** (-8.0 * (h + 1) / self.n_heads) for h in range(self.n_heads)])
        self.register_buffer("slopes", slopes.view(1, self.n_heads, 1, 1), persistent=False)
        self._cached_len = 0

    def _get_bias(self, T, device, dtype):
        if T > self._cached_len or not hasattr(self, "alibi_bias") or self.alibi_bias.device != device:
            pos = torch.arange(T, device=device, dtype=torch.float32)
            dist = (pos.unsqueeze(1) - pos.unsqueeze(0)).clamp(min=0.0).view(1, 1, T, T)
            causal = torch.triu(torch.full((T, T), float("-inf"), device=device), diagonal=1).view(1, 1, T, T)
            self.register_buffer("alibi_bias", -self.slopes.to(device) * dist + causal, persistent=False)
            self._cached_len = T
        return self.alibi_bias[:, :, :T, :T].to(dtype)

    def forward(self, x, freqs_cos, freqs_sin):
        B, T, _ = x.shape
        q = self.wq(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = repeat_kv(self.wk(x).view(B, T, self.n_kv_heads, self.head_dim), self.n_rep).transpose(1, 2)
        v = repeat_kv(self.wv(x).view(B, T, self.n_kv_heads, self.head_dim), self.n_rep).transpose(1, 2)
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=self._get_bias(T, x.device, q.dtype),
            dropout_p=self.dropout if self.training else 0.0, is_causal=False,
        )
        return self.resid_dropout(self.wo(out.transpose(1, 2).contiguous().view(B, T, -1)))


class FoXAttention(nn.Module):
    """FoX / Forgetting Transformer (Lin et al., ICLR 2025): RoPE + data-dependent cumulative forget gate."""
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        self.n_rep = self.n_heads // self.n_kv_heads
        self.head_dim = args.dim // args.n_heads
        self.dropout = args.dropout
        self.wq = nn.Linear(args.dim, args.dim, bias=False)
        self.wk = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(args.dim, args.dim, bias=False)
        self.w_forget = nn.Linear(args.dim, self.n_heads, bias=True)
        self.resid_dropout = nn.Dropout(args.dropout)
        self._cached_len = 0

    def _get_causal(self, T, device):
        if T > self._cached_len or not hasattr(self, "causal") or self.causal.device != device:
            c = torch.triu(torch.full((T, T), float("-inf"), device=device), diagonal=1).view(1, 1, T, T)
            self.register_buffer("causal", c, persistent=False)
            self._cached_len = T
        return self.causal[:, :, :T, :T]

    def forward(self, x, freqs_cos, freqs_sin):
        B, T, _ = x.shape
        q = self.wq(x).view(B, T, self.n_heads, self.head_dim)
        k = self.wk(x).view(B, T, self.n_kv_heads, self.head_dim)
        v = self.wv(x).view(B, T, self.n_kv_heads, self.head_dim)
        q, k = apply_rotary_emb(q, k, freqs_cos, freqs_sin)
        q = q.transpose(1, 2)
        k = repeat_kv(k, self.n_rep).transpose(1, 2)
        v = repeat_kv(v, self.n_rep).transpose(1, 2)
        log_f = F.logsigmoid(self.w_forget(x).float()).transpose(1, 2)
        c = torch.cumsum(log_f, dim=-1)
        decay_bias = (c.unsqueeze(-1) - c.unsqueeze(-2)).clamp(max=0.0) + self._get_causal(T, x.device)
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=decay_bias.to(q.dtype),
            dropout_p=self.dropout if self.training else 0.0, is_causal=False,
        )
        return self.resid_dropout(self.wo(out.transpose(1, 2).contiguous().view(B, T, -1)))


def build_baseline(kind: str, args: ModelArgs = None):
    if args is None:
        args = ModelArgs(dim=256, n_layers=6, n_heads=4, n_kv_heads=4, vocab_size=4096, multiple_of=32, max_seq_len=512, dropout=0.15)
    m = Transformer(args)
    for layer in m.layers:
        layer.attention = ALiBiAttention(args) if kind == "alibi" else FoXAttention(args)
    m.apply(m._init_weights)
    for pn, p in m.named_parameters():
        if pn.endswith("w3.weight") or pn.endswith("wo.weight"):
            nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * args.n_layers))
    return m, args


def load_any_checkpoint(path: str, device: str = "cpu"):
    """Load Pure Llama 2, EPR Attention, ALiBi, or FoX checkpoints uniformly."""
    ckpt_path = os.path.join(path, "ckpt.pt") if os.path.isdir(path) else path
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    valid_fields = set(ModelArgs.__dataclass_fields__.keys())
    raw_args = ckpt.get("model_args", None)
    if raw_args is None:
        margs = ModelArgs(dim=256, n_layers=6, n_heads=4, n_kv_heads=4, vocab_size=4096, multiple_of=32, max_seq_len=512, dropout=0.0)
    else:
        margs = ModelArgs(**{k: v for k, v in raw_args.items() if k in valid_fields})
        margs.dropout = 0.0

    kind = ckpt.get("kind", ckpt.get("baseline_kind", None))
    if kind in ("alibi", "fox"):
        model, _ = build_baseline(kind, margs)
    else:
        model = Transformer(margs)

    sd = {k.replace("_orig_mod.", ""): v for k, v in ckpt["model"].items()}
    model.load_state_dict(sd, strict=True)
    model.to(device).eval()
    return model, margs, ckpt


def train_baseline(kind: str, out_dir: str, seed: int = 1337, dataset: str = "wikitext2"):
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    torch.manual_seed(seed)
    np.random.seed(seed)
    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    train_data = np.memmap(f"data/{dataset}/train.bin", dtype=np.uint16, mode="r")
    val_data = np.memmap(f"data/{dataset}/val.bin", dtype=np.uint16, mode="r")
    B, T = 32, 512

    def get_batch(split, rng=None):
        d = train_data if split == "train" else val_data
        mx = len(d) - T - 1
        ix = rng.randint(0, mx, size=(B,)) if rng is not None else torch.randint(mx, (B,)).numpy()
        x = torch.stack([torch.from_numpy(d[i : i + T].astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy(d[i + 1 : i + 1 + T].astype(np.int64)) for i in ix])
        return (x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)) if device == "cuda" else (x.to(device), y.to(device))

    raw_model, margs = build_baseline(kind)
    raw_model.to(device)
    dec = sum(p.numel() for p in raw_model.parameters() if p.dim() >= 2)
    print(f"\n=== Training {kind.upper()} (seed={seed}) | decayed params: {dec:,} ===")
    opt = raw_model.configure_optimizers(0.1, 8e-4, (0.9, 0.95), "cuda" if device == "cuda" else "cpu")
    model = torch.compile(raw_model) if device == "cuda" else raw_model

    best_val = 1e9
    t_start = time.time()
    for it in range(3001):
        lr = 8e-4 * (it + 1) / 201 if it < 200 else 8e-5 + 0.5 * (8e-4 - 8e-5) * (1 + math.cos(math.pi * (it - 200) / 2800))
        for g in opt.param_groups:
            g["lr"] = lr
        if it % 250 == 0:
            model.eval()
            rng = np.random.RandomState(42)
            with torch.no_grad():
                if device == "cuda":
                    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                        v_losses = [(model(*get_batch("val", rng)), raw_model.last_loss.item())[1] for _ in range(20)]
                else:
                    v_losses = [(model(*get_batch("val", rng)), raw_model.last_loss.item())[1] for _ in range(20)]
            vl = float(np.mean(v_losses))
            if vl < best_val and it > 0:
                best_val = vl
                os.makedirs(out_dir, exist_ok=True)
                torch.save(
                    {
                        "model": raw_model.state_dict(),
                        "kind": kind,
                        "baseline_kind": kind,
                        "model_args": margs.__dict__,
                        "config": {"dataset": dataset, "seed": seed},
                        "iter_num": it,
                        "best_val_loss": best_val,
                    },
                    f"{out_dir}/ckpt.pt",
                )
            print(f"[{kind.upper()}] step {it:4d}: val loss {vl:.4f}, val ppl {math.exp(vl):.2f}, elapsed {time.time()-t_start:.1f}s")
            model.train()
        if it == 3000:
            break
        x, y = get_batch("train")
        if device == "cuda":
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                model(x, y)
                loss = raw_model.last_loss
        else:
            model(x, y)
            loss = raw_model.last_loss
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)

    return raw_model


if __name__ == "__main__":
    seed = 1337
    kinds = ["alibi", "fox"]
    out_prefix = "out-6m"
    for a in sys.argv[1:]:
        if a.startswith("--seed="):
            seed = int(a.split("=", 1)[1])
        elif a.startswith("--kinds="):
            kinds = a.split("=", 1)[1].split(",")
        elif a.startswith("--out_prefix="):
            out_prefix = a.split("=", 1)[1]
    for k in kinds:
        suffix = f"-s{seed}" if seed != 1337 else ""
        train_baseline(k, f"{out_prefix}-{k}{suffix}", seed=seed)
