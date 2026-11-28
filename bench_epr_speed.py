"""
Hardware Speed & Memory Benchmark for Unified EPR Attention (v2.0):
  1. Verifies `EPRKVCache` asymmetric memory savings (including `tau_damped` buffer accounting)
  2. Benchmarks Pure Llama 2 (FA2) vs EPR Distilled vs EPR Elastic-RBF vs EPR Tunneling
     across 6M, 28M, and optional `--include_1b` (TinyLlama-1.1B GQA config).
"""

import sys
import time
import torch
from model import ModelArgs, Transformer, EPRKVCache


def bench_model(name, dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len=1024, bs=4):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    print(f"\n--- {name} (dim={dim}, {n_layers}L, {n_heads}Q/{n_kv_heads}KV, T={seq_len}, B={bs}) ---")
    variants = [
        ("Pure Llama 2 (FA2/SDPA)",     dict(use_epr=False)),
        ("EPR v2.0 Distilled (Flex)",   dict(use_epr=True, use_flex_attention=True)),
        ("EPR + Elastic-RBF (Flex)",    dict(use_epr=True, use_elastic_rbf=True, use_flex_attention=True)),
        ("EPR + Res. Tunneling (Flex)", dict(use_epr=True, use_resonant_tunneling=True, use_flex_attention=True)),
    ]
    for label, kw in variants:
        if device == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        args = ModelArgs(
            dim=dim, n_layers=n_layers, n_heads=n_heads, n_kv_heads=n_kv_heads,
            vocab_size=vocab_size, max_seq_len=max(2048, seq_len), dropout=0.0, **kw,
        )
        m = Transformer(args).to(device=device, dtype=dtype)
        n_params = sum(p.numel() for p in m.parameters()) / 1e6
        opt = torch.optim.AdamW(m.parameters(), lr=3e-4, fused=(device == "cuda"))
        x = torch.randint(0, vocab_size, (bs, seq_len), device=device)

        for _ in range(3):
            m(x, x)
            m.last_loss.backward(); opt.step(); opt.zero_grad(set_to_none=True)
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        iters = 10 if device == "cuda" else 2
        for _ in range(iters):
            m(x, x)
            m.last_loss.backward(); opt.step(); opt.zero_grad(set_to_none=True)
        if device == "cuda":
            torch.cuda.synchronize()
        tok_s = (iters * bs * seq_len) / (time.time() - t0)
        peak_mb = (torch.cuda.max_memory_allocated() / (1024 ** 2)) if device == "cuda" else 0.0
        print(f"  {label:28s} | Params: {n_params:6.2f}M | Train: {tok_s:8.0f} tok/s | Peak VRAM: {peak_mb:7.1f} MB")
        del m, opt


def bench_kvcache():
    print("\n--- Stateful Asymmetric EPRKVCache Memory Savings (window_size=128, incl. tau_damped) ---")
    for name, L, H, H_kv, D, n_wh_kv, T in [
        ("6M Edge MHA (4H, 1 WH, 2K)",      6,  4, 4, 64, 1, 2048),
        ("6M Edge MHA (4H, 1 WH, 8K)",      6,  4, 4, 64, 1, 8192),
        ("28M Edge MHA (8H, 1 WH, 8K)",     6,  8, 8, 64, 1, 8192),
        ("1.1B GQA (32Q/4KV, 1 WH grp, 8K)", 22, 32, 4, 64, 1, 8192),
    ]:
        cache = EPRKVCache(
            n_layers=L, n_heads=H, n_kv_heads=H_kv, head_dim=D,
            n_wormhole_kv_heads=n_wh_kv, window_size=128, max_seq_len=T,
        )
        std_mb = cache.standard_kv_bytes(T) / (1024 ** 2)
        epr_mb = cache.memory_bytes(active_seq_len=T) / (1024 ** 2)
        saved = cache.savings_ratio(T) * 100.0
        print(f"  {name:34s} | Standard KV: {std_mb:6.2f} MB -> EPR KV: {epr_mb:6.2f} MB | Saved: {saved:5.1f}%")


if __name__ == "__main__":
    bench_kvcache()
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        bench_model("6M EPR", 256, 6, 4, 4, 4096, seq_len=1024, bs=8)
        bench_model("28M EPR", 512, 6, 8, 8, 16384, seq_len=1024, bs=4)
        if "--include_1b" in sys.argv:
            bench_model("1.1B TinyLlama GQA", 2048, 22, 32, 4, 32000, seq_len=1024, bs=2)
    else:
        print("\n(Run on CUDA GPU such as NVIDIA L4/A100 to benchmark FlexAttention Triton throughput)")
