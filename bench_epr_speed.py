"""
Quick L4 Benchmark for EPR Attention (v7):
  1. Verifies model.py FlexAttention vs SDPA training & prefill speedup on 6M / 28M
  2. Verifies EPRKVCache asymmetric memory savings (75% - 87.5% at 4K / 8K context)
"""
import time
import torch
from model import ModelArgs, Transformer, EPRKVCache


def bench_model(name, dim, n_layers, n_heads, vocab_size, seq_len=1024, bs=4):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    print(f"\n--- {name} (dim={dim}, {n_layers}L, {n_heads}H, T={seq_len}, B={bs}) ---")
    for label, use_grav, use_flex in [
        ("Pure Llama 2 (FA2/SDPA)", False, False),
        ("EPR Attention (Old SDPA)", True, False),
        ("EPR Attention (FlexAttn)", True, True),
    ]:
        if device == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        args = ModelArgs(
            dim=dim, n_layers=n_layers, n_heads=n_heads, vocab_size=vocab_size,
            max_seq_len=max(2048, seq_len), dropout=0.0,
            use_unified_gravity=use_grav, use_flex_attention=use_flex,
        )
        m = Transformer(args).to(device=device, dtype=dtype)
        opt = torch.optim.AdamW(m.parameters(), lr=3e-4)
        x = torch.randint(0, vocab_size, (bs, seq_len), device=device)

        for _ in range(2):  # warmup + Triton compile
            m(x, x)
            m.last_loss.backward(); opt.step(); opt.zero_grad(set_to_none=True)
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        iters = 8 if device == "cuda" else 2
        for _ in range(iters):
            m(x, x)
            m.last_loss.backward(); opt.step(); opt.zero_grad(set_to_none=True)
        if device == "cuda":
            torch.cuda.synchronize()
        tok_s = (iters * bs * seq_len) / (time.time() - t0)
        peak_mb = (torch.cuda.max_memory_allocated() / (1024 ** 2)) if device == "cuda" else 0.0
        print(f"  {label:25s} | Train: {tok_s:8.0f} tok/s | Peak VRAM: {peak_mb:6.1f} MB")
        del m, opt


def bench_kvcache():
    print("\n--- Asymmetric Edge EPRKVCache Memory Savings (window_size=128) ---")
    for name, L, H_kv, D, n_wh, T in [
        ("6M Edge (4H, 1 Wormhole, 4K)",   6, 4, 64, 1, 4096),
        ("28M Edge (8H, 1 Wormhole, 8K)",  6, 8, 64, 1, 8192),
        ("1.1B GQA (4KV, 1 WH grp, 8K)",  22, 4, 64, 1, 8192),
    ]:
        cache = EPRKVCache(n_layers=L, n_kv_heads=H_kv, head_dim=D, n_wormhole_kv_heads=n_wh, window_size=128, max_seq_len=T)
        std_mb = cache.standard_kv_bytes(T) / (1024 ** 2)
        epr_mb = cache.memory_bytes() / (1024 ** 2)
        saved = cache.savings_ratio(T) * 100.0
        print(f"  {name:30s} | Standard KV: {std_mb:6.2f} MB -> EPR KV: {epr_mb:6.2f} MB | Saved: {saved:5.1f}%")


if __name__ == "__main__":
    bench_kvcache()
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        bench_model("6M EPR", 256, 6, 4, 4096, seq_len=1024, bs=8)
        bench_model("28M EPR", 512, 6, 8, 16000, seq_len=1024, bs=4)
    else:
        print("\n(Run on CUDA GPU such as NVIDIA L4 to benchmark FlexAttention Triton speedup)")
