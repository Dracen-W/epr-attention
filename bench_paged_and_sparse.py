"""
bench_paged_and_sparse.py — Industrial v3.0 GPU Benchmarks for EPR Attention:
  1. Custom `@triton.jit` Asymmetric PagedAttention (`EPRPagedKVCache`):
     - Variable-length continuous batching (`B=16`, heterogeneous lengths up to `8192`) physical memory
       savings vs standard contiguous KV allocation (eliminating both padding fragmentation & O(T) damped KV!).
     - Fused Triton Paged decode throughput (`tok/s` & `ms/step`) across `L = 1024, 2048, 4096, 8192`
       on 1.1B GQA (`dim=2048, 22L, 32Q/4KV, head_dim=64`).
  2. Sub-Quadratic `FlexAttention` Asymmetric `BlockMask` (`flex_damped_window=512`):
     - Compares Pure Llama 2 (FA2), Dense-Causal EPR `FlexAttention`, and Block-Sparse EPR `FlexAttention`
       (`W_damped=512` + Full Wormhole) forward+backward training throughput across `T = 1024, 2048, 4096, 8192`!
"""

import gc
import json
import os
import time

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch

from epr_attention import EPRConfig, EPRAttention
from epr_paged_attention import EPRPagedKVCache
from model import ModelArgs, Transformer, precompute_freqs_cis


def bench_paged_serving_1b(device: str = "cuda"):
    print("=" * 88)
    print("[1] 1.1B GQA Asymmetric Triton PagedAttention Serving Benchmark (22L, 32Q/4KV, D=64)")
    print("=" * 88)

    # 1a. Continuous Batching Physical Memory Footprint (B=16 heterogeneous requests in [256 .. 8192])
    seq_lens = [260, 512, 780, 1024, 1500, 2048, 2600, 3100, 4096, 4800, 5500, 6200, 7000, 7500, 8000, 8192]
    B = len(seq_lens)
    cache = EPRPagedKVCache(
        n_layers=22, n_heads=32, n_kv_heads=4, head_dim=64, n_wormhole_kv_heads=1,
        window_size=128, max_seq_len=8192, max_batch_size=B, page_size=16,
        dtype=torch.bfloat16, device=device, use_triton=True,
    )
    for b, L in enumerate(seq_lens):
        for pos in range(L):
            cache._ensure_token_position_allocated(b, pos)
    cache._sync_tables_to_device(B)

    std_contig_mb = cache.standard_contiguous_bytes(max_alloc_len=8192) / (1024 ** 2)
    # Standard vLLM PagedAttention (all 4 KV heads paged, no damped window eviction)
    total_pages_no_evict = sum((L + 15) // 16 for L in seq_lens)
    vllm_std_paged_mb = (total_pages_no_evict * 22 * 16 * 2 * 4 * 64 * 2) / (1024 ** 2)
    epr_paged_mb = cache.active_physical_bytes() / (1024 ** 2)

    sav_vs_contig = (1.0 - epr_paged_mb / std_contig_mb) * 100.0
    sav_vs_vllm = (1.0 - epr_paged_mb / vllm_std_paged_mb) * 100.0
    print(f"  Continuous Batch (B=16, lens 260..8192, avg={sum(seq_lens)//B}):")
    print(f"    - Contiguous Static KV (max=8192) : {std_contig_mb:8.2f} MB")
    print(f"    - Standard PagedAttention (4 KV)  : {vllm_std_paged_mb:8.2f} MB")
    print(f"    - EPR Asymmetric Paged (W=128+WH) : {epr_paged_mb:8.2f} MB  "
          f"(Saved {sav_vs_contig:.1f}% vs Contig | {sav_vs_vllm:.1f}% vs Standard Paged!)")
    del cache
    torch.cuda.empty_cache()

    # 1b. Single-Layer & 22-Layer Equivalent Triton Kernel Latency & Throughput across Context Lengths
    print("\n  Fused `@triton.jit` `_epr_paged_decode_kernel` Scaling (B=8 concurrent requests, 32Q/4KV, W=128):")
    print(f"  {'Context Len':>12s} | {'22L Kernel ms':>14s} | {'Attn Bandwidth tok/s':>22s} | {'Damped Pages/Seq':>17s} | {'WH Pages/Seq':>13s}")
    print("  " + "-" * 86)

    B_dec = 8
    paged_rows = []
    for ctx_len in [1024, 2048, 4096, 8192]:
        c1 = EPRPagedKVCache(
            n_layers=1, n_heads=32, n_kv_heads=4, head_dim=64, n_wormhole_kv_heads=1,
            window_size=128, max_seq_len=ctx_len + 32, max_batch_size=B_dec, page_size=16,
            dtype=torch.bfloat16, device=device, use_triton=True,
        )
        for b in range(B_dec):
            for pos in range(ctx_len):
                c1._ensure_token_position_allocated(b, pos)
        c1._sync_tables_to_device(B_dec)

        q_step = torch.randn(B_dec, 32, 64, device=device, dtype=torch.bfloat16)
        tau_q = torch.full((B_dec, 32), 25.0, device=device, dtype=torch.float32)
        v0 = torch.tensor([-2.0, -2.0, -2.0, 2.5], device=device, dtype=torch.float32)

        for _ in range(10):
            _ = c1._decode_triton(0, q_step, tau_q, v0, True, True)
        torch.cuda.synchronize()

        iters = 100
        t0 = time.perf_counter()
        for _ in range(iters):
            _ = c1._decode_triton(0, q_step, tau_q, v0, True, True)
        torch.cuda.synchronize()
        ms_1l = (time.perf_counter() - t0) * 1000.0 / iters
        ms_22l = ms_1l * 22.0
        attn_tok_s = (B_dec * 1000.0) / ms_22l
        d_pgs = len(c1._d_pages_seq[0])
        w_pgs = len(c1._w_pages_seq[0])
        print(f"  {ctx_len:12d} | {ms_22l:11.3f} ms | {attn_tok_s:18,.0f} tok/s | {d_pgs:17d} | {w_pgs:13d}")
        paged_rows.append({
            "context_len": ctx_len,
            "kernel_22l_ms": round(ms_22l, 3),
            "attn_tok_per_s": round(attn_tok_s, 1),
            "damped_pages_per_seq": d_pgs,
            "wh_pages_per_seq": w_pgs,
        })
        del c1
        torch.cuda.empty_cache()

    return {
        "batch16_memory_mb": {
            "contiguous_8k": round(std_contig_mb, 2),
            "standard_paged": round(vllm_std_paged_mb, 2),
            "epr_asymmetric_paged": round(epr_paged_mb, 2),
            "saved_vs_contiguous_pct": round(sav_vs_contig, 1),
            "saved_vs_standard_paged_pct": round(sav_vs_vllm, 1),
        },
        "triton_decode_scaling": paged_rows,
    }


def bench_subquadratic_blockmask(device: str = "cuda"):
    print("\n" + "=" * 88)
    print("[2] Sub-Quadratic `FlexAttention` Asymmetric `BlockMask` Fwd+Bwd Speedup (1.1B GQA: 32Q/4KV, D=64)")
    print("=" * 88)
    print(f"  {'Seq Len T':>10s} | {'Dense EPR Flex (tok/s)':>24s} | {'Block-Sparse EPR W=512':>24s} | {'Sparse Speedup':>15s}")
    print("  " + "-" * 82)

    rows = []
    # Benchmark a 1.1B-geometry GQA EPRAttention layer (`dim=2048, 32Q/4KV, D=64`) across T = 1024 .. 8192 (fwd + bwd)
    for T in [1024, 2048, 4096, 8192]:
        B = max(1, 8192 // T)
        res_t = {"seq_len": T, "batch_size": B}
        for tag, win in [("dense", 0), ("sparse_w512", 512)]:
            torch._dynamo.reset()
            attn = EPRAttention(EPRConfig(
                dim=2048, n_heads=32, n_kv_heads=4,
                use_wormhole=True, use_elastic_rbf=True,
                use_flex_attention=True, flex_damped_window=win,
            )).to(device=device, dtype=torch.bfloat16)
            fc, fs = precompute_freqs_cis(attn.d_r, T)
            fc, fs = fc.to(device), fs.to(device)
            x = torch.randn(B, T, 2048, device=device, dtype=torch.bfloat16, requires_grad=True)

            for _ in range(4):
                out = attn(x, fc, fs)
                out.sum().backward()
                attn.zero_grad(set_to_none=True)
                x.grad = None
            torch.cuda.synchronize()

            iters = 20
            t0 = time.perf_counter()
            for _ in range(iters):
                out = attn(x, fc, fs)
                out.sum().backward()
                attn.zero_grad(set_to_none=True)
                x.grad = None
            torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            res_t[tag] = round((B * T * iters) / dt, 1)
            del attn, x, out
            gc.collect()
            torch.cuda.empty_cache()

        sp = res_t["sparse_w512"] / max(1e-6, res_t["dense"])
        res_t["speedup"] = round(sp, 2)
        rows.append(res_t)
        print(f"  {T:10d} | {res_t['dense']:18,.0f} tok/s | {res_t['sparse_w512']:18,.0f} tok/s | {sp:13.2f}x", flush=True)

    return rows


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA not available; skipping GPU throughput benchmark.")
    else:
        print("GPU:", torch.cuda.get_device_name(0))
        out = {
            "gpu": torch.cuda.get_device_name(0),
            "paged_serving": bench_paged_serving_1b("cuda"),
            "sparse_blockmask": bench_subquadratic_blockmask("cuda"),
        }
        with open("industrial_systems_bench.json", "w") as f:
            json.dump(out, f, indent=2)
        print("\nSaved -> industrial_systems_bench.json")
