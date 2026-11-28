# Raw `v3.0` Industrialization & `1.1B`-Scale Telemetry Archive

All raw JSON telemetry and unedited NVIDIA L4 (`22.03 GiB`, `PyTorch 2.14.1+cu130`, `bfloat16`) execution logs for the `v3.0` industrialization & `1.1B` evaluation suite:

| File | Description |
| :--- | :--- |
| [`industrial_systems_bench.json`](industrial_systems_bench.json) | `1.1B GQA` (`22L, dim=2048, 32Q/4KV, D=64`) `@triton.jit` Asymmetric PagedAttention serving memory (`379.02 MB` vs `1,357.12 MB` Standard PagedAttention = **`-72.1%`**, vs `2,816.00 MB` Contiguous = **`-86.5%`**), fused decode latency (`1K..8K`), and sub-quadratic `FlexAttention` `BlockMask` (`W=512`) Fwd+Bwd throughput scaling (`1.63×` speedup at `T=8192`). |
| [`industrial_1b_results.json`](industrial_1b_results.json) | Complete `1.1B` evaluation telemetry: **Track A** (`pure_llama_1b` vs `epr_elastic_1b` from-scratch pretraining on `FineWeb-Edu`, `300 steps × 16,384 tokens/step`, `4K` zero-shot extrapolation buckets) and **Track B** (`TinyLlama-1.1B-3T` untouched anchor vs `base_1b_cpt` vs `epr_elastic_1b_cpt` `-22,166,760`-param surgery + `300 steps × 32,768 tokens/step` continued pretraining on `ARC-Easy`, `HellaSwag`, `WinoGrande`, `PIQA`, `BoolQ`, `4K` zero-shot extrapolation, `4K` Passkey NIAH, and `W=256` `71.9%`-KV-evicted `4K` evaluation). |
| [`v3_stage1_stage2_trackA_1b.log`](v3_stage1_stage2_trackA_1b.log) | Raw NVIDIA L4 stdout/stderr log covering Stage 1 (9-pillar CUDA/Triton audit), Stage 2 (`bench_paged_and_sparse.py`), and Stage 3 Track A (`1.1B` from-scratch `FineWeb-Edu` runs). |
| [`v3_track_b.log`](v3_track_b.log) | Raw NVIDIA L4 stdout/stderr log covering Stage 3 Track B (`TinyLlama-1.1B` anchor, `base_1b_cpt`, `epr_elastic_1b_cpt` surgery, 5 zero-shot benchmarks, `4K` extrapolation, `W=256` eviction, and `bf16` checkpoint exports). |
