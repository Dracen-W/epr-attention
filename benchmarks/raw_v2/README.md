# Raw Experimental Telemetry Archive (`v2.0` Unified Benchmark Suite)

All files in this folder are direct, unmodified outputs collected on **NVIDIA L4 (22.0 GiB VRAM, `PyTorch 2.14.1+cu130`, `bfloat16`, `torch.compile=True`)** for the unified `v2.0` codebase. Every number and figure in [`README.md`](../../README.md) and [`README_zh.md`](../../README_zh.md) is derived directly from these files.

## Contents

| File | Description |
| :--- | :--- |
| [`cuda_l4_audit_and_speed.log`](cuda_l4_audit_and_speed.log) | Full output of `audit_all.py` (8 pillars, including CUDA `bf16` `FlexAttention` parity `1.95e-03` & `0.00e+00` future leak) and `bench_epr_speed.py` (6M & 28M `FlexAttention` throughput + `EPRKVCache` memory savings). |
| [`phase1_8models_s1337.json`](phase1_8models_s1337.json) | Complete JSON metrics for all 8 architectures at `seed=1337` (`Pure Llama 2`, `ALiBi`, `FoX`, `EPR v2.0 Distilled`, `EPR + Mass-Elastic RBF`, `EPR + Resonant Tunneling (1 WH)`, `EPR + Pure Tunneling (0 WH)`, `EPR Full`): parameter counts, training time, window & 384-seq bucketed PPL, 200-trial needle recall (`±95% CI`), standard & counterfactual 2-key associative recall, OOD overconfidence, and zero-cost vacuum abstention AUROC. |
| [`phase1_training_raw.log`](phase1_training_raw.log) | Complete raw step-by-step training stdout across all 8 models (`Step 0 → 3000`). |
| [`phase1_step_and_wallclock_curves.json`](phase1_step_and_wallclock_curves.json) | Parsed step-by-step and wall-clock convergence trajectories (`val_loss`, `val_ppl`, `wall_sec`, `median_step_ms`) from `phase1_training_raw.log`. |
| [`phase2_multiseed.json`](phase2_multiseed.json) | Additional random-seed runs (`seeds = 2026, 42`) for `Pure Llama 2`, `FoX`, `EPR v2.0 Distilled`, and `EPR + Mass-Elastic RBF`. |
| [`phase2_multiseed_and_kv.log`](phase2_multiseed_and_kv.log) | Raw training & CLI benchmark log for Phase 2, including `Dynamic NTK`, `YaRN`, `eval_kv_eviction.py` (`W = ∞, 512, 256, 128, 64` with stateful `EPRKVCache` decode `tok/s`), and `eval_ood_hallucination.py`. |
| [`summary_3seed_mean_std.json`](summary_3seed_mean_std.json) | Aggregated 3-seed (`1337, 2026, 42`) `Mean ± Sample Std` (`ddof=1`) across validation PPL, un-diluted extrapolation buckets, and 200-trial needle recall. |
