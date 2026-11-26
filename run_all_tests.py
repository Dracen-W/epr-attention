"""
================================================================================
ONE-CLICK MASTER BENCHMARK & AUDIT RUNNER (v6 Wormhole-Gravity)
================================================================================
Runs end-to-end verification, training, post-training fp64 causality audit,
physics diagnostics, 4x length extrapolation, 3x needle-in-a-haystack recall,
Damped Pauli Repulsion sampling, and WMT14 En-De / Zero-Shot De<->Fr translation.

Usage:
  python3 run_all_tests.py --mode=audit       # ~3s:   6-pillar fp64 audit only (CPU/GPU)
  python3 run_all_tests.py --mode=wikitext2   # ~8m:   6M WikiText-2 Base vs v6 + full evals
  python3 run_all_tests.py --mode=ende        # ~12m:  6.8M WMT14 En->De BLEU showdown
  python3 run_all_tests.py --mode=zeroshot    # ~15m:  7.9M Trilingual Zero-Shot De<->Fr
  python3 run_all_tests.py --mode=tinystories # ~13m:  6M TinyStories 6k-step convergence
  python3 run_all_tests.py --mode=fineweb     # ~25m:  6.8M FineWeb-Edu 15k-step extreme
  python3 run_all_tests.py --mode=28m         # ~42m:  28M RealWorld 16K (8-head) showcase
  python3 run_all_tests.py --mode=all         # Run audit + wikitext2 + ende + zeroshot
================================================================================
"""
import argparse
import subprocess
import sys


def run_cmd(cmd, title=None):
    if title:
        print("\n" + "=" * 80)
        print(f">>> {title}")
        print("=" * 80)
    print(f"$ {' '.join(cmd)}")
    res = subprocess.run(cmd)
    if res.returncode != 0:
        raise SystemExit(f"Command failed with exit code {res.returncode}: {' '.join(cmd)}")


def run_audit(ckpt=None):
    cmd = [sys.executable, "audit_all.py"]
    if ckpt:
        cmd.append(f"--ckpt={ckpt}")
    run_cmd(cmd, f"6-Pillar fp64 Causality & Math Parity Audit ({ckpt or 'Fresh Models'})")


def run_lm_pair(tag, prep_script, cfg, base_dir, v6_dir, prompt, pauli_alpha=1.5):
    if prep_script:
        run_cmd([sys.executable, prep_script], f"[{tag}] Preparing Dataset")
    run_cmd([sys.executable, "train.py", cfg, "--use_unified_gravity=False", f"--out_dir={base_dir}"],
            f"[{tag}] Training Pure Llama 2 Baseline ({base_dir})")
    run_cmd([sys.executable, "train.py", cfg, "--use_unified_gravity=True", "--use_wormhole=True", f"--out_dir={v6_dir}"],
            f"[{tag}] Training v6-Wormhole ER=EPR ({v6_dir})")
    run_audit(f"{v6_dir}/ckpt.pt")
    run_cmd([sys.executable, "inspect_fields.py", f"{v6_dir}/ckpt.pt"], f"[{tag}] Physics Parameter Inspection")
    run_cmd([sys.executable, "eval_extrapolation.py", base_dir, v6_dir, "--lengths=256,512,1024,2048"],
            f"[{tag}] 4x Zero-Shot Length Extrapolation (256 -> 2048)")
    run_cmd([sys.executable, "eval_needle.py", base_dir, v6_dir, "--distances=64,128,256,512,1024,1536"],
            f"[{tag}] 3x Distant Needle-in-a-Haystack Recall (64 -> 1536)")
    run_cmd([sys.executable, "sample.py", f"--out_dir={base_dir}", f"--start={prompt}",
             "--num_samples=1", "--max_new_tokens=180", "--temperature=0.7", "--top_k=40", "--pauli_alpha=0.0", "--seed=42"],
            f"[{tag}] Sample Generation — Pure Llama 2")
    run_cmd([sys.executable, "sample.py", f"--out_dir={v6_dir}", f"--start={prompt}",
             "--num_samples=1", "--max_new_tokens=180", "--temperature=0.7", "--top_k=40",
             f"--pauli_alpha={pauli_alpha}", "--pauli_halflife=16.0", "--seed=42"],
            f"[{tag}] Sample Generation — v6-Wormhole + Damped Pauli Repulsion")


def run_translation_pair(tag, prep_script, cfg, base_dir, v6_dir, eval_script):
    run_cmd([sys.executable, prep_script], f"[{tag}] Preparing WMT14 Translation Dataset")
    run_cmd([sys.executable, "train.py", cfg, "--use_unified_gravity=False", f"--out_dir={base_dir}"],
            f"[{tag}] Training Pure Llama 2 Baseline ({base_dir})")
    run_cmd([sys.executable, "train.py", cfg, "--use_unified_gravity=True", "--use_wormhole=True", f"--out_dir={v6_dir}"],
            f"[{tag}] Training v6-Wormhole ER=EPR ({v6_dir})")
    run_audit(f"{v6_dir}/ckpt.pt")
    run_cmd([sys.executable, eval_script, base_dir, v6_dir], f"[{tag}] sacreBLEU & Side-by-Side Translation Evaluation")


def main():
    parser = argparse.ArgumentParser(description="One-Click Master Benchmark & Audit Suite for v6-Wormhole")
    parser.add_argument(
        "--mode",
        type=str,
        default="wikitext2",
        choices=["audit", "wikitext2", "tinystories", "fineweb", "28m", "ende", "zeroshot", "all"],
        help="Benchmark suite to run (default: wikitext2)",
    )
    args = parser.parse_args()

    # Always start with the 6-pillar fp64 causality & math parity audit
    run_audit()
    if args.mode == "audit":
        return

    if args.mode in ("wikitext2", "all"):
        run_lm_pair(
            "6M WikiText-2", None, "config/train_wikitext2_colab_scale.py",
            "out-6m-base", "out-6m-v6-wormhole", "In 1945 , the", pauli_alpha=1.8,
        )
    if args.mode == "tinystories":
        run_lm_pair(
            "6M TinyStories", "scripts/prepare_tinystories_bpe.py", "config/train_tinystories_6m_converge.py",
            "out-ts-6m-base", "out-ts-6m-v6-wormhole",
            "Once upon a time, there was a little girl named Lily who found a secret key", pauli_alpha=1.0,
        )
    if args.mode == "fineweb":
        run_lm_pair(
            "6.8M FineWeb-Edu", "scripts/prepare_fineweb_edu.py", "config/train_fineweb_edu_extreme.py",
            "out-fw-6m-base", "out-fw-6m-v6-wormhole",
            "Photosynthesis is the process by which plants", pauli_alpha=1.5,
        )
    if args.mode == "28m":
        run_lm_pair(
            "28M RealWorld 16K", "scripts/prepare_realworld_16k.py", "config/train_realworld_28m.py",
            "out-rw-28m-base", "out-rw-28m-v6-wormhole",
            "In July 1969, NASA launched Apollo 11", pauli_alpha=1.4,
        )
    if args.mode in ("ende", "all"):
        run_translation_pair(
            "6.8M WMT14 En->De", "scripts/prepare_wmt14_ende.py", "config/train_wmt14_ende_6m.py",
            "out-ende-6m-base", "out-ende-6m-v6-wormhole", "eval_bleu_wmt14.py",
        )
    if args.mode in ("zeroshot", "all"):
        run_translation_pair(
            "7.9M Trilingual Zero-Shot De<->Fr", "scripts/prepare_wmt14_zeroshot.py", "config/train_wmt14_trilingual_7m.py",
            "out-tri-7m-base", "out-tri-7m-v6-wormhole", "eval_zeroshot_defr.py",
        )

    print("\n" + "=" * 80)
    print(f"ONE-CLICK BENCHMARK SUITE COMPLETED SUCCESSFULLY (mode={args.mode})!")
    print("=" * 80)


if __name__ == "__main__":
    main()
