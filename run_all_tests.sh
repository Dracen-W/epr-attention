#!/usr/bin/env bash
# One-Click Benchmark & Audit Wrapper for Unified Fiber-Gravity & ER=EPR Wormhole Attention (v6)
# Usage:
#   bash run_all_tests.sh            # Default: 6M WikiText-2 full suite (audit + train + extrap + needle + sample)
#   bash run_all_tests.sh audit      # 6-pillar fp64 causality & math parity audit only (~3s)
#   bash run_all_tests.sh ende       # 6.8M WMT14 En->De translation BLEU showdown
#   bash run_all_tests.sh zeroshot   # 7.9M Trilingual Zero-Shot De<->Fr translation showdown
#   bash run_all_tests.sh 28m        # 28M RealWorld 16K (8-head) full showcase
#   bash run_all_tests.sh all        # Run audit + wikitext2 + ende + zeroshot
set -euo pipefail

MODE="${1:-wikitext2}"
python3 run_all_tests.py --mode="${MODE}"
