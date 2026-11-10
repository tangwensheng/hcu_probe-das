#!/bin/bash
# Compatibility entry point for the original Gemma 3 1B example.
#
# Kept so existing commands and scripts keep working; the implementation now
# lives in run_with_probe.sh with profiles/gemma3_pretrain.sh.
#
# Usage:
#   HCU_PROBE_CONFIG=/abs/L0.json \
#   HCU_PROBE_MONITOR_CONFIG=/abs/monitor.json \
#   ./train_gemma3_pretrain_with_probe.sh <recipe> <dryrun|train> [hostfile] [nodes]
set -euo pipefail

CURRENT_DIR=$( cd "$( dirname "$0" )" && pwd )

exec "${CURRENT_DIR}/run_with_probe.sh" \
    gemma3_pretrain "${1:-}" "${2:-}" "${3:-}" "${4:-}"
