#!/bin/bash
set -e

# Output directory for memory profiles
OUTDIR="results/torch_memory_profiles"
mkdir -p "$OUTDIR"

# 3 stages x 3 context lengths = 9 tasks, 4 GPUs parallel
parallel --jobs 4 \
    'CUDA_VISIBLE_DEVICES=$(( ({%} - 1) )) \
        uv run python -m cs336_systems.basic_lm_memory_profile \
        --size_name large --context_length {2} \
        --save_path '\"$OUTDIR\"'/memory_snapshot \
        --stage {1} \
        > '\"$OUTDIR\"'/log_large_{2}_{1}.txt 2>&1 \
        && echo "[DONE] ctx={2} stage={1}" \
        || echo "[FAIL] ctx={2} stage={1}"' \
    ::: fw fw_with_grad full_train \
    ::: 128 256 512
echo "All profiling jobs completed. Results in $OUTDIR/"