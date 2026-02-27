#!/bin/bash
set -e

# Output directory for nsys profiles
OUTDIR="results/nsys_profiles"
mkdir -p "$OUTDIR"

# 4 model sizes x 4 context lengths = 16 tasks, 4 GPUs parallel
parallel --jobs 4 \
    'CUDA_VISIBLE_DEVICES=$(( ({%} - 1) )) nsys profile \
        --capture-range=cudaProfilerApi \
        --capture-range-end=stop \
        -o '"$OUTDIR"'/profile_{1}_{2} \
        -f true \
        uv run python -m cs336_systems.basic_lm_nsys_profile_run \
        --sizes {1} --context_length {2} \
        > '"$OUTDIR"'/log_{1}_{2}.txt 2>&1 \
        && echo "[DONE] size={1} ctx={2}" \
        || echo "[FAIL] size={1} ctx={2}"' \
    ::: small medium large xl \
    ::: 128 256 512 1024

echo "All profiling jobs completed. Results in $OUTDIR/"
