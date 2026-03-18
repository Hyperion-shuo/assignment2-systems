#!/bin/bash
set -e

# Output directory for memory profiles
OUTDIR="results/torch_memory_profiles"
mkdir -p "$OUTDIR"

NPROC=4
SIZE="medium"
CTX=256
STAGE="full_train"

echo "========================================"
echo "Memory profile comparison: ZeRO vs no-ZeRO"
echo "  Model size: $SIZE, Context length: $CTX, Stage: $STAGE"
echo "========================================"

# 1. Baseline: without ZeRO (DDP only, multi-GPU via torchrun)
echo "[START] Baseline (no ZeRO, DDP $NPROC GPUs) ..."
uv run torchrun --nproc_per_node=$NPROC \
    -m cs336_systems.ddp_lm_memory_profile \
    --size_name $SIZE --context_length $CTX \
    --save_path "$OUTDIR/memory_snapshot" \
    --stage $STAGE \
    > "$OUTDIR/log_${SIZE}_cl${CTX}_${STAGE}_no_zero.txt" 2>&1 \
    && echo "[DONE] Baseline (no ZeRO, DDP $NPROC GPUs)" \
    || echo "[FAIL] Baseline (no ZeRO, DDP $NPROC GPUs)"

# 2. With ZeRO (multi-GPU via torchrun)
echo "[START] ZeRO ($NPROC GPUs) ..."
uv run torchrun --nproc_per_node=$NPROC \
    -m cs336_systems.ddp_lm_memory_profile \
    --size_name $SIZE --context_length $CTX \
    --save_path "$OUTDIR/memory_snapshot" \
    --stage $STAGE \
    --use_zero \
    > "$OUTDIR/log_${SIZE}_cl${CTX}_${STAGE}_zero.txt" 2>&1 \
    && echo "[DONE] ZeRO ($NPROC GPUs)" \
    || echo "[FAIL] ZeRO ($NPROC GPUs)"

echo ""
echo "All profiling jobs completed. Results in $OUTDIR/"
echo "  Baseline snapshots: ${OUTDIR}/memory_snapshot_${SIZE}_cl${CTX}_${STAGE}_fp32_rank*.pickle"
echo "  ZeRO snapshots:    ${OUTDIR}/memory_snapshot_${SIZE}_cl${CTX}_${STAGE}_fp32_zero_rank*.pickle"
