#!/bin/bash
set -e

OUTPUT_FILE="results/allreduce_benchmark/results.csv"
rm -f "$OUTPUT_FILE"
mkdir -p "$(dirname "$OUTPUT_FILE")"

# num_elements for float32 (4 bytes each):
#   262144    = 1 MB
#   2621440   = 10 MB
#   26214400  = 100 MB
#   268435456 = 1 GB
BACKENDS="gloo nccl"
WORLD_SIZES="2 4"
NUM_ELEMENTS="262144 2621440 26214400 268435456"

for backend in $BACKENDS; do
  for ws in $WORLD_SIZES; do
    for ne in $NUM_ELEMENTS; do
      echo ">>> backend=$backend, world_size=$ws, num_elements=$ne"
      uv run python -m cs336_systems.benchmark_all_reduce \
        --backend "$backend" \
        --world_size "$ws" \
        --num_elements "$ne" \
        --warm_up 5 \
        --num_iters 10 \
        --output_file "$OUTPUT_FILE" \
        || echo "[FAIL] backend=$backend, world_size=$ws, num_elements=$ne"
    done
  done
done

echo "All benchmarks completed. Results saved to $OUTPUT_FILE"

# bash my_scripts/benchmark_all_reduce.sh to run
