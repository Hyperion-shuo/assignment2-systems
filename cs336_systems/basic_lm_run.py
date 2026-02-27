import argparse
import torch
import numpy as np
import csv
import math

from typing import Callable
from cs336_basics.model import BasicsTransformerLM
from cs336_basics.data import get_batch
from cs336_systems.utils import benchmark, benchmark_split


MODEL_CONFIGS = {
    "small":  {"d_model": 768,  "d_ff": 3072,  "num_layers": 12, "num_heads": 12},
    "medium": {"d_model": 1024, "d_ff": 4096,  "num_layers": 24, "num_heads": 16},
    "large":  {"d_model": 1280, "d_ff": 5120,  "num_layers": 36, "num_heads": 20},
    "xl":     {"d_model": 1600, "d_ff": 6400,  "num_layers": 48, "num_heads": 25},
    # "2.7B":   {"d_model": 2560, "d_ff": 10240, "num_layers": 32, "num_heads": 32},
}


def lm_forward_operation(
    model: BasicsTransformerLM,
    x: torch.Tensor,
    y: torch.Tensor,
) -> Callable:
    def run():
        # with torch.no_grad():
        logits = model(x)
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)), y.view(-1)
        )
        return loss.item()

    return run


def lm_backward_operation(
    model: BasicsTransformerLM,
    x: torch.Tensor,
    y: torch.Tensor,
) -> Callable:
    start_fw = torch.cuda.Event(enable_timing=True)
    end_fw = torch.cuda.Event(enable_timing=True)
    end_bw = torch.cuda.Event(enable_timing=True)

    def run():
        model.zero_grad()
        start_fw.record()
        logits = model(x)
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)), y.view(-1)
        )
        end_fw.record()
        loss.backward()
        end_bw.record()
        torch.cuda.synchronize()
        return {
            "fw": start_fw.elapsed_time(end_fw),
            "bw": end_fw.elapsed_time(end_bw),
            "fwbw": start_fw.elapsed_time(end_bw),
        }

    return run

#  uv run python -m cs336_systems.basic_lm_run
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab_size", type=int, default=10000)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--context_length", type=int, default=256)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--warmups", type=int, default=5, help="Number of warmup runs for benchmarking")
    parser.add_argument(
        "--sizes", type=str, nargs="+",
        default=list(MODEL_CONFIGS.keys()),
        help="Model sizes to benchmark, e.g. --sizes small medium",
    )
    parser.add_argument(
        "--output", type=str, default="results/basic_lm_benchmark_results",
        help="Output CSV file path",
    )
    args = parser.parse_args()

    results = []

    for size_name in args.sizes:
        config = MODEL_CONFIGS[size_name]
        print(f"\n{'=' * 60}")
        print(f"Benchmarking model size: {size_name}")
        print(f"  d_model={config['d_model']}, d_ff={config['d_ff']}, "
              f"num_layers={config['num_layers']}, num_heads={config['num_heads']}")
        print(f"{'=' * 60}")

        # Create model
        model = BasicsTransformerLM(
            vocab_size=args.vocab_size,
            context_length=args.context_length,
            d_model=config["d_model"],
            num_layers=config["num_layers"],
            num_heads=config["num_heads"],
            d_ff=config["d_ff"],
            rope_theta=10000.0,
        ).to(args.device)

        # Pre-generate batch data on GPU (exclude data loading from timing)
        dataset = np.random.randint(
            0, args.vocab_size,
            size=(args.batch_size * args.context_length + 1,),
            dtype=np.int64,
        )
        x, y = get_batch(dataset, args.batch_size, args.context_length, args.device)

        # Benchmark forward
        fw_mean, fw_std = benchmark(
            f"{size_name}_forward",
            lm_forward_operation(model, x, y),
            args.warmups
        )
        print(f"  Forward:          {fw_mean:.2f} ± {fw_std:.2f} ms")

        # Benchmark backward with CUDA event split timing
        split = benchmark_split(
            f"{size_name}_backward",
            lm_backward_operation(model, x, y),
            args.warmups
        )
        bw_only_mean, bw_only_std = split["bw"]
        bw_mean, bw_std = split["fwbw"]
        print(f"  Backward:         {bw_only_mean:.2f} ± {bw_only_std:.2f} ms")
        print(f"  Forward+Backward: {bw_mean:.2f} ± {bw_std:.2f} ms")

        results.append({
            "size": size_name,
            "d_model": config["d_model"],
            "d_ff": config["d_ff"],
            "num_layers": config["num_layers"],
            "num_heads": config["num_heads"],
            "fw_mean_ms": round(fw_mean, 2),
            "fw_std_ms": round(fw_std, 2),
            "bw_only_mean_ms": round(bw_only_mean, 2),
            "bw_only_std_ms": round(bw_only_std, 2),
            "fwbw_mean_ms": round(bw_mean, 2),
            "fwbw_std_ms": round(bw_std, 2),
        })

        # Free GPU memory before next model
        del model, x, y
        torch.cuda.empty_cache()

    # Save results to CSV
    fieldnames = [
        "size", "d_model", "d_ff", "num_layers", "num_heads",
        "fw_mean_ms", "fw_std_ms", "bw_only_mean_ms", "bw_only_std_ms",
        "fwbw_mean_ms", "fwbw_std_ms", 
    ]
    # output name add warmup iter
    output_name = f"{args.output}_warmups{args.warmups}.csv"
    with open(output_name, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"\nResults saved to {output_name}")

    # Print summary table
    print(f"\n{'Size':<10} {'FW Mean':<12} {'FW Std':<12} "
          f"{'BW Mean':<12} {'BW Std':<12}"
          f"{'FW+BW Mean':<12} {'FW+BW Std':<12} ")
    print("-" * 82)
    for r in results:
        print(f"{r['size']:<10} {r['fw_mean_ms']:<12} {r['fw_std_ms']:<12} "
              f"{r['bw_only_mean_ms']:<12} {r['bw_only_std_ms']:<12}"
              f"{r['fwbw_mean_ms']:<12} {r['fwbw_std_ms']:<12} ")