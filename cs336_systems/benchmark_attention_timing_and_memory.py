import argparse
from os import times
from tracemalloc import start
import torch
import time
import math
import csv
import traceback

from torch import Tensor
from jaxtyping import Float, Bool
from einops import einsum
from cs336_basics.nn_utils import softmax


def compute_mean_std(data: list[float]) -> tuple[float, float]:
    """Compute mean and sample standard deviation of a list of floats."""
    n = len(data)
    mean = sum(data) / n
    std = (sum((x - mean) ** 2 for x in data) / (n - 1)) ** 0.5 if n > 1 else 0.0
    return mean, std


def scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys    d_k"],
    V: Float[Tensor, " ... keys    d_v"],
    mask: Bool[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " ... queries d_v"]:
    d_k = K.shape[-1]
    attention_scores = einsum(Q, K, "... query d_k, ... key d_k -> ... query key") / math.sqrt(d_k)

    if mask is not None:
        attention_scores = torch.where(mask, attention_scores, float("-inf"))

    attention_weights = softmax(attention_scores, dim=-1)

    return einsum(attention_weights, V, "... query key, ... key d_v -> ... query d_v")


def benchmark_forward(attn_func, Q, K, V, mask, num_warmups=5, num_trials=100):
    """Benchmark forward pass timing (ms). Returns (mean, std)."""
    for _ in range(num_warmups):
        with torch.no_grad():
            _ = attn_func(Q, K, V, mask)
        torch.cuda.synchronize()

    times_list = []
    for _ in range(num_trials):
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.no_grad():
            _ = attn_func(Q, K, V, mask)
        torch.cuda.synchronize()
        end = time.perf_counter()
        times_list.append((end - start) * 1000)

    return compute_mean_std(times_list)


def benchmark_memory(attn_func, Q, K, V, mask, num_warmups=5, num_trials=100):
    """Benchmark memory saved for backward (MB).
    Returns (no_grad_mean, with_grad_mean, saved_mean, no_grad_std, with_grad_std, saved_std).
    """
    for _ in range(num_warmups):
        with torch.no_grad():
            output_no_grad = attn_func(Q, K, V, mask)
        torch.cuda.synchronize()
        del output_no_grad
        output = attn_func(Q, K, V, mask)
        del output

    mem_no_grad_list = []
    mem_with_grad_list = []
    mem_saved_list = []
    for _ in range(num_trials):
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        with torch.no_grad():
            output_no_grad = attn_func(Q, K, V, mask)
        torch.cuda.synchronize()
        mem_no_grad = torch.cuda.memory_allocated()

        del output_no_grad
        torch.cuda.empty_cache()

        torch.cuda.synchronize()
        output = attn_func(Q, K, V, mask)
        torch.cuda.synchronize()
        mem_with_grad = torch.cuda.memory_allocated()

        mem_saved = mem_with_grad - mem_no_grad

        mem_no_grad_list.append(mem_no_grad / (1024 ** 2))
        mem_with_grad_list.append(mem_with_grad / (1024 ** 2))
        mem_saved_list.append(mem_saved / (1024 ** 2))

        del output
        torch.cuda.empty_cache()

    no_grad_mean, no_grad_std = compute_mean_std(mem_no_grad_list)
    with_grad_mean, with_grad_std = compute_mean_std(mem_with_grad_list)
    saved_mean, saved_std = compute_mean_std(mem_saved_list)

    return no_grad_mean, with_grad_mean, saved_mean, no_grad_std, with_grad_std, saved_std


def benchmark_backward(attn_func, Q, K, V, mask, num_warmups=5, num_trials=100):
    """Benchmark backward pass timing (ms).
    Returns (fw_mean, fw_std, bw_mean, bw_std).
    """
    for _ in range(num_warmups):
        Q.grad = None
        K.grad = None
        V.grad = None
        output = attn_func(Q, K, V, mask)
        loss = output.sum()
        loss.backward()
        torch.cuda.synchronize()

    times_fw = []
    times_bw = []
    for _ in range(num_trials):
        Q.grad = None
        K.grad = None
        V.grad = None

        torch.cuda.synchronize()
        start_fw = time.perf_counter()
        output = attn_func(Q, K, V, mask)
        loss = output.sum()
        torch.cuda.synchronize()

        torch.cuda.synchronize()
        start_bw = time.perf_counter()
        loss.backward()
        torch.cuda.synchronize()
        end = time.perf_counter()
        times_fw.append((start_bw - start_fw) * 1000)
        times_bw.append((end - start_bw) * 1000)

    fw_mean, fw_std = compute_mean_std(times_fw)
    bw_mean, bw_std = compute_mean_std(times_bw)
    return fw_mean, fw_std, bw_mean, bw_std


# uv run python -m cs336_systems.benchmark_attention_timing_and_memory
if __name__ == "__main__":
    # torch/_inductor/compile_fx.py:194: UserWarning:
    # TensorFloat32 tensor cores for float32 matrix multiplication available but not enabled. 
    # Consider setting `torch.set_float32_matmul_precision('high')` for better performance.
    torch.set_float32_matmul_precision('high')
    
    parser = argparse.ArgumentParser(description="Benchmark scaled dot-product attention")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num_warmups", type=int, default=10) # compile version may need more warmups
    parser.add_argument("--num_trials", type=int, default=100)
    parser.add_argument("--torch_compile", action="store_true", help="Whether to torch.compile the attention function")
    parser.add_argument("--output", type=str, default="results/attention/attention_benchmark_results.csv")
    args = parser.parse_args()

    device = args.device
    batch_size = args.batch_size
    embedding_dims = [16, 32, 64, 128]
    seq_lengths = [256, 1024, 4096, 8192, 16384]
    
    attn_func = scaled_dot_product_attention
    if args.torch_compile:
        # 使用 mode='reduce-overhead' 对于小 batch 通常更快，
        # 或默认 mode='default'
        attn_func = torch.compile(scaled_dot_product_attention)

    results = []

    for d_k in embedding_dims:
        for seq_len in seq_lengths:
            print(f"\n{'=' * 60}")
            print(f"d_k={d_k}, seq_len={seq_len}, batch_size={batch_size}")
            print(f"{'=' * 60}")

            try:
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()

                Q = torch.randn(batch_size, seq_len, d_k, device=device, requires_grad=True)
                K = torch.randn(batch_size, seq_len, d_k, device=device, requires_grad=True)
                V = torch.randn(batch_size, seq_len, d_k, device=device, requires_grad=True)
                mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device))

                # Forward benchmark
                fw_mean, fw_std = benchmark_forward(
                    attn_func, Q, K, V, mask, num_warmups=args.num_warmups, num_trials=args.num_trials
                )
                print(f"  Forward:  {fw_mean:.3f} ± {fw_std:.3f} ms")

                # Backward benchmark
                fw_grad_mean, fw_grad_std, bw_mean, bw_std = benchmark_backward(
                    attn_func, Q, K, V, mask, num_warmups=args.num_warmups, num_trials=args.num_trials
                )
                print(f"  FW(grad): {fw_grad_mean:.3f} ± {fw_grad_std:.3f} ms")
                print(f"  Backward: {bw_mean:.3f} ± {bw_std:.3f} ms")

                # Memory benchmark
                no_grad_mem, with_grad_mem, saved_mem, no_grad_mem_std, with_grad_mem_std, saved_mem_std = benchmark_memory(
                    attn_func, Q, K, V, mask, num_warmups=args.num_warmups, num_trials=args.num_trials
                )
                print(f"  Mem no_grad:  {no_grad_mem:.2f} ± {no_grad_mem_std:.2f} MB")
                print(f"  Mem w/ grad:  {with_grad_mem:.2f} ± {with_grad_mem_std:.2f} MB")
                print(f"  Mem saved:    {saved_mem:.2f} ± {saved_mem_std:.2f} MB")

                results.append({
                    "d_k": d_k,
                    "seq_len": seq_len,
                    "fw_mean_ms": round(fw_mean, 3),
                    "fw_std_ms": round(fw_std, 3),
                    "fw_grad_mean_ms": round(fw_grad_mean, 3),
                    "fw_grad_std_ms": round(fw_grad_std, 3),
                    "bw_mean_ms": round(bw_mean, 3),
                    "bw_std_ms": round(bw_std, 3),
                    "mem_no_grad_MB": round(no_grad_mem, 2),
                    "mem_with_grad_MB": round(with_grad_mem, 2),
                    "mem_saved_MB": round(saved_mem, 2),
                    "OOM": False,
                })

            except torch.cuda.OutOfMemoryError:
                print(f"  *** OOM at d_k={d_k}, seq_len={seq_len} ***")
                results.append({
                    "d_k": d_k,
                    "seq_len": seq_len,
                    "fw_mean_ms": None,
                    "fw_std_ms": None,
                    "fw_grad_mean_ms": None,
                    "fw_grad_std_ms": None,
                    "bw_mean_ms": None,
                    "bw_std_ms": None,
                    "mem_no_grad_MB": None,
                    "mem_with_grad_MB": None,
                    "mem_saved_MB": None,
                    "OOM": True,
                })
                torch.cuda.empty_cache()

            finally:
                for var_name in ['Q', 'K', 'V', 'mask']:
                    if var_name in dir():
                        exec(f"del {var_name}")
                torch.cuda.empty_cache()

    # Save results to CSV
    fieldnames = [
        "d_k", "seq_len",
        "fw_mean_ms", "fw_std_ms",
        "fw_grad_mean_ms", "fw_grad_std_ms",
        "bw_mean_ms", "bw_std_ms",
        "mem_no_grad_MB", "mem_with_grad_MB", "mem_saved_MB",
        "OOM",
    ]
    if args.torch_compile:
        output_path = args.output.replace(".csv", "_torch_compile.csv")
    else:
        output_path = args.output
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"\nResults saved to {output_path}")
    # Print summary table
    header = f"{'d_k':>6} {'seq_len':>8} {'FW(ms)':>12} {'FWgrad(ms)':>12} {'BW(ms)':>12} {'NoGrad(MB)':>12} {'W/Grad(MB)':>12} {'Saved(MB)':>12} {'OOM':>6}"
    print(f"\n{header}")
    print("-" * len(header))
    for r in results:
        fw = f"{r['fw_mean_ms']:.3f}" if r['fw_mean_ms'] is not None else "N/A"
        fwg = f"{r['fw_grad_mean_ms']:.3f}" if r['fw_grad_mean_ms'] is not None else "N/A"
        bw = f"{r['bw_mean_ms']:.3f}" if r['bw_mean_ms'] is not None else "N/A"
        ng = f"{r['mem_no_grad_MB']:.2f}" if r['mem_no_grad_MB'] is not None else "N/A"
        wg = f"{r['mem_with_grad_MB']:.2f}" if r['mem_with_grad_MB'] is not None else "N/A"
        sv = f"{r['mem_saved_MB']:.2f}" if r['mem_saved_MB'] is not None else "N/A"
        print(f"{r['d_k']:>6} {r['seq_len']:>8} {fw:>12} {fwg:>12} {bw:>12} {ng:>12} {wg:>12} {sv:>12} {str(r['OOM']):>6}")