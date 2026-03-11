import os
import csv
import time
import argparse

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from cs336_systems.collective_communication_utils import setup, cleanup
from cs336_systems.utils import mean, std


def all_reduce(rank: int, world_size: int, num_elements: int, backend: str,
               warm_up: int, num_iters: int, result_tensor: torch.Tensor):
    use_cuda = backend == 'nccl'
    if use_cuda:
        torch.cuda.set_device(rank)
        device = torch.device(f'cuda:{rank}')
    else:
        device = torch.device('cpu')

    setup(rank, world_size, backend)

    tensor = torch.randn(num_elements, device=device)

    # Warmup
    for _ in range(warm_up):
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        if use_cuda:
            torch.cuda.synchronize()
    dist.barrier()

    # Benchmark
    durations = []
    for _ in range(num_iters):
        start = time.perf_counter()
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        if use_cuda:
            torch.cuda.synchronize()
        dist.barrier()
        durations.append(time.perf_counter() - start)

    # Rank 0 writes results to shared tensor
    if rank == 0:
        size_bytes = tensor.element_size() * tensor.numel()
        # Ring all-reduce: each GPU sends & receives (world_size-1) chunks in each of 2 phases
        sent_bytes = size_bytes * 2 * (world_size - 1) / world_size
        dur_mean = mean(durations)
        dur_std = std(durations)
        bw_mean = sent_bytes / dur_mean / (1024 ** 2)  # MB/s
        bw_std = bw_mean * (dur_std / dur_mean) if dur_mean > 0 else 0.0
        result_tensor[0] = dur_mean * 1000  # convert to ms
        result_tensor[1] = dur_std * 1000
        result_tensor[2] = bw_mean
        result_tensor[3] = bw_std

    cleanup()


def run_benchmark(backend, world_size, num_elements, warm_up, num_iters):
    result_tensor = torch.zeros(4)  # [dur_mean, dur_std, bw_mean, bw_std]
    result_tensor.share_memory_()
    mp.spawn(
        fn=all_reduce,
        args=(world_size, num_elements, backend, warm_up, num_iters, result_tensor),
        nprocs=world_size,
        join=True,
    )
    return result_tensor.tolist()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--backend', type=str, default='gloo')
    parser.add_argument('--world_size', type=int, default=4)
    parser.add_argument('--num_elements', type=int, default=1024, help='number of float32 elements')
    parser.add_argument('--warm_up', type=int, default=5)
    parser.add_argument('--num_iters', type=int, default=5)
    parser.add_argument('--output_file', type=str, default='results/allreduce_benchmark/results.csv')
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)

    dur_mean, dur_std, bw_mean, bw_std = run_benchmark(
        args.backend, args.world_size, args.num_elements, args.warm_up, args.num_iters,
    )

    size_mb = args.num_elements * 4 / (1024 ** 2)  # float32 = 4 bytes
    print(f"backend={args.backend}, world_size={args.world_size}, "
          f"size={size_mb:.2f}MB, duration={dur_mean:.4f}±{dur_std:.4f}ms, "
          f"bandwidth={bw_mean:.2f}±{bw_std:.2f}MB/s")

    # Append to CSV
    write_header = not os.path.exists(args.output_file)
    with open(args.output_file, 'a', newline='') as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(['backend', 'world_size', 'num_elements', 'size_mb',
                             'duration_mean_ms', 'duration_std_ms', 'bandwidth_mean_MBps', 'bandwidth_std_MBps'])
        writer.writerow([args.backend, args.world_size, args.num_elements, f'{size_mb:.4f}',
                         f'{dur_mean:.4f}', f'{dur_std:.4f}', f'{bw_mean:.2f}', f'{bw_std:.2f}'])