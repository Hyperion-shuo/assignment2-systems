import torch
import time
import math
import numpy as np
from typing import Callable

def mean(lst: list[float]) -> float:
    return sum(lst) / len(lst) if lst else 0.0

def std(lst: list[float]) -> float:
    if len(lst) < 2:
        return 0.0
    mean_val = mean(lst)
    variance = sum((x - mean_val) ** 2 for x in lst) / (len(lst) - 1)
    return math.sqrt(variance)

# can also use pytorch benchmark
# https://docs.pytorch.org/tutorials/recipes/recipes/benchmark.html
def benchmark(description: str, run: Callable, num_warmups: int = 5, num_trials: int = 10):
    for _ in range(num_warmups):
        run()
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        
    times: list[float] = []
    for trial in range(num_trials):
        start_time = time.perf_counter()
        
        run()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            
        end_time = time.perf_counter()
        times.append((end_time - start_time) * 1000)
        
    mean_time = mean(times)
    std_time = std(times)
    return mean_time, std_time

def benchmark_split(description: str, run: Callable, num_warmups: int = 5, num_trials: int = 10):
    """Like benchmark, but run() should return a dict of {segment_name: time_ms}."""
    for _ in range(num_warmups):
        run()
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    all_times: dict[str, list[float]] = {}
    for _ in range(num_trials):
        segment_times = run()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        for k, v in segment_times.items():
            all_times.setdefault(k, []).append(v)
    
    return {k: (mean(v), std(v)) for k, v in all_times.items()}
