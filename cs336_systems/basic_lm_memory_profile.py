import argparse
import torch
import numpy as np
import csv
import math
import cs336_basics
from contextlib import nullcontext

from typing import Callable
from torch import Tensor
from cs336_basics.model import BasicsTransformerLM
from cs336_basics.data import get_batch
from cs336_basics.optimizer import AdamW
from cs336_basics.nn_utils import softmax
from jaxtyping import Float, Bool
from einops import einsum


MODEL_CONFIGS = {
    "small":  {"d_model": 768,  "d_ff": 3072,  "num_layers": 12, "num_heads": 12},
    "medium": {"d_model": 1024, "d_ff": 4096,  "num_layers": 24, "num_heads": 16},
    "large":  {"d_model": 1280, "d_ff": 5120,  "num_layers": 36, "num_heads": 20},
    "xl":     {"d_model": 1600, "d_ff": 6400,  "num_layers": 48, "num_heads": 25},
    "2.7B":   {"d_model": 2560, "d_ff": 10240, "num_layers": 32, "num_heads": 32},
}

def run_lm(vocab_size, context_length, batch_size, device, steps, config, save_path, mixed_precision=False, stage='fw'):
    # stage: 'fw' = forward only (no grad, no optimizer)
    #        'fw_with_grad' = forward only (with grad & optimizer allocated, but no backward/step)
    #        'full_train' = full training loop (forward + backward + optimizer.step)
    # Create model
    model = BasicsTransformerLM(
        vocab_size=vocab_size,
        context_length=context_length,
        d_model=config["d_model"],
        num_layers=config["num_layers"],
        num_heads=config["num_heads"],
        d_ff=config["d_ff"],
        rope_theta=10000.0,
    ).to(device)
    
    need_optimizer = stage in ('fw_with_grad', 'full_train')
    if need_optimizer:
        optimizer = AdamW(model.parameters(), lr=1e-4)

    # Pre-generate batch data on GPU (exclude data loading from timing)
    dataset = np.random.randint(
        0, vocab_size,
        size=(batch_size * context_length + 1,),
        dtype=np.int64,
    )
    x, y = get_batch(dataset, batch_size, context_length, device)
    
    ctx = torch.autocast(device_type='cuda', dtype=torch.bfloat16) if mixed_precision else nullcontext()

    # Warm-up runs (no memory recording)
    warmup_steps = min(5, steps)
    for step in range(warmup_steps):
        with ctx:
            if stage == 'full_train':
                model.zero_grad()
                logits = model(x)
                loss = torch.nn.functional.cross_entropy(
                    logits.view(-1, logits.size(-1)), y.view(-1)
                )
                loss.backward()
                optimizer.step()
            elif stage == 'fw_with_grad':
                model.zero_grad()
                logits = model(x)
                loss = torch.nn.functional.cross_entropy(
                    logits.view(-1, logits.size(-1)), y.view(-1)
                )
            else:  # fw
                with torch.no_grad():
                    logits = model(x)
                    loss = torch.nn.functional.cross_entropy(
                        logits.view(-1, logits.size(-1)), y.view(-1)
                    )
    
    # Clear cache after warmup to get a clean snapshot
    torch.cuda.empty_cache()

    # Start recording memory history
    torch.cuda.memory._record_memory_history(max_entries=1000000)

    # Profiled runs
    for step in range(warmup_steps, steps):
        with ctx:
            if stage == 'full_train':
                model.zero_grad()
                logits = model(x)
                loss = torch.nn.functional.cross_entropy(
                    logits.view(-1, logits.size(-1)), y.view(-1)
                )
                loss.backward()
                optimizer.step()
            elif stage == 'fw_with_grad':
                model.zero_grad()
                logits = model(x)
                loss = torch.nn.functional.cross_entropy(
                    logits.view(-1, logits.size(-1)), y.view(-1)
                )
            else:  # fw
                with torch.no_grad():
                    logits = model(x)
                    loss = torch.nn.functional.cross_entropy(
                        logits.view(-1, logits.size(-1)), y.view(-1)
                    )

    torch.cuda.memory._dump_snapshot(save_path)
    torch.cuda.memory._record_memory_history(enabled=None)
    
        

#  uv run python -m cs336_systems.basic_lm_run
# translate piclkle to html:
# python -m torch.cuda._memory_viz trace_plot memory_snapshot_large_cl512_fw.pickle -o memory_snapshot_large_cl512_fw.html
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab_size", type=int, default=10000)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--context_length", type=int, default=256)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--steps", type=int, default=6, help="Number of gradient steps")
    parser.add_argument("--mixed_precision", action="store_true", help="Whether to use mixed precision (bf16) for benchmarking")
    parser.add_argument("--save_path", type=str, default="results/torch_memory_profiles/memory_snapshot", help="Path to save memory snapshot")
    parser.add_argument("--stage", type=str, default="fw",
        choices=["fw", "fw_with_grad", "full_train"],
        help="fw: forward only (no grad); fw_with_grad: forward with grad & optimizer; full_train: full training loop")
    parser.add_argument(
        "--size_name", type=str, default="large",
        choices=MODEL_CONFIGS.keys(),
        help="Model size to run, e.g. --size_name large",
    )
    
    args = parser.parse_args()

    size_name = args.size_name
    config = MODEL_CONFIGS[size_name]
    print(f"\n{'=' * 60}")
    print(f"Benchmarking model size: {size_name}")
    print(f"  d_model={config['d_model']}, d_ff={config['d_ff']}, "
            f"num_layers={config['num_layers']}, num_heads={config['num_heads']}")
    print(f"{'=' * 60}")
    
    stage = args.stage
    precision = "bf16" if args.mixed_precision else "fp32"
    save_path = f"{args.save_path}_{size_name}_cl{args.context_length}_{stage}_{precision}.pickle"

    run_lm(args.vocab_size, args.context_length, args.batch_size, args.device, args.steps, config, save_path, 
           mixed_precision=args.mixed_precision, stage=stage)
    