import argparse
import torch
import numpy as np
import csv
import math
import torch.cuda.nvtx as nvtx
import cs336_basics

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
    # "2.7B":   {"d_model": 2560, "d_ff": 10240, "num_layers": 32, "num_heads": 32},
}

@nvtx.range("scaled dot product attention")
def annotated_scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys    d_k"],
    V: Float[Tensor, " ... keys    d_v"],
    mask: Bool[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " ... queries d_v"]:
    """Scaled dot-product attention.

    This function implements Eq. 1 of the Transformer paper.

    Args:
        Q: Tensor of queries, may have any number of leading dimensions.
        K: Tensor of keys, sharing leading dimensions with Q.
        V: Tensor of values, sharding leading dimensions with Q and K.
        mask: An (optional) mask of shape (..., seq_len, seq_len).
            Attention scores for positions with a mask value of `False` should
            be masked out, i.e., not affect the softmaxed attention probabilities.

    Returns:
        torch.FloatTensor of shape (..., seq_len, value_dimension)
        with the output of running your scaled dot product attention
        implementation with the provided key, query, and value tensors.
    """

    d_k = K.shape[-1]
    with nvtx.range("computing attention scores"):
        attention_scores = einsum(Q, K, "... query d_k, ... key d_k -> ... query key") / math.sqrt(d_k)

    if mask is not None:
        attention_scores = torch.where(mask, attention_scores, float("-inf"))

    with nvtx.range("computing softmax"):
        attention_weights = softmax(attention_scores, dim=-1)  # Softmax over the key dimension

    with nvtx.range("final matmul"):
        values = einsum(attention_weights, V, "... query key, ... key d_v ->  ... query d_v")
    
    return values

cs336_basics.model.scaled_dot_product_attention = annotated_scaled_dot_product_attention

def run_lm(vocab_size, context_length, batch_size, device, steps, config):
    # Create model
    with nvtx.range("define_model"):
        model = BasicsTransformerLM(
            vocab_size=vocab_size,
            context_length=context_length,
            d_model=config["d_model"],
            num_layers=config["num_layers"],
            num_heads=config["num_heads"],
            d_ff=config["d_ff"],
            rope_theta=10000.0,
        ).to(device)
    
    with nvtx.range("define_optimizer"):
        optimizer = AdamW(model.parameters(), lr=1e-4)

    with nvtx.range("define_input"):
    # Pre-generate batch data on GPU (exclude data loading from timing)
        dataset = np.random.randint(
            0, vocab_size,
            size=(batch_size * context_length + 1,),
            dtype=np.int64,
        )
        x, y = get_batch(dataset, batch_size, context_length, device)
    
    for step in range(steps):
        if step > 10:
            # start profiling after 10 warmup iterations
            torch.cuda.cudart().cudaProfilerStart()
        
        nvtx.range_push(f"step_{step}")
        
        model.zero_grad()
        
        with nvtx.range("forward"):
            logits = model(x)
        
        with nvtx.range("loss"):
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)), y.view(-1)
            )
        
        with nvtx.range("backward"):
            loss.backward()
        
        with nvtx.range("optimizer_step"):
            optimizer.step()
        
        nvtx.range_pop()

#  uv run python -m cs336_systems.basic_lm_run
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab_size", type=int, default=10000)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--context_length", type=int, default=256)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--steps", type=int, default=20, help="Number of gradient steps")
    parser.add_argument(
        "--sizes", type=str, nargs="+",
        default=list(MODEL_CONFIGS.keys()),
        help="Model sizes to benchmark, e.g. --sizes small medium",
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

        run_lm(args.vocab_size, args.context_length, args.batch_size, args.device, args.steps, config)