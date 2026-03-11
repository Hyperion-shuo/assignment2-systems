from copy import deepcopy
import argparse

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim

from cs336_systems.collective_communication_utils import setup, cleanup


def my_toy_model():
    return nn.Sequential(
        nn.Linear(16, 32),
        nn.ReLU(),
        nn.Linear(32, 16),
    )

    
def test_naive_ddp_train(rank, world_size, model_class, dataset, global_batch_size, epochs, backend):
    setup(rank, world_size, backend)

    if torch.cuda.is_available() and backend == 'nccl':
        device = torch.device(f'cuda:{rank}')
    else:
        device = torch.device('cpu')
    if torch.cuda.is_available():
        torch.cuda.set_device(rank)

    # Create a toy model and move it to the proper device.
    model = model_class().to(device)

    # broadcast model parameters from rank 0 to all other ranks to ensure they start with the same weights
    for param in model.parameters():
        dist.broadcast(param.data, src=0)
        
    none_ddp_model = deepcopy(model)  # Each rank has its own model instance

    # Define a simple loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    none_ddp_model_optimizer = optim.SGD(none_ddp_model.parameters(), lr=0.01)
    local_batch_size = global_batch_size // world_size

    # Training loop
    for epoch in range(epochs):
        # local rank batch
        ids = torch.arange(rank * local_batch_size, (rank + 1) * local_batch_size) + epoch * global_batch_size
        batch = dataset[ids]
        batch = batch.to(device)
        optimizer.zero_grad()
        outputs = model(batch)
        loss = criterion(outputs, batch)  # Using input as target for simplicity
        loss.backward()
        
        # all reduce gradients across all ranks
        for param in model.parameters():
            dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
            param.grad.data /= world_size  # Average the gradients
        optimizer.step()
        
    if torch.cuda.is_available() and backend == 'nccl':
        torch.cuda.synchronize() 
    dist.barrier()  # Ensure all ranks have finished training before validation
        
    # run local model on the same global batch
    if rank == 0:
        for epoch in range(epochs):
            ids = torch.arange(epoch * global_batch_size, (epoch + 1) * global_batch_size)
            batch = dataset[ids]
            batch = batch.to(device)
        
            none_ddp_model_optimizer.zero_grad()
            local_outputs = none_ddp_model(batch)
            local_loss = criterion(local_outputs, batch)
            local_loss.backward()
            none_ddp_model_optimizer.step()

        if torch.cuda.is_available() and backend == 'nccl':
            torch.cuda.synchronize() 
        # dist.barrier()  # Ensure all ranks have finished training before validation
    
        # compare named param and print name and dim
        for (name, param), (none_ddp_name, none_ddp_param) in zip(model.named_parameters(), none_ddp_model.named_parameters()):
            assert torch.allclose(param.data, none_ddp_param.data, atol=1e-4), f"Rank {rank}: Model parameters diverged from non-DDP model after training"
            print(f"Rank {rank}: {name} ({param.shape}) matches {none_ddp_name} ({none_ddp_param.shape})")

    dist.barrier()
    cleanup()


def run(model_class, global_batch_size=16, epochs=4, backend='nccl', world_size=4):
    dataset = torch.randn(global_batch_size * epochs, 16)
    
    mp.spawn(
        test_naive_ddp_train,
        args=(world_size, model_class, dataset, global_batch_size, epochs, backend,),
        nprocs=world_size,
        join=True,
    )


# uv run python -m cs336_systems.naive_ddp_train
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Naive DDP Training")
    parser.add_argument("--global_batch_size", type=int, default=16, help="Total batch size across all processes")
    parser.add_argument("--epochs", type=int, default=4, help="Number of training epochs")
    parser.add_argument("--world_size", type=int, default=4, help="Number of processes for distributed training")
    parser.add_argument("--backend", type=str, default="nccl", choices=["gloo", "nccl"], help="Distributed backend to use")
    args = parser.parse_args()
    
    global_batch_size = args.global_batch_size
    epochs = args.epochs
    world_size = args.world_size
    backend = args.backend
    assert args.global_batch_size % world_size == 0, "Global batch size must be divisible by world size"
    
    run(my_toy_model, global_batch_size=global_batch_size, epochs=epochs, backend=backend, world_size=world_size)