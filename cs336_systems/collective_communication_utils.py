import os
import torch as th
import torch.nn as nn
import torch.distributed as dist

def setup(rank: int, world_size: int, backend: str = 'gloo'):
    # Specify where master lives (rank 0), used to coordinate (actual data goes through NCCL)
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "15623"
    dist.init_process_group(backend, rank=rank, world_size=world_size)


def cleanup():
    th.distributed.destroy_process_group()

# Backward hooks
# using register_post_accumulate_grad_hook
# Asynchronous communication
# using async_op=True to return handles and use handles.wait to get results
class ddp_wrapper(nn.Module):
    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module
        
        # Broadcast parameters from rank 0 to ensure consistent initialization across ranks
        for param in module.parameters():
            dist.broadcast(param.data, src=0)
            
        # register hooks for all parameters that require gradients
        hook_fn = self.make_ddp_hook_fn()
        self.hook_handles = []
        for name, param in module.named_parameters():
            if param.requires_grad:
                hook_handle= param.register_post_accumulate_grad_hook(hook_fn)
                self.hook_handles.append(hook_handle)
                # dist.broadcast(param.data, src=0)
        self.pending_handles = []
                
    # 闭包，torch 规定hook只能传入一个参数
    # 通过闭包实现向make_ddp_hook_fn传入多个参数达到向hook传入多个参数的目的
    def make_ddp_hook_fn(self):
        def hook(param: th.Tensor):
            # perform all reduce on grad and update param.grad with the result
            if param.grad is not None:
                handle = dist.all_reduce(param.grad, op=dist.ReduceOp.AVG, async_op=True)
                self.pending_handles.append(handle)
        return hook
    
    def forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)
    
    def finish_gradient_synchronization(self):
        # Handles are ordered from last layer to first layer, since backward
        # propagation computes gradients in that order, triggering hooks accordingly.
        # Thus we wait on the earliest-issued handles first, which are most likely
        # already complete, minimizing blocking time.
        for handle in self.pending_handles:
            handle.wait()
        self.pending_handles.clear()

# params_with_grad = [p for p in transformer.parameters() if p.grad is not None]
# grads_list = [p.grad.data for p in params_with_grad]
# flat_grads = th._utils._flatten_dense_tensors(grads_list)
# dist.all_reduce(flat_grads, op=dist.ReduceOp.SUM)
# flat_grads /= world_size
# unflattened_grads = th._utils._unflatten_dense_tensors(flat_grads, grads_list)

class ddp_bucket_wrapper(nn.Module):
    def __init__(self, module: nn.Module, bucket_size_mb: float):
        super().__init__()
        self.module = module
        self.bucket_size_mb = bucket_size_mb
        self.buckets = []
        self.param_to_bucket = {}
        
        # Broadcast parameters from rank 0 to ensure consistent initialization across ranks
        for param in module.parameters():
            dist.broadcast(param.data, src=0)
        
        self.build_buckets()
        # register hooks for all parameters that require gradients
        hook_fn = self.make_ddp_hook_fn()
        for bucket in self.buckets:
            for param in bucket["params"]:
                if param.requires_grad:
                    hook_handle= param.register_post_accumulate_grad_hook(hook_fn)
                     
    # 闭包，torch 规定hook只能传入一个参数
    # 通过闭包实现向make_ddp_hook_fn传入多个参数达到向hook传入多个参数的目的
    def make_ddp_hook_fn(self):
        def hook(param: th.Tensor):
            # perform all reduce on grad and update param.grad with the result
            if param.grad is not None:
                bucket = self.param_to_bucket[param]
                bucket["ready_count"] += 1
                
                if bucket["ready_count"] == len(bucket["params"]):
                    # all params in the bucket are ready, perform all reduce
                    grads_list = [p.grad for p in bucket["params"]]
                    flat_grads = th._utils._flatten_dense_tensors(grads_list)
                    bucket["handle"] = dist.all_reduce(flat_grads, op=dist.ReduceOp.AVG, async_op=True)
                    bucket["flat_buffer"] = flat_grads
        return hook
    
    def build_buckets(self):
        param_with_grad = reversed([p for p in self.module.parameters() if p.requires_grad])
        current_bucket = []
        current_bucket_size = 0
        for param in param_with_grad:
            param_size_mb = float(param.numel() * param.element_size() / (1024 * 1024))
            if param_size_mb > self.bucket_size_mb:
                if len(current_bucket) > 0:
                    self._create_bucket(current_bucket)
                    current_bucket = []
                    current_bucket_size = 0
                self._create_bucket([param])
                continue
            
            if current_bucket_size + param_size_mb > self.bucket_size_mb:
                self._create_bucket(current_bucket)
                current_bucket = []
                current_bucket_size = 0
            
            current_bucket.append(param)
            current_bucket_size += param_size_mb
            
        if len(current_bucket) > 0:
            self._create_bucket(current_bucket)
    
    def _create_bucket(self, param_list: list[nn.Parameter]):
        bucket_id = len(self.buckets)
        bucket_info = {
            "bucket_id": bucket_id,
            "params": param_list,
            "handle": None,
            "ready_count": 0,
            "flat_buffer": None,
        }
        self.buckets.append(bucket_info)
        
        for p in param_list:
            self.param_to_bucket[p] = bucket_info
    
    def forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)
    
    def finish_gradient_synchronization(self):
        for bucket in self.buckets:
            if bucket["handle"] is not None:
                bucket["handle"].wait()
                flat_grads = bucket["flat_buffer"]
                params = bucket["params"]
                unflattened_grads = th._utils._unflatten_dense_tensors(flat_grads, [p.grad for p in params])
                # Unflatten back into individual gradients
                for p, unflat_g in zip(params, unflattened_grads):
                    p.grad.copy_(unflat_g)
                
                # reset
                bucket["handle"] = None
                bucket["ready_count"] = 0
                bucket["flat_buffer"] = None
            else:
                bucket["ready_count"] = 0