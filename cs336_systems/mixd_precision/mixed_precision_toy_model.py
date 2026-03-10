import torch
import torch.nn as nn

class ToyModel(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.fc1 = nn.Linear(in_features, 10, bias=False)
        self.ln = nn.LayerNorm(10)
        self.fc2 = nn.Linear(10, out_features, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x):
        print(f"x.dtype: {x.dtype}")
        x = self.relu(self.fc1(x))
        print(f"After fc1 and relu, x.dtype: {x.dtype}")
        x = self.ln(x)
        print(f"After LayerNorm, x.dtype: {x.dtype}")
        x = self.fc2(x)
        print(f"After fc2, x.dtype: {x.dtype}")
        return x
    
    
def main():
    model = ToyModel(1000, 1000).cuda()
    x = torch.randn(1, 1000, dtype=torch.float32).cuda()
    # try fp16 and bf16 here
    # cast_dtype = torch.float16
    cast_dtype = torch.bfloat16
    with torch.autocast(device_type='cuda', dtype=cast_dtype):
        # model's dtype
        for name, param in model.named_parameters():
            print(f"{name} dtype: {param.dtype}")
        y = model(x)
        # loss
        loss = y.sum()
        print(f"Loss: {loss.item()}, loss.dtype: {loss.dtype}")
        # backward
        loss.backward()
        # gradients dtype
        for name, param in model.named_parameters():
            print(f"{name} grad dtype: {param.grad.dtype}")

if __name__ == "__main__":
    main()