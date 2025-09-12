import torch

class ToyModel(torch.nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.fc1 = torch.nn.Linear(in_features, 10, bias=False)
        self.ln = torch.nn.LayerNorm(10)
        self.fc2 = torch.nn.Linear(10, out_features, bias=False)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x)) # fp16 cause linear
        print(f"type_relu = {x.dtype}")
        x = self.ln(x) # fp32 cause
        print(f"type_ln = {x.dtype}")
        x = self.fc2(x)
        print(f"type_fc2 = {x.dtype}")
        return x

batch = 30
in_features = 10
out_features = 20
m = ToyModel(in_features, out_features).cuda()
x = torch.zeros([batch, in_features], device='cuda')

print("Default")
m.forward(x)

print("Autocast f16")
# dtype = torch.float16
dtype = torch.bfloat16
x = torch.zeros([batch, in_features], dtype=dtype, device='cuda')
with torch.autocast('cuda', dtype=dtype):
    output = m(x)
    print(f"output.dtype = {output.dtype}")
