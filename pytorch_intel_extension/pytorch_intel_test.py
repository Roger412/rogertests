import torch
import torch.nn as nn
import intel_extension_for_pytorch as ipex

# Define a basic model
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear = nn.Linear(16, 8)

    def forward(self, x):
        return self.linear(x)

model = Model()
model.eval()  # Set the model to evaluation mode for inference
dtype = torch.float32  # Can change to bfloat16, etc.

# Create dummy input
data = torch.randn(1, 16, dtype=dtype)

##### Run on GPU ######
model = model.to('xpu')
data = data.to('xpu')

# Optimize for XPU with IPEX
model = ipex.optimize(model, dtype=dtype)

# Run inference with profiling
with torch.no_grad(), torch.xpu.amp.autocast(enabled=True, dtype=dtype, cache_enabled=False):
    model = torch.jit.trace(model, data)
    model = torch.jit.freeze(model)
    output = model(data)

print("✅ Inference done. Output:")
print(output)
