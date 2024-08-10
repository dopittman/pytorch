import torch

# Running tensors and pytorch objects on the GPUs (and making faster computations)

### Getting a GPU
#### Use Google Colab/Colab Pro to run commands on their machine

# Sets the device to "cuda" if it is available, otherwise cpu
device = "cuda" if torch.cuda.is_available() else "cpu"

# Putting tensors and modals on the GPU

# Create a tensor  (default on the CPU)
tensor = torch.tensor([1,2,3,])
print(tensor, tensor.device)

# Moves tensor "to" GPU
tensor_on_gpu = tensor.to(device)
print(tensor_on_gpu)

### Numpy only works with CPU, not the GPU
### If tensor is on GPU we can't transform it to NumPy

# To fix the GPU tensor with NumPy issue, we first set it to CPU

tensor_back_on_cpu = tensor_on_gpu.cpu().numpy()
print(tensor_back_on_cpu)
