import torch
import numpy as np

# Data in Numpy, want in Pytorch tensor
# do this with torch.from_numpy(ndarray)
# Pytorch tensor -> Numpy -> torch.Tensor.humpy()

array = np.arange(1.0, 8.0)
tensor = torch.from_numpy(array) # warning: when converting from numpy -> pytorch, pytorch reflects numpy's default datatype of float64 unless specified otherwise

print(array, tensor)
#[1. 2. 3. 4. 5. 6. 7.] tensor([1., 2., 3., 4., 5., 6., 7.], dtype=torch.float64)


print(array.dtype, tensor.dtype)
# float64 torch.float64

# Change value of array after declaration does NOT change tensor value
array = array + 1
print(array, tensor)

# Tensor to Numpy
z = torch.ones(7)
numpy_tensor = z.numpy()
print(z, numpy_tensor)
# # tensor([1., 1., 1., 1., 1., 1., 1.]) [1. 1. 1. 1. 1. 1. 1.]

print(z.dtype, numpy_tensor.dtype)
# torch.float32 float32

# Change the tensor after declaration does not change the array made from numpy, they do not share memory
z = z + 1
print (z, numpy_tensor)



# Reproducibility - trying to take the random out of random)
# How a Neural Network learns is
### start with Random numbers -> Tensor Operations -> update random numbers to try and make
### them better represntations of the data -> again -> again -> again...

### To reduce the randomness in Neural Networks and PyTorch comes the concept of a **random seed**
### What the random seed does is 'flavor' the randomness

# Create two random tensors
random_tensor_A = torch.rand(3, 4)
random_tensor_B = torch.rand(3, 4)

print(random_tensor_A)
print(random_tensor_B)

print(random_tensor_A == random_tensor_B)

# make random but reproducible tensors

# set the random seed
RANDOM_SEED = 42
### Setting the manual seed, you usually have to set the manual seed each time you call each time
torch.manual_seed(RANDOM_SEED)
random_tensor_C = torch.rand(3, 4)

torch.manual_seed(RANDOM_SEED)
random_tensor_D = torch.rand(3, 4)
print(random_tensor_C)
print(random_tensor_D)
print(random_tensor_C == random_tensor_D)




