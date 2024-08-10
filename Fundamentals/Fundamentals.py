import torch

print(torch.__version__)

# scalar

scalar = torch.tensor(7)
print(scalar)

# Get tensor back as Python int
print(scalar.item())


# Vector
vector = torch.tensor([7, 7])
print(vector)
print(vector.ndim)
print(vector.shape)

# MATRIX
MATRIX = torch.tensor(([7,8], [9,10]))
print(MATRIX)
print(MATRIX.ndim)
print(MATRIX.shape)
print(MATRIX[0])
print(MATRIX[1])

# TENSOR
TENSOR = torch.tensor([[[1,2,3], [4,5,6], [8, 9, 10]]])
print(TENSOR)
print(TENSOR.ndim)
print(TENSOR.shape)
print(TENSOR[0])


### Random Tensors

#### Create a random tensor of size 3,4

random_tensor = torch.rand(3, 4)
print(random_tensor)
print(random_tensor.ndim)


#### Create a random tensor with a similar shape to an image tensor

random_image_tensor = torch.rand(size=(224,224, 3))  # height, width, color channels
print(random_image_tensor.ndim, random_image_tensor.size())

print(torch.Size([224,224,3]))

# Zeros and ones

#### Create a tensor of all zeros
zeros = torch.zeros(3, 4)
print(zeros)

#### Create a tensor of all ones
ones = torch.ones(3, 4)
print(ones)
# default type is a float32
print(ones.dtype)


### Creating a range of tensors and tensors-like

# Use torch.range() - will be deprecated soon, use torch.arange() instead
one_to_ten = torch.arange(0, 10)
print(one_to_ten)


# Creating tensors_like - makes a tensor with the same dim and shape as the input
ten_zeros = torch.zeros_like(input=one_to_ten)
print(ten_zeros)

# Float 32 Tensor
float_32_tensor = torch.tensor([3.0, 6.0, 9.0],
                               dtype=None,  # what datatype the tensor is (i.e. float32, float64, float16, etc.)
                               device="cuda",  # what device your tensor is on, defaults to CPU by default, can also be "cuda" and use the GPU
                               requires_grad=False)  # If you want PyTorch to track the gradient
print(float_32_tensor)
print(float_32_tensor.dtype)

# One way to fix "Tensor is not right datatype error"
float_16_tensor = float_32_tensor.type(torch.float16)
print(float_16_tensor, float_16_tensor.dtype)

print("multiply tensor: ", float_16_tensor * float_32_tensor)
# received: tensor([ 9., 36., 81.], device='cuda:0')
# Some operations will run even if datatypes don't match, other will throw datatype error

int_32_tensor = torch.tensor([3, 6, 9], dtype=torch.int32, device="cuda")

print(int_32_tensor * float_32_tensor)

# Getting information from tensor (tensor attributes)

# 1. Tensor not right datatype - to do get datatype from a tensor, can use "tensor.dtype"
# 2. Tensor not right shape - to get shape from a tensor use 'tensor.shape'
# 3. Tensor not on right device - to get shape from a tensor use 'tensor.device'

# Create a tensor
some_tensor = torch.rand(3, 4)

# Find out details
print(some_tensor)
print(f"Datatype of tensor: {some_tensor.dtype}")
print(f"Shape of tensor: {some_tensor.shape}")
print(f"Device of tensor: {some_tensor.device}")

# Manipulating Tensors (tensor operations)

# ## Tensor operations include:
# * Addition
# * Subtraction
# * Multiplication
# * Division
# * Matrix Multiplication

### Addition

tensor = torch.rand(3, 4)
