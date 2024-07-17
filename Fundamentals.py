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

