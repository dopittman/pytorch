import torch


# Manipulating Tensors (tensor operations)

# ## Tensor operations include:
# * Addition
# * Subtraction
# * Multiplication
# * Division
# * Matrix Multiplication

### Addition

tensor = torch.tensor([1, 2, 3])
plusTen = tensor + 10

print(plusTen)

### Subtraction

minusTen = tensor - 10
print(minusTen)

### Multiply

timesTen = tensor * 10
print(timesTen)

### Division

divTen = tensor / 10
print(divTen)

### PyTorch inbuilt functions - generally use the python functions though

print(torch.add(tensor, 10))
print(torch.mul(tensor, 10))


## Matrix Multiplication

### There are two main ways of performing matrix multiplication in neural networks and deep learning

### * Element wise multiplication

print(f"{tensor} * {tensor}")
print(f"Equals {tensor * tensor}")

### * Matrix multiplication (dot product)

#### * There are two main rules for matrix multiplication
##### * The **inner dimensions** must match:
###### * (3,2) @ (3,2) - won't work
###### * (2,3) @ (3,2) - will work
###### * (3,2) @ (2,3) - will work

##### * The resulting matrix has the shape of the **outer dimensions**:
###### * (2,3) @ (3,2) -> (2,2)
###### * (3,2) @ (2,3) -> (3,3)


##### torch.matmul = matrix multiplication
print(torch.matmul(tensor, tensor))

#### Matrix Multiplication by hand"
#### tensor = tensor([1,2,3]) * tensor([1,2,3]) = 1*1 + 2*2 + 3*3 = tensor([14])


## Matrix Multiplication Continued

## One of the most common errors in deep learning is shape errors

tensor_A = torch.tensor([[1,2,], [3,4], [5, 6]])

tensor_B = torch.tensor([[7,10,], [8,11], [9, 12]])

# print(torch.matmul(tensor_A, tensor_B))
# This returns the error
# RuntimeError: mat1 and mat2 shapes cannot be multiplied (3x2 and 3x2)
# This is because the inner values do not match 2 != 3 so the tensors cannot be multiplied

# To fix our tensor shape issues, we can manipulate the shape of one of our tensors using a **transpose**

# A transpose switches the axis or dimensions of a given tensor

print('Transposed B', tensor_B.T, tensor_B.T.shape)
print('OG Tensor', tensor_B, tensor_B.shape)

# Tensor_B.T is now a shape of [2,3] (was [3,2]), while the original tensor_B remains unchanged

# Using the **transpose** we are able to now multiply the tensors
print(torch.mm(tensor_A, tensor_B.T))
print(torch.mm(tensor_A, tensor_B.T).shape)
# Returns [3,3] since MatMul returns a tensor with the dimensions of the outside dimensions

# The matrix multiplication works when tensor_b is transposed

print(f"Original shapes: tensor_A={tensor_A.shape}, tensor_B={tensor_B.shape}")
print(f"New shapes: tensor_A={tensor_A.shape} same shape as above, tensor_B={tensor_B.T.shape} different shape from above")
print(f"Multiplying: tensor_A={tensor_A.shape} @ tensor_B={tensor_B.T.shape} <- inner dimensions must match")
output = torch.mm(tensor_A, tensor_B.T)
print(f" output: {output}")
print(f"Output Shape: {output.shape}:")


# Tensor Aggregation - Finding the min, max, mean, sum, etc

x = torch.arange(0, 100, 10)
print("x: ", x)

## Min
print("min: ", torch.min(x), x.min())

## Max
print("max: ", torch.max(x), x.max())

## Average - find the mean
## torch.mean requires the type to be float32 (long)
print("mean: ", torch.mean(x.type(torch.float32)), x.type(torch.float32).mean())

## Find the sum
print("sum: ", torch.sum(x), x.sum())


# Positional min and max

## Find the position in tensor that has minimum value, returns index position of target tensor where min val occurs
print("argmin: ", x.argmin())

## Find the position in tensor that has max value, returns index position of target tensor where max val occurs
print("argmax: ", x.argmax())


# Reshaping, stacking, squeezing, and unsqueezing tensors

## Reshaping - reshapes an input tensor to a defined shape
## View - return a view of an input tensor of certain shape but keep same memory as original
## Stacking - return multiple tensors on top of each other vertically (vstack) or side-by-side (hstack)
## Squeeze - removes all `1` dimensions from a tensor
## Unsqueeze - add a `1` dimension to a target tensor
## Permute - return a view of the input with the dimensions permuted (swapped) in a certain way

r = torch.arange(1., 10.)

print(r, r.shape)

### Add an extra dimension
r_reshaped = r.reshape([1, 9])
print(r_reshaped, r_reshaped.shape)

### Change the view - z is just a different view of r - view of tensor uses the same memory as the input
z = r.view(1,9)
print(z, z.shape)
#### since z and r are the same in memory, changing an element in z also changes it in r
z[:,0] = 5
print(z, r)

### Stack tensors on top of each other
x_stacked = torch.stack([r,r,r,r], dim=0)
print(x_stacked)
x_stacked = torch.stack([r,r,r,r], dim=1)
print(x_stacked)

### Squeeze - torch.squeeze - removes all dimensions of input size 1
print(r_reshaped)
print(r_reshaped.shape)
print(r_reshaped.squeeze())
print(r_reshaped.squeeze().shape)

### Unsqueeze - torch.unsqueeze addsa a single dimension to a target tensor at a specific dim
r_squeezed = r_reshaped.squeeze()
print(r_squeezed)
print(r_squeezed.shape)

print(r_squeezed.unsqueeze(dim=0))
print(r_squeezed.unsqueeze(dim=0).shape)

### Permute - torch.permute - rearranges the dimensions of a target tensor in a specific order
r_original = torch.rand(size=(224,224,3)) # [height, width, color_channels]

#### Permute original tensor to rearrange the axis (or dim) order
r_permuted = r_original.permute(2, 0, 1) # shifts axis 0>1, 1>2, 2>0
print(r_original.shape)
print(r_permuted.shape)
#### Permute just changes the view of a tensor in memory, so we get [224,224,3] and [3,224,224] but they are the same in memory

