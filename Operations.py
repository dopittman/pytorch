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



