import torch

x = torch.arange(1,10).reshape(1, 3, 3)
print(x)
print(x.shape)

# Indexing on tensor
print(x[0])

# tensor([[1, 2, 3],
#        [4, 5, 6],
#        [7, 8, 9]])

#index on middle bracket (dim=1)
print(x[0, 0])
print(x[0][0]) # these two return the same thing
# tensor([1, 2, 3])
# tensor([1, 2, 3])

print(x[0][0][0])
# tensor(1)
print(x[0][0][1])
# tensor(2)

print(x[0][2][2])
# tensor(9)

# you can also use ":" to select all of a target dimension
print(x[:,0])
# tensor([[1, 2, 3]])

# get all values of 0th and 1st dimension, but only the index 1 of 2nd dimension
print(x[:, :, 1])
# tensor([[2, 5, 8]])

# get all vals of 0th dim but only 1 index of 1st and 2nd dim
print(x[:, 1, 1])
# tensor([5])

# get index 0 of 0th and 1st dim, and all vals of 2nd dim
print(x[0, 0, :])
# tensor([1, 2, 3])

print(x[:, :, 2])
# tensor([[3, 6, 9]])

print(x[0, :, 2])
# tensor([3, 6, 9])