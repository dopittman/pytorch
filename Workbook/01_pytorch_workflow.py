import numpy as np

# Exploring an e2e pytorch workflow

what_were_covering = {1: 'data (prepare and load)',
                      2: "build model",
                      3: "fitting the model to data (training)",
                      4: "making predictions and evaluating a model (inference)",
                      5: "saving and loading a model",
                      6: "putting it all together"}

import torch
from torch import nn ## nn contains all off Pytorchs building blocks for neural network
import matplotlib.pyplot as plt

# Check PyTorch version
print(torch.__version__)

## 1. Data (preparing and loading)

# Machine learning is a game of two parts:
# 1. Get data into numerical representation
# 2. BBuild a model to learn patterns in that numerical representation

# To showcase this, let's create some known data using the linear regression formula.
# linear regression formula: Y = a+bX   where X is the explanatory variable, and Y is the dependent variable,
#   the slope of the line is "b" and "a" is the intercept (the value of y when x = 0)

# We'll use a linear regression formula to make a straight line with known parameters

# Y = a+bX
# Create *known* parameters
weight = 0.7 # weight = "b"
bias = 0.3 # bias = "a"

# Create
start = 0
end = 1
step = 0.02
X = torch.arange(start, end, step).unsqueeze(dim=1) # usually a matrix or Tensor
y = weight * X + bias

print(X[:10], y[:10])
print(len(X), len(y))

### Spliting data into training and test sets (one of the most important concepts)

# Let's create a training and testing set with our data

# Create a basic train/test split

train_split = int(0.8 * len(X)) # get 80% of X tensor length
print(train_split)  # eval: 40

X_train, y_train = X[:train_split], y[:train_split]  # use indexing to get everythiong from 0-train_split
X_test, y_test = X[train_split:], y[train_split:]  # use indexing to get everything from train_split onwards

# Length of train and test sets
print(len(X_train), len(y_train), len(X_test), len(y_test))  # eval: 40 40 10 10

# how might we better visualize our data?
# Visualize, visualize, visualize

def plot_predictions(train_data=X_train, train_labels=y_train, test_data=X_test, test_labels=y_test, predictions=None):

    """Plots training data, test data, and predictions."""
    plt.figure(figsize=(10, 7))

    #plot training data in blue
    plt.scatter(train_data, train_labels, c="b", s=4, label="Training Data")

     # Plot test data in green
    plt.scatter(test_data, test_labels, c="g", s=4, label="Testing Data")

    # Are there predictions
    if predictions is not None:
        # Plot the predictions if they exist
        plt.scatter(test_data, predictions, c="r", s=4, label="Predictions")

    #  Show the legend
    plt.legend(prop={'size': 14})

# need the below command to show the plot
    plt.show()


plot_predictions()



### 2. Build model

### What our model does:
### Start with random values (weight and bias)
### Look at training data and adjsut the random values to better represent (or get closer to) the
### ideal values (the weight and bias values we used to create the data)

### How does it do so?
### Through two amin algorithms
### Gradient Descent
### Backpropogation

# Create Linear-regression model class Y = a + bX

## nn.Module = Base class for all neural network modules
## Your models should also subclass this Module
class LinearRegressionModel(nn.Module): ## Almost everything in PyTorch inherits from nn.Module
    def __init__(self):
        super().__init__()
        ### Start with random values for weights and bias
        self.weights = nn.Parameter(torch.randn(1, # <- start with a random weight and try to adjust it to the ideal weight
                                                requires_grad=True, # <- can this parameter be updated via gradient descent
                                                dtype=torch.float))  # <- PyTorch loves the datatype torch.float32

        self.bias = nn.Parameter(torch.randn(1, # <- start with a random bias and try to adjust it to the ideal bias
                                             requires_grad=True, # <- can this parameter be updated via gradient descent
                                             dtype=torch.float))  # <- Again, PyTorch loves torch.float32 data type


        # forward method to define computation in the model
    def forward(self, x: torch.Tensor) -> torch.Tensor:  # <- x is the input data
        return x * self.weights + self.bias  # this is the linear regression formula





### Pytorch model building essentials

# torch.nn - contains all the building blocks for computational graphs (a neural network can be considered a computational graph)
# torch.nn.Parameter - what params should our model try and learn, often a PyTorch layer from torch.nn will set these for us
# torch.nn.Module - The base class for all neural network modules, if you subclass it you should overwrite forward
# torch.optim - this is where the optimizers in PyTorch live, will help with improving gradient descent and reducing the loss
# def forward() - All nn.Modules require you to overwrite forward. this method defines what happens in the forward computation

# torch.utils.data.Dataset - Represents a mao between key(label) and sample(features) pairs of your data. Such as images and their associated labels
# torch.utils.data.DataLoader - Creates a Python iterable over a torch Dataset (allows you to iterate over your data)


### Checking the contents of our PyTorch model
### We can check what's in our model using .Parameters


# Create a random seed
torch.manual_seed(42)

#Create an instance of the model we created (this is a subclass of nn.Module)
model_0 = LinearRegressionModel()

print(list(model_0.parameters()))
# eval: tensor([0.1940], device='cuda:0', requires_grad=True), Parameter containing:
# tensor([0.1391], device='cuda:0', requires_grad=True)]

# List named params
print(model_0.state_dict())
# eval: OrderedDict({'weights': tensor([0.1940], device='cuda:0'), 'bias': tensor([0.1391], device='cuda:0')})
# we want the above values to get closer to the weight and bias we set at the top (weight: 0.7, bias: 0.3)
# in most cases we won't know what the ideal values are

### Making predictoins using torch.inference mode
# To check our models predictive power, lets see how well it predicts y_test based on x_test
# When we pass data through our model, it runs through the forward method


# Make predictions with Model
### We want this prediction to get as close to the test data as possible
with torch.inference_mode(): # <- you can also use torch.no_grad() but inference mode is preffered and quicker
    y_preds = model_0(X_test)

print(y_preds)
# eval: tensor([[0.3982],
#         [0.4049],
#         [0.4116],
#         [0.4184],
#         [0.4251],
#         [0.4318],
#         [0.4386],
#         [0.4453],
#         [0.4520],
#         [0.4588]])

plot_predictions(predictions=y_preds)
# we want the red dots to get as close to the green dots as possible





# 3. Training Model
# The whole idea of training is to move the model from some *unknown* params (these may be random) to some *known* params
# or in other words from a poor representation of the data to a better representation

# One way to measure how poor or wrong your model's predictions are you can use a loss function (criterion or cost function)

# Things we need to train a model:
### Loss function - a function to measure how wrong the model's predicitons are to the ideal outputs, smaller is better
### Optimizer - takes in account the loss of a model and adjusts the parameters (eg. weight and bias) to improve loss func
# And speficially for PyTorch,
### Training loop
### Test loop



# Setup a loss function
# L1Loss - Finds mean absolute error (MAE) between each element in input x and target y
loss_fn = nn.L1Loss()


# Set up an optimizer (stochastic gradient descent)
### Optimzer takes in 2 things:
### #params - the model params we want to optimize
### learning rate (lr) -defines how big/small the optimizer changes the parameters with each step. Smaller lr, smaller changes, vice versa
### possibly the most important hyperparameter you can set)
optimizer = torch.optim.SGD(model_0.parameters(), lr=0.01)


# Building a training loop in PyTorch
### Steps:
### Loop through the data
### Forward pass (Forward propogation) - this involves the data moving through the models forward function(s) to make predictions on data
### Calculate the loss (compare forward pass predictions to ground truth labels
### Optimizer zero grad
### loss backward - moves backwards through the network to calculate the gradients of each param of our models with respect to the loss (backpropogation)
### Optmize step - use the optimizer to adjust our models parmeters to try and improve the loss (gradient descent)

print('starting state:', model_0.state_dict())
# An epoch is one loop through the data (this is a hyperparameter because we set it ourselves)
epochs = 200

# track different values
epoch_count = []
loss_values = []
test_loss_values = []


### Training Loop
# 0. Loop through the data
for epoch in range(epochs):
    # set the model to training mode
    model_0.train() # train mode in PyTorch sets all params that require gradients to require gradients

    # 1. Forward pass
    y_pred = model_0(X_train)

    # 2. Calculate loss - takes input first, target second
    loss = loss_fn(y_pred, y_train)

    # 3. Optimizer zero grad
    ###
    optimizer.zero_grad()

    # 4. Perform backpropgation on the loss with respect to the params of the model
    loss.backward()

    # 5.  step the optimizer - perform gradient descent
    optimizer.step() # By default, how the optimizer changes will accumulate through the loop, so we have to
                     # zero them above in step 3 fopr the next iteration through the loop


### Testing
    model_0.eval() # turns off different settins in the model not needed for  evaluation/testing
    with torch.inference_mode(): # turns off gradient tracking and a couple more things behind the scenes
        #  1. Do forward pass:
        test_pred = model_0(X_test)

        # 2. Calculate the loss
        test_loss = loss_fn(test_pred, y_test) # difference between test predictions and labels

    # Print out what's happening
    if epoch % 10 == 0:
        epoch_count.append(epoch)
        loss_values.append(loss)
        test_loss_values.append(test_loss)

        print(f"Epoch: {epoch} | MAE Train Loss: {loss} | Test loss: {test_loss}")

        print('state', model_0.state_dict())

## Plot the new predictions
# with torch.inference_mode():
#     y_preds_new = model_0(X_test)
#
#     plot_predictions(predictions=y_preds_new)

# Plot the loss curve
plt.plot(epoch_count, np.array(torch.tensor(loss_values).numpy()), label="Train Loss")
plt.plot(epoch_count, test_loss_values, label="Test Loss")
plt.title("Training Loss and Test Loss Curves")
plt.ylabel("Loss")
plt.xlabel("Epochs")
plt.legend()
plt.show()