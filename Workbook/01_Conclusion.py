# Putting it together

# import pytorch and matplotlib
import torch
from torch import nn
import matplotlib.pyplot as plt
from pathlib import Path

# Check Pytorch verssion
print(torch.__version__)

# Create device-agnostic code, if we have access to GPU we use it, otherwise CPU

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device: {device}")

# Create some data using linear regression formula (y = bx + a)
weight = 0.7
bias = 0.3

# Create range values
start = 0
end = 1
step = 0.02

# Create X and Y
X = torch.arange(start, end, step).unsqueeze(dim=1) # without unsqueeze, errors will pop up due to dimension mismatch
y = weight * X + bias

print(X[:10], y[:10])

# Split the data
train_split = int(0.8 * len(X))
X_train, y_train = X[:train_split], y[:train_split]
X_test, y_test = X[train_split:], y[train_split:]

print(len(X_train), len(y_train))
print(len(X_test), len(y_test))

# Plot the data

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


plot_predictions(X_train, y_train, X_test, y_test)

## Building a PyTorch Linear model by subclassing nn.Module

class LinearRegressionModelV2(nn.Module):
    def __init__(self):
        super().__init__()
        # Use nn.Linear() for creating model params / also called linear transform, probe transform, and many others
        self.linear_layer = nn.Linear(1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_layer(x)



# Set the manual seed for reporducability

torch.manual_seed(42)
model_1 = LinearRegressionModelV2()
print(model_1, model_1.state_dict())


# Check the model current device
print(next(model_1.parameters()).device)

# Set the model to use the target device
model_1.to(device)
print(next(model_1.parameters()).device)


# Training Code
### For training we need:
### Loss function
### Optimizer
### Training Loop
### Testing Loop

# Setup loss function
loss_fn = nn.L1Loss() # Same as MAE

# Setup our optimizer
optimizer = torch.optim.SGD(params=model_1.parameters(), lr=0.01)

# Let's write a training loop
torch.manual_seed(42)

epochs = 200

# Put data on target device for device-agnostic code
X_train = X_train.to(device)
y_train = y_train.to(device)
X_test = X_test.to(device)
y_test = y_test.to(device)

for epoch in range(epochs):
    model_1.train()

    # 1. Forward pass
    y_pred = model_1(X_train)

    # Calc loss
    loss = loss_fn(y_pred, y_train)

    # Optimizer zero grad
    optimizer.zero_grad()

    # Perform Backpropagation
    loss.backward()

    # Optimizer step
    optimizer.step()

    # Testing
    model_1.eval()
    with torch.inference_mode():
        test_pred = model_1(X_test)

        test_loss = loss_fn(test_pred, y_test)

    if epoch % 10 == 0:
        print(f"Epoch is {epoch} | Loss is {loss} | test loss is {test_loss}")

print(f"state_dict: {model_1.state_dict()}")

# Eval:
# state_dict: OrderedDict({'linear_layer.weight': tensor([[0.6968]], device='cuda:0'), 'linear_layer.bias': tensor([0.3025], device='cuda:0')})


# Making and evaluating predictions

# Turn model into eval mode
model_1.eval()

with torch.inference_mode():
    y_preds = model_1(X_test)

    # Check out predictions visually
    plot_predictions(predictions=y_preds.cpu()) # Matplotlib requires us to change "preds" to cpu() device


# Save and Load model

# 1. Create model directory
MODEL_PATH = Path('models')

MODEL_PATH.mkdir(parents=True, exist_ok=True)

# 2. Create model save path
MODEL_NAME = "01_pytorch_workflow_model_1.pth"  # PyTorch objects are saved as .pt or pth
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

print(f"Saving model to {MODEL_SAVE_PATH}")
torch.save(obj=model_1.state_dict(), f=MODEL_SAVE_PATH)

# Loading a model
### Since we saved state_dict instead of entire model we'll create a new model class the load in state_dict

### To load in a saved state_dict we need to instantiate a new model class
loaded_model_1 = LinearRegressionModelV2()
print(loaded_model_1.state_dict())
# Load the saved state_dict of model_0 (will update the new instance with updated params)
loaded_model_1.load_state_dict(torch.load(f=MODEL_SAVE_PATH))

# Put loaded model to device
loaded_model_1.to(device)

print(f" dict: {(loaded_model_1.state_dict())}")


# Make predictions with our loaded model
loaded_model_1.eval()
with torch.inference_mode():
    loaded_model_preds = loaded_model_1(X_test)
print(loaded_model_preds)

# Make some model preds (This is just to make sure that the original preds are present)
model_1.eval()
with torch.inference_mode():
    y_preds = model_1(X_test)


# compare loaded model preds with original preds
print(y_preds == loaded_model_preds)
# eval: tensor([[True],
#         [True],
#         [True],
#         [True],
#         [True],
#         [True],
#         [True],
#         [True],
#         [True],
#         [True]])