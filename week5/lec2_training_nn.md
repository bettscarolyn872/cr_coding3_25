## How Networks Learn

The entire process of deep learning is a cycle repeated thousands of times, enabling the network to adjust its internal patterns ($w$ and $b$).

- Neural networks learn by adjusting weights
- Training loop (repeated many times):
    1. Forward pass: Input data flows through the network to generate a Prediction
    2. Calculate loss: How wrong were we? The prediction is compared to the true label (Target) to measure the Error
    3. Backward pass: Calculating gradients is done by the backpropagation algorithm using the error to calculate (the direction of steepest increase in error)
    4. Update: Adjust weights to reduce loss where the Optimizer uses the gradients to adjust the weights and biases

- It's like learning to throw darts:
    - Throw a dart (forward)
    - See how far from target/error (loss)
    - Adjust aim (update weights)
    - Try again -> repeat the cycle

---

Now that we understand how networks learn conceptually, let's look at the practical framework we use for every PyTorch project. Whether you're building a simple linear model or a deep neural network with ReLU activations, the PyTorch workflow is always the same:

![01_a_pytorch_workflow](https://git.arts.ac.uk/user-attachments/assets/71d5797c-fc1e-4f74-bc5e-623f47fc5c73)

1. **Get data ready** (DataLoaders, transforms)

<img width="1402" alt="01-machine-learning-a-game-of-two-parts" src="https://git.arts.ac.uk/user-attachments/assets/47eefe7d-7256-45d3-9baa-cc7074780c0d" />

2. Build or pick a pretrained **model** (Linear, Sequential, etc.)
3. **Fit** the model (Training loop)
4. **Evaluate** (Test accuracy)
5. **Improve** (Adjust hyperparameters, architecture)
6. **Save and load** (Preserve your work)

_Part of today’s lecture (including some content and images) was adapted from PyTorch Fundamentals. Images without separate credit are from [PyTorch Fundamentals](https://www.learnpytorch.io/01_pytorch_workflow/). You can also practice with the [PyTorch Workflow notebook](https://www.learnpytorch.io/01_pytorch_workflow/). Just keep in mind that the workflow notebook shows a linear regression example, but we're using the exact same process for our neural network with ReLU. The framework scales from simple to complex!_

Today we'll focus especially on steps 3-6. You already learned about building models in Lecture 1 (step 2), and we've loaded MNIST data (step 1). Now let's dive into the training process.

## Loss Functions: Measuring "How Wrong"
 
- Loss function quantifies the model's prediction error
- Lower loss = better predictions
  
- Different tasks need different losses:
  
| Task Type                          | Common Loss Function                         | What it Measures                                                                          | PyTorch Example         |
|------------------------------------|----------------------------------------------|-------------------------------------------------------------------------------------------|-------------------------|
| Regression (Predicting a value)    | Mean Squared Error (MSE) or L1Loss (MAE)     | The squared or absolute distance between prediction and target (how far off the numbers are). | `nn.L1Loss()` / `nn.MSELoss()` |
| Classification (Predicting a category) | Cross‑Entropy Loss (CE)                      | The difference between the model's predicted probability distribution and the true target distribution (penalises low probability on the correct class). | `nn.CrossEntropyLoss()`   |

### Softmax: The Final Activation for Classification

We use a special function called Softmax on the final layer of a classification model (like MNIST):

- Softmax converts the network's raw scores (logits) into a **probability distribution**.
- It ensures all output probabilities are positive and **sum up to 1.0.**
- This mutually exclusive output is required by the **Cross-Entropy Loss** function.

## 2. Optimizer: How to Update Weights

The **Optimizer** is the algorithm that determines how the model will use the calculated Gradient to adjust its parameters ($w$ and $b$).

| Optimizer                                | Method                                                                 | Learning Rate (lr)                         | When to Use                                             |
|------------------------------------------|------------------------------------------------------------------------|--------------------------------------------|---------------------------------------------------------|
| SGD (Stochastic Gradient Descent)        | Simple; takes steps proportional to the gradient.                      | The step size for each update. Too high → unstable; too low → slow. | Good starting point; useful for teaching concepts and sometimes preferred for large‑scale or well‑tuned training. |
| Adam (Adaptive Moment Estimation)        | Adaptive; adjusts the effective step size per parameter using moments. | Typical default: 0.001                      | Often the default choice for faster convergence and robust performance with minimal tuning. |


```python
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
```

## The Training Loop

- This is the heart of deep learning
 
```python
for epoch in range(num_epochs):
    for batch in dataloader:
        # 1. Forward pass
        outputs = model(inputs)
        loss = loss_function(outputs, targets)
        
        # 2. Backward pass
        optimizer.zero_grad()  # Clear old gradients
        loss.backward()        # Calculate new gradients
        
        # 3. Update weights
        optimizer.step()       # Apply gradients
        
    print(f'Epoch {epoch}, Loss: {loss.item()}')
```

## Data Loading: Handling Data Efficiently

Training on one image at a time is slow. The `DataLoader` solves this by grouping data into **batches**.

- **Batching**: Processing 64 images at once is more efficient for the computer (especially GPUs) than processing them one by one.
- **Shuffling**: The DataLoader randomizes the data order (shuffle=True), which helps prevent the model from learning the order of samples.
- **Shape**: When you grab a batch of 64 MNIST images, the shape is 4D: `[Batch, Channel, Height, Width]`, e.g., `$[64, 1, 28, 28]$`.
 
- DataLoader handles batching and shuffling:

```python
from torch.utils.data import DataLoader

# Load MNIST
train_dataset = torchvision.datasets.MNIST(
    root='./data', train=True, 
    transform=transforms.ToTensor(), download=True)

# Create DataLoader
train_loader = DataLoader(
    train_dataset, 
    batch_size=64,      # 64 images at a time
    shuffle=True        # Randomize order
)
```

- Why batch? More efficient than one image at a time!


## Complete Example: MNIST Classifier

 
```python
# 1. Model
model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(784, 128), nn.ReLU(),
    nn.Linear(128, 64), nn.ReLU(),
    nn.Linear(64, 10)
)

# 2. Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

# 3. Training Loop
for epoch in range(5):
    for images, labels in train_loader:
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

## Monitoring Training: How Do We Know It's Working?

- Track metrics during training:
    - Loss decreasing? Great!
    - Loss stuck? Try lowering the learning rate, changing the optimizer, or checking for bugs (data/labels, gradients)
    - Loss exploding? Try reducing the learning rate
    - Be aware of overtraining or underfitting

- Always test on unseen data:
    - Training accuracy: How well on training data
    - Validation/ Test accuracy: How well on new data
    - Gap between them? Might be overfitting

- Visualize predictions to verify: to inspect sample inputs, predicted outputs, and errors (confusion matrix/ sample images), which will help catch systematic mistakes

<img width="502" alt="generalization_curve_overfitting" src="https://git.arts.ac.uk/user-attachments/assets/478046c3-0029-4566-ac15-ab9ea99695af" />

[Image credit/ also great resource to check out: Machine Learning Crash Course by Google](https://developers.google.com/machine-learning/crash-course/overfitting/overfitting?_gl=1*1mflb7v*_up*MQ..*_ga*MTYyMjA1MzcwNy4xNzYxNzQzODk0*_ga_SM8HXJ53K2*czE3NjE3NDM4OTQkbzEkZzAkdDE3NjE3NDM4OTQkajYwJGwwJGgw#fitting)

## Evaluation: Testing on Unseen Data

```
# Set model to evaluation mode
model.eval()

# Turn off gradients for evaluation
with torch.no_grad():
    correct = 0
    total = 0
    
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    print(f'Test Accuracy: {accuracy:.2f}%')
```

## Saving and Loading Models

```
# Save your trained model
torch.save(model.state_dict(), 'mnist_model.pth')
print("Model saved!")

# Load the model later
new_model = MNISTNet()  # Create model structure
new_model.load_state_dict(torch.load('mnist_model.pth'))
new_model.eval()  # Set to evaluation mode
print("Model loaded!")
```

## Debugging Neural Networks/ Common Pitfalls

- Loss is NaN or Infinity:
    - Learning rate too high
    - Try a smaller learning rate

- Loss not decreasing:
    - Learning rate too small
    - Model too simple
    - Check data is loaded correctly

- 10% accuracy on MNIST (random guessing):
    - Wrong loss function
    - Model always predicts the same class
    - Data preprocessing issue

- Remember: Start simple, add complexity!

---

# In-Class Assignment 2

- Please complete the exercise at the end of Lab 1, Lab 2, and Lab 3
- This is In-Class Assignment #2 out of 4
- They are clearly labelled. For example: "In-class Assignment 2/4 Part 1/3 (Part of Your Assessment Hand-In)"
- For hand-in, please complete the cells with each task assigned, run the cells, and save the notebooks with all of your outputs.
- Finally, record a quick (<5 minutes) walk-through video, briefly explaining what you did for each task.

---

# Today's Lab

- Three Jupyter notebooks to complete:

- Lab 1: PyTorch Basics
    - Tensors and operations
    - Automatic differentiation

- Lab 2: Building Neural Networks
    - Linear layers
    - Activation functions
    - Sequential models

- Lab 3: Training Process
    - Complete training loop
    - MNIST classification
    - Evaluation and visualization

- Work at your own pace 
