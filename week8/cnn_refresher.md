# Week 6 & 7 CNN Refresher

The CNN fundamentals you've learned are the foundation for all modern computer vision. Let's do a brief review of what we've covered so far.

## Question 1: Why do CNNs work better than fully connected networks for images?

<details>
<summary>Click to reveal answer</summary>
<br>
<br>
Here is an illustration of a fully connected layer vs. a convolutional layer:

<img width="889" alt="fully_connected_layer_vs_convolutional_layer" src="https://git.arts.ac.uk/user-attachments/assets/292c9eab-3dfe-4399-9cf7-eff14e2b80e2" />

[Image Source: Fully Connected Layer vs. Convolutional Layer: Explained](https://builtin.com/machine-learning/fully-connected-layer)


CNNs excel at image processing for three fundamental reasons:

1. **Preserve Spatial Information**
   - Fully connected networks flatten images (28×28 2D grid/matrix -> 1D vector of length 784), destroying the 2D structure
   - CNNs maintain spatial relationships throughout most of the network
   - Neighboring pixels stay neighbors in feature maps

2. **Parameter Efficiency Through Weight Sharing**
   - FC network: Every pixel connects to every neuron (massive parameters)
   - CNN: Same small kernel (e.g., 3×3) slides across entire image
   - Example: MNIST CNN uses ~100K parameters vs FC's ~600K parameters

3. **Translation Invariance**
   - FC network: The Cat in the top-left is completely different from the cat in the bottom-right because FC layers treat every pixel as a separate input without spatial context
   - CNN: CNNs use filters (feature detectors) that slide across the entire image. So, the same filter that detects a vertical edge or a cat’s ear in one part of the image can detect it somewhere else too.
   - A vertical edge detector works equally well in any position, regardless of where they are in the image

**Biological Inspiration**: Just like our visual cortex processes local regions first (edges, textures) before understanding whole objects, CNNs build up complex understanding from simple local features.

</details>

---

## Question 2: What are the differences between filters and feature maps in CNNs?

<details>
<summary>Click to reveal answer</summary>
<br>
<br>
   
**1. Input Images**
- An image is a grid of pixel values
- Grayscale image = 1 channel (2D matrix)
- Color image = 3 channels (RGB: Red, Green, Blue)

**2. What is the Filter/Kernel?**

<img width="681" alt="kernel working" src="https://git.arts.ac.uk/user-attachments/assets/5a2f3019-86f6-430f-bcfe-c5a6b5bcb362" />

[Image source and a great explanation on how kernels work: Kernels (Filters) in convolutional neural network](https://www.geeksforgeeks.org/deep-learning/kernels-filters-in-convolutional-neural-network/)

- A small 2D grid of numbers. For example: 3x3
- Scans the image to detect patterns like edges or textures
- CNNs have many filters, each learns different features

For example, here is a filter shape: `[16, 1, 3, 3]`
- 16 filters
- 1 input channel
- Each filter is 3x3 in size

**3. What is a Feature Map?**
- Each filter produces one 2D feature map by sliding over the image
- Feature maps show where the filter's pattern appears
- With 16 filters, we get 16 feature maps stacked together

==> **A feature map is the output produced after applying a filter/kernel to an input image or a previous layer's output.** 


</details>


---

## Question 3: What's the typical architecture for a CNN?

<details>
<summary>Click to reveal answer</summary>
<br>
<br>

### Here is the Standard CNN Architecture

<img width="893" alt="standard_cnn_architecture" src="https://git.arts.ac.uk/user-attachments/assets/802a84a5-e3b7-43be-999b-16d0f07d43ea" />

[Image Source: Convolutional Neural Networks: A Comprehensive Evaluation and Benchmarking of Pooling Layer Variants](https://www.mdpi.com/2073-8994/16/11/1516)


The standard CNN architecture follows this pattern:

```
Input Image
    ↓
[Conv → ReLU → Pool] × N         (Feature Extraction)
    ↓
Flatten                         (Convert feature maps into a vector)
    ↓
[FC → ReLU] × M                 (Classification with fully connected layers)
    ↓
Output (Softmax)                (Class probabilities)

```

**Concrete Example (MNIST CNN)**:
```
Input Image: 28×28×1 (grayscale)
    ↓
Conv1: 32 filters of size 3×3 → ReLU activation  
    ↓
MaxPooling: 2×2 window, stride 2  
    Output shape: 14×14×32  
    ↓
Conv2: 64 filters of size 3×3 → ReLU activation  
    ↓
MaxPooling: 2×2 window, stride 2  
    Output shape: 7×7×64  
    ↓
Flatten: Converts feature maps (7×7×64) into a vector of length 7×7×64 = 3,136  
    ↓
Fully Connected Layer 1 (FC1): 128 neurons → ReLU  
    ↓
Fully Connected Layer 2 (FC2): 10 neurons (one per class) → Softmax to produce class probabilities

```

**Key Takeaways**:
- Spatial dimensions decrease (28 → 14 → 7): This happens because of pooling layers or strided convolutions (capturing the strongest details) and helps to reduce computational cost and capture more abstract features over larger regions
- Number of channels increases (1 → 32 → 64): More channels mean the network can learn a richer set of features at each spatial location.
- When it comes to Feature Extraction vs. Decision Making: **Convolutional layers act as feature extractors**, where they detect edges, textures, shapes, etc. **Fully connected layers act as classifiers** that use extracted features to make decisions about the input.
   - Feature extraction (Conv layers) -> Decision making (FC layers)
- Pooling provides downsampling and some translation invariance: Pooling layers reduce the spatial size of feature maps, which makes the network more efficient and makes it less sensitive to small translations or shifts in the input

### Understanding Tensor Dimensions in Convolutional Neural Networks (CNNs)

This guide clarifies the different tensor dimensions involved in CNNs, focusing on how 2D, 3D, and 4D tensors arise in the process.

#### 1. Single Image, Single Filter

- **Input image:**  
  2D matrix of pixel values (Height × Width), e.g., 28×28 pixels.

- **Kernel (filter):**  
  Small 2D matrix, e.g., 3×3.

- **Output feature map:**  
  Result of convolution → a **2D** matrix (Height' × Width'), e.g., 26×26.


#### 2. Single Image, Multiple Filters

- Apply multiple filters (e.g., 16 filters) to the same input image.
- Each filter produces one **2D feature map**.
- Stack all feature maps along a new dimension (channels) → forms a **3D tensor**:

#### 3. Batch of Images, Multiple Filters

- CNNs typically process multiple images simultaneously (batch processing).
- For a batch size of, say, 32 images:
  - Each image produces a **3D tensor** (Height' × Width' × Number of filters).
- Stack outputs along batch dimension → results in a **4D tensor**:


#### Summary Table of Tensor Dimensions

| Scenario                     | Tensor Rank | Shape Example              | Explanation                           |
|-----------------------------|-------------|----------------------------|-------------------------------------|
| Single image, single filter  | 2D          | (Height × Width)           | One feature map                     |
| Single image, multiple filters | 3D          | (Height' × Width' × Filters) | Multiple feature maps stacked       |
| Batch of images, multiple filters | 4D          | (Batch × Filters × Height' × Width') | Batch dimension added                |


#### Key Points to Remember

- **Batch size** adds the first dimension in input/output tensors.  
- **Channels/filters** represent depth in feature maps.  
- **Each filter produces one 2D feature map.**  
- Stacking multiple 2D feature maps creates a 3D tensor for one image.  
- Stacking over batch size creates a 4D tensor overall.

</details>

---

## Question 4: What do early vs late layers detect?

**What kind of features would you expect at different depths of the network?**

<details>
<summary>Click to reveal answer</summary>
<br>
<br>

<img width="622" alt="cnn layers" src="https://git.arts.ac.uk/user-attachments/assets/5e774b94-3600-4116-b07d-f0f02762a854" />

[Image source: Convolutional Neural Networks (CNNs) in 5 minutes](https://glassboxmedicine.com/2020/08/03/convolutional-neural-networks-cnns-in-5-minutes/)


CNNs learn a **hierarchical representation** of visual features:

**Early Layers (1-2)**:
- Simple edges and gradients
- Basic colors and brightness changes
- Similar to Sobel/Canny filters from OpenCV
- Small receptive fields (3×3 to 5×5)

**Middle Layers (3-5)**:
- Corners and curves
- Simple textures (stripes, dots)
- Basic shapes (circles, rectangles)
- Medium receptive fields

**Deep Layers (6-8)**:
- Object parts (eyes, wheels, faces)
- Complex textures
- Semantic components
- Large receptive fields

**Final Layers**:
- Complete objects (cats, cars, people)
- Scene understanding
- Abstract concepts
- Global receptive field

**Visualization Example**:
```
Layer 1: | — / \ (edges)
Layer 3: ◠ ◡ ⌒ ○ (curves, simple shapes)
Layer 5: 👁 🚗 ✋ (object parts)
Layer 7: 🐱 🚗 👤 (whole objects)
```


</details>

---

## PyTorch CNN Cheatsheet

### 1. Essential Imports
```
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
```

### 1.25 Setting the Random Seed

#### Random Seed

We didn't talk about this in the previous notebooks, but you may have seen this in the code. Setting a random seed before starting training a CNN (or any neural network) in PyTorch is done for **reproducibility**. Here's why:

1. **Randomness in Training**: Training a convolutional neural network involves many operations that use randomness, such as:

- Initialization of network weights
- Random shuffling of the training data
- Augmentation techniques like random cropping, flipping, etc.
- Dropout layers during training

2. **Reproducibility**: Without fixing the random seed, every time you run your code, these random operations produce different results. This leads to slightly different model behavior and performance in each run.

3. **Debugging and Experiment Comparison**:

- By setting the random seed, you make sure that the random numbers generated are the same every time. This helps in debugging because you can consistently reproduce errors or issues. It allows fair comparison between different model versions or hyperparameters since all randomness is controlled.

- Multiple Random Sources: PyTorch uses multiple sources of randomness:
  - CPU operations (`torch.manual_seed`)
  - NumPy operations (`np.random.seed`)
  - CUDA (GPU) operations (`torch.cuda.manual_seed`)

```
torch.manual_seed(42)             # You can use any number here. People often use 42 because of the book The Hitchhiker's Guide to the Galaxy
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
```

### 1.5 Load and Prepare Training Data

When you use datasets like MNIST from `torchvision.datasets`, they are already split into a training set (`train=True`) and a test set (`train=False`).

```
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # Normalize grayscale images
])

train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)
```
If it's a dataset you created, you can create an 80/20 split by doing something like this:
```
from torch.utils.data import random_split

dataset = CustomDataset(...)  # your full dataset
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
```

### 2. Basic CNN Architecture

Here, you are:
- Setting up Convolution and Fully Connected layers

```
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Fully connected layers
        self.fc1 = nn.Linear(64 * 7 * 7, 128)  # Adjust size based on input
        self.fc2 = nn.Linear(128, 10)
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        # Conv block 1
        x = self.pool(F.relu(self.conv1(x)))  # 28x28 -> 14x14
        
        # Conv block 2  
        x = self.pool(F.relu(self.conv2(x)))  # 14x14 -> 7x7
        
        # Flatten
        x = x.view(-1, 64 * 7 * 7)
        
        # FC layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
```

#### Here are the Common Layer Parameters in the Code

**Conv2d**
```
nn.Conv2d(in_channels,    # Number of input channels (1 for grayscale, 3 for RGB)
          out_channels,   # Number of filters/kernels
          kernel_size,    # Size of kernel (3 or (3,3))
          stride=1,       # Step size for sliding window
          padding=0,      # Padding to add (1 for 'same' with 3x3)
          dilation=1,     # Spacing between kernel elements
          groups=1,       # For grouped convolutions
          bias=True)      # Include bias term
```

**MaxPool2d**
```
nn.MaxPool2d(kernel_size,   # Size of pooling window
             stride=None,    # Step size (defaults to kernel_size)
             padding=0,      # Padding to add
             dilation=1,     # Spacing between elements
             return_indices=False)  # For unpooling
```

### 3. Training Loop Essentials

Here, you are:
- Setting up the model, loss function, and optimizer
- Moving the model and data to the appropriate device (CPU or GPU)
- Writing the training loop with forward pass, loss calculation, backward pass, and optimizer step
- Here, you don't need a softmax function because `nn.CrossEntropyLoss` runs it internally during training. The output layer will produce raw scores (logits) without softmax.


```
# Setup
model = SimpleCNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Training
model.train()
for epoch in range(num_epochs):
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        # Forward
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

### 4. Evaluation

In the Evaluation step, here is what is happening here’s what happens step-by-step:

- `model.eval()`: This sets the model to evaluation mode, which disables certain layers like dropout and batch normalization from behaving differently during training.
- `with torch.no_grad()`: This context disables gradient calculation, reducing memory usage and speeding up computation since gradients are not needed during evaluation.
- Iterate over `test_loader`: The model processes batches of test images to produce outputs (logits).
- Calculate accuracy: For each batch, you compare the predicted class with the true labels. Predicted class is usually obtained by taking the index of the maximum logit value: `_, predicted = torch.max(outputs, 1)` Then count how many predictions match the true labels and accumulate this count. Finally, calculate accuracy as the ratio of correct predictions over total samples.
   - Here, the underscore `_` is a common Python convention used as a throwaway variable. It means that we don't care about the first output of `torch.max` here.

`torch.max(outputs, 1)` returns two things:

1. The maximum values along dimension 1 (which we don't need in this case).
2. The indices of those maximum values (which correspond to predicted class labels).

Since we only want the indices (predicted classes), we assign the first output to `_` to indicate it's ignored.

```
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
accuracy = correct / total
print(f'Accuracy: {accuracy:.4f}')

```

## Common Patterns and Tips When Working with CNNs in PyTorch:

### Adding Batch Normalization

- Batch Normalization (`BatchNorm`) is a technique used in neural networks to make training faster and more stable. It is typically added right after convolution layers to stabilize and speed up training.
- During training, the distribution of inputs to each layer can change as the parameters of previous layers update. This is called internal covariate shift.
- `BatchNorm` normalizes the output of a layer for each mini-batch, keeping the mean close to 0 and the variance close to 1. This helps the network learn more effectively by stabilizing the input distributions to layers.
- How it works: For each mini-batch, BatchNorm calculates the mean and variance of the activations. It normalizes the activations using these statistics. Then, it scales and shifts the normalized values using learnable parameters (gamma and beta) so the network can still represent the needed transformations.
  
Benefits of Batch Normalization
- Faster training: Allows higher learning rates without instability.
- Regularization effect: Acts like a mild noise injection, which can reduce overfitting.
- Less sensitivity to initialization: Makes training less dependent on careful weight initialization.

Typically, BatchNorm is applied after convolutional or fully connected layers and before non-linearities like ReLU.

```
self.bn1 = nn.BatchNorm2d(32)  # After conv1
# In forward pass:
x = self.bn1(self.conv1(x))

```

### Adding Dropout

- Dropout is a regularization technique used during training to prevent overfitting by randomly “dropping out” (setting to zero) a fraction of neurons in a layer.
- This forces the network to not rely too heavily on any single neuron and encourages it to learn more robust, distributed representations.
- During training, dropout randomly disables neurons with a certain probability (e.g., 0.5). During evaluation, dropout is turned off and all neurons are used.
- Dropout is typically applied after fully connected layers, but can also be used after convolutional layers.

**Benefits of Dropout**
- Reduces overfitting by promoting redundancy in feature learning.
- Improves generalization on unseen data.
- Simple to implement and effective in practice.

**Example usage:**

```
self.dropout = nn.Dropout(p=0.5)  # Drop 50% of neurons
# In forward pass after activation:
x = self.dropout(x)
```

### Using Learning Rate Scheduling

- Learning rate scheduling adjusts the learning rate during training to improve convergence and final performance
- The learning rate determines how big a step the optimizer takes when updating model parameters. Changing it during training can help escape plateaus and fine-tune the model
- Common scheduling strategies include step decay (reducing the rate at fixed epochs) and cosine annealing (smoothly decreasing the rate)
- Learning rate schedulers are typically applied after each epoch or batch during training
  
**Benefits of Learning Rate Scheduling:**

- Enables faster initial training with higher learning rates
- Helps achieve better accuracy by lowering the rate for fine-tuning
- Can prevent oscillations or divergence during training

Example usage with PyTorch:
```
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
# In training loop after each epoch:
scheduler.step()
```

### Using Sequential for Clean Code
- `nn.Sequential` is a container module that allows you to stack layers or operations together in a simple, ordered way. Instead of defining each layer separately and writing the forward pass manually, you can group them inside `nn.Sequential` to create a pipeline where the output of one layer is automatically passed as input to the next:

```
self.features = nn.Sequential(
    nn.Conv2d(1, 32, 3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2, 2),
    nn.Conv2d(32, 64, 3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2, 2)
)
```

### Getting Intermediate Features
- Intermediate features are the outputs produced by layers inside a neural network before reaching the final output. In a CNN, these typically refer to the feature maps generated by convolutional layers after activation functions.

These features capture different levels of information about the input:

- Early layers detect simple patterns like edges or textures.
- Middle layers combine these simple patterns into more complex shapes or parts.
- Deeper layers extract high-level concepts or object parts.

Accessing intermediate features is useful for:

- Understanding what the network is learning at various stages.
- Visualizing the progression from raw input to final prediction.
- Debugging and improving model design.

In code, saving intermediate features usually involves storing outputs from certain layers during the forward pass and returning them alongside the final output.

Saving intermediate outputs can be useful for visualization or debugging:

```
def forward(self, x):
    features = []
    x = self.pool(F.relu(self.conv1(x)))
    features.append(x)  # Save intermediate output
    x = self.pool(F.relu(self.conv2(x)))
    features.append(x)
    # ... continue forward pass
    return x, features  # Return both output and features
```

### Debugging Commands

- Print model architecture to check layers:
```
print(model)
```

- Count total trainable parameters:
```
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params:,}")
```

- Check output shapes of layers with dummy inputs:
```
x = torch.randn(1, 1, 28, 28)  # Dummy input
for layer in model.features:
    x = layer(x)
    print(f"{layer.__class__.__name__}: {x.shape}")
```

- Visualize convolutional filters (weights):
```
filters = model.conv1.weight.data.cpu()
# Plot filters using matplotlib
```


**Tips**:
- Use ReLU (not sigmoid/tanh)
- Add `BatchNorm` after Conv
- Use Dropout in FC layers
- Consider Global Average Pooling instead of Flatten+FC
