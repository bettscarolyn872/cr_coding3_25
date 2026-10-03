# Lecture 2: Building, Training, and Understanding CNNs

Now that we understand HOW CNNs work, let's see them in action. We'll build one, train it, and peek inside to see what they actually learn.

---

# Building a CNN for MNIST

Last lecture, we designed this architecture. Now let's build it:

```
Input (28×28×1)
    ↓
Conv1 (32 filters, 3×3) → ReLU → MaxPool (2×2)
    ↓
Conv2 (64 filters, 3×3) → ReLU → MaxPool (2×2)
    ↓
Flatten → FC1 (128) → ReLU → FC2 (10) → Softmax
```

### PyTorch Implementation

```python
import torch.nn as nn

class MNISTConvNet(nn.Module):
    def __init__(self):
        super(MNISTConvNet, self).__init__()
        
        # Feature extraction layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        
        # Classification layers
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        
    def forward(self, x):
        # First conv block
        x = self.pool(F.relu(self.conv1(x)))  # 28×28 → 14×14
        
        # Second conv block  
        x = self.pool(F.relu(self.conv2(x)))  # 14×14 → 7×7
        
        # Flatten and classify
        x = x.view(-1, 64 * 7 * 7)  # Flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
```

### Key Design Decisions

| **Choice** | **Why?** |
|------------|----------|
| **32→64 filters** | Gradually increase capacity as we go deeper |
| **3×3 kernels** | Small enough to be efficient, large enough to capture patterns |
| **Padding=1** | Preserves spatial dimensions (with stride=1) |
| **2×2 MaxPool** | Standard choice for 2x downsampling |
| **128 hidden units** | Sufficient for MNIST's 10 classes |

### Understanding the Dimensions

Let's trace how data flows through our CNN:

```
Input:         [batch, 1, 28, 28]    (grayscale images)
After conv1:   [batch, 32, 28, 28]   (32 feature maps)
After pool1:   [batch, 32, 14, 14]   (downsampled 2x)
After conv2:   [batch, 64, 14, 14]   (64 feature maps)
After pool2:   [batch, 64, 7, 7]     (downsampled again)
After flatten: [batch, 3136]         (64×7×7 = 3,136)
After fc1:     [batch, 128]          
Output:        [batch, 10]           (10 digit classes)
```

In [Lab 3](https://drive.google.com/file/d/1NCpw-Wdpp5iammXcJSBGTZoG1tOABSyg/view?usp=sharing), we implement this exact architecture and train it.

---

# Training and Performance 

## The Training Process

Let's train our CNN and watch it learn.

**Training Setup:**
```python
# Loss function: CrossEntropyLoss for multi-class classification
criterion = nn.CrossEntropyLoss()

# Optimizer: Adam (adaptive learning rates)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(num_epochs):
    for images, labels in train_loader:
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

## Watching Performance Improve

**Training Progression:**
```
Epoch 1: Accuracy 95.2% (already better than random)
Epoch 2: Accuracy 98.1% (learning edges and shapes)
Epoch 3: Accuracy 98.9% (refining feature detectors)
Epoch 4: Accuracy 99.2% (fine-tuning patterns)
Epoch 5: Accuracy 99.3% (near-perfect on MNIST)
```

## What Makes CNNs Train So Well?

**1. Efficient Learning**
- Fewer parameters to optimize
- Shared weights across images = faster convergence
- Natural inductive bias for vision

**2. Meaningful Gradients**
- Local connections = focused learning
- Each filter learns one specific pattern
- Gradients don't vanish as easily

**3. Data Efficiency**
- Parameter sharing = implicit data augmentation
- Can learn from fewer examples
- Generalizes better to new data

## Training Tips

**Start Simple:**
- Begin with proven architectures
- Use standard optimizers (Adam, SGD with momentum)
- Monitor both loss and accuracy

**Common Issues:**
- Overfitting → Add dropout, data augmentation
- Slow convergence → Adjust learning rate
- Poor accuracy → Check data preprocessing

In [Lab 3](https://drive.google.com/file/d/1NCpw-Wdpp5iammXcJSBGTZoG1tOABSyg/view?usp=sharing), you'll train this exact model and see the progression.

---

# Visualizing What CNNs Learn 

## Layer 1: Edge Detectors Emerge

After training, let's look at what the first convolutional layer learned:

```
# Extract learned kernels from conv1
kernels = model.conv1.weight.data.cpu()
# Shape: [32, 1, 3, 3] = a 4D tensor with 32 filters, each 3×3
```
- After training, the first convolutional layer has learned 32 filters (kernels).
- Each filter is a small 3×3 matrix of numbers (weights).
- These filters act like feature detectors that scan the input image to find specific patterns.
- Here:
    - `.data`: Accesses the raw tensor data of the weights, detached from the computation graph. This means you get the actual values without tracking gradients (used for visualization or analysis).
    - `.cpu()`: Moves the tensor from the GPU to the CPU memory. If your model is trained on a GPU, the weights reside in GPU memory by default. To process or visualize these weights using libraries that work on the CPU (like Matplotlib), you need to move them to the CPU first. You want the weights on the CPU to easily convert them to NumPy arrays or visualize them. This is just a CPU copy that is used only temporarily for tasks like visualization or saving.

**How do we see what the filters learned?** 
- You can look at the actual weights of these filters by accessing `model.conv1.weight`
- The shape `[32, 1, 3, 3]` means:
    - 32 filters (one for each feature map output)
    - 1 input channel (for grayscale images)
    - Each filter is 3×3 pixels in size

**What kinds of patterns do these filters learn?**
Early layers in CNNs usually learn to detect basic visual features like:
- Vertical edges: Lines running up and down,
- Horizontal edges: Lines running left to right,
- Diagonal edges: Lines at an angle,
- Corners: Points where edges meet,
- Center-surround: Patterns highlighting contrast between center and surroundings, like blobs or spots.

**Why is this important?**
- These simple patterns are the building blocks for recognizing more complex shapes and objects later in the network.
- The network discovers these useful filters automatically during training — you don’t have to design them manually.
- Visual analogy: Imagine each filter as a small stencil that highlights certain shapes in the image. Some stencils highlight vertical lines, some horizontal, and so on.

[Lab 3](https://drive.google.com/file/d/1NCpw-Wdpp5iammXcJSBGTZoG1tOABSyg/view?usp=sharing) shows the actual learned filters from your trained network.

## From Random to Structured

**Before Training (Random Initialization):**
- When a CNN starts training, its filters (kernels) are initialized with random values.
- These random values look like noise. There are no meaningful patterns.
- At this point, filters don’t detect anything useful because they haven't learned yet.

**After Training (Learned Features):**
- As training progresses, the CNN adjusts the filter values to detect important features in the data.
- The first convolutional layer often learns to detect simple patterns like edges (horizontal, vertical, diagonal).
- These learned filters look organized and structured, like the classic edge detectors, Sobel or Canny filters.
- The CNN automatically discovers these useful detectors from the data without manual design.
    - This is amazing because:
        - Traditional computer vision required manually creating edge detectors.
        - CNNs learn these fundamental features on their own through training.
        - This automatic feature learning is a major reason CNNs are so powerful.

Think of it like starting with a messy, jumbled toolset (random weights) -> Training organizes and sharpens these tools into effective instruments (edge detectors).

## Visualizing Feature Maps

Remember, when do we get feature maps?
- Feature maps are the outputs of convolutional layers after the kernels (filters) slide over the input.
- Each kernel scans the entire input (image or previous layer’s feature maps) using convolution.
- At each position, the kernel performs element-wise multiplication and sums the results, producing one value in the feature map.
- As the kernel moves across all positions, it creates a 2D feature map showing where and how strongly the pattern was detected.

What happens when an image passes through these filters?

**Original Image -> Conv1 Feature Maps:**
```
Input digit "3" -> 32 different feature maps if you have 32 filters
- Map 1: Vertical edges highlighted
- Map 2: Top curves emphasized  
- Map 3: Bottom curves detected
- Map 4-32: Various other patterns
```

Each feature map shows WHERE specific patterns were detected.

## Filter Responses in Action

**Example: Vertical Edge Detector**
- Strong response on the left/right edges of digits
- No response in uniform areas
- Negative response on opposite edges

**Example: Curve Detector**
- Activates on rounded parts of 0, 6, 8, 9
- Less activation on straight digits like 1, 7

## Going Deeper: What Layer 2 Sees

Layer 2 combines Layer 1 features into more complex patterns:

**Layer 1**: Edges and simple patterns
**Layer 2**: Combinations like:
- Curves (multiple edges forming arcs)
- Junctions (where lines meet)
- Loops (closed curves)
- Textures (repeated patterns)

In [Lab 2](https://drive.google.com/file/d/1OaC7SgypikIRN6-zw4lrNyYFhjjdppZZ/view?usp=sharing), we visualize feature maps at each layer to see this progression.

### What Different Layers Learn

**Layer-by-Layer Feature Hierarchy:**

| **Layer** | **What It Detects** | **Example on "8"** |
|-----------|--------------------|--------------------|
| **Input** | Raw pixels | Full digit image |
| **Conv1** | Edges, gradients | Top/bottom circles outlined |
| **Conv2** | Shapes, corners | Two circular parts detected |
| **FC layers** | Complete patterns | "Two stacked circles = 8" |

This hierarchical learning is why CNNs work so well.

---

# Practical Tips for Your CNNs

**1. Start Simple**
- 2-3 conv layers are often sufficient for simple tasks
- Add complexity only if needed

**2. Common Pitfalls**
- Forgetting to flatten before FC layers
- Wrong dimension calculations
- Not enough pooling (network too large)

**3. Hyperparameter Guidelines**
- Kernel size: 3×3 is usually best
- Filters: Start with 32/64, double after pooling
- Pooling: 2×2 with stride 2 is standard

**4. When to Use Pretrained Models**
- Small dataset? Use transfer learning
- Standard task? Start with proven architecture
- Research? Design your own

## Architecture for Different Image Sizes

| **Input Size** | **Architecture Suggestion** |
|----------------|---------------------------|
| Small-size images like MNIST digits (28×28) | 2-3 conv layers enough |
| Medium-size images like typical ImageNet input (224×224) | 5-10 conv layers (VGG-like) |
| Large-size images, high resolution, like medical imaging, satellite photos (512×512+) | Consider strided convolutions |

_Remember: The best way to understand CNNs is to train one yourself! While the practice tasks in the lab notebooks are not required, they will help you deepen your understanding of how to train CNNs._
