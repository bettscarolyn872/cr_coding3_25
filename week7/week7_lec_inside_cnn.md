# Week 7 Lecture: Inside CNNs: What Makes Them Work

## Overview
Today, we will dive deep into CNN architectures, understand what they learn, and trace their evolution. 

---

# Visualizing What CNNs Learn

**Quick Reminder:**
- **Convolution**: Sliding a small filter over the image to detect patterns
- **Filters/Kernels**: The learnable weights that detect features
- **Feature Maps**: The output after applying filters to an image

Let's start with a simple reminder:

```
import torch
import torch.nn as nn
import torchvision.models as models
import matplotlib.pyplot as plt
import numpy as np

# A simple Conv layer
conv = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3)
print(f"Filter shape: {conv.weight.shape}")  # [16, 1, 3, 3]
print(f"Number of parameters: {conv.weight.numel() + conv.bias.numel()}")
```

**What this tells us:**
- `[16, 1, 3, 3]` means: 16 different filters, each looking at 1 input channel, with size 3×3
- Each filter is just a 3×3 grid of learnable numbers
- Total parameters = (3×3×1×16) + 16 bias terms = 160 parameters
- `conv.weight` contains the weights of all the filters in the convolutional layer.
- `conv.bias` contains the bias terms for each filter.
- `numel()` returns the total number of elements in a tensor.

But here's the key question: **What patterns do these numbers learn to detect?**

## What Do Trained Filters Look Like?

When we train a CNN, the random numbers in these filters gradually change to detect useful patterns. Let's look at filters from a network that's already been trained on millions of images:

Let's load a pre-trained model and examine its first layer filters with a VGG16 model.

VGG16 is a popular deep convolutional neural network designed for image recognition tasks. It’s known for its simple and consistent structure, which helped improve accuracy on challenging datasets like ImageNet.

Key points about VGG16:
- Depth: It has 16 layers with learnable weights (13 convolutional layers + 3 fully connected layers).
- Small filters: Uses only small 3×3 convolutional filters throughout the network.
- Layer stacking: Stacks multiple convolutional layers before each pooling layer to gradually learn complex features.
- Pooling: Uses max pooling layers to reduce the size of feature maps and control computation.
- Fully connected layers: At the end, it has fully connected layers to perform classification.
- Uniform design: The consistent use of 3×3 filters and simple architecture makes it easy to understand and implement.

Why VGG16 is important:
- It showed that deeper networks with small filters work better than shallow networks with large filters.
- Its straightforward design influenced many later models.
- Despite being older, it still serves as a strong baseline in computer vision.
- Think of VGG16 as building with many small building blocks (3×3 filters), stacking them layer by layer, to create detailed and powerful image understanding.

```
# import
import torch
import torchvision.models as models
import matplotlib.pyplot as plt
import numpy as np

# Load pre-trained VGG16
vgg16 = models.vgg16(pretrained=True)
vgg16.eval()

# Extract first layer filters
first_conv_layer = vgg16.features[0]
filters = first_conv_layer.weight.data.clone()
print(f"First layer filters shape: {filters.shape}")  # [64, 3, 3, 3]

# Visualize some filters
fig, axes = plt.subplots(4, 8, figsize=(12, 6))
axes = axes.ravel()

for i in range(32):
    # Get the i-th filter (3x3x3 for RGB)
    filter_img = filters[i].cpu().numpy()
    # Normalize to [0, 1] for display
    filter_img = (filter_img - filter_img.min()) / (filter_img.max() - filter_img.min())
    # Transpose to HWC format for matplotlib
    filter_img = np.transpose(filter_img, (1, 2, 0))
    
    axes[i].imshow(filter_img)
    axes[i].axis('off')
    axes[i].set_title(f'Filter {i}', fontsize=8)

plt.suptitle('VGG16 First Layer Filters (3×3 RGB)', fontsize=14)
plt.tight_layout()
plt.show()
```

<img width="1191" alt="vgg16_output" src="https://git.arts.ac.uk/user-attachments/assets/92c74bb6-5b7b-4721-bd36-a573f91322d0" />


**What are we seeing?**
- Each small square shows one of the 3×3 RGB filters learned by VGG16’s first convolutional layer.
- The colors and patterns correspond to the weights the network uses to detect specific features in images.
- These filters act as
    - **Edge detectors**: Filters that look for vertical, horizontal, or diagonal edges
    - **Color detectors**: Filters sensitive to red, green, or blue
    - **Corner detectors**: Filters that activate on corners or intersections
    - **Texture detectors**: Filters for specific patterns like dots or grids

**This is remarkable because:**
1. We never told the network to look for edges
2. These patterns emerged naturally from training on images
3. They're similar to hand-designed filters in classical computer vision
4. The network discovered that edges are fundamental building blocks for understanding images

## Going Deeper: What Do Different Layers Learn?

As we go deeper into the network, neurons combine simple patterns into more complex ones. It's like building with LEGO blocks:
- **Layer 1**: Individual blocks (edges)
- **Layer 2-3**: Simple shapes (corners, curves)
- **Layer 4-5**: Complex parts (eyes, wheels)
- **Layer 6+**: Complete objects (faces, cars)

Let's visualize this hierarchy:

```
import torchvision.transforms as transforms
from PIL import Image
import requests
from io import BytesIO

# Load a sample image
def load_image_from_url(url, size=(224, 224)):
    response = requests.get(url)
    img = Image.open(BytesIO(response.content)).convert('RGB')
    
    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    return transform(img).unsqueeze(0)

# Example image: a corgi
img_url = "https://www.akc.org/wp-content/uploads/2017/11/Pembroke-Welsh-Corgi-standing-outdoors-in-the-fall-400x267.jpg"
input_img = load_image_from_url(img_url)

# Hook to capture activations
activations = {}

def get_activation(name):
    def hook(model, input, output):
        activations[name] = output.detach()
    return hook

# Register hooks at different layers
layers_to_visualize = {
    'conv1': vgg16.features[0],   # First conv layer
    'conv5': vgg16.features[10],  # Middle layer
    'conv10': vgg16.features[20], # Deeper layer
}

for name, layer in layers_to_visualize.items():
    layer.register_forward_hook(get_activation(name))

# Forward pass
with torch.no_grad():
    output = vgg16(input_img)

# Visualize activations
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for idx, (name, activation) in enumerate(activations.items()):
    # Take the first few channels and average them
    act = activation[0, :16].mean(dim=0).cpu().numpy()
    
    axes[idx].imshow(act, cmap='viridis')
    axes[idx].set_title(f'{name} activations\nShape: {list(activation.shape)}')
    axes[idx].axis('off')

plt.tight_layout()
plt.show()
```

<img width="1240" alt="corgi_convo" src="https://git.arts.ac.uk/user-attachments/assets/c68c174b-cb73-45a7-a18d-5ecf9435b109" />


**What the code did:**
1. Loaded a real image (a corgi photo).
- The image was downloaded from the internet and preprocessed (resized, cropped, and normalized*) to fit the model input requirements.
- *Normalization refers to adjusting the pixel values of the input image so they have a consistent scale and distribution that matches what the pretrained model expects.

2. Passed the image through a pretrained CNN (VGG16).
- The CNN processed the image layer by layer, extracting features at different levels of complexity.

3. Captured feature maps (activations) from multiple layers using hooks.
- Hooks are special functions that let us peek inside the network to see what patterns each layer detects when processing the image.
- A feature map is the output produced when a convolutional filter (kernel) scans over an input image or the output of a previous layer.

4. Visualized the average activations of 16 channels from three different layers:
- Early layer: detects simple features like edges and colors
- Middle layer: detects more complex shapes and textures
- Deeper layer: detects object parts or semantic concepts

**How to interpret the visualizations:**
- The images you see show where and how strongly certain features were detected in the input image at each layer
- Early layers highlight basic structures like edges and color gradients
- Middle layers show more detailed textures and shapes
- Deep layers capture parts of objects, helping the network recognize high-level concepts (like “corgi ears” or “fur patterns”)

**Why this matters:**
- It helps us understand what the CNN “sees” internally
- Demonstrates the hierarchical nature of CNN feature extraction
- Shows that CNNs learn meaningful representations automatically from data


## Feature Hierarchy in CNNs

CNN Feature Hierarchy:
Layer 1-2    -> Edges, colors, simple textures
Layer 3-5    -> Corners, curves, simple shapes
Layer 6-8    -> Object parts (eyes, wheels, legs)
Layer 9-12   -> Full objects (faces, cars, animals)
Layer 13-16  -> Scenes and contexts

**Why this hierarchy makes sense:**
- **Compositionality**: Complex objects are made of simpler parts
- **Reusability**: An edge detector useful for cars is also useful for buildings
- **Efficiency**: Share low-level features across many high-level concepts
- **Biological inspiration**: Similar to how human vision works!

Think of it like learning to read:
1. First, you learn individual strokes and curves
2. Then you combine them into letters
3. Letters form words
4. Words create sentences
5. Sentences convey meaning

CNNs do the same with visual features.

---

# Understanding Receptive Fields

## What is a Receptive Field?

Imagine you're looking through a camera. You can only see a frame/part of a scene. That's your "field of view". Similarly, neurons in a CNN can only "see" a small patch of the input image. This patch is called the **receptive field**.

**Why this matters:**
- A neuron detecting a cat needs to "see" the whole cat
- A neuron detecting an edge only needs to see a small area
- Deeper layers need larger receptive fields to detect complex objects

Let's visualize this concept:

```
# import
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# Visual demonstration of receptive fields
import torch.nn.functional as F

def visualize_receptive_field():
    # Create a simple CNN
    class SimpleCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(1, 1, kernel_size=3, padding=1)
            self.conv2 = nn.Conv2d(1, 1, kernel_size=3, padding=1)
            self.pool = nn.MaxPool2d(2, 2)
            
        def forward(self, x):
            x = self.conv1(x)
            x = self.pool(x)
            x = self.conv2(x)
            return x
    
    # Create input with single white pixel
    input_img = torch.zeros(1, 1, 16, 16)
    input_img[0, 0, 8, 8] = 1  # Center pixel
    
    model = SimpleCNN()
    # Set weights to 1 for visualization
    with torch.no_grad():
        for conv in [model.conv1, model.conv2]:
            conv.weight.fill_(1/9)  # Average pooling effect
            conv.bias.zero_()
    
    # Forward pass
    layer1_out = model.conv1(input_img)
    pool_out = model.pool(layer1_out)
    final_out = model.conv2(pool_out)
    
    # Visualize
    fig, axes = plt.subplots(1, 4, figsize=(12, 3))
    
    imgs = [input_img[0, 0], layer1_out[0, 0], pool_out[0, 0], final_out[0, 0]]
    titles = ['Input\n(16×16)', 'After Conv1\n(16×16)', 'After Pool\n(8×8)', 'After Conv2\n(8×8)']
    
    for ax, img, title in zip(axes, imgs, titles):
        im = ax.imshow(img.detach().numpy(), cmap='hot')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        
    plt.suptitle('Receptive Field Growth Through Layers')
    plt.tight_layout()
    plt.show()

visualize_receptive_field()
```

<img width="1207" alt="Receptive Field Growth Through Layers" src="https://git.arts.ac.uk/user-attachments/assets/3d853346-51c5-4ce8-ad9c-d4cf1c3bd362" />

**What this visualization shows:**
1. Input (16×16): 
The original input image is mostly empty (all zeros) except for one white pixel in the center. This helps us trace how information spreads through the network.

2. After Conv1 (16×16):
The first convolution applies a 3×3 filter over the input. Because of this, each neuron in this layer “sees” a 3×3 patch of the input. So the influence of that one white pixel spreads to a small 3×3 area in the feature map.

3. After Pool (8×8):
Max pooling downsamples the feature map by a factor of 2, reducing spatial size but increasing the effective receptive field. Now, each neuron in this smaller map corresponds to a larger area (roughly 2×2) in the previous layer’s output and thus a bigger patch in the original input.

4. After Conv2 (8×8):
Another convolution with a 3×3 kernel further increases the receptive field. Each neuron now covers a wider area of the original input image, combining information from a larger patch.

This spreading effect is the **receptive field growing**.

**Why does this matter?**
- Receptive field size determines how much context a neuron sees:
- Early layers focus on local details (edges, textures), while deeper layers integrate broader context (object parts, whole objects).
- Understanding receptive fields helps design networks: You want neurons in deeper layers to have receptive fields big enough to capture entire objects or important regions.
- Pooling layers reduce spatial size but increase the receptive field: This trade-off is crucial for balancing detail and computational efficiency.

### Calculating Receptive Fields

Understanding receptive field math helps you design better architectures. Here's a simple formula:

**For a single conv layer:**
- Receptive Field = Kernel Size

**For multiple layers:**
- Each layer adds to the receptive field
- Pooling/striding multiplies the effect

Let's see this in action:

```
def calculate_receptive_field(layers):
    """
    Calculate receptive field for a sequence of layers
    layers: list of (kernel_size, stride, padding) tuples
    """
    rf = 1  # Initial receptive field
    stride_total = 1  # Cumulative stride
    
    print(f"{'Layer':<20} {'Operation':<25} {'RF':<10} {'Stride':<10}")
    print("-" * 65)
    print(f"{'Input':<20} {'-':<25} {rf:<10} {stride_total:<10}")
    
    for i, (k, s, p) in enumerate(layers):
        rf = rf + (k - 1) * stride_total
        stride_total *= s
        
        op_type = f"Conv{k}×{k}, stride={s}, pad={p}"
        print(f"{'Layer ' + str(i+1):<20} {op_type:<25} {rf:<10} {stride_total:<10}")
    
    return rf, stride_total

# Example: VGG-style architecture
vgg_layers = [
    (3, 1, 1),  # Conv3×3
    (3, 1, 1),  # Conv3×3
    (2, 2, 0),  # MaxPool2×2
    (3, 1, 1),  # Conv3×3
    (3, 1, 1),  # Conv3×3
    (2, 2, 0),  # MaxPool2×2
]

final_rf, final_stride = calculate_receptive_field(vgg_layers)
print(f"\nFinal receptive field: {final_rf}×{final_rf} pixels")
print(f"Final stride: {final_stride}")
print(f"A 224×224 image becomes: {224//final_stride}×{224//final_stride}")
```

The output is
```
Layer                Operation                 RF         Stride    
-----------------------------------------------------------------
Input                -                         1          1         
Layer 1              Conv3×3, stride=1, pad=1  3          1         
Layer 2              Conv3×3, stride=1, pad=1  5          1         
Layer 3              Conv2×2, stride=2, pad=0  6          2         
Layer 4              Conv3×3, stride=1, pad=1  10         2         
Layer 5              Conv3×3, stride=1, pad=1  14         2         
Layer 6              Conv2×2, stride=2, pad=0  16         4         

Final receptive field: 16×16 pixels
Final stride: 4
A 224×224 image becomes: 56×56
```

- Receptive field: After all layers, each neuron in the output feature map corresponds to a 16×16 pixel region in the original input image.
- Stride: The downsampling factor is 4, meaning the spatial dimensions of the feature map are reduced by a factor of 4 compared to the input.
- Output size: For an input image of size 224 x 224, dividing by a stride of 4, gives an output size of approximately 56 x 56.


## Why Receptive Fields Matter

Let's see why getting the receptive field right is crucial for detection:

```
# Demonstrate object size vs receptive field
def show_receptive_field_importance():
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    # Create sample "image" with objects of different sizes
    img = np.zeros((224, 224, 3))
    
    # Small object (10×10)
    img[50:60, 50:60, 0] = 1  # Red small object
    
    # Medium object (40×40)
    img[100:140, 100:140, 1] = 1  # Green medium object
    
    # Large object (80×80)
    img[140:220, 140:220, 2] = 1  # Blue large object
    
    axes[0].imshow(img)
    axes[0].set_title("Input Image\nObjects: 10×10, 40×40, 80×80")
    axes[0].axis('off')
    
    # Show receptive fields
    rf_sizes = [5, 40, 100]
    colors = ['red', 'green', 'blue']
    
    for i, (rf, color) in enumerate(zip(rf_sizes[1:], colors[1:]), 1):
        ax = axes[i]
        ax.imshow(img, alpha=0.3)
        
        # Draw receptive field
        center = 112
        rect = plt.Rectangle((center-rf//2, center-rf//2), rf, rf, 
                           fill=False, edgecolor=color, linewidth=3)
        ax.add_patch(rect)
        ax.set_title(f"Receptive Field: {rf}×{rf}")
        ax.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    print("Key Insights:")
    print("- Early layers (small RF): Detect small features like edges")
    print("- Middle layers (medium RF): Detect parts of objects")
    print("- Deep layers (large RF): Detect whole objects")
    print("- Object detection needs RF ≥ object size!")

show_receptive_field_importance()
```

<img width="1188" alt="object size v receptive field" src="https://git.arts.ac.uk/user-attachments/assets/fac7f7e0-40af-4145-8ce7-59886616949d" />

- Early CNN layers have small receptive fields (e.g., 5×5 pixels) and are good at detecting small features like edges.
- Middle layers have medium receptive fields (e.g., 40×40 pixels) and can detect parts of objects.
- Deep layers have large receptive fields (e.g., 100×100 pixels) and can recognize whole objects.
- To detect an object reliably, the receptive field size of neurons needs to be at least as large as the object’s size.
- If the receptive field is too small compared to the object, neurons won’t capture enough context to identify it.

**Real-world implications:**
1. **Too small RF**: Can't see the whole object → poor detection
2. **Too large RF**: Wastes computation, includes irrelevant background
3. **Just right**: Efficiently captures object features

**Example**: To detect faces (typically 50-100 pixels):
- Early layers with 3×3 RF won't see the whole face
- Need to stack enough layers to reach ~100×100 RF
- This is why face detection networks are deep

---

# CNN Architecture Evolution

Now, let's trace how CNN architectures evolved over time. Each breakthrough solved specific problems and pushed the field forward.

## LeNet-5 (1998): The Pioneer

**Historical Context**: In the 1990s, people weren't sure if neural networks could work for vision. Yann LeCun proved them wrong with LeNet-5, which could read handwritten digits on checks!

**Key Insights from LeNet**:
- Convolution + Pooling pattern works
- Gradual reduction in spatial size (28→14→10→5)
- Gradual increase in channels (1→6→16)
- End with fully connected layers for classification

**Why LeNet Mattered**:
- First practical CNN application (check reading)
- Established the conv -> pool -> conv -> pool -> FC pattern
- Showed that weight sharing (convolution) beats fully connected
- Only 60K parameters could achieve 99%+ on MNIST

## AlexNet (2012): The Game Changer

**The ImageNet Moment**: For years after LeNet, CNNs were largely ignored. Then in 2012, AlexNet crushed the ImageNet competition, dropping error rates from 26% to 15%. This started the deep learning revolution!

**Why did AlexNet succeed where others failed?**
1. **GPUs**: Made training deep networks feasible
2. **ReLU**: Solved the vanishing gradient problem
3. **Dropout**: Prevented overfitting on large networks
4. **Data Augmentation**: Created more training examples

**Before AlexNet:**
- The vanishing gradient problem was a major challenge in training deep neural networks before AlexNet
- AlexNet helped overcome this by using ReLU activation functions, which mitigate vanishing gradients better than older activations like sigmoid or tanh
- This innovation was key to enabling deeper networks and better performance

**AlexNet's Architecture**:
- 5 convolutional layers (much deeper than LeNet)
- 60 million parameters (1000× larger than LeNet)
- Trained on 1.2 million images (vs 60K for LeNet)
- First network to use GPU training effectively

## VGGNet (2014): Simplicity and Depth

**The Design Philosophy**: While AlexNet used various kernel sizes (11×11, 5×5, 3×3), VGG asked: "What if we only use 3×3?" This radical simplicity led to better performance.

**VGG's Key Insight**: Two 3×3 convolutions have the same receptive field as one 5×5, but with:
- Fewer parameters
- More non-linearity (two ReLUs instead of one)
- Better gradient flow
- Instead of using one big window (for example, 7×7) to look at the image, VGG uses several smaller windows (3×3) in sequence
    - This is like looking carefully through smaller lenses multiple times rather than one big lens once.
    - It’s more efficient and helps the network learn better.
- More non-linearity:
    - Each convolution is followed by an activation function (like ReLU).
    - Stacking multiple small conv layers means more ReLU activations between them.
    - More activations add more non-linearity, which helps the network learn more complex patterns.
- VGG proved that deeper is better, and simple, uniform architectures can work extremely well

**VGG's Design Rules**:
1. All convolutions are 3×3 (except the first layer in some variants)
2. Double channels after each pooling
3. Multiple convs before pooling for deeper features
4. Very deep: VGG-16 has 16 layers, VGG-19 has 19

## ResNet (2015): The Revolution

**The Problem**: VGG showed that deeper is better, but there was a limit. Networks deeper than ~20 layers actually performed WORSE. Not due to overfitting, but optimization difficulty.

**ResNet's Breakthrough**: 
- Enabled training networks with 100+ layers (ResNet-152)
- Won ImageNet 2015 with 3.57% error (better than human performance!)
- Skip connections are now used in almost all modern architectures
- Showed that depth is not a limitation anymore

**How ResNet Works:**
- Deep networks often struggle because information and gradients can get lost across many layers
- Skip connections create shortcuts that let information and gradients bypass some layers, making learning easier
- Instead of learning a completely new transformation, the network learns the residual (difference) from the input
- If the best action is to keep the input unchanged, the skip connection makes that easy to learn
- Without skip connections, very deep networks have difficulty training due to vanishing gradients
- Skip connections enable effective training of very deep networks (50, 100+ layers), driving the success of ResNet and modern architectures

<img width="991" alt="How Skip Connections Help Gradient Flow in Deep Networks" src="https://git.arts.ac.uk/user-attachments/assets/fd79af4c-8ded-49b9-8d17-d8a3053b4844" />

The figure compares two scenarios during training a deep neural network:

**Left Side: Traditional Deep Network (No Skip Connections)**
- The layers are stacked one after another with no shortcuts
- The arrows show how gradients flow backward during training
- As the network gets deeper, gradients often become very small or vanish before reaching early layers
- This vanishing gradient problem makes it hard for early layers to learn effectively
- The red text “Gradients vanish” highlights this issue

**Right Side: ResNet with Skip Connections**
- Here, skip connections (red dashed arrows) create shortcuts allowing gradients to flow directly from deeper layers back to earlier ones.
- This direct path helps gradients bypass some layers, preventing them from vanishing.
- The green text “Gradients flow directly” emphasizes that skip connections help maintain strong gradient signals.
- Skip connections also let the network learn residual functions, making training much easier, especially for very deep networks.

**Why Skip Connections Work**:
1. **Gradient Highway**: Gradients can flow directly backward through shortcuts
2. **Identity Mapping**: Easy to learn when layers should do nothing
3. **Ensemble Effect**: Like having multiple paths of different lengths
4. **Feature Reuse**: Later layers can use both processed and raw features

## Different Architectures for Different Tasks

So far, we've focused on classification: "What's in this image?" But computer vision has many other tasks. The beauty of CNNs is that we can modify them for different purposes.

**Common Computer Vision Tasks**:
1. **Classification**: What is in the image? (one label per image)
2. **Segmentation**: What is each pixel? (label every pixel)
3. **Detection**: Where are the objects? (find and label multiple objects)
4. **Generation**: Create new images (we'll see this in later courses)

Next week, we will dive into object detection with YOLO. 

---

# Practical Training Tips

## Why Initialization Matters
- Too small → Signals vanish → No learning
- Too large → Signals explode → Unstable training  
- Just right → Steady gradient flow → Good learning

## Common Troubleshooting

If:                          Then:
- Loss = NaN                  Learning rate too high -> Try ÷10
- Loss not decreasing         LR too low or high -> Try 0.001, 0.0001, 0.00001
- Overfitting quickly         Model too big -> Add dropout, reduce channels
- Underfitting                Model too small -> Add layers/channels
- Training too slow           Batch size too small -> Use GPU, larger batches
- Gradient explosion          Bad init or no BN -> Add BatchNorm
- Poor accuracy               Data issues -> Check normalization, augmentation

## Pro Tips for Training CNNs
- Start simple: Get a small model working first. Test with 3–5 convolutional layers before going deeper.
- Gradual channels: Increase the number of filters gradually (e.g., 32 → 64 → 128) rather than jumping abruptly (e.g., 32 → 512).
- Use 3×3 convolutions: Stack multiple small kernels to achieve larger receptive fields efficiently.
- Apply Batch Normalization: Place batch normalization layers after convolution and before activation to stabilize training.
- Use skip connections: For networks deeper than 20 layers, incorporate skip (residual) connections to improve gradient flow.
- Overfit one batch: Try training your model on a very small dataset (like 10 samples). If it can’t achieve near-zero loss or perfect accuracy, there’s likely a problem, such as:
    - Bugs in data loading, model definition, or loss calculation
    - Incorrect optimizer or learning rate settings
    - Model architecture issues
    - Data preprocessing mistakes
- Visualize everything: Monitor losses, gradients, and activations to understand training behavior and diagnose issues.
- Use proven architectures: Start with well-established models instead of building from scratch unless necessary.

### Learning Rate is Crucial

Learning Rate Guidelines:
- Too High (>0.1): Loss explodes, NaN values
- Good (0.001): Steady decrease, smooth curves
- Too Low (<0.0001): Very slow progress, gets stuck

Modern approach: Start at 0.001, reduce when loss plateaus

---

# Today's Lab

Today's lab notebook is adapted for Google Colab. You can find it [here](https://drive.google.com/file/d/1WiiCk1Pc_Ltb4KMuaMPIJjoS2aHUPLmU/view?usp=sharing).
