# Introduction to Convolutional Neural Networks (CNNs)

## Lecture Overview

Welcome back! Last week, we explored neural networks:
- A network is just a series of matrix multiplications
- The training loop involves forward, loss, backward, and update
- We learned about fully connected layers (also called dense layers), where every input connects to every output.

This is the core of how all neural networks learn.

Today, we'll explore why CNNs are great for computer vision tasks and why standard neural networks struggle with images.

---

# Part 1: The Problem with Images

## Why Standard Neural Networks Fail at Vision

Now, let's talk about images. When we feed an image into a standard fully connected neural network (also called a dense neural network), we do something destructive: we flatten it. A 28×28 pixel image of a hand-drawn digit becomes a single, long vector of 784 numbers. Why is this a huge problem?

| **Flaw** | **Explanation** |
|----------|-----------------|
| **Massive Parameters** | If we take a slightly larger 100×100 RGB image, that's 100 × 100 × 3 channels = 30,000 input values. The first layer alone might need millions of parameters. This is inefficient and very slow to train. |
| **Loss of Spatial Data** | Flattening destroys the most important information: spatial locality. When you turn a 2D grid into a 1D vector, the network loses the knowledge that "pixel A is right next to pixel B." It treats distant pixels the same way it treats neighbors. |
| **No Translation Invariance** | If a fully connected network learns that a cat's eye is a certain pattern of pixels in the top-left corner, it can only recognize the eye there. If the cat moves slightly to the bottom-right, the whole pattern changes, and the network treats it as a completely new object. |

## Why CNNs for Images?

### The Inspiration: Biological & Computational Inspiration

Remember how we talked about neurons and neural networks are modeled after our human brain? CNN, Convolutional Neural Network, was inspired by how our own visual cortex works. When you look at an image, your brain doesn't process every pixel simultaneously. It focuses on local features: small edges, lines, and corners. That's the core idea.

| **Advantage** | **Benefit** |
|---------------|-------------|
| **Sparse Connectivity** | We don't connect every input pixel to every neuron. Each neuron in the first layer only "looks" at a small, local area of the image (a Local Receptive Field), drastically cutting down the weight count. |
| **Parameter Sharing (The Game Changer)** | If a kernel (our feature detector) learns how to spot a vertical edge in the top-left, we assume that the same vertical edge detector will work everywhere else in the image. We share those same weights across the entire image. This is the source of Translation Invariance and massive parameter reduction. |
| **Preservation of Spatial Hierarchy** | We keep the image as a 3D volume (Height × Width × Channels). We only flatten it right at the very end, once all the meaningful features have been extracted. |

### Introducing the Convolutional Layer

The key difference is the operation itself. Instead of a massive, one-off matrix multiplication, the CNN uses the Convolution Operation. Think of it as a small, focused lens sliding across the image.

Remember a few weeks back, we talked about **kernels**? This is kernels at work.

A fully connected layer is a type of neural network layer where each neuron is connected to every neuron in the previous layer, enabling global feature learning. A convolutional layer, by contrast, uses filters that connect only to local regions of the input, allowing it to detect spatial patterns like edges or textures while using fewer parameters.

This is why standard fully connected networks are terrible at scaling up to real-world images and are blind to movement.

Here is an illustration of a fully connected layer vs. a convolutional layer:

<img width="889" alt="fully_connected_layer_vs_convolutional_layer" src="https://git.arts.ac.uk/user-attachments/assets/292c9eab-3dfe-4399-9cf7-eff14e2b80e2" />

[Image Source: Fully Connected Layer vs. Convolutional Layer: Explained](https://builtin.com/machine-learning/fully-connected-layer)

### Real-World Impact

| Architecture* | Year | Key Innovation | Impact / Significance |
|--------------|------|----------------|---------------------|
| [LeNet-5](https://d2l.ai/chapter_convolutional-neural-networks/lenet.html) | 1998 | The foundation: Introduced the core structure of CNNs (Conv layers → Pooling layers → Fully Connected layers). | Shows the fundamental building blocks of modern CNNs, proving they could effectively recognize handwritten digits. |
| [AlexNet](https://www.pinecone.io/learn/series/image-search/imagenet/) | 2012 | Deep learning boom: Used Rectified Linear Units (ReLU) as the activation function and extensive use of Data Augmentation and GPU processing. | Showed that deeper networks were possible and practical for complex tasks like the ImageNet challenge, kicking off the deep learning revolution. Dropped [ImageNet](https://www.image-net.org/) error rate from 26% to 15% |
| [VGG](https://medium.com/@siddheshb008/vgg-net-architecture-explained-71179310050f) | 2014 | Simplicity and depth: Showed that using very small (3×3) convolutional filters consistently and simply stacking many layers could achieve state-of-the-art results. | Demonstrated the power of uniform, smaller filters and the benefit of increased depth (up to 19 layers). |
| [ResNet](https://viso.ai/deep-learning/resnet-residual-neural-network/) | 2015 | The solution to vanishing gradients: Introduced Residual Blocks using skip connections (or identity mappings). | Solved the major problem of training ultra-deep networks (up to 152 layers) by allowing gradients to bypass layers, enabling incredible performance gains. |

- **Today**: Powers face recognition, medical imaging, self-driving cars
- **Your phone**: Every photo filter, face detection, and portrait mode uses CNNs

_*Architecture_: Model architecture is the organized design or blueprint of a machine learning model that defines how its layers and components are arranged and connected. It determines how the model processes input data to learn patterns and make predictions. For example, CNNs have a specific architecture designed for processing grid-like data such as images, using layers like convolutional and pooling layers to capture spatial features. LLMs (Large Language Models), on the other hand, typically use the Transformer architecture, which relies on attention mechanisms to handle sequential data like text, enabling them to capture long-range dependencies effectively.

---

# Part 2: The Convolution Operation

## The Kernel (or Filter): A Pattern Detector

The kernel is a small square matrix, usually 3×3 or 5×5, that moves (or slides) across the input image to perform the convolution operation. It acts like a lens focusing on a small region of the image at a time, extracting features by computing dot products between the kernel values and the input pixels covered by the kernel.

<img width="681" alt="kernel working" src="https://git.arts.ac.uk/user-attachments/assets/5a2f3019-86f6-430f-bcfe-c5a6b5bcb362" />

[Image source and a great explanation on how kernels work: Kernels (Filters) in convolutional neural network](https://www.geeksforgeeks.org/deep-learning/kernels-filters-in-convolutional-neural-network/)

The kernel is our feature detector. It holds the weights we want to learn.

| **Element** | **Size** | **Role** |
|-------------|----------|----------|
| **Kernel/Filter** | 3×3 or 5×5 | Contains the learned weights (the pattern it detects). |
| **Feature Map** | Smaller than the input | The output of the convolution: it shows where the pattern was found and how strongly. |
| **neuron** | Scalar (single number) | The single output value computed by applying the filter at one location |

**Visual Intuition**: Imagine the input image is a large spreadsheet, and the kernel is a 3×3 magnifying glass. We place the magnifying glass over a part of the spreadsheet, do some math, and write the single result to a new, smaller spreadsheet called the Feature Map.

## The Mechanics of Convolution (Step-by-Step)

How does the math work?

1. **Placement**: Place the 3×3 kernel over the corresponding 3×3 patch in the input image.
2. **Element-wise Multiplication**: Multiply each of the nine pixel values in the image patch by the corresponding weight in the kernel.
3. **Summation**: Add all nine results together.
4. **Output**: This single number becomes the top-left pixel in the Feature Map.
5. **Sliding (Stride)**: Move the kernel one step (or more) to the right and repeat the entire process until you've covered the whole image.

A simple example: a kernel with a positive number in the center and negative numbers on the sides might learn to detect a bright dot on a dark background.

**Example of element-wise multiplication with a 3×3 edge detection kernel:**
```
Kernel:          Image patch:        Calculation:
[-1  0  1]       [10  10  20]       (-1×10) + (0×10) + (1×20) +
[-2  0  2]   ×   [10  10  20]   =   (-2×10) + (0×10) + (2×20) +
[-1  0  1]       [10  10  20]       (-1×10) + (0×10) + (1×20)
                                   = -10 + 0 + 20 - 20 + 0 + 40 - 10 + 0 + 20
                                   = 40 
```
The value 40 from the convolution calculation represents the strength of the response to the vertical edge filter (the kernel). What you are seeing in the kernel is a Sobel filter. It is designed to detect vertical edges. The zero in the center means the filter looks at differences around the central pixel but doesn’t weigh the center pixel itself. It highlights the differences in pixel intensity in the horizontal direction. Positive and negative weights in the kernel emphasize changes from left to right. 

During today's [Lab 1](https://drive.google.com/file/d/1xTnhoWVUFQbNYxzvcVpp8eNkIKH8RVJZ/view?usp=sharing), you will see the implementation of this step-by-step process and visualize each multiplication.

## Hyperparameters: Stride and Padding

We have two key knobs to turn when setting up a convolution:

- **Stride**: This is the size of the step the kernel takes. A stride of 1 means it moves one pixel at a time. A stride of 2 means it skips a pixel.
  - **Effect**: A larger stride drastically reduces the size of the output Feature Map, saving computation.

Here is what stride looks like in PyTorch:
```
import torch.nn as nn

# Example: 2D convolution with stride 2
conv = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=2)
```

- **Padding**: When the kernel is near the edge, it might not have enough pixels to cover its full 3×3 area. To solve this, we often add zero-value pixels around the border. Padding means adding a border of pixels around the input image. If padding = 1, it means adding 1 pixel all around. 
  - **Effect**: "Same" Padding ensures the output Feature Map has the same height and width as the input image. This is very common.
  - Why do we pad?: Because when you apply a 3×3 filter on a 3×3 image without padding, the filter can only fit in a limited number of positions.
    - Without padding: output size is smaller than input.
    - With padding: output size can be the same as input.

```
Original input (3×3):

a b c
d e f
g h i

After padding with zeros (1 pixel):

0 0 0 0 0
0 a b c 0
0 d e f 0
0 g h i 0
0 0 0 0 0

New size: 5×5
```

Here is what padding looks like in PyTorch:
```
import torch.nn as nn

# Example: 2D convolution with padding of 1 pixel
conv = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
```

## Multiple Channels and Multiple Filters

Images aren't just one layer. They also have depth (or channels): Red, Green, and Blue.

- **Multiple Channels**: If the input image is 28×28×3 (RGB), our kernel must also be 3×3×3 (because the kernel needs to match the depth, i.e., the number of channels of the input image, to properly process all the information simultaneously). We perform the element-wise multiplication across all three depth dimensions and sum all 27 results into a single value for the feature map.

- **Multiple Filters**: A single CNN layer doesn't just use one filter. One filter (kernel) detects one type of pattern in the input, so one filter might detect vertical edges, another might detect corners, and another might detect color gradients, etc. Each filter produces a feature map (a 2D output showing where a pattern was detected by a filter) by sliding over the input and calculating how strongly its pattern matches at each location. The CNN layer might use 32, 64, 128 filters, or even more. When you have, say, 64 filters, you get 64 feature maps. These feature maps are stacked together along the depth dimension to form a new 3D tensor.

In this code:
```
import torch.nn as nn

# Example: 2D convolution with padding of 1 pixel
conv = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
```

Here, Multiple Channels:
- `in_channels=3` means the input has 3 channels.
- This typically corresponds to an RGB image with 3 color channels: Red, Green, and Blue.
- Each input channel represents a separate layer of data that the convolution operates on simultaneously.

Multiple Filters:
- `out_channels=16` means the convolutional layer has 16 filters.
- Each filter is a separate 3D kernel that spans all 3 input channels (depth = 3) with spatial size 3×3 -> the kernel is shaped (3, 3, 3)
- Each filter produces one output feature map by sliding over the input and combining information from all input channels.
- Since there are 16 filters, the layer outputs 16 feature maps stacked together.

---

# Part 3: The Pooling Layer 

## Purpose of Pooling: Downsampling & Robustness

Pooling layers are one of the building blocks of Convolutional Neural Networks. Where Convolutional layers extract features from images, Pooling layers consolidate the features learned by CNNs. Its purpose is to gradually shrink the representation’s spatial dimension to minimize the number of parameters and computations in the network.

After convolution, we have feature maps that are still quite large. The Pooling Layer is the tool we use to downsample or compress this information.

### Max Pooling (The Most Common)

The most common type is Max Pooling.

- **Mechanics**: We use another sliding window, typically 2×2, with a stride of 2.
- **Action**: At each 2×2 stop, we only keep the maximum value and throw away the other three.
- **Meaning**: This is a summary. We are saying: 'We don't need to know exactly where the feature was, just that the strongest activation for that feature occurred somewhere in this 2×2 region.'

```
Original:        Max Pool (2×2):
[1  3  2  4]     [6  4]
[5  6  1  2]  →  [2  3]
[1  2  3  1]     
[0  1  2  3]
```

The standard foundational block of a CNN is: 
**Convolution** (extracts features like edges, textures) -> **Activation** (ReLU: applies non-linearity. ReLU sets negative values to zero, keeping positive ones.) -> **Pooling** (reduces spatial size, retains important info)

We repeat this to build deep networks.

In [Lab 2](https://drive.google.com/file/d/1OaC7SgypikIRN6-zw4lrNyYFhjjdppZZ/view?usp=sharing), we'll build this complete block in PyTorch and see how dimensions change at each step.

<img width="637" alt="max_pooling_initial_feature_map" src="https://git.arts.ac.uk/user-attachments/assets/029c63e0-0807-4dc2-9b93-1a09836dcdcb" />
<img width="634" alt="max_pooling_first_operation" src="https://git.arts.ac.uk/user-attachments/assets/4b287da4-ecdc-4f69-b012-202bb892d0aa" />
<img width="636" alt="max_pooling_second_operation" src="https://git.arts.ac.uk/user-attachments/assets/645d1d02-b3e7-4a92-948d-7cf68db4327a" />
<img width="636" alt="max_pooling_third_operation" src="https://git.arts.ac.uk/user-attachments/assets/0d7f33da-d9ae-4864-a3e8-4a1d5aa1dfe1" />
<img width="637" alt="max_pooling_final_output" src="https://git.arts.ac.uk/user-attachments/assets/1d02ed30-f00a-4feb-9e27-82bc21eadfc2" />

[Image Source: What is a max pooling layer in CNN?](https://www.educative.io/answers/what-is-a-max-pooling-layer-in-cnn)

### Effects of Pooling

**Benefits:**
- Reduces spatial dimensions
- Makes features more robust to small translations
- Reduces computation for subsequent layers

**Trade-off:**
- Loses some spatial information
- Modern architectures sometimes reduce or eliminate pooling. Some modern networks replace pooling with other techniques like strided convolutions to improve learning, but this is an advanced topic.

---

# Part 4: Flattening and Classification 

## The Transition to a Decision

So far, we have taken an image, run it through several stacks of Conv and Pool layers, and now we have a small stack of deep feature maps. These maps contain all the high-level patterns our network has learned to detect. Now, we need to turn these patterns into a final decision—a probability for each class.

| **Operation** | **Input → Output** | **Role** |
|---------------|-------------------|----------|
| **Flatten** | 7×7×64 volume → 3136-long vector | Converts the 3D feature data into a 1D vector suitable for standard fully connected layers. |

## The Fully Connected Head

Once the data is flattened, we feed it into a traditional Fully Connected (FC) Layer: the same type of layer we studied last week. The purpose of this final part, often called the FC Head or Classifier, is to take all the extracted features and learn the final, complex rules for classification.

- The FC Layers combine all the detected patterns (e.g., 'ear' + 'whiskers' + 'fur texture') to produce the final classification.
- The final layer is the size of our classes (e.g., 10 for digits 0-9).
- We apply the Softmax function to the final output to get a clean probability distribution, where all the values sum to 1.0. Softmax is an activation function that converts a vector of raw scores into probabilities that sum to 1, highlighting the most likely class.

Sample PyTorch code would look something like this:

You first flatten the input outside the model (e.g., in your data preprocessing or training loop):

```
# Suppose x is a batch of images or feature maps with shape:
# (batch_size, channels, height, width)

# Flatten all dimensions except batch size
x = x.view(x.size(0), -1)  
# or equivalently:
x = torch.flatten(x, start_dim=1)
```

Then, define and run the rest inside the model:

```
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleMNISTModel(nn.Module):
    def __init__(self):
        super(SimpleMNISTModel, self).__init__()
        
        # Step 1 (already done above): Input feature is flattened to a 1D vector of size 3136.
        # So the first dense layer input size = 3136
        # This usually comes from flattening the output of convolutional layers or other feature extractors.
        
        # Step 2: Dense Layer 1 (128 neurons)
        # The 3,136-dimensional vector is passed through a fully connected layer with 128 neurons.
        # Each neuron computes a weighted sum of inputs plus bias.
        # ReLU activation adds non-linearity, setting negative values to zero.
        self.fc1 = nn.Linear(in_features=3136, out_features=128)
        self.relu1 = nn.ReLU()
        
        # Step 3: Dropout after the first dense layer
        # Dropout is a regularization technique used during training to randomly set some neuron outputs to zero to prevent overfitting in neural networks.
        # Helps the network generalize better by preventing reliance on specific neurons.
        self.dropout1 = nn.Dropout(p=0.5)
        
        # Step 4: Dense Layer 2 (64 neurons)
        # The output of the first dense layer is passed to another dense layer with 64 neurons.
        # Again, ReLU introduces non-linearity.
        self.fc2 = nn.Linear(in_features=128, out_features=64)
        self.relu2 = nn.ReLU()
        
        # Step 5: Dropout after the second dense layer
        self.dropout2 = nn.Dropout(p=0.5)
        
        # Step 6: Output layer (10 classes for MNIST)
        # Final dense layer with 10 neurons, one for each digit class (0 to 9).
        # Outputs raw scores (logits) for each class.
        self.output = nn.Linear(in_features=64, out_features=10)
        
        # Step 7: Softmax will be applied in the forward method or with the loss function
        # Softmax converts the raw scores into probabilities that sum to 1.
        # Each value represents the model’s confidence that the input belongs to that class.
        
    def forward(self, x):
        # x is expected to be flattened already to shape (batch_size, 3136)
        
        x = self.fc1(x)            # Dense Layer 1
        x = self.relu1(x)          # ReLU activation
        x = self.dropout1(x)       # Dropout
        
        x = self.fc2(x)            # Dense Layer 2
        x = self.relu2(x)          # ReLU activation
        x = self.dropout2(x)       # Dropout
        
        x = self.output(x)         # Output layer (logits)
        
        # Softmax is applied outside the model during inference; CrossEntropyLoss applies it internally during training.
        # So if you apply softmax here, it can lead to incorrect results.
        
        return x                   # x contains raw output scores (logits) for each class before softmax


# Example usage:
# model = SimpleMNISTModel()
# input_tensor = torch.randn(32, 3136) # batch of 32 samples, flattened input
# output_logits = model(input_tensor)
# print(output_logits.shape) # Should be (32, 10)
```

## Here is the Standard CNN Architecture

<img width="893" alt="standard_cnn_architecture" src="https://git.arts.ac.uk/user-attachments/assets/802a84a5-e3b7-43be-999b-16d0f07d43ea" />

[Image Source: Convolutional Neural Networks: A Comprehensive Evaluation and Benchmarking of Pooling Layer Variants](https://www.mdpi.com/2073-8994/16/11/1516)

---

# Feature Detection Concept & Hierarchical Learning

## The Hierarchical Feature Learning

This is the most beautiful part of the CNN: it automatically learns to detect features in a sensible hierarchy, starting simple and getting more complex as it goes deeper.

| **Network Depth** | **Feature Learned** |
|-------------------|---------------------|
| **Layer 1 (Shallow)** | **Simple Edges**: Horizontal, vertical, diagonal lines, and color gradients. (The kernels look like basic light/dark patterns.) |
| **Layer 2 (Mid-Depth)** | **Textures & Corners**: Combinations of edges to form checkerboard patterns, simple curves, and corners. |
| **Layer 3+ (Deep)** | **Object Parts**: Combinations of textures and corners to form meaningful parts: eyes, wheels, ears, wings, or specific text fonts. |
| **Final FC Head** | **The Whole Object**: Uses the presence and location of the parts (e.g., two eyes, one nose, one mouth) to make the final prediction (e.g., "Face"). |

### Visualization Example

Imagine recognizing a face:
```
Input Image -> Layer 1: Edges of eyes, nose, mouth
            -> Layer 2: Eye shapes, nose shape  
            -> Layer 3: Face parts with relationships
            -> Layer 4: Complete face detection
            -> FC Head: "This is a face!" (95% confidence)
```

## Connection to OpenCV and Traditional CV

Remember the kernels from Week 3 (Canny edge detector, Gaussian blur)?
- Early CNN filters often behave like edge detectors you saw in OpenCV
- Deeper filters compose those edges into parts and objects
- But the network discovers and tunes them automatically from data

This is important because:
- No manual feature engineering needed
- Learns optimal features for the specific task
- Can discover patterns humans might miss

The CNN is not just solving the image problem. It is automating the process of feature engineering. It learns its own feature detectors (the kernels) through the training process. This is why it is a significant advancement in computer vision.

---

# Quick Cheatsheet on CNN 

1. **CNNs use**:
   - **Sparse Connectivity**: Local receptive fields
   - **Parameter Sharing**: Same detector everywhere
   - **Spatial Hierarchy**: Keep 3D structure until the end

2. **Core Operations**:
   - **Convolution**: Small pattern detector sliding across the image
   - **Pooling**: Downsampling for efficiency and robustness
   - **Flattening + FC**: Final decision making

3. **Architecture Flow**:
   ```
   Image -> [Conv -> ReLU -> Pool] × N -> Flatten -> FC -> Softmax -> Output
   ```

4. **Hierarchical Features**: 
   - Automatically learns from edges → textures → parts → objects
   - No manual feature engineering needed

## Common questions:

1. **"How many layers should my CNN have?"**
   - Start with 2-3 conv blocks for simple tasks (MNIST)
   - Modern networks use 50-150+ layers for complex tasks
   - Depth helps, but diminishing returns + harder to train

2. **"How does the network decide what features to learn?"**
   - Random initialization provides different starting points
   - Backpropagation adjusts kernels to minimize loss
   - Features that help reduce error are reinforced

3. **"What's the difference between kernel, filter, and feature detector?"**
   - All refer to the same thing: the learnable convolution weights
   - "Kernel" = mathematical term
   - "Filter" = signal processing term  
   - "Feature detector" = what it does

4. **"Why is padding important?"**
   - Without padding, feature maps shrink with each layer
   - "Same" padding preserves spatial dimensions
   - Allows deeper networks without losing all spatial info

5. **"Can we visualize what CNNs learn?"**
   - Yes! Feature visualization shows what activates each filter
   - Early layers: edges, colors
   - Deep layers: complex patterns, object parts
   - Helps debug and understand the network
     
---

## Resources for Further Learning

1. **Interactive Visualizations**:
   - [CNN Explainer](https://poloclub.github.io/cnn-explainer/)
   - [ConvNetJS](https://cs.stanford.edu/people/karpathy/convnetjs/)
   - [Feature Visualization](https://distill.pub/2017/feature-visualization/)

2. **Video Resources**:
   - [3Blue1Brown: "But what is a convolution?"](https://www.youtube.com/watch?v=KuXjwB4LzSA)
   - [CS231n Lectures (Stanford)](https://cs231n.stanford.edu/schedule.html)
   - [Andrew Ng's Deep Learning Course](https://www.andrewng.org/courses/)
  
3. **Reading**:
   - [A Deep Dive into the world of Convolutional Neural Networks](https://medium.com/@raushanagrawal/a-deep-dive-into-the-world-of-convolutional-neural-networks-580b122e0031)
