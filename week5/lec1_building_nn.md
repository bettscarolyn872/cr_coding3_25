# Week 5 Lecture
Week 5: 6 October 30, 2025

## Today's Agenda

- Recap: Neural Networks from Week 4
- Lecture 1: Building Your First Neural Network
    - Introduction to PyTorch
    - Tensors and operations
    - Building blocks: layers and activations
- Lecture 2: Training Neural Networks  
    - The training loop
    - Loss functions and optimizers
    - Building and training an MNIST classifier
- Lab Time:
    - Three hands-on notebooks
    - Start Assignment 2

## Quick Recap
### What We Learned in Week 4

**Neural Networks: Inspired by Biological Neurons**

- Neural networks are comprised of neurons

<img width="838" alt="neuron_computation" src="https://git.arts.ac.uk/user-attachments/assets/7e20de02-6408-49cb-b509-ad0e2808c40b" />

[Input → W×input + b → Activation → Output](http://premvishnoi.medium.com/understanding-artificial-neurons-the-core-of-deep-learning-0d05a9286c08)

- They learn patterns from data through layers
- They are basically layers of mathematical operations
- Each layer:
  - Takes input data
  - Multiplies by weights
  - Adds bias
  - Applies activation function
- It's just matrix multiplication + functions!

<img width="996" alt="training_loop" src="https://git.arts.ac.uk/user-attachments/assets/72d3d418-d681-47f4-bb39-08161bc63a18" />

[Image: A training loop consisting of feedforward and backpropagation](https://www.analog.com/en/resources/analog-dialogue/articles/training-convolutional-neural-networks-what-is-machine-learning-part-2.html)


- MNIST: Handwritten digit recognition (28×28 pixels)
- We saw a demo but didn't build one ourselves
- Today: We build and train our own!

<img width="778" alt="neural_network" src="https://git.arts.ac.uk/user-attachments/assets/120f7b51-120b-49c6-ac18-58c218607f3b" />

[Image source](https://medium.com/diaryofawannapreneur/deep-learning-for-computer-vision-for-the-average-person-861661d8aa61)


# Building Your First Neural Network

## Why PyTorch?

_Part of today’s lecture (including some content and images) was adapted from PyTorch Fundamentals. Images without separate credit are from [PyTorch Fundamentals](https://www.learnpytorch.io/00_pytorch_fundamentals/)_

- What is PyTorch?
    - Python library for machine learning
    - Created by Facebook AI Research and is the most used deep learning framework on Paper with Code, a website for tracking machine learning research papers and code repositories attached with them
    - Used by researchers and industry, like Tesla and Meta
    - It takes care of many things, such as GPU acceleration (making your code run faster), behind the scenes

## Working with PyTorch

To start, import torch and check the version we are using:

```
import torch
torch.__version__
```

## Introduction to Tensors

### What are tensors?

- A tensor is a multi-dimensional array of numbers (like a NumPy array) with a shape and a data type
- PyTorch tensors also support GPU (faster) and automatic gradients
- Fundamental building block of machine learning
- They represent data in a numerical way

For example, you could represent an image as a tensor with shape `[3, 224, 224]`, which would mean `[colour_channels, height, width]`, as in the image has `3` colour channels (red, green, blue), a height of `224` pixels, and a width of `224` pixels.

![00-tensor-shape-example-of-image](https://git.arts.ac.uk/user-attachments/assets/6b8fa2a9-71ac-40b6-a240-321689ee2f3c)

- This means that the tensor would have three dimensions, one for `color_channels`, `height`, and `width`.

### Key Tensor Properties:

- **Shape**: the size in each dimension. For example: `torch.zeros(3, 224, 224).shape` → (3, 224, 224)
    - It builds a tensor of zeros with shape (3, 224, 224)
    - First number (3): number of channels (e.g., R, G, B)
    - Second number (224): image height (pixels)
    - Third number (224): image width (pixels)

- **Rank** / `ndim`: number of dimensions (axes). For example, a scalar has rank 0, a vector has rank 1, etc.
- **Dtype**: data type of elements (`float32`, `int64`, `bool`). For example, `torch.tensor([1.0, 2.0]`, `dtype=torch.float32)`
- **Device**: where it lives (`cpu` or `cuda`). For example, `tensor.to('cuda')` or `tensor.device`

- Data type may not seem like a big deal, but this is one of the most common issues you may come across. For example, if one of the tensors is `torch.float32` and the other is `torch.float16`. PyTorch likes tensors to be the same format.
- Similar to devices. If one of your tensors is on the CPU and the other is on the GPU, you will also run into errors since PyTorch likes calculations between tensors to be on the same device.

### Tensor dimensions:

- 0D: Scalar (single number)
- 1D: Vector [1, 2, 3, 4]
- 2D: Matrix (like a grayscale image)
- 3D: Cube (like color image: H×W×Channels)
- 4D: Batch of images

### 0D: Scalar (single number)

- It's a single numeric value (for example: loss = 2.3, learning rate = 0.001)
- Zero-dimensional tensor

If we run

``` # Scalar
scalar = torch.tensor(7)
scalar
```

The output would be `tensor(7)`. This means that although `scalar` is a single number, it's of type `torch.Tensor`. 

To check the dimensions of a tensor, we use the `ndim` attribute: `scalar.ndim`, which will return `0`.

### 1D: Vector 

- A vector is a single-dimensional tensor, but can contain many numbers. As in, you could have a vector `[3, 2]` to describe `[bedrooms, bathrooms]` in your house. Or you could have `[3, 2, 2]` to describe `[bedrooms, bathrooms, car_parks]` in your house.

The important trend here is that a vector is flexible in what it can represent (the same with tensors).

```
# Vector
vector = torch.tensor([7, 3])
vector
```
will output `tensor([7,3])`

If you run
```
# Check the number of dimensions of vector
vector.ndim
```
It will output `1`

**Pro Tip**: You can tell the number of dimensions a tensor in PyTorch has by the number of square brackets on the outside (`[`), and you only need to count one side. Another important concept for tensors is their `shape` attribute. The shape tells you how the elements inside them are arranged. 

If you run
```
# Check shape of vector
vector.shape
```
You will get `torch.Size([2])`, which means our vector has a shape of `([2])`. This is because of the two elements we placed inside the square brackets of `([7, 3])`.

### 2D: Matrix (like a grayscale image)

- Why is a matrix like a grayscale image? Because when you add color channels, you need more dimensions. For example, an RGB image is 3 x H x W or H x W x 3, but each channel is still a 2D matrix of intensities. 

```
# Matrix
MATRIX = torch.tensor([[7, 8], 
                       [9, 10]])
MATRIX
```

will return:
```
tensor([[ 7,  8],
        [ 9, 10]])
```

Matrices are as flexible as vectors, except they've got an extra dimension. 
```
# Check number of dimensions
MATRIX.ndim
```
which will return `2`.

When you run
```
MATRIX.shape
```
You will get output `torch.Size([2, 2])`, because `MATRIX` is two elements deep and two elements wide.

#### Dimensions vs. Shape

- This is easy to mix up
- **Dimensions (or rank)** = how many axes a tensor has (how many numbers you need to pick an element).
    - Examples: 0D scalar (no axes), 1D vector (one axis), 2D matrix/image (two axes), 3D image with channels (three axes), 4D batch of images (four axes).
- **Shape** = the size along each axis
    - A list or tuple of numbers that tells you how many elements are on each axis.
    - Example: shape (3, 224, 224) means three axes (so the tensor is 3D) and each axis has lengths 3, 224, and 224, respectively.

- Think about it this way:
    - **Dimensions** = how many drawers are in a filing cabinet (how many levels of indexing).
    - **Shape** = how many files are in each drawer (the size of each level).
    - If you have 3 drawers, with 224 folders in each, and 224 pages per folder → dimensions = 3, shape = (3, 224, 224).

Here is another example:
- Scalar: dimensions = 0, shape = () or torch.Size([])
- Vector with 4 numbers: dimensions = 1, shape = (4,)
- Grayscale image 28×28: dimensions = 2, shape = (28, 28)
- RGB image 3×224×224 (C, H, W): dimensions = 3, shape = (3, 224, 224)
- Batch of 32 RGB images: dimensions = 4, shape = (32, 3, 224, 224)

Quick code checks in PyTorch:
```
t = torch.zeros(3, 224, 224)
t.dim()    # returns 3  (number of dimensions)
t.shape    # returns torch.Size([3, 224, 224])  (sizes along each dimension)
len(t.shape)  # also 3
```

Understanding shape is important because shape errors are one of the most common errors. Because much of deep learning is multiplying and performing operations on matrices, and matrices have a strict rule about what shapes and sizes can be combined, one of the most common errors you'll run into in deep learning is shape mismatches. You will fix this with a **transpose** (switch the dimensions of a given tensor). More on this later.

Quick takeaway: Dimensions = how many axes. Shape = how big each axis is (a list of sizes for those axes).

### 3D & 4D Tensors

- You will also see 3D and 4D tensors in real machine learning workflows (images, batches, videos):
- Why 3D tensors matter:
    - Example: a single RGB image in PyTorch is 3D with shape (C, H, W), e.g., (3, 224, 224).
    - Interpretation: three axes are Channel, Height, and Width.
    - Each element is a pixel intensity for a given channel.
    - Other 3D uses: time series with features across time (features × time × 1), or stacked grayscale images (channels > 1).
- Why 4D tensors matter:
    - Example: a batch of RGB images is 4D with shape (B, C, H, W) — e.g., (32, 3, 224, 224).
    - Interpretation: four axes are Batch, Channel, Height, and Width. Models process batches for efficiency and gradient stability.
    - Other 4D uses: video frames can be (B, T, C, H, W) but sometimes reshaped, or CNN feature maps across batches.

Short code examples you can run
```
import torch
img = torch.zeros(3, 224, 224)        # single RGB image (C, H, W)
batch = torch.zeros(32, 3, 224, 224)  # batch of 32 images (B, C, H, W)
print(img.shape, img.dim())   # (3,224,224) 3
print(batch.shape, batch.dim()) # (32,3,224,224) 4
```
- Quick indexing examples:
    - First channel of an image: `img[0]`
    - First image in batch: `batch[0]`
    - Pixel at `(y=10,x=20)` of first image: `batch[0, :, 10, 20]` (all channels at that pixel)

- One or two caveats to mention:
    - PyTorch convention: channels-first = (C, H, W); some libraries (and image files) use (H, W, C)
    - Be explicit about ordering.
    - Batches are for efficiency: training uses batches (not single images) to compute gradients and make training stable.

## Creating Tensors

### Working with Tensors in PyTorch

You can create a tensor like this:
```
# Tensor
TENSOR = torch.tensor([[[1, 2, 3],
                        [3, 6, 9],
                        [2, 4, 5]]])
TENSOR
```
which will output
```
tensor([[[1, 2, 3],
         [3, 6, 9],
         [2, 4, 5]]])
```
Tensors can represent almost anything; for example, the tensor we just created could be the sales numbers of a steak and almond butter. 


<img width="683" alt="steak" src="https://git.arts.ac.uk/user-attachments/assets/73c974ac-a384-4196-93dc-5e363c1bc4ad" />



To check its dimensions and shape:
```
# Check number of dimensions for TENSOR
TENSOR.ndim # will return 3

# Check shape of TENSOR
TENSOR.shape # will return torch.Size([1, 3, 3])
```

The dimensions go outer to inner -> this means that there is 1 dimension of 3 by 3.


<img width="1253" alt="00-pytorch-different-tensor-dimensions" src="https://git.arts.ac.uk/user-attachments/assets/1c7846ab-1f4e-4211-80a2-8edc0ba19193" />


You might've noticed me using lowercase letters for `scalar` and `vector` and uppercase letters for `MATRIX` and `TENSOR`. This was on purpose. In practice, you'll often see scalars and vectors denoted as lowercase letters such as `y` or `a`. And matrices and tensors are denoted as uppercase letters such as `X` or `W`.

You also might notice the names matrix and tensor used interchangeably. This is common. Since in PyTorch, you're often dealing with torch. Tensors (hence the tensor name), however, the shape and dimensions of what's inside will dictate what it actually is.

### Tensor Manipulations (Tensor Operations)

In deep learning, data (images, text, video, audio, protein structures, etc) gets represented as tensors.

A model learns by investigating those tensors and performing a series of operations (could be 1,000,000s+) on tensors to create a representation of the patterns in the input data.

These operations can combine:

- Addition
- Subtraction
- Multiplication (element-wise)
- Division
- Matrix multiplication

For example:

```
# Create tensors
a = torch.tensor([1, 2, 3, 4])          # From list
b = torch.zeros(3, 3)                   # 3×3 zeros
c = torch.rand(2, 3, 4)                 # Random 3D tensor

# Basic operations
d = a + 10                              # Add scalar
e = a * b                               # Element-wise multiply
f = torch.matmul(b, b)                  # Matrix multiply

# Reshape
g = a.reshape(2, 2)                     # Change dimensions
```

### Zeros and ones

Sometimes you will want to fill tensors with zeros or ones for masking (like masking some of the values in one tensor with zeros to let a model know not to learn them).

For example, here is creating a tensor full of zeros with `torch.zeros()`. Note the `size` parameter comes into play.

```
# Create a tensor of all zeros
zeros = torch.zeros(size=(3, 4))
zeros, zeros.dtype
```
will return 
```
(tensor([[0., 0., 0., 0.],
         [0., 0., 0., 0.],
         [0., 0., 0., 0.]]),
 torch.float32)
```
We can do the same to create a tensor of all ones, except using `torch.ones()` instead.

```
# Create a tensor of all ones
ones = torch.ones(size=(3, 4))
ones, ones.dtype
```
will return
```
(tensor([[1., 1., 1., 1.],
         [1., 1., 1., 1.],
         [1., 1., 1., 1.]]),
 torch.float32)
```



### To recap tensors:

<img width="689" alt="tensor_summary" src="https://git.arts.ac.uk/user-attachments/assets/d7105a38-77bd-43a6-8be7-4c96b033df3e" />

![00-scalar-vector-matrix-tensor](https://git.arts.ac.uk/user-attachments/assets/93082e39-5023-468d-8227-680776d08079)

-----

## Automatic Differentiation (The Magic of `Autograd`)

### Why Gradients? The Learning Compass

In Week 4, we saw that a neural network learns by adjusting its **weights** and **biases**. But how does it know *which way* to adjust them?

The answer is **Gradients**.

  * **Analogy:** If your Loss Function tells you how far away you are from the target (the **error**), the Gradient tells you the **direction** and **steepness** of the path back toward the minimum error. It's the compass pointing downhill.
  * **The Problem:** Calculating these gradients by hand requires complex calculus (differentiation) for every single weight in a network that can have millions of parameters.
  * **The Solution: PyTorch's `Autograd`:** This is PyTorch’s superpower. It automatically tracks every operation performed on a tensor and calculates the corresponding gradients for you.

The *gradient* and *gradient descent* are not the same, but they are critically dependent on each other.

| Term            | What it Is                                                                                                                                               | Analogy                                                        |
|-----------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------|
| Gradient        | A vector (compass) that points in the direction of the steepest increase in the loss function. When training, we move in the opposite direction of the gradient to find the lowest loss. The **calculated value**, a vector of derivatives, that tells you the direction of steepest _increase_ in the loss function. It is the required piece of information needed to know how to adjust the weights `(w)` and biases `(b)`. | The **map and compass** tell you which way is up the hill.    |
| Gradient Descent| The **algorithm** (method) that uses the gradient to iteratively adjust the parameters (weights and biases) to minimize the loss. In ML, this is the update mechanism that uses the gradient to determine the size and direction of the **step** taken.                           | The **act of walking downhill**, taking steps based on the gradient's direction and a chosen step size **(learning rate)**. |


In short, the Gradient is the math (the calculated direction), and Gradient Descent is the method (the process of taking steps).

When training neural networks, the most frequently used algorithm is backpropagation. In this algorithm, parameters (model weights) are adjusted according to the gradient of the loss function with respect to the given parameter.

To compute those gradients, PyTorch has a built-in differentiation engine called `torch.autograd`. It supports automatic computation of the gradient for any computational graph.

Consider the simplest one-layer neural network, with input `x`, parameters `w` and `b`, and some loss function. It can be defined in PyTorch in the following manner:

```
import torch

x = torch.ones(5)  # input tensor
y = torch.zeros(3)  # expected output
w = torch.randn(5, 3, requires_grad=True)
b = torch.randn(3, requires_grad=True)
z = torch.matmul(x, w)+b
loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y)
```

<img width="808" alt="tensors_functions_computational graph" src="https://git.arts.ac.uk/user-attachments/assets/36d6a24c-98ee-498c-897a-38e5c099930d" />

[Image credit: PyTorch documentation on Automatic Differentiation with `torch.autograd`](https://docs.pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.html?__hstc=196695859.2f3f33a24b44870ec4a577029c49e44b.1757462400127.1757462400128.1757462400129.1&__hssc=196695859.1.1757462400130&__hsfp=2400812917)

In this graphic, you are seeing that the parameters `$w$` and `$b$` are being applied to the input, and the node **CE** stands for a crucial component in the training process.

Here is a clarification of what the computational graph means, specifically for training a neural network:

The graph models the mathematical steps required to get from the input data to the final error (loss).

### 1. The Forward Pass (Input to Output)

This sequence models a single **Linear Layer** operation:

* **$x$ (Input Tensor):** This represents your input data (e.g., a batch of flattened images).
* **$w$ (Weights) & $b$ (Bias):** These are the **Parameters** of the network—the values the model learns and needs to **optimize**.
* **$*$ (Multiplication):** The input $x$ is multiplied by the weight $w$ (specifically, this is matrix multiplication: $x \times w$).
* **$+$ (Addition):** The result of the multiplication is added to the bias $b$.
* **$z$ (Output/Logits):** This is the final prediction of the linear layer before it is compared against the true answer.

### 2. The Loss Calculation

The second part of the graph compares the prediction to the truth:

* **$y$ (True Label):** This is the expected output (the correct answer for the input $x$).
* **CE (Cross-Entropy):** This stands for **Cross-Entropy Loss**.
    * Cross‑entropy is a loss function that measures how different two probability distributions are. In machine learning, it quantifies the distance between the model’s predicted probability distribution over classes and the true distribution (usually a one‑hot vector for the correct class). Lower cross‑entropy means the model’s predicted probabilities are closer to the true labels.
    * **Cross-Entropy Loss** is the standard **Loss Function** used in **classification** tasks (like MNIST). It measures the difference between the model's output ($z$) and the true label ($y$).
* **$\text{loss}$:** This is the final **scalar value** that quantifies how wrong the model's prediction was. The entire purpose of training is to minimize this value.

In summary, the graph shows:

$$\text{Loss} = \text{CrossEntropy}(\; (x \times w + b)\;,\; y\;)$$

And remember what we discussed last week? Once the loss is calculated, the information doesn't just stop there; it's used to initiate the backward pass, where backpropagation occurs to prepare for the model update.

This completes the full cycle of the Training Loop.


### Autograd in Action (Live Demo from Lab 1)

To use Autograd, you only need to tell PyTorch which tensors are **learnable parameters** by setting `requires_grad=True`.

**Step 1: Setting up the parameter**

We start with a simple number, $x$, that we want to track.

```python
import torch

# This is our parameter. PyTorch will track every operation involving it.
x = torch.tensor(3.0, requires_grad=True) 

print(f"Initial X: {x}")
# Initial X: 3.0
```

**Step 2: Performing a calculation**

Let's calculate $y = x^2$. PyTorch records this operation in the background.

```python
# Our calculation (the "forward pass")
y = x ** 2
print(f"Result Y (X^2): {y}")
# Result Y (X^2): 9.0
```

**Step 3: Finding the Gradient**

To find how $y$ changes with respect to $x$ (the derivative, $\frac{dy}{dx}$), we call `.backward()`.

```python
# This is the "backward pass" - calculating the gradient
y.backward()

# The gradient is stored in x.grad
print(f"Gradient (dy/dx): {x.grad}")
# Gradient (dy/dx): 6.0
```

> **Quick Check:** Mathematically, the derivative of $x^2$ is $2x$. Since $x=3$, the gradient is $2 \times 3 = 6$. PyTorch did the calculus for us\!

This process: creating tensors, running calculations, and calling `.backward()` is the core of how our neural network will learn. We'll revisit this in Lecture 2 when we build the full **Training Loop**.

-----

## Building Blocks: Layers, Activations, and `nn.Sequential`

### Linear Layers: The Core Math

A single **Linear Layer** (also known as a Fully-Connected Layer or Dense Layer) is the fundamental unit of most neural networks. Its job is simple: **take a set of input vectors and transform them into a new set of output vectors of potentially different dimensionality.**

The layer does this by making use of a weight matrix and an optional bias term, and brings about a linear transformation of the form 

* **Input ($\text{X}$):** Your data (e.g., a flattened image vector).
* **Weights ($\text{W}$):** The **learnable parameters** PyTorch adjusts to find patterns. This is just a matrix.
* **Bias ($\text{b}$):** Another **learnable parameter** that slightly shifts the output line, adding flexibility.

Where x is the input vector and W is the weight matrix multiplied by it, b is an optional bias term that is added to the result of this matrix multiplication to yield the linearly transformed vector y, and hence the name linear layer.

<img width="581" alt="linear_layer" src="https://git.arts.ac.uk/user-attachments/assets/e84b68d1-f403-4556-8a8e-e948649833c0" />


[Image credit](https://www.scaler.com/topics/pytorch/pytorch-linear-pytorch-embedding/)

The linear layer applies a linear transformation to the input vector using a weight matrix (and optional bias); each neuron computes one component of the output.

As a result, all possible connection-layer-to-layer connections are present; hence, every input of the input vector influences every output of the output vector.

Note that in the above figure, although the output is represented as a single node, it only means that the output vector can be of any dimensionality, the same or different from the dimensionality of the input vector.

Also, using the linear layer to construct deep neural networks is needed more than using the layer to transform the input vector. We need to add some form of activation to the $y$ vector by adding an activation layer.

This is needed to introduce non-linearity in the network structure to enable to network to learn non-linear relationships. But for this, the network shall transform into one big linear transformation, and it will not be possible to model non-linear relationships with such a network.

**Demo: Seeing the Math in Action** (Live Demo from Lab 2, Part 2)

We can simulate this operation manually using basic tensor multiplication:

```
# We want 3 input features -> 2 output features
input_features = 3
output_features = 2

# 1. Create weights (3x2 matrix) and bias (1x2 vector)
weights = torch.randn(input_features, output_features)
bias = torch.randn(output_features)

# 2. Create input data (4 samples, 3 features each)
input_data = torch.randn(4, input_features) 

# 3. Apply the linear transformation
output = input_data @ weights + bias

print(f"Input Shape: {input_data.shape}")   # (4, 3)
print(f"Output Shape: {output.shape}")      # (4, 2)
```
This is the same operation that `nn.Linear(in_features=3, out_features=2)` does internally (though PyTorch stores weight with shape `(out_features, in_features)` and applies it accordingly).

**PyTorch's Official Layer (`nn.Linear`)**

Since doing this manually is tedious, PyTorch provides the super clean `nn.Linear` class, which handles the weight initialization, bias handling, and all the gradient tracking for us\!

```python
import torch.nn as nn

# The clean way: nn.Linear(input_size, output_size)
layer = nn.Linear(3, 2) 

# Use it like a function:
input_data = torch.randn(4, 3)
output = layer(input_data)

print(f"PyTorch Output Shape: {output.shape}") # (4, 2)
```

### Activation Functions: Adding Non-Linearity

If we only used Linear Layers, the entire neural network would just be one giant, complex linear equation. It couldn't learn curves, boundaries, or anything complex, no matter how many layers you stacked\!

There are linear activation functions. However, in deep learning, we use non-linear activation functions to generate complex results (patterns).

Without these non-linear functions (like ReLU), stacking multiple layers would still only result in a single linear function, making it impossible for the network to learn the complex curves and relationships found in real-world data.

**Activation Functions** are simple, non-linear mathematical functions placed between layers to introduce the necessary complexity.

| Function | What it Does | Why we use it |
| :--- | :--- | :--- |
| **ReLU** (Rectified Linear Unit) | Changes all **negative** values to **zero**, leaving positive values unchanged. | It's computationally simple and highly effective for deep networks. **Our default choice.** |
| **Sigmoid** | "Squashes" any value into a range between **0 and 1**. | Useful in the **final output layer** for binary classification (is it $X$ or not $X$?). |
| **Tanh** (Hyperbolic Tangent) | "Squashes" any value into a range between **-1 and 1**. | Similar to Sigmoid, but centered at zero, which can sometimes help training. |


<img width="747" alt="derivative_of_activation_functions" src="https://git.arts.ac.uk/user-attachments/assets/9a797e81-e2a4-4580-b66c-e6af646af654" />

[Image resource and also a great read: Activation Functions in Neural Networks](https://medium.com/data-science/activation-functions-neural-networks-1cbd9f8d91d6)

### Softmax

Hidden-layer activations (ReLU, Sigmoid, Tanh) are applied elementwise to each neuron individually and independently to introduce nonlinearity inside the network. 

By contrast, Softmax is _not_ an elementwise hidden activation. It operates on the whole output vector and turns raw scores (logits) into a probability distribution over classes. In practice, you use ReLU/Tanh inside the network, then use Softmax only at the final layer when you need class probabilities.

Use ReLU/Tanh inside the model to learn features; use Softmax at the end to convert final scores into class probabilities. Example:

```
# hidden layers use elementwise activations
hidden = torch.relu(linear1(x))        # elementwise ReLU
hidden = torch.tanh(linear2(hidden))   # elementwise Tanh

# final linear gives raw class scores (logits)
logits = final_linear(hidden)

# softmax converts logits to probabilities (used for prediction/interpretation)
probs = torch.softmax(logits, dim=1)
```

#### Why and When We Need Softmax

You need Softmax when your model is performing Multi-Class Classification, which means assigning an input to exactly one of three or more possible categories.

**Goal: Mutually Exclusive Probabilities**

1. Output Format: Softmax converts the raw output scores (called logits) from the final linear layer into a probability distribution.
2. Constraint: It ensures that all probabilities are between 0 and 1, and their sum equals 100% (or 1.0).
3. Example (MNIST): For the MNIST model, the last layer produces 10 scores. Softmax converts these into 10 probabilities where the highest probability is the model's prediction. (e.g., $95\%$ chance of being a '7', $2\%$ chance of being a '1', etc.)

#### When We Do Not Need Softmax

You should not use Softmax when the output is not a mutually exclusive probability distribution. In these cases, you typically use a Linear or Sigmoid activation function in the final layer.

1. **Regression (Predicting a Continuous Value)**
    - Goal: To predict a raw, unconstrained number (e.g., price, temperature, or age).
    - Final Activation: Linear Activation (or no activation).
    - Why Not Softmax? Softmax constrains the output between 0 and 1 (and forces the outputs to sum to 1). If you need to predict a price of $500,000, Softmax would ruin the result.

2. **Multi-Label Classification (Predicting Multiple Categories)**
    - Goal: To assign an input to **zero**, **one**, or **multiple** categories simultaneously.
    - Example: Classifying an image that might contain a "Cat" **AND** a "Dog" **AND** a "Horse."
    - Final Activation: Sigmoid.
    - Why Not Softmax? Softmax forces the probabilities to sum to 1.0. If a picture has a cat (90%) and a horse (80%), Softmax would falsely push the probabilities down for all labels to make the sum 1.0. Sigmoid treats each label independently.

### Stacking Layers with `nn.Sequential`

To build a complete neural network, we stack Linear layers and activation functions like building blocks.

The easiest way to do this in PyTorch is with `nn.Sequential`. This container passes the input tensor through each module in order, one after the other.

Example (MNIST flattened input):

```
import torch.nn as nn

model = nn.Sequential(
    nn.Linear(784, 128),    # Input: 28×28 = 784 flattened pixels → 128 hidden units
    nn.ReLU(),              # Adds nonlinearity
    nn.Linear(128, 64),     # Hidden layer
    nn.ReLU(),              # Adds nonlinearity
    nn.Linear(64, 10)       # Output layer: 10 class scores (digits 0–9)
)
```
You can pick any positive integer, but round numbers (32/64/128) are faster on real hardware, make scaling easier to reason about, and are the common practical choice. A random number like 59 will work, but offers no practical advantage. Models will still train. While in doubt, pick a standard size (32, 64, 128...), then tune based on performance. 

This `nn.Sequential` model is now a single object that takes your input tensor, pushes it through all the layers and activations, and spits out the final prediction\!
