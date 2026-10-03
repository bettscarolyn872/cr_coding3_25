# Quick Recap So Far

## Machine Learning & Neural Networks Fundamentals

### What is Machine Learning?

**Machine Learning** is teaching computers to learn patterns from data, rather than explicitly programming rules.

| Traditional Programming | Machine Learning |
|:---|:---|
| **Input:** Data + Rules | **Input:** Data + Expected Output |
| **Output:** Answer | **Output:** Rules (Model) |
| Example: `if pixel > 128: bright` | Example: Learn what makes a cat photo |

### What are Neural Networks?

**Neural Networks** are computing systems inspired by biological brains, consisting of:

1. **Neurons (Nodes)**: Basic units that receive input, process it, and produce output
2. **Connections (Weights)**: Links between neurons with learnable strengths
3. **Layers**: Groups of neurons processing information at different levels
```
Input → [Hidden Layer 1] → [Hidden Layer 2] → Output
Photo →    [Features]    →   [Patterns]    → "Cat"
```
Each layer processes the information at a higher level of abstraction, helping the network understand and classify the input.

### The Learning Process: How Neural Networks Improve

Neural networks learn through iterative optimization using **gradient descent**:

1. **Initialize**: Start with random weights (the network knows nothing).
2. **Predict (Forward Pass)**: Feed data forward through the network to generate predictions.
3. **Measure Error**: Compare predictions with true answers using a loss function to quantify the error.
4. **Calculate Gradients (Backward Pass)**: Use backpropagation to compute gradients—how much each weight contributes to the error.
5. **Adjust Weights (Gradient Descent)**: Update the weights by moving them in the direction that reduces the error, using the gradients.
6. **Repeat**: Perform this cycle thousands of times with different examples until the network learns to make accurate predictions.

![neural network](https://git.arts.ac.uk/user-attachments/assets/e52d1f01-3242-4aeb-acfa-1034497564b0)

[Image Source: https://www.analog.com/en/resources/analog-dialogue/articles/training-convolutional-neural-networks-what-is-machine-learning-part-2.html](https://www.analog.com/en/resources/analog-dialogue/articles/training-convolutional-neural-networks-what-is-machine-learning-part-2.html)

Think of it like learning to throw darts:
- First throw: Random (miss badly)
- Measure: How far from the bullseye?
- Adjust: Change angle/force based on error
- Practice: Repeat until accurate

A **gradient** is a vector of partial derivatives that indicates the direction and rate of the steepest increase of a function.

In the context of neural networks:

- The function is the **loss function**, which measures how far the network’s predictions are from the true labels.
- The gradient tells us **how much** and in **which direction** each weight in the network should be changed to reduce the loss.
- Each element of the gradient corresponds to one weight and shows the sensitivity of the loss to that weight.
- Think of it like a map showing the steepest uphill slope; since we want to minimize loss, we move against the gradient (downhill) to find better weights.

What does **“direction”** mean here?
- The gradient is a vector, meaning it has both magnitude (how big the change is) and direction (which way to move).
- Each component of the gradient corresponds to a specific weight in the network.
- The direction for each weight tells you whether that weight should be increased or decreased to reduce the loss.

During the **Forward Pass (step 2)**, this is what is happening:

![Deep-networks-graphic](https://git.arts.ac.uk/user-attachments/assets/19635916-7ebe-4423-a10e-5d4073cfc15a)

[Image Source: https://www.quantamagazine.org/deep-neural-networks-help-to-explain-living-brains-20201028/](https://www.quantamagazine.org/deep-neural-networks-help-to-explain-living-brains-20201028/)

### Key Concepts We've Covered:

- **Supervised Learning**: Learning from labeled examples ("this is a cat")
- **Unsupervised Learning**: No labeled examples are given to the model. The model learns patterns or structure from data without any labels or explicit outputs.
- **Semi-Supervised Learning**: Only some labeled examples are given to the model. The model learns from a combination of labeled and unlabeled data, using the labeled examples to guide learning.
- **Reinforcement Learning**: The model learns by interacting with an environment and gets rewards or penalties based on its actions to maximize cumulative reward.
- **Training Data**: Examples used to teach the network
- **Validation Data**: Separate examples to check if learning generalizes
- **Overfitting**: Memorizing training data instead of learning patterns
- **Epochs**: Complete passes through all training data
- **Batch Size**: How many examples to process before updating weights

---

## Week 5: The Magic of Non-Linearity (The Why Behind Activation Functions like ReLU)

During Week 5, we saw a linear model and a non-linear model. 

<img width="564" alt="linear vs nonlinear activation function" src="https://git.arts.ac.uk/user-attachments/assets/1e0079be-50a6-47d2-a99c-0a0939ada5cd" />

[Image Source: https://studymachinelearning.com/activation-functions-in-neural-network/](https://studymachinelearning.com/activation-functions-in-neural-network/)

On the left, the linear model separates the two groups with a straight line. This works only when the data can be divided by a simple boundary. However, many real-world problems, like images, have complex patterns that don’t fit this simple rule. 

On the right, the non-linear model uses a curved boundary to enclose one group, showing how it can capture more complicated relationships. This flexibility is why non-linearity is essential in deep neural networks for solving complex tasks.

**Non-linearity is the key reason a deep neural network can solve complex computer vision problems.** Without it, the network is fundamentally limited to only learning straight-line relationships, which cannot capture the complex patterns found in real-world images.

---

## 1. The Core Problem: Separating Data

Imagine you are trying to separate data points in a 2D space:

| Model Type | What it does | Limitation |
| :--- | :--- | :--- |
| **Linear Model (No Activation)** | It can only draw a **straight line** to separate your data. | It cannot learn boundaries that curve or wrap around data clusters. No matter how many linear layers you stack, you only ever perform one long linear operation. |
| **Non-Linear Model (with ReLU, etc.)** | It can draw a **curved, complex boundary** to separate your data. | This is what allows us to distinguish objects like a 'cat' from a 'dog' in an image. **The activation function introduces the necessary *bend* in the line** and makes it non-linear. |

---

## 2. The Equation: Where Non-Linearity Fits In

A single layer in a neural network performs two distinct steps:

### A. Linear Step (The Calculation)

The input ($X$) is multiplied by the layer's weights ($W$), and a bias ($b$) is added.

$$Z = (X \cdot W) + b$$

### B. Non-Linear Step (The Magic)

We pass the result ($Z$) through an **Activation Function** ($\sigma$, e.g., ReLU).

$$A = \sigma(Z)$$

The activation function breaks the linear constraint, giving the network the ability to learn complex, non-straight relationships. This allows each stacked layer to learn a new, independent pattern.

---

## 3. The Activation Function: ReLU in Computer Vision

The Rectified Linear Unit (**ReLU**) is the workhorse of most modern deep learning models, including CNNs. It efficiently introduces non-linearity and helps networks train effectively.

<img width="572" alt="ReLU" src="https://git.arts.ac.uk/user-attachments/assets/4b5f7c50-f227-4127-b2e5-d0f39637fa58" />

[Imaged credit: https://www.geeksforgeeks.org/deep-learning/relu-activation-function-in-deep-learning/](https://www.geeksforgeeks.org/deep-learning/relu-activation-function-in-deep-learning/)


$$
\text{ReLU}(z) = \max(0, z)
$$

| Property | Explanation | Importance for Computer Vision |
| :--- | :--- | :--- |
| **The "Bend"** | The sudden change in the function at zero (from flat to diagonal) is the **non-linearity**. | It allows the network to learn hierarchical features: one layer detects simple edges, the next combines them into corners, the next into eyes or noses. |
| **Sparsity** | It forces any negative input to become **zero**, effectively switching off that "neuron" for that specific input. | This makes the network more computationally efficient and often speeds up training by focusing the learning on positive, active features. |
| **Efficiency** | It's incredibly fast to compute (just a simple comparison to zero). | Critical for performance when running large CNNs on many high-resolution images. |

As we transition into **Convolutional Neural Networks (CNNs)**, remember that their power is built on this simple concept:

1.  **Convolutional Layers** are smart feature extractors (a form of linear operation). The convolution operation itself is linear because it involves applying a set of weights (filters or kernels) to the input through weighted sums (dot products) at different spatial locations.
2.  **ReLU Activation** (after the convolution) makes the extracted features useful by allowing the network to combine them non-linearly.

This methodical combination is what enables CNNs to perform the powerful object recognition we need for creative robotics.

---

# PyTorch & NN Cheatsheet: Vocabulary, Workflow, & Code

This cheat sheet summarizes the essential building blocks and processes you need to understand for building and training neural networks in PyTorch.

---

## 1. Core Vocabulary & Building Blocks

| Term | Definition | PyTorch Code / Role |
| :--- | :--- | :--- |
| **Tensor** | The fundamental data structure in PyTorch. It's like a NumPy array, but is optimized for deep learning (runs on GPU, supports autograd). | `torch.Tensor` or `torch.rand()`, `torch.zeros()`. |
| **Weight ($W$)** | Numerical values in the network that are **learned** during training. They determine the strength of the connection between neurons. | Automatically created within `nn.Linear()` or `nn.Conv2d()`. |
| **Bias ($b$)** | An extra numerical value added to the linear transformation. It allows the model to shift the activation function (the decision boundary). | Automatically created within `nn.Linear()` or `nn.Conv2d()`. |
| **Layer** | A collection of interconnected neurons that performs a mathematical transformation on the input data. | `nn.Linear()` (for fully connected layers) or `nn.Sequential()`. |
| **Activation Function ($\sigma$)** | A non-linear function applied after a linear layer (e.g., **ReLU**). It gives the network the ability to learn complex, curved patterns. | `nn.ReLU()` (most common), `nn.Sigmoid()`, `nn.Softmax()`. |
| **`nn.Module`** | The base class for all neural network modules. You subclass it to create custom models, or use pre-built modules like `nn.Linear`. | All layers and models inherit from this. |

### Code Example: Building a Simple Neural Network

```
import torch.nn as nn

# Model for MNIST (784 features in, 10 classes out)
# Remember MNIST has digits 0 through 9, so 10 categories -> 10 output classes in total
# Since each MNIST image is 28×28 pixels, flattening it into a vector gives 28 × 28 = 784 features.

model = nn.Sequential(
    # 1. Linear Layer (input features -> 128 hidden units)
    nn.Linear(in_features=784, out_features=128), 
    
    # 2. Non-linear Activation (Required for learning complexity!)
    nn.ReLU(),
    
    # 3. Another Linear Layer (hidden units -> 10 output classes)
    nn.Linear(in_features=128, out_features=10)
)

print(model)
```

- 128 hidden units: This is an arbitrary choice you make when designing the network. It determines how many neurons are in the hidden layer and affects model capacity and complexity. You can adjust this number based on experimentation or resource constraints.
- `in_features=128` in the 3rd linear layer: This matches the number of outputs from the previous layer (the hidden units), so the input size of this layer must be the same as the number of hidden units output by the previous layer.

---

## 2. The PyTorch Workflow (The 6 Steps of Any Project)


![pytorch_workflow](https://git.arts.ac.uk/user-attachments/assets/25e6b1ae-7d28-4bf2-9042-be1e9a5ea9f8)

[Image Credit: PyTorch Workflow Fundamentals](https://www.learnpytorch.io/01_pytorch_workflow/)


Every PyTorch project, from a simple linear model to a complex CNN, follows this standard structure:

1.  **Get Data Ready:** Load, transform, and package data into **DataLoaders** (which handle batching).
2.  **Build Model:** Define the network's structure using `nn.Sequential` or subclassing `nn.Module`.
3.  **Define Loss Function:** Choose a function to measure the error. Common choices: **`nn.CrossEntropyLoss()`** (for multiclass classification like MNIST).
4.  **Define Optimizer (types of gradient descent algorithms):** Choose an algorithm to update weights based on gradients. Common choices: **`optim.SGD`** or **`optim.Adam`**.
5.  **Training Loop:** Execute the **4 Steps of Learning** (Forward, Loss, Backward, Update) over multiple **Epochs**.
6.  **Make Predictions/Evaluate:** Test the trained model's performance on unseen data.

---

## 3. The PyTorch Training Loop (The 4 Steps of Learning)

A neural network learns by repeating this cycle thousands of times, enabling it to adjust its internal weights and biases:

| Step | What Happens | Why It Matters |
| :--- | :--- | :--- |
| **1. Forward Pass** | Input data flows through the network to generate a **Prediction** (e.g., "It's a 7"). | This is the *use* of the current, untrained model. |
| **2. Calculate Loss** | The **Prediction** is compared to the true label (**Target**) to measure the **Error** (Loss). | This tells us *how wrong* the model was. The goal is to minimize this number. |
| **3. Backward Pass** | The **Backpropagation** algorithm calculates the **gradients** (slopes) of the loss with respect to every weight. | This tells the network the *direction* to adjust each weight to reduce the error. |
| **4. Update Weights** | The **Optimizer** (e.g., SGD, Adam) uses the gradients to adjust the network's weights and biases. | This is the actual *learning* step, improving the model for the next cycle. |

### Code Example: The Core Training Loop (Inside one Epoch)

```
# Assume 'model', 'loss_fn', and 'optimizer' are defined
# Assume 'dataloader' is loaded with your training data

for batch_X, batch_y in dataloader:
    # --- Step 1 & 2: Forward Pass & Calculate Loss ---
    # 1. Zero the gradients from the last step (PyTorch accumulates gradients by default)
    optimizer.zero_grad() 
    
    # 2. Forward pass: generate prediction
    y_pred = model(batch_X)
    
    # 3. Calculate loss
    loss = loss_fn(y_pred, batch_y)
    
    # --- Step 3 & 4: Backward Pass & Update ---
    # 4. Backward pass: calculate gradient
    loss.backward()
    
    # 5. Update weights: adjust weights based on the gradients
    optimizer.step() 

# The model's weights are now slightly better at the end of this batch!
```

---

## 4. PyTorch Functions & Modules

| Type | PyTorch Module/Function | Purpose |
| :--- | :--- | :--- |
| **Layer** | `nn.Linear(in_features, out_features)` | Performs the linear transformation ($W \cdot X + b$). |
| **Container** | `nn.Sequential()` | Stacks layers and activations into a complete model, passing output sequentially. |
| **Activation** | `nn.ReLU()` | Rectified Linear Unit: $\max(0, z)$. Introduces non-linearity. |
| **Loss** | `nn.CrossEntropyLoss()` | Best for multi-class classification (like MNIST digits). |
| **Optimizer** | `optim.SGD(model.parameters(), lr)` | Stochastic Gradient Descent. Updates model weights using a learning rate (`lr`). |
| **Data** | `torch.zeros()`, `torch.rand()` | Creates Tensors. |
| **Autograd**| `tensor.requires_grad_(True)` | Tags a tensor so PyTorch tracks its operations for gradient calculation. |
