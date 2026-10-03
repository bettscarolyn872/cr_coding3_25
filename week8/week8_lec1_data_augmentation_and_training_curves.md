# Week 8 Lecture 1: Data Augmentation and Learning How to Read Your Training Results

## Why Data Augmentation?

### The Data Hunger Problem  
CNNs require lots of labeled data to learn well, but collecting and labeling large datasets is expensive and time-consuming. Data augmentation helps us create more training examples by applying transformations that don’t change the label.

### What is Data Augmentation?  
It means applying label-preserving transformations like flips, rotations, or brightness changes to existing images so the model sees more diverse examples.

<img width="958" alt="data augmentation" src="https://git.arts.ac.uk/user-attachments/assets/7aac962c-2793-4ce6-ad51-bede0dd6070e" />

[Image source: What are Data augmentation techniques: 2024 update](https://ubiai.tools/what-are-the-advantages-anddisadvantages-of-data-augmentation-2023-update/)

### Common Types of Augmentation  
- Horizontal Flip  
- Vertical Flip  
- Rotation  
- Color Jitter (brightness, contrast, saturation)  
- Random Crop  
- Gaussian Blur  

### When to Use Each Augmentation  
Not all augmentations suit every dataset. 

For example:  
- Flips work for natural images but not for text or medical scans
- Rotations are good for objects without a fixed orientation, but not for faces or buildings 
- Color jitter helps with varying lighting conditions, but not for medical images

### Today’s Lab Notebook  
Let's have a look at [today's lab notebook 1](https://colab.research.google.com/drive/1F8PjIgfkPEwTjRO5AYu5QG84LLCl1vbW?usp=sharing).

---

## How to Read CNN / Machine Learning Training Results

- There are three common dynamics that you are likely to observe in learning curves. They are:
  - Underfit
  - Overfit
  - Good Fit

### 1. Understand the Key Metrics

#### Common metrics during training:
- **Loss**  
  - Quantifies how well the model is performing. Lower is better.  
  - Examples: Cross-entropy loss (classification), Mean Squared Error (regression).

- **Accuracy**  
  - Percentage of correctly predicted samples. Higher is better.  

- **Validation metrics**  
  - Metrics computed on validation (or test) data. Helps check if the model generalizes well.

### 2. Understanding the Training and Validation Graphs

#### Loss Curves show how well the model fits the training and validation data:
- The **training loss** measures how well the model fits the training data. It should generally decrease steadily as the model learns.
- The **validation loss** tells us how well the model performs on unseen data. Ideally, it should decrease and stay close to the training loss.
- If the validation loss starts increasing while the training loss keeps decreasing, it may be **overfitting**.

#### Accuracy Curves measure the percentage of correct predictions on training and validation sets:
- Both accuracies should increase during training.  
- **Training accuracy** shows the percentage of correct predictions on the training set. It typically improves over time.
- **Validation accuracy** measures performance on new data and is a better indicator of how the model will perform in real-world scenarios.
- A small gap between training and validation accuracy suggests **good generalization**. A large gap means the model might be **overfitting**.

<img width="844" alt="underfit model without sufficient capacity" src="https://git.arts.ac.uk/user-attachments/assets/a9bb14c5-3ee7-4eab-9371-40dc5b1a086c" />

[Image Source: How to use Learning Curves to Diagnose Machine Learning Model Performance](https://machinelearningmastery.com/learning-curves-for-diagnosing-machine-learning-model-performance/)

**What is happening here:**
- Both training and validation losses are very low and decrease gradually
- However, the training loss is consistently lower than the validation loss by a noticeable margin
- The validation loss decreases slowly and remains higher than the training loss, without converging closer
- The difference between training and validation loss does not increase sharply, so there is no sign of overfitting

**Interpretation:**
- This is an example of an underfitting model: the model is not complex enough (lacks capacity) to fully capture the patterns in the data
- The slow decrease and low magnitude of losses indicate that the model is learning but not improving rapidly or sufficiently
- The gap between training and validation loss suggests the model could benefit from being bigger or more powerful (more layers, neurons, etc.)

**What to do next:**
- Increase model capacity by adding more layers or neurons
- Train longer or with better optimization settings
- Consider feature engineering ( creating or transforming input data features to make them more useful for the model, like data augmentation) or using more informative inputs.

### 3. Look at Training vs. Validation Curves

#### What to plot:
- Training loss and accuracy over epochs  
- Validation loss and accuracy over epochs

#### What to observe:
- **Convergence:**  
  Do the training and validation losses decrease steadily?  
- **Overfitting:**  
  Training loss keeps decreasing, but validation loss stops improving or rises → overfitting.  
- **Underfitting:**  
  Both training and validation loss remain high → model too simple or training insufficient.

<img width="839" alt="underfit model that needs more training" src="https://git.arts.ac.uk/user-attachments/assets/28cc47c2-ef5e-4f4b-85f5-8a044b7b1a0f" />

[Image Source: How to use Learning Curves to Diagnose Machine Learning Model Performance](https://machinelearningmastery.com/learning-curves-for-diagnosing-machine-learning-model-performance/)

This graph shows a different pattern compared to the first one, and it suggests that the underfitting issue might be due to insufficient training time, meaning the model needs more training.

**What this graph is telling us:**

- The losses start relatively high (~1.07 for train, ~1.06 for val), showing the model is initially performing poorly.  
- Both training and validation losses decrease steadily and significantly over 50 epochs —> the model is learning.  
- The close parallel between training and validation losses indicates the model generalizes reasonably well so far.  
- Since the losses have not plateaued yet, training for more epochs will likely continue improving the model.

In this graph, the model is clearly learning because both losses keep going down steadily. The fact that they started high but are decreasing means the model just needs more time to train.
This differs from the first graph, where losses were already low but barely improving, suggesting the model was too simple.

**What to do next:**
- Continue training longer. The model hasn’t converged yet.  
- Monitor if losses plateau after more epochs. If they do but are still high, consider increasing capacity by making the model more complex or powerful (add more layers to the network, add more neurons in each layer, use more advanced architectures, etc.)  
- If validation loss starts increasing later, watch out for overfitting.
  
<img width="843" alt="overfit" src="https://git.arts.ac.uk/user-attachments/assets/708f133f-7c04-4ed8-990b-3b28b51a45ae" />

[Image Source: How to use Learning Curves to Diagnose Machine Learning Model Performance](https://machinelearningmastery.com/learning-curves-for-diagnosing-machine-learning-model-performance/)

**What is going on here:**
- Both training and validation losses start high and decrease rapidly during the early epochs, indicating the model is learning
- After around epoch 50, the training loss continues to decrease steadily, showing the model keeps improving on the training data
- However, the validation loss plateaus and even fluctuates around a higher value, not decreasing in tandem with training loss

**This growing gap between training and validation loss is a classic sign of overfitting:**
- The model fits the training data very well but does not generalize as well to new, unseen data
- The model is likely too complex relative to the amount or variability of data it has
- It memorizes training examples instead of learning patterns that generalize
- Noise or peculiarities in training data are learned as if they were important features

**What to do next:**
- Early Stopping: Stop training when validation loss stops improving to prevent overfitting.
- Regularization Techniques such as adding dropout layers, applying batch normalization, etc.
- More Data or Data Augmentation: Increase training data size to help the model generalize better. Use augmentation techniques like rotations, flips, or color changes for images.
- Simplify Model: Reduce the number of layers or neurons to decrease capacity

<img width="842" alt="good fit" src="https://git.arts.ac.uk/user-attachments/assets/d870a223-3199-4515-97fc-f23e82efbe04" />

[Image Source: How to use Learning Curves to Diagnose Machine Learning Model Performance](https://machinelearningmastery.com/learning-curves-for-diagnosing-machine-learning-model-performance/)

This graph represents the ideal scenario when training deep learning models, showing a reliable and well-performing model.

**What is happening here?**
- Both training loss (blue line) and validation loss (orange line) start high and decrease sharply in early epochs
- After about 10 epochs, both losses stabilize and fluctuate closely around the same low value (~0.4)
- There is no large gap between training and validation loss throughout the training process
- Loss curves are smooth without erratic spikes, indicating stable learning

**Why is this a good fit?**
- The model learns the training data well (training loss decreases)
- The model generalizes well to unseen data (validation loss follows training loss closely)
- No sign of overfitting (validation loss does not increase or diverge)
- No sign of underfitting (losses are low and stable)

**What makes a model a good fit?**
- Balanced complexity: Model is complex enough to capture patterns but not too complex to memorize noise
- Sufficient training: The model has trained long enough to learn meaningful features
- Good generalization: Validation performance mirrors training performance closely
- Stable training: Loss curves are smooth, indicating consistent learning without instability

**What to do next when you see such a curve?**
- You might consider stopping training (early stopping) since the model has converged
- Optionally fine-tune hyperparameters for slight improvements
- Evaluate on test data to confirm performance
- Deploy or use the model confidently for predictions

So... If: 
- Training and validation losses decrease together -> Model learns and generalizes well
- Small gap between losses -> No overfitting or underfitting
- Loss values stabilize at a low level -> Model has converged


### 4. Signs of Good Training

- Both training and validation loss decrease smoothly.  
- Training and validation accuracies both increase and are close to each other.  
- No sudden spikes or erratic behavior in either curve.

### 5. What to Watch For

- If training loss decreases but validation loss plateaus or increases, consider adding regularization or data augmentation.  
- If both losses are high, the model may be underfitting — try increasing model capacity or training longer.

Using these graphs, you can monitor your model’s learning progress and make informed decisions about tuning or adjusting your training process.

### 6. Typical Behavior and What It Means

| Pattern                             | Meaning                                   | Action                              |
|-----------------------------------|-------------------------------------------|-----------------------------------|
| Training loss and validation loss both decrease and stabilize | Good training, model is learning well | Continue training or stop early   |
| Training loss decreases, validation loss increases           | Overfitting detected                      | Use regularization, dropout, or get more data |
| Both losses remain high                                         | Underfitting or problem in training      | Increase model capacity, learn rate, or epochs |

### Signs of Overfitting vs Underfitting

| Condition          | What Happens                   | What It Means                  |
|--------------------|-------------------------------|-------------------------------|
| Underfitting       | High train & val loss          | Model too simple or not trained enough |
| Good fit           | Low train & val loss, close together  | Model learning well            |
| Overfitting        | Low train loss, high val loss  | Model memorizes training data but fails to generalize |


### 7. Example Walkthrough

```
Epoch 1: Train Loss=1.2, Val Loss=1.3, Train Acc=50%, Val Acc=48%
Epoch 5: Train Loss=0.5, Val Loss=0.6, Train Acc=80%, Val Acc=78%
Epoch 10: Train Loss=0.2, Val Loss=0.7, Train Acc=95%, Val Acc=75%
```

#### What is Loss?

- **Loss** is a number that tells us how far off the model’s predictions are from the true answers.  
- Think of it like a "mistake score": higher means more mistakes, lower means fewer mistakes.

#### What Does “High” or “Low” Loss Mean?

- **High loss** means the model is making many mistakes or is confused.  
- **Low loss** means the model is making fewer mistakes and is getting better.

#### How High is High? How Low is Low?

- The exact numbers depend on the problem and how loss is calculated, but here’s a simple example:
  - Imagine a classification problem with 10 classes (like recognizing digits 0-9):
    - If the model guesses randomly, the loss might be around 2.3.  
    - If the loss starts at around 2.3, it means the model is just guessing.  
    - As the model learns, the loss should go **down** from 2.3 towards 0.  
    - When the loss gets closer to 0, it means the model is making very few mistakes.

#### What Should You Look For?

- Don’t focus too much on the exact number. Focus on whether the loss is **getting smaller over time**.  
- If the loss stays high or doesn’t get better, it means the model is struggling and you might need to change something (like training longer or using a better model).  
- If the training loss keeps going down but the validation loss goes up, it may mean the model is just memorizing training data (this is **overfitting**).

Think of it as learning to throw balls into buckets blindfolded:
- At first, you miss most of the time (high loss).  
- With practice, you miss less (loss decreases).  
- If you only practice in one room and then try another room and miss more, that’s like overfitting.

### What this tells us:

- **Epoch 1:**  
  - Training and validation losses are high (1.2 and 1.3), and accuracies are low (50% train, 48% val).  
  - The model is just starting to learn, and its predictions are only slightly better than random guessing.

- **Epoch 5:**  
  - Training loss drops significantly to 0.5, and training accuracy improves to 80%.  
  - Validation loss also decreases to 0.6, and validation accuracy improves to 78%.  
  - This indicates good progress—the model is learning patterns that generalize reasonably well.

- **Epoch 10:**  
  - Training loss further decreases to 0.2, with training accuracy reaching 95%.  
  - However, validation loss increases slightly to 0.7, and validation accuracy drops to 75%.  
  - This divergence suggests **overfitting**: the model fits training data very well but does not generalize as well to new unseen data.

### Key points to remember:

- Look for both loss and accuracy trends on training and validation sets, not just one metric.  
- A widening gap between training and validation performance (accuracy or loss) often signals overfitting.  
- Early stopping or regularization techniques might be needed when overfitting appears.  
- If both training and validation perform poorly, the model may need more capacity or longer training.

