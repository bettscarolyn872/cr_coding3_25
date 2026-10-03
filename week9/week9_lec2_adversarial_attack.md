# Week 9 Lecture 2: Adversarial Attacks and Robustness

## Introduction: When AI Vision Fails

### The Brittle Nature of Neural Networks

Despite achieving superhuman performance on many vision tasks, deep learning models can be surprisingly fragile. A tiny, imperceptible change to an image can cause a state-of-the-art face detection system to completely fail. This vulnerability has profound implications for robotics, security systems, and any application where reliable computer vision is critical.

For example, in a 2019 Guardian article, [Facial recognition fails on race, government study says](https://www.bbc.co.uk/news/technology-50865437), a US government study suggests facial recognition algorithms are far less accurate at identifying African-American and Asian faces compared to Caucasian faces. These implications can be very detrimental and traumatic for those who were falsely identified. 

Here is a graphic from the same BBC article illustrating the potential issues:

<img width="421" alt="racial profiling" src="https://git.arts.ac.uk/user-attachments/assets/c63f6b5b-2e9c-46f4-be54-84e430b2c22a" />


---

## Understanding Adversarial Examples

### What Are Adversarial Examples?

Adversarial examples are inputs designed to cause machine learning models to make mistakes. They are created by applying small, intentional perturbations to legitimate inputs. The key insight is that these perturbations are often imperceptible to humans but can completely change the model's output.

For example, this is becoming significantly more common:

<img width="640" alt="poetry attack" src="https://git.arts.ac.uk/user-attachments/assets/243ac629-346c-40f4-8403-b15ee5ad3d2b" />

[Link to the paper here.](https://arxiv.org/pdf/2511.15304)

<img width="741" alt="poison llm" src="https://git.arts.ac.uk/user-attachments/assets/0b7dfc7c-b460-4b1d-ab28-6f6c9e99bd53" />

[Link to the Claude.ai study](https://www.anthropic.com/research/small-samples-poison)

### Why Do They Work?

Neural networks operate in high-dimensional spaces where they learn complex decision boundaries. These boundaries, while effective for normal inputs, can be surprisingly close to correctly classified examples. By pushing inputs across these boundaries with minimal changes, we can flip the model's predictions.

Key factors that enable adversarial examples:
- **Linearity in high dimensions**: Even though neural networks have nonlinear activation functions, they exhibit surprisingly linear behavior in high-dimensional spaces
- **Overconfidence**: Models often assign high confidence to their predictions, even when slightly perturbed
- **Generalization vs. Robustness tradeoff**: Models optimized for accuracy on clean data may not be robust to small perturbations

### Types of Adversarial Attacks

**- By Knowledge Level**
  - **White-box attacks**: The attacker has full access to the model architecture, weights, and gradients. These are the most powerful attacks.
  - **Black-box attacks**: The attacker can only query the model and observe outputs. Despite limited information, effective attacks are still possible through techniques like:
    - Transfer attacks (using adversarial examples from one model against another)
    - Query-based attacks (estimating gradients through repeated queries)

**- By Goal**  
  - **Untargeted attacks**: The goal is simply to cause misclassification, regardless of the resulting class.
  - **Targeted attacks**: The attacker wants the model to classify the input as a specific incorrect class.

**- By Domain**
  - **Digital attacks**: Perturbations applied directly to digital images before feeding them to the model.
  - **Physical-world attacks**: Adversarial patterns that survive real-world transformations like printing, photographing, and varying lighting conditions. 

Examples include:
- Adversarial patches that can be printed and placed in scenes
- Adversarial makeup patterns
- Modified road signs that fool autonomous vehicles
- Humans doing terrible things

![waymo](https://git.arts.ac.uk/user-attachments/assets/d529d718-7dbf-4c0b-8a68-d2344e03dda8)

[Here is the link to the video](https://www.youtube.com/shorts/KStahAcim_E)


---

## How Adversarial Attacks Work

### The Mathematics Behind Attacks

At its core, creating adversarial examples is about finding the smallest possible changes that cause the biggest possible confusion to the model. Think of it like finding the model's "weak spots."

The challenge is balancing two goals:
- Make changes that fool the model (effectiveness)
- Keep changes invisible to humans (stealth)

This creates an optimization puzzle: What's the tiniest nudge we can give each pixel to maximally confuse the AI?

### Fast Gradient Sign Method (FGSM)

<img width="717" alt="fgsm example" src="https://git.arts.ac.uk/user-attachments/assets/bde10fa6-c18c-4d3e-acc9-8958ba1c1eb1" />

[Image Source](https://www.researchgate.net/figure/A-FGSM-attack-example-on-MNIST-A-image-in-MNIST-which-should-be-classified-to-2-is_fig4_329757198)

One of the simplest yet effective attacks is FGSM, introduced by [Goodfellow et al](https://arxiv.org/abs/1412.6572). Think of it like this: imagine you're trying to confuse someone who's looking at a picture. FGSM figures out the tiniest changes to make to each pixel that will cause maximum confusion to the AI, while keeping the changes so small that humans won't notice.

The key insight: Instead of random changes, we make calculated adjustments. Each pixel is nudged in the specific direction that most confuses the model, but only by a tiny amount.

Essentially, in adversarial attacks like the Fast Gradient Sign Method (FGSM), carefully crafted adversarial noise—small perturbations are added to the input data to "fool" or "poison" the computer vision model. This noise is designed to exploit vulnerabilities in the model so that it makes incorrect predictions, even though the changes are often imperceptible to humans. The model processes this altered input and produces wrong outputs, which is why these attacks are called adversarial. They work against the model’s intended behavior.

### More Advanced Attack Methods

**Projected Gradient Descent (PGD)**: An iterative version of FGSM that takes multiple small steps, projecting back to the allowed perturbation budget after each step. Generally stronger than FGSM.

**Carlini & Wagner (C&W)**: Formulates the attack as an optimization problem that minimizes perturbation size while ensuring misclassification. Often produces smaller perturbations than FGSM/PGD.

**DeepFool**: Finds the minimal perturbation needed to cross the nearest decision boundary.

### Physical World Attacks

Creating attacks that work in the physical world requires addressing additional challenges:
- Varying viewpoints and distances
- Lighting changes
- Camera characteristics
- Printing/display artifacts

Researchers have developed "robust" adversarial examples that maintain effectiveness despite these transformations. The Expectation Over Transformation (EOT) method optimizes adversarial perturbations (subtle and intentional small changes or modifications made to data or input) to work across a distribution of possible physical transformations.

---

## Real-World Applications and Implications

### Anti-Surveillance and Privacy Protection

The same vulnerabilities that concern security researchers have been adopted by privacy advocates and protesters. 

![cvdazzle-06-copyright-adam-harvey-2020](https://git.arts.ac.uk/user-attachments/assets/622abf16-698d-4b38-ad36-f2185a57e76b)

[Artist Adam Harvey's work CV Dazzle](https://adam.harvey.studio/cvdazzle)

CV Dazzle is a form of camouflage from computer vision created in 2010 by artist Adam Harvery's. It was his master's thesis at New York University’s Interactive Telecommunications Program. Unlike traditional camouflage, such as disruptive-pattern material, that hides the wearer from human observation, CV Dazzle is designed to break machine vision systems while still remaining perceptible to human observers. It is the first documented camouflage technique to successfully attack a computer vision algorithm.

Computer Vision Dazzle (CV Dazzle) pioneered the use of makeup and hairstyling to evade face detection systems. Key techniques include:

- **Breaking facial symmetry**: Asymmetric patterns disrupt the facial landmarks that algorithms rely on.
- **Contrast manipulation**: Dark and light regions placed strategically can confuse feature detectors.
- **Key point obscuration**: Covering or altering the appearance of critical facial landmarks like the nose bridge or eye corners.

Similar real-world deployments:
- Hong Kong protesters (2019) used reflective face paint and asymmetric makeup
- Juggalo face paint has been found to confuse some facial recognition systems
- LED glasses that emit infrared light invisible to humans but blinding to cameras

### Implications for Robotics

For robotic systems that rely on computer vision, adversarial vulnerabilities present unique challenges:

- **Safety-critical failures**: A robot that fails to detect a stop sign or misidentifies a human could cause serious harm.
- **Manipulation by bad actors**: Adversarial patches could be used to manipulate robot behavior in warehouses, factories, or public spaces.
- **Sensor fusion as defense**: Combining multiple sensor modalities (vision, lidar, radar) can provide robustness against single-sensor attacks.

### The Arms Race

As defenses improve, so do attacks. This creates an ongoing cycle:
1. New attack methods are discovered
2. Defenses are developed to counter these attacks
3. Attackers find ways to circumvent the defenses
4. The cycle continues

Current trends show that achieving perfect robustness remains elusive. The best approaches combine multiple defense strategies rather than relying on any single method.

---

## Connection to Data Augmentation

### Augmentation as Implicit Robustness

Data augmentation, which you studied in Week 8, shares interesting connections with adversarial robustness:

- **Standard augmentation** (rotations, crops, color jitter) helps models generalize to natural variations but provides limited adversarial robustness.
- **Adversarial training** can be viewed as augmentation with "worst-case" examples rather than random transformations.

### Key Differences

While both involve modifying training data, their goals differ:
- Augmentation: Improve generalization to natural distribution shifts
- Adversarial training: Defend against worst-case perturbations

The perturbations in adversarial training are:
- Optimized to maximize loss (not random)
- Often imperceptible (unlike many augmentations)
- Computed dynamically during training

### Adversarial Training in Practice

The basic idea is elegant: if we want our model to handle adversarial examples, we should show it adversarial examples during training. It's like training a security guard by showing them both normal visitors and people trying to sneak in.

The process works like this:
1. Train normally with clean images
2. Generate adversarial versions of those same images
3. Train the model to correctly classify both versions
4. Repeat throughout training

This forces the model to learn features that are robust to small perturbations, not just features that work on clean data.

### Trade-offs in Robust Training

Training for adversarial robustness often comes with costs:
- Reduced accuracy on clean images
- Increased computational requirements
- Need for larger models to maintain performance

Research suggests this trade-off may be fundamental, not just a limitation of current methods.

---

## Building Robust Vision Systems

### Defense Strategies

No single defense provides complete protection, but combining approaches improves robustness:
- **Adversarial Training**: Include adversarial examples during training. Most effective but computationally expensive.
- **Gradient Masking**: Make gradients less useful to attackers. Warning: Often gives a false sense of security.
- **Input Preprocessing**: Detect or remove adversarial perturbations before classification. Examples include:
  - Denoising
  - Compression
  - Randomized smoothing

- **Ensemble Methods**: Combine predictions from multiple models trained differently.
- **Certified Defenses**: Provide mathematical guarantees about robustness within certain perturbation bounds.

### Design Principles for Robust Robotic Systems

When building vision systems for robotics, consider:

1. **Multi-modal sensing**: Don't rely solely on vision. Combine with lidar, ultrasonic, or other sensors.
2. **Fail-safe behaviors**: Design systems to fail gracefully when vision is compromised.
3. **Anomaly detection**: Flag inputs that seem unusual or adversarial.
4. **Human oversight**: Keep humans in the loop for critical decisions.
5. **Regular updates**: As new attacks emerge, systems need updates to maintain security.

### Ethical Considerations

The dual-use nature of adversarial research raises important questions:
- **Publishing attack methods**: Should researchers publish attacks that could be misused?
- **Privacy vs. security**: Anti-surveillance techniques protect privacy but could enable malicious actors.
- **Responsible disclosure**: How should vulnerabilities in deployed systems be handled?

### Future Directions

Emerging areas in adversarial robustness include:
- Robustness in other domains (NLP, audio, reinforcement learning)
- Adversarial examples as a tool for understanding neural networks
- Connections to neuroscience and biological vision
- Social and legal frameworks for adversarial AI

Adversarial examples reveal fundamental characteristics of how neural networks process information. For robotics applications, understanding these vulnerabilities is crucial for building reliable systems. While perfect robustness remains elusive, combining multiple defense strategies and maintaining awareness of the threat landscape helps create more resilient vision systems.

Key takeaways:
- Small perturbations can fool powerful models
- Physical-world attacks pose real threats to deployed systems
- Robustness often trades off with standard accuracy
- Multi-layered defenses work better than single approaches
- The adversarial arms race continues to evolve

As you develop vision systems for robotics, remember that robustness **is not an afterthought** but a critical design consideration from the start.

# Today's Lab

You can download it [here](https://git.arts.ac.uk/c-lin/CR-Coding-3-2025/blob/main/week9/week9_lab_breaking_face_detection.ipynb) if you haven't downloaded the files from the class repository yet. 
