# Week 8 Lecture 2: Object Detection Introduction

## Overview  
Today, we’ll move from asking “what is in this image?” to “what is it and where is it?” —> a crucial capability for robots that interact with the world.

---

## From Classification to Detection

- Image classification assigns a single label to an image, while object detection identifies and locates multiple objects within an image.
- Classification is simpler, focusing on one prominent object vs. detection is more complex, requiring bounding boxes for precise localization.
- Applications include autonomous driving, retail analytics, and medical image analysis.
- Key architectures include CNNs for classification and models like YOLO and Faster R-CNN for detection.
- Both techniques enhance machine vision, though object detection offers more detailed insights into image content.

### Understanding Image Classification
Image classification in computer vision is a foundational task for which an annotator attributes a label (or a category) to an entire data piece (i.e., image or video frame). In essence, the model learns to recognize patterns and features within an image that are indicative of a particular class. For instance, a well-trained image classification model can differentiate between various animal species, classify everyday objects, or even diagnose diseases in medical images.

The fundamental challenge of an image classification technique lies in feature extraction. The model must identify distinguishing characteristics within the image that define its class. This often involves using convolutional neural networks (CNNs) that are adept at capturing hierarchical features like edges, textures, and shapes. These features are acquired via supervised learning, which involves training the model using a labeled machine learning dataset. 

### Understanding Object Detection
While image classification focuses on assigning a single label to an entire image, object detection models go a step beyond by recognizing and pinpointing the positions of numerous objects within an image. In other words, object detection not only categorizes the objects present but also draws bounding boxes around them to indicate their exact location.

Object detection has gained immense importance due to its wide range of applications. From advanced driver assistance systems (ADAS) that enable cars to perceive their surroundings to retail analytics (take a look at the ZARA case) that track product placement, object detection serves as a versatile tool.

The complexity of object detection stems from its dual requirements of categorization and localization. This has led to the development of architectures like Faster R-CNN, YOLO (You Only Look Once), and SSD (Single Shot MultiBox Detector), each with its unique approach to solving this intricate challenge.

- **Image Classification:** Answers "What objects are in this image?"  
- **Object Detection:** Answers "What objects are in this image, and where are they located?"
  
<img width="664" alt="image classification and object detection" src="https://git.arts.ac.uk/user-attachments/assets/0d4a34c3-c48c-4d2c-8287-fe38697be865" />

[Image source and reference: What’s the Difference Between Image Classification & Object Detection?](https://labelyourdata.com/articles/object-detection-vs-image-classification)

### Classification, Localization, and Object Detection

This image illustrates the key differences between three important computer vision tasks:

- **Classification**
- **Classification + Localization**
- **Object Detection**

#### 1. Classification (Left Image)

- **What it does:** The model looks at the entire image and predicts the main object or category present, for example, "BUS".
- **Output:** A single label describing the whole image.
- **Limitations:** It does not say *where* the object is in the image, only *what* is present

#### 2. Classification + Localization (Middle Image)

- **Localization** in computer vision refers to the task of identifying the position of an object within an image, usually by drawing a bounding box around it.
- **What it does:** The model not only classifies the object but also predicts its location via a bounding box around the object.
- **Output:**  
  - A label ("BUS")  
  - A bounding box showing the approximate position of the bus in the image.
- **Limitations:** Usually assumes only one main object per image. It cannot handle multiple objects or different classes simultaneously.

#### 3. Object Detection (Right Image)
- **What it does:** Detects and classifies multiple objects in an image, each with its own bounding box.
- **Output:**  
  - Multiple labels (e.g., "BUS", "CAR")  
  - Multiple bounding boxes, one for each detected object.
- **Advantages:** Can find and classify many objects of different types within the same image and locate them precisely.

## Summary Table

| Task                         | Output                          | Number of Objects | Object Locations       |
|------------------------------|--------------------------------|-------------------|-----------------------|
| Classification               | Single label                   | Usually 1         | No                    |
| Classification + Localization | Label + bounding box          | Usually 1         | Yes                   |
| Object Detection             | Multiple labels + multiple boxes | Multiple          | Yes                   |


**Why detection matters for robotics:**  
- Computer vision systems has evolved from just recognizing what is in an image to precisely locating multiple objects in complex scenes
- Spatial awareness for grasping and manipulation  
- Handling multiple objects simultaneously  
- Avoiding collisions by knowing the exact obstacle locations  
- Path planning and navigation  
- Interaction with specific objects

## How Object Detection Works

### Key Concepts

<img width="1155" alt="bounding box" src="https://git.arts.ac.uk/user-attachments/assets/77f8abd5-c4ab-4dcb-89e7-440b2bf54acf" />

[Image Source: An Introduction to Bounding Boxes + Best Practices](https://superb-ai.com/en/resources/blog/an-introduction-to-bounding-boxes-best-practices)

- **Bounding Boxes:**
  - A bounding box is an imaginary box that is used to enclose an object of interest in an image or video. The bounding box acts as a point of reference for object recognition and is used to identify the location and size of the object within the image.
  - The bounding box is defined by its center of the box coordinates (x,y) relative to the bounds and its width and height (w,h). These coordinates and dimensions provide a frame that encloses the object of interest.


<img width="743" alt="object detection class labels example" src="https://git.arts.ac.uk/user-attachments/assets/a0c431bf-3ef1-4fa7-9119-8964f734429e" />

[Image Source: Finding Label Errors in Object Detection Datasets](https://docs.cleanlab.ai/stable/tutorials/object_detection.html)

- **Class Labels:** The categories or types assigned to detected objects within an image. Each bounding box drawn around an object is associated with a class label that identifies what the object is, such as "person," "chair," "car," or "cup." These labels help the system and users understand the nature of each detected object.

<img width="812" alt="confidence score" src="https://git.arts.ac.uk/user-attachments/assets/498b6c8f-189f-4d30-8a33-61f58c265526" />

[Image Source: One-Stage Object Detection](https://machinethink.net/blog/object-detection/)

- **Confidence Scores:** How sure is the model about the detection? Confidence Score in object detection is a value (often between 0 and 1) that represents the model’s estimated probability that a detected object actually belongs to the predicted class. It indicates how confident the model is about each specific detection.

The confidence score is calculated by the detection model during inference based on internal probabilities, not directly from mean average precision (mAP).

**Mean Average Precision (mAP)** is commonly used to analyze the performance of object detection and segmentation systems. Many object detection algorithms, such as Faster R-CNN, MobileNet SSD, and YOLO, use mAP to evaluate their models. The mAP is also used across several benchmark challenges such as Pascal, VOC, COCO, and more. 
- mAP & IoU = metrics used to evaluate overall model performance across many detections and images.

<img width="646" alt="ground truth box v predicted box" src="https://git.arts.ac.uk/user-attachments/assets/0f96e165-a9bc-4ca3-bd69-8d49dd236136" />

- Ground Truth Box (green): This is the actual, correct bounding box manually labeled on the object in the image (the “true” position and size)
- Predicted Box (red): This is the bounding box predicted by the detection model, where it thinks the object is
- IoU measures how much the predicted box overlaps with the ground truth box

<img width="683" alt="iou" src="https://git.arts.ac.uk/user-attachments/assets/7c8b8a87-af09-43a1-a968-1dc574e160e7" />

[Images Source: Mean Average Precision (mAP) Explained: Everything You Need to Know](https://www.v7labs.com/blog/mean-average-precision)

- **Intersection over Union (IoU):** Intersection over Union (IoU) is a measure of overlap between predicted bounding boxes and ground truth boxes, used both in evaluation metrics like mAP and in processes such as Non-Maximum Suppression.
- IoU tells us how accurate the predicted box is in locating the object compared to the true box.
- Values range from 0 to 1:
  - IoU = 1: Perfect prediction, boxes exactly match.
  - IoU close to 0: Poor prediction, little or no overlap.
- In object detection tasks, a threshold IoU (e.g., 0.5) is often set to decide if a detected box is considered a correct detection or a false positive.

### Two Families of Detectors

<img width="710" alt="one stage v two stage detectors " src="https://git.arts.ac.uk/user-attachments/assets/1c49a279-3da9-4ca9-88ef-ee9c6e6f67e3" />

[Image Source](https://www.researchgate.net/figure/Two-stage-vs-one-stage-object-detection-models_fig3_353284602)

#### Two-Stage Detectors:
- (e.g., R-CNN family) Generate many region proposals, then classify them. They are accurate but slower.
- How they work: These methods first generate a set of candidate regions in the image that may contain objects (called region proposals). Then, each proposal is processed separately by a classifier to determine the object class and refine the bounding box.
- Characteristics:
  - Typically, more accurate because they carefully analyze each proposed region
  - The multi-step process makes them computationally heavier and slower
  - Examples include the R-CNN family (R-CNN, Fast R-CNN, Faster R-CNN)
- Use cases: Suitable for applications where accuracy is critical and some delay is acceptable, such as offline image analysis or high-quality detection tasks

#### One-Stage Detectors:
- (e.g., YOLO) Predict bounding boxes and classes in a single pass. They are faster, suitable for real-time robotics.
- How they work: These models simultaneously predict bounding boxes and class probabilities directly from the entire image in a single forward pass through the network.
- Characteristics:
  - Much faster due to the single-step approach.
  - Slightly lower accuracy compared to two-stage methods but improving rapidly.
  - Well-suited for real-time applications like robotics, video surveillance, or autonomous vehicles where speed is essential.
- Examples: YOLO (You Only Look Once) and SSD (Single Shot MultiBox Detector).

## YOLO

<img width="537" alt="yolo" src="https://git.arts.ac.uk/user-attachments/assets/8c3ce884-96d0-4242-996a-4ed6a3e10a4e" />

[Image Source: YOLO object detection with OpenCV](https://pyimagesearch.com/2018/11/12/yolo-object-detection-with-opencv/)

### YOLO’s Key Idea: Detection as Regression

- Dividing the Image:
YOLO splits the input image into a grid made up of ( $S$ x $S$ ) cells (for example, ( 7 x 7 ) cells). Think of this like cutting the image into small squares.
- Grid Cell Predictions:
Each cell is responsible for detecting objects whose centers fall inside it. For every cell, YOLO predicts:
  - Bounding boxes: These are rectangles that try to tightly surround objects. Each cell predicts multiple boxes (usually $B$ boxes), giving different guesses.
  - Confidence scores: Each bounding box has a confidence value that estimates how likely it is that the box contains an object and how accurate the box is.
  - Class probabilities: For each bounding box, the cell predicts the likelihood of the object belonging to each possible class (like "person," "car," or "dog").
- Single Forward Pass:
YOLO processes the whole image through its convolutional neural network just once and outputs all bounding boxes, confidence scores, and class probabilities simultaneously. This is very fast compared to methods that look at parts of the image one by one.
- Why This Matters: Because YOLO treats detection as a single regression problem, directly predicting bounding boxes and class probabilities from the image pixels. It can run in real-time, making it great for applications like robotics, video surveillance, or self-driving cars.

This approach contrasts with older methods that generate many candidate regions and analyze them separately, which makes YOLO both simpler and faster.

### YOLO x OpenCV

- YOLO and OpenCV are related but serve different purposes in computer vision workflows:
  - **YOLO (You Only Look Once)** is a deep learning-based object detection model that predicts bounding boxes and class labels for objects in images or video frames.
  - **OpenCV (Open Source Computer Vision Library)** is a powerful library that provides tools for image and video processing, including reading/writing images and videos, drawing shapes (like bounding boxes), camera interfacing, and basic computer vision algorithms.
- How they work together: You use YOLO to perform the actual object detection (running the neural network to find and classify objects)
- You use OpenCV to handle tasks around those detections, such as:
  - Capturing video frames from a webcam or video file
  - Preprocessing images before feeding them to YOLO
  - Drawing bounding boxes and class labels on images based on YOLO’s output
  - Displaying the annotated images or saving them
  - Handling user input during real-time detection loops

YOLO is the detection engine, while OpenCV is the toolkit that helps you prepare data for YOLO, visualize its results, and build complete applications involving image/video input and output. They complement each other in creating practical object detection systems.

### Non-Maximum Suppression (NMS)
- Object detectors often produce multiple overlapping bounding boxes for the same object
- NMS is a technique used to filter out redundant boxes, keeping only the most confident one
- It helps improve detection clarity and avoids multiple boxes cluttering the same object by keeping only the highest confidence box and removing others with high overlap
- This results in clean, distinct detections

### Using YOLO in Practice

- Load pre-trained YOLO models (YOLOv8 is popular)
- Use the YOLO model to find and identify objects in images or videos
- Extract bounding boxes, confidence scores, and class labels
- Draw boxes on images or use detections for robot actions
- Object detection enables robots to perceive and interact with their environment effectively
- Real-time performance is critical. YOLO balances speed and accuracy well
- Data augmentation and proper training improve detection robustness

---

## Robotics Case Studies

### 1. Amazon Robotics: Proteus Autonomous Mobile Robot

<img width="775" alt="proteus" src="https://git.arts.ac.uk/user-attachments/assets/a6737921-6dc3-4d54-a64f-c0e6aebbfbab" />

[Image and Video from Robots' Guide](https://robotsguide.com/robots/proteus)

**Company**: Amazon Robotics  
**Robot**: Proteus  
**Problem Solved**: Safely navigating warehouses alongside human workers while moving heavy carts

**Technical Details**:
- Proteus uses advanced computer vision and object detection technologies as part of its navigation and safety systems. This allows it to operate autonomously and safely around both human employees and other objects in an unrestricted warehouse environment. 
- Detects: humans, other robots, carts, packages, obstacles
- 360-degree camera coverage with 8 cameras
- Custom-trained on millions of warehouse images
- Processes all cameras simultaneously on embedded GPUs

**How Object Detection Helps**:
- Maintains safe distance from humans (detection + tracking)
- Identifies cart handles for autonomous coupling
- Distinguishes between stationary and moving obstacles
- Reads floor markers and signs for navigation

### 2. ABB Robotics: YuMi Collaborative Assembly

<img width="747" alt="YuMi" src="https://git.arts.ac.uk/user-attachments/assets/581d1fba-7f9a-44d4-874e-98c0cee88408" />

[Image and Video from Robots' Guide](https://robotsguide.com/robots/yumi)

**Company**: ABB Robotics  
**Product**: YuMi dual-arm robot with integrated vision  
**Problem Solved**: Assembling small electronics alongside human workers

**Technical Details**:
- The ABB YuMi robot is designed to use object detection and is often equipped with integrated or external vision systems for this purpose. The robot's standard configuration includes high-resolution cameras in its end-effectors for part recognition and accurate grasping. 
- Identifies 50+ different electronic components
- Achieves 99.2% accuracy at 25 FPS
- Integrated with force sensors for safe collaboration
- Trained on synthetic data + real assembly line images

**How Object Detection Helps**:
- Locates tiny components (resistors, capacitors) on cluttered workbench
- Tracks human hands to avoid collisions (also soft robotic arms)
- Verifies assembly correctness by detecting component placement
- Adapts to new product variants with minimal retraining

### 3. Train YOLOv8 on Custom Dataset – A Complete Tutorial by Learn OpenCV

<img width="736" alt="annotated pothole dataset images" src="https://git.arts.ac.uk/user-attachments/assets/94d97c21-acc4-442c-b96f-f97a01ec7745" />

[Image Source: Learn OpenCV](https://learnopencv.com/train-yolov8-on-custom-dataset/)

**Platform**: [LearnOpenCV Website](https://learnopencv.com/train-yolov8-on-custom-dataset/)

**Technical Details:**
- In this tutorial, you will see how they train the YOLOv8 on a custom pothole dataset, which mainly contains small objects that can be difficult to detect. The tutorial covers the dataset and trains three different YOLOv8 models: YOLOv8n (Nano model), YOLOv8s (Small model), and YOLOv8m (Medium model).

### 4. teamLab: Athletics Forest

<img width="815" alt="teamlab" src="https://git.arts.ac.uk/user-attachments/assets/4f0db70e-f47a-4f47-9733-3d8af1141f2b" />

[Image and video sourced: teamlab](https://www.youtube.com/watch?v=NrOiBZcIGrc)

**Artists**: teamLab  
**Installation**: Athletics Forest  
**Description**: An Interactive space where robotic projectors track and respond to multiple visitors

**Technical Details**:
- teamLab: Athletics Forest uses various sensor-based technologies, which would encompass techniques like object detection, to create its interactive environments. The installations are designed to dynamically shift and transform in real-time in response to the actions and movements of visitors.
- The artworks are generated by computer programs that continuously render the environment, and they rely on the physical presence and interactions of people to change the space. For instance, in the "Aerial Climbing through a Flock of Colored Birds" exhibit, the physical structures and the projected elements change based on visitors' movements and balance, suggesting the use of sensors to track positions and interactions within the space. 
- 30+ ceiling-mounted robotic projectors
- Creates personalized visual effects for each person
- 60 FPS detection enabling real-time reactive projections
- Custom training for detecting people from an overhead view

**How Object Detection Enables the Art**:
- Each person gets a unique visual "aura" that follows them
- Detects interactions between people (proximity, gestures)
- Adjusts projections to avoid projecting on faces
- Creates emergent patterns from collective movement

### 5. Kat/the.poet.engineer on IG

Kat’s work with TouchDesigner often incorporates real-time interactive elements, and it is common for such projects to use technologies like object detection to respond dynamically to audience movement or presence. While not all of her TouchDesigner projects necessarily use object detection, some installations may integrate it to track people’s positions or gestures, enabling the visuals or sound to change based on user interaction. This combination of creative coding and sensing technology allows her to create immersive, responsive environments.

<img width="1068" alt="poetic engineer 1" src="https://git.arts.ac.uk/user-attachments/assets/0a64267a-9caa-4e01-8391-7010b7a8ddb1" />

This work uses object detection or hand tracking techniques to identify and "see" the hand. The system captures the hand’s position and movements, then overlays interactive graphical elements that respond to the hand’s gestures. This allows the hand to function like a control interface, turning it into a dynamic, kinematic dashboard for interaction within the installation.

<img width="373" alt="poetic engineer 2" src="https://git.arts.ac.uk/user-attachments/assets/2883c549-e8d6-4150-a310-0b17bc311f8b" />

Here, the camera first uses object detection to identify the hand’s presence and location. Then, with pose/keypoint detection, it tracks specific points like fingertips and joints. Gesture classification interprets these movements, such as pinching or rotating fingers, as commands. This combination enables creative interactive works like a gestural font editor, where specific finger gestures control how characters are warped or adjusted in real time.

[Image Source: the.poet.engineer](https://www.instagram.com/the.poet.engineer/?hl=en)

---

## Lab

For YOLO, we have two lab notebooks:
- [week8_lab2_yolo_webcam.py](https://git.arts.ac.uk/c-lin/CR-Coding-3-2025/blob/main/week8/week8_lab2_yolo_webcam.py)
- [week8_lab3_detection.py](https://git.arts.ac.uk/c-lin/CR-Coding-3-2025/blob/main/week8/week8_lab3_detection.py)

---

## Next Week

Ilia will be guest lecturing next week. We will also have a discussion session on your final mini project idea. So please come with ideas ❤️ They can be in the early phase. 
