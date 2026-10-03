"""
Week 8 Lab 3: Understanding Bounding Boxes and Confidence Scores

In this lab, we'll:
1. Analyze what bounding boxes really represent
2. Understand confidence scores and thresholds
3. Visualize IoU (Intersection over Union)
4. Explore Non-Maximum Suppression (NMS)
5. Analyze detection quality metrics
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
import time
from collections import defaultdict

print("Week 8 Lab 3: Understanding Bounding Boxes and Confidence Scores")
print("=" * 60)

# Load YOLO model
model = YOLO('yolov8n.pt')

# Part 1: Anatomy of a Detection

# Creates a simple image with colored rectangles to simulate objects and runs the YOLO model on it. 
# It prints bounding box coordinates in two formats, confidence scores, class IDs, 
# and box properties like area and aspect ratio to help understand detection output.

print("\n1. Anatomy of a Detection")
print("-" * 40)

# Load a sample image
print("Loading sample image...")
# Create a simple test image with shapes
test_img = np.ones((480, 640, 3), dtype=np.uint8) * 255

# Add some objects (colored rectangles to simulate objects)
cv2.rectangle(test_img, (100, 100), (200, 300), (0, 0, 255), -1)  # Red "object"
cv2.rectangle(test_img, (400, 200), (550, 400), (0, 255, 0), -1)  # Green "object"

# For real testing, load an actual image:
# test_img = cv2.imread('test_image.jpg')

# Run detection
results = model(test_img)

print("\nUnderstanding Detection Output:")
print("-" * 40)

for result in results:
    if result.boxes is not None:
        for i, box in enumerate(result.boxes):
            print(f"\nDetection {i+1}:")
            
            # Bounding box formats
            xyxy = box.xyxy[0].cpu().numpy()  # x1, y1, x2, y2
            xywh = box.xywh[0].cpu().numpy()  # x_center, y_center, width, height
            
            print(f"  Format xyxy: [{xyxy[0]:.1f}, {xyxy[1]:.1f}, {xyxy[2]:.1f}, {xyxy[3]:.1f}]")
            print(f"  Format xywh: [{xywh[0]:.1f}, {xywh[1]:.1f}, {xywh[2]:.1f}, {xywh[3]:.1f}]")
            
            # Confidence and class
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            class_name = model.names[cls]
            
            print(f"  Class: {class_name} (ID: {cls})")
            print(f"  Confidence: {conf:.3f} ({conf*100:.1f}%)")
            
            # Box properties
            area = xywh[2] * xywh[3]
            aspect_ratio = xywh[2] / xywh[3] if xywh[3] > 0 else 0
            
            print(f"  Area: {area:.0f} pixels²")
            print(f"  Aspect ratio: {aspect_ratio:.2f}")


# Part 2: Confidence Score Analysis

# Captures a frame from the webcam and runs detections at different confidence thresholds. 
# It displays how varying the threshold affects the number and quality of detected objects 
# by drawing bounding boxes and labels on copies of the image for visual comparison.


print("\n\n2. Confidence Score Analysis")
print("-" * 40)

def analyze_confidence_thresholds(image, thresholds=[0.1, 0.25, 0.5, 0.7, 0.9]):
    # You set the threshold numbers here. You can choose any values, with maximum of 1
    """
    Analyze how different confidence thresholds affect detections
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.ravel()
    
    for idx, threshold in enumerate(thresholds):
        # Run detection with specific threshold
        results = model(image, conf=threshold)
        
        # Copy image for drawing
        img_copy = image.copy()
        num_detections = 0
        
        # Draw detections
        for result in results:
            if result.boxes is not None:
                for box in result.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    
                    # Draw box
                    cv2.rectangle(img_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    # Draw label
                    label = f"{model.names[cls]}: {conf:.2f}"
                    cv2.putText(img_copy, label, (x1, y1-5),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    
                    num_detections += 1
        
        # Display
        axes[idx].imshow(cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB))
        axes[idx].set_title(f"Threshold: {threshold}\nDetections: {num_detections}")
        axes[idx].axis('off')
    
    # Hide extra subplot if any
    if len(thresholds) < len(axes):
        for j in range(len(thresholds), len(axes)):
            axes[j].axis('off')
    
    plt.suptitle("Effect of Confidence Threshold on Detections", fontsize=16)
    plt.tight_layout()
    plt.show()

print("Testing different confidence thresholds...")
print("(Using webcam capture for real objects)")

# Warm up webcam and capture one frame for analysis with better brightness
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam")
else:
    # Warm-up frames to allow auto exposure adjustment
    for _ in range(10):
        ret, frame = cap.read()
        if not ret:
            print("Warning: Couldn't read frame during warm-up")
    
    ret, frame = cap.read()
    cap.release()

    if ret:
        # Optional brightness/contrast correction if image still dark
        alpha = 1.3  # Contrast control (1.0-3.0)
        beta = 30    # Brightness control (0-100)
        adjusted_frame = cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)
        
        analyze_confidence_thresholds(adjusted_frame)
    else:
        print("Could not capture frame from webcam; using test image instead.")
        analyze_confidence_thresholds(test_img)


# Part 3: Understanding IoU (Intersection over Union)

# Defines a function to calculate IoU between two bounding boxes and visualizes 
# several scenarios with different overlaps to illustrate how IoU measures the accuracy of detections and their overlap.

print("\n3. Understanding IoU (Intersection over Union)")
print("-" * 40)

def calculate_iou(box1, box2):
    """
    Calculate IoU between two boxes
    Boxes in format [x1, y1, x2, y2]
    """
    # Calculate intersection
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    
    # Calculate areas
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    # Calculate union
    union = area1 + area2 - intersection
    
    # Calculate IoU
    iou = intersection / union if union > 0 else 0
    
    return iou

def visualize_iou_examples():
    """
    Visualize different IoU scenarios
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.ravel()
    
    # Define different box scenarios
    scenarios = [
        {
            'box1': [100, 100, 300, 300],
            'box2': [100, 100, 300, 300],
            'title': 'Perfect Match\nIoU = 1.0'
        },
        {
            'box1': [100, 100, 300, 300],
            'box2': [150, 150, 350, 350],
            'title': 'Good Overlap\nIoU ≈ 0.47'
        },
        {
            'box1': [100, 100, 300, 300],
            'box2': [250, 250, 450, 450],
            'title': 'Small Overlap\nIoU ≈ 0.07'
        },
        {
            'box1': [100, 100, 300, 300],
            'box2': [350, 100, 550, 300],
            'title': 'No Overlap\nIoU = 0.0'
        },
        {
            'box1': [100, 100, 400, 200],
            'box2': [150, 50, 350, 250],
            'title': 'Different Shapes\nIoU ≈ 0.29'
        },
        {
            'box1': [100, 100, 200, 200],
            'box2': [50, 50, 250, 250],
            'title': 'Small Inside Large\nIoU ≈ 0.17'
        }
    ]
    
    for idx, scenario in enumerate(scenarios):
        # Create blank image
        img = np.ones((500, 600, 3), dtype=np.uint8) * 255
        
        # Draw boxes
        box1 = scenario['box1']
        box2 = scenario['box2']
        
        # Draw box1 in red
        cv2.rectangle(img, (box1[0], box1[1]), (box1[2], box1[3]), 
                     (0, 0, 255), 3)
        cv2.putText(img, "Box 1", (box1[0], box1[1]-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # Draw box2 in blue
        cv2.rectangle(img, (box2[0], box2[1]), (box2[2], box2[3]), 
                     (255, 0, 0), 3)
        cv2.putText(img, "Box 2", (box2[0], box2[1]-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        
        # Calculate actual IoU
        iou = calculate_iou(box1, box2)
        
        # Display
        axes[idx].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        axes[idx].set_title(f"{scenario['title']}\nActual IoU = {iou:.3f}")
        axes[idx].axis('off')
    
    plt.suptitle("IoU (Intersection over Union) Examples", fontsize=16)
    plt.tight_layout()
    plt.show()

visualize_iou_examples()


# Part 4: Non-Maximum Suppression

# Implements a manual NMS algorithm to remove duplicate overlapping detections based on IoU and confidence scores. 
# It demonstrates the effect by showing bounding boxes before and after applying NMS.

print("\n4. Non-Maximum Suppression (NMS)")
print("-" * 40)

def manual_nms(boxes, scores, iou_threshold=0.5):
    """
    Perform NMS manually to understand the algorithm
    """
    # Sort boxes by confidence score (descending)
    indices = np.argsort(scores)[::-1]
    
    keep = []
    
    while len(indices) > 0:
        # Keep the box with highest confidence
        current = indices[0]
        keep.append(current)
        
        # Calculate IoU with remaining boxes
        ious = []
        for idx in indices[1:]:
            iou = calculate_iou(boxes[current], boxes[idx])
            ious.append(iou)
        
        # Remove boxes with high IoU (they're duplicates)
        indices = indices[1:]  # Remove current box
        if len(ious) > 0:
            indices = indices[np.array(ious) < iou_threshold]
    
    return keep

# Demonstrate NMS
def demonstrate_nms():
    """
    Show how NMS removes duplicate detections
    """
    print("\nDemonstrating NMS Effect:")
    print("-" * 30)
    
    # Create synthetic overlapping detections
    # Simulate multiple detections of the same object
    boxes = [
        [100, 100, 200, 200],  # Main detection
        [105, 95, 205, 195],   # Slightly offset
        [95, 105, 195, 205],   # Slightly offset
        [110, 110, 210, 210],  # Slightly offset
        [300, 300, 400, 400],  # Different object
        [305, 295, 405, 395],  # Slightly offset
    ]
    
    scores = [0.95, 0.92, 0.88, 0.85, 0.93, 0.90]
    
    print(f"Before NMS: {len(boxes)} detections")
    
    # Apply NMS
    keep_indices = manual_nms(boxes, scores, iou_threshold=0.5)
    
    print(f"After NMS: {len(keep_indices)} detections")
    print(f"Kept boxes: {keep_indices}")
    
    # Visualize
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Before NMS
    img1 = np.ones((500, 500, 3), dtype=np.uint8) * 255
    for i, (box, score) in enumerate(zip(boxes, scores)):
        color = (255 - i*30, 0, i*30)  # Gradient from red to blue
        cv2.rectangle(img1, (box[0], box[1]), (box[2], box[3]), color, 2)
        cv2.putText(img1, f"{score:.2f}", (box[0], box[1]-5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    ax1.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
    ax1.set_title(f"Before NMS: {len(boxes)} boxes")
    ax1.axis('off')
    
    # After NMS
    img2 = np.ones((500, 500, 3), dtype=np.uint8) * 255
    for idx in keep_indices:
        box = boxes[idx]
        score = scores[idx]
        cv2.rectangle(img2, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 3)
        cv2.putText(img2, f"{score:.2f}", (box[0], box[1]-5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    ax2.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
    ax2.set_title(f"After NMS: {len(keep_indices)} boxes")
    ax2.axis('off')
    
    plt.tight_layout()
    plt.show()

demonstrate_nms()


# Part 5: Detection Quality Metrics

# Defines a class that tracks detections over multiple frames, recording confidence scores and class counts. 
# It provides methods to plot detection frequency, confidence distributions, and averages to analyze model performance over time.

print("\n5. Detection Quality Metrics")
print("-" * 40)

class DetectionAnalyzer:
    """
    Analyze detection quality over time
    """
    def __init__(self):
        self.detections_history = []
        self.confidence_history = defaultdict(list)
        self.class_counts = defaultdict(int)
        
    def add_frame_detections(self, results):
        """
        Add detections from a frame
        """
        frame_detections = []
        
        for result in results:
            if result.boxes is not None:
                for box in result.boxes:
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    class_name = model.names[cls]
                    
                    frame_detections.append({
                        'class': class_name,
                        'confidence': conf,
                        'box': box.xyxy[0].cpu().numpy()
                    })
                    
                    self.confidence_history[class_name].append(conf)
                    self.class_counts[class_name] += 1
        
        self.detections_history.append(frame_detections)
        return len(frame_detections)
    
    def plot_analysis(self):
        """
        Plot detection analysis
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. Detections per frame over time
        detections_per_frame = [len(d) for d in self.detections_history]
        ax1.plot(detections_per_frame)
        ax1.set_title("Detections per Frame Over Time")
        ax1.set_xlabel("Frame")
        ax1.set_ylabel("Number of Detections")
        ax1.grid(True, alpha=0.3)
        
        # 2. Confidence distribution by class
        for class_name, confidences in self.confidence_history.items():
            if len(confidences) > 5:  # Only plot classes with enough detections
                ax2.hist(confidences, alpha=0.5, label=class_name, bins=20)
        ax2.set_title("Confidence Distribution by Class")
        ax2.set_xlabel("Confidence Score")
        ax2.set_ylabel("Frequency")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Class frequency
        classes = list(self.class_counts.keys())
        counts = list(self.class_counts.values())
        ax3.bar(classes, counts)
        ax3.set_title("Detection Frequency by Class")
        ax3.set_xlabel("Class")
        ax3.set_ylabel("Total Detections")
        ax3.tick_params(axis='x', rotation=45)
        
        # 4. Average confidence by class
        avg_conf = {cls: np.mean(confs) for cls, confs in self.confidence_history.items()}
        classes = list(avg_conf.keys())
        avg_values = list(avg_conf.values())
        ax4.bar(classes, avg_values)
        ax4.set_title("Average Confidence by Class")
        ax4.set_xlabel("Class")
        ax4.set_ylabel("Average Confidence")
        ax4.tick_params(axis='x', rotation=45)
        ax4.set_ylim(0, 1)
        
        plt.tight_layout()
        plt.show()


# Part 6: Live Analysis

# Uses the webcam to capture live video frames, runs YOLO detection on each frame, 
# annotates the detections, and displays them in real-time. It collects 
# detection data for analysis and prints summary statistics when stopped.

print("\n6. Live Detection Analysis")
print("-" * 40)
print("Let's analyze detections from your webcam!")
print("Press 'q' to stop and see analysis")

analyzer = DetectionAnalyzer()
cap = cv2.VideoCapture(0)

frame_count = 0
start_time = time.time()

try:
    while frame_count < 300:  # Analyze 300 frames or until 'q' pressed
        ret, frame = cap.read()
        if not ret:
            break
        
        # Run detection
        results = model(frame, conf=0.25)
        
        # Add to analyzer
        num_detections = analyzer.add_frame_detections(results)
        
        # Draw detections
        annotated_frame = results[0].plot()
        
        # Add frame info
        cv2.putText(annotated_frame, f"Frame: {frame_count}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(annotated_frame, f"Detections: {num_detections}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.imshow('Detection Analysis', annotated_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        frame_count += 1

except KeyboardInterrupt:
    pass

finally:
    cap.release()
    cv2.destroyAllWindows()
    
    duration = time.time() - start_time
    print(f"\nAnalyzed {frame_count} frames in {duration:.1f} seconds")
    print(f"Average FPS: {frame_count/duration:.1f}")
    
    if frame_count > 0:
        print("\nGenerating analysis plots...")
        analyzer.plot_analysis()

