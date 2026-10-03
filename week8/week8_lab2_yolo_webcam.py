"""
Week 8 Lab 2: Implementing YOLO with Webcam Feed
Real-time Object Detection

In this lab, we'll:
1. Install and load YOLO
2. Set up webcam preprocessing
3. Run real-time object detection
4. Visualize results with bounding boxes
"""

import cv2
import numpy as np
import time
from collections import deque

# First, let's install the required package
# You may need to upgrade pip:
# (comment out the below line when you copy and paste this into your Command Line)
# pip install --upgrade pip

# Run this install in your terminal:
# (comment out the below line when you copy and paste this into your Command Line)
# pip install ultralytics


# Part 1: Load YOLO Model

print("\n1. Loading YOLO Model")
print("-" * 30)

from ultralytics import YOLO

# Load a pre-trained model
# We'll use YOLOv8n (nano) for speed on most computers
model = YOLO('yolov8n.pt')  # This will download on first run (~6MB)

print("Model loaded successfully!")
print(f"Model type: YOLOv8 nano")
print(f"Number of classes: {len(model.names)}")
print(f"Some classes: {list(model.names.values())[:10]}...")

# Part 2: Webcam Setup

print("\n2. Setting up Webcam")
print("-" * 30)

# Initialize webcam
cap = cv2.VideoCapture(0)  # 0 = default camera

# Check if camera opened successfully
if not cap.isOpened():
    print("Error: Could not open webcam")
    print("Please check:")
    print("- Camera is connected")
    print("- No other application is using the camera")
    print("- You have camera permissions enabled")
    exit()

# Set camera properties for better performance
# The FPS setting asks the camera how many frames per second to capture, 
# but the real speed depends on how fast the computer processes each frame.
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)

print("Webcam initialized.")
print(f"Resolution: {int(cap.get(3))}x{int(cap.get(4))}")
print(f"FPS: {int(cap.get(5))}")


# Part 3: Preprocessing Pipeline
print("\n3. Creating Preprocessing Pipeline")
print("-" * 30)

def preprocess_frame(frame):
    """
    Preprocess frame for better detection
    This mimics augmentation techniques from lecture
    """
    # 1. Resize for consistent input
    # YOLO works with various sizes, but 640x640 is standard
    # We'll keep aspect ratio and pad if needed
    height, width = frame.shape[:2]
    
    # Calculate scaling to fit in 640x640
    scale = min(640/width, 640/height)
    new_width = int(width * scale)
    new_height = int(height * scale)
    
    # Resize image
    resized = cv2.resize(frame, (new_width, new_height))
    
    # Create padded image
    padded = np.zeros((640, 640, 3), dtype=np.uint8)
    # Calculate padding
    y_offset = (640 - new_height) // 2
    x_offset = (640 - new_width) // 2
    
    # Place resized image in center
    padded[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = resized
    
    return padded, scale, x_offset, y_offset

print("Preprocessing functions ready!")


# Part 4: Detection Colors
# Generate random colors for each class
np.random.seed(42)  # For consistent colors
COLORS = np.random.randint(0, 255, size=(len(model.names), 3), dtype=np.uint8)

# Part 5: Main Detection Loop
print("\n4. Starting Detection Loop...")
print("-" * 30)
print("\nControls:")
print("- Press 'q' to quit")
print("- Press 's' to save screenshot")
print("- Press 'c' to toggle confidence display")
print("- Press 'f' to toggle FPS display")
print("\nLook for common objects: person, chair, cup, cell phone, etc.")

# Performance tracking
fps_history = deque(maxlen=30)  # Store last 30 FPS values
show_confidence = True
show_fps = True
frame_count = 0
screenshot_count = 0

print("\nStarting camera feed...\n")

try:
    while True:
        # Start time for FPS calculation
        start_time = time.time()
        
        # Capture frame
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
        
        # Preprocess frame
        processed_frame, scale, x_offset, y_offset = preprocess_frame(frame)
        
        # Run YOLO detection
        results = model(processed_frame, conf=0.25)  # 0.25 confidence threshold
        
        # Process detections
        for result in results:
            boxes = result.boxes
            
            if boxes is not None:
                # Get box coordinates, confidence, and class
                for box in boxes:
                    # Get coordinates (x1, y1, x2, y2)
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    
                    # Convert back to original image coordinates
                    x1 = int((x1 - x_offset) / scale)
                    y1 = int((y1 - y_offset) / scale)
                    x2 = int((x2 - x_offset) / scale)
                    y2 = int((y2 - y_offset) / scale)
                    
                    # Get confidence and class
                    confidence = float(box.conf[0])
                    class_id = int(box.cls[0])
                    class_name = model.names[class_id]
                    
                    # Get color for this class
                    color = COLORS[class_id].tolist()
                    
                    # Draw bounding box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    
                    # Create label
                    label = f"{class_name}"
                    if show_confidence:
                        label += f" {confidence:.2f}"
                    
                    # Draw label background
                    label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                    cv2.rectangle(frame, (x1, y1 - label_size[1] - 4), 
                                (x1 + label_size[0], y1), color, -1)
                    
                    # Draw label text
                    cv2.putText(frame, label, (x1, y1 - 2), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Calculate FPS
        end_time = time.time()
        fps = 1 / (end_time - start_time)
        fps_history.append(fps)
        avg_fps = np.mean(fps_history)
        
        # Draw FPS if enabled
        if show_fps:
            fps_text = f"FPS: {avg_fps:.1f}"
            cv2.putText(frame, fps_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Draw help text
        cv2.putText(frame, "Press 'q' to quit, 's' for screenshot", (10, frame.shape[0] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Display frame
        cv2.imshow('YOLO Object Detection', frame)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord('s'):
            # Save screenshot
            filename = f'yolo_detection_{screenshot_count}.jpg'
            cv2.imwrite(filename, frame)
            print(f"Screenshot saved as {filename}")
            screenshot_count += 1
        elif key == ord('c'):
            show_confidence = not show_confidence
            print(f"Confidence display: {'ON' if show_confidence else 'OFF'}")
        elif key == ord('f'):
            show_fps = not show_fps
            print(f"FPS display: {'ON' if show_fps else 'OFF'}")
        
        frame_count += 1
        
        # Print statistics every 100 frames
        if frame_count % 100 == 0:
            print(f"Processed {frame_count} frames | Avg FPS: {avg_fps:.1f}")

except KeyboardInterrupt:
    print("\n\nDetection stopped by user")

finally:
    # Clean up
    print("\nCleaning up...")
    cap.release()
    cv2.destroyAllWindows()
    
    print(f"\nSession Statistics:")
    print(f"- Total frames processed: {frame_count}")
    print(f"- Average FPS: {np.mean(fps_history):.1f}")
    print(f"- Screenshots saved: {screenshot_count}")

