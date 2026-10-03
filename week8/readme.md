# This Week's Labs

## Lab 1 Data Augmentation and CNN Workflow
You can find this lab on Google Colab: [click here to access and make a copy for your Google Drive](https://colab.research.google.com/drive/1F8PjIgfkPEwTjRO5AYu5QG84LLCl1vbW?usp=sharing)

## Lab 2 + 3 (`.py` files)

Here are some additional exercises if you want to play further:

### Lab 2 YOLO with Webcam:

1. PERFORMANCE TESTING:
   - Try different YOLO models (yolov8s.pt, yolov8m.pt)
   - Compare FPS and accuracy
   - Which is best for your computer?

2. CONFIDENCE THRESHOLD:
   - Modify the conf parameter in model()
   - Try 0.1, 0.5, 0.7
   - What happens to false positives/negatives?

3. CLASS FILTERING:
   - Modify code to only show specific classes
   - Example: Only detect 'person' and 'cell phone'
   - Hint: Check class_name before drawing

4. DETECTION ZONES:
   - Create a "detection zone" in the center of the frame
   - Only show detections within this zone
   - Draw the zone boundary

5. OBJECT COUNTING:
   - Count how many objects of each type are detected
   - Display counts on screen
   - Track maximum count seen

6. ROBOT BEHAVIOR:
   - Print "STOP!" if a person is detected too close (large bounding box)
   - Print "Turn left/right" based on the person's position
   - This simulates basic robot reactions!

Can you make the camera "follow" a specific object by printing movement commands based on the object's position?

### Lab 3 Bounding Boxes and Confidence Scores

1. IoU THRESHOLD EXPERIMENT:
   - Modify the NMS IoU threshold (try 0.3, 0.7, 0.9)
   - How does it affect duplicate removal?
   - What's the trade-off?

2. CONFIDENCE ANALYSIS:
   - Track confidence scores for specific objects over time
   - Do they vary with distance? Lighting? Occlusion?
   - Create a plot showing the relationship

3. FALSE POSITIVE STUDY:
   - Find cases where YOLO detects incorrectly
   - What confidence scores do false positives have?
   - Can you find a pattern?

4. DETECTION STABILITY:
   - Track the same object across frames
   - Does the bounding box "jitter"?
   - Calculate IoU between consecutive frames

5. CUSTOM METRICS:
   - Create a "detection quality score" combining:
     * Confidence
     * Box stability (low jitter)
     * Consistency (detected in most frames)

6. MULTI-MODEL COMPARISON:
   - Run YOLOv8n and YOLOv8s on same video
   - Compare their confidence distributions
   - Which is more consistent?

Additional Challenge:
Implement a "detection smoother" that:
- Tracks objects across frames
- Averages bounding boxes over time
- Only shows stable detections
This would be useful for robots to avoid reacting to brief false detections.
