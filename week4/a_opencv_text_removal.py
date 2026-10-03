# Today's lab is adapted from https://opencv.org/blog/text-detection-and-removal-using-opencv/#h-text-removal-using-opencv

import cv2
import numpy as np
import os

# !! NOTE: 
# Before you start, please download the East Model (the .pb file) and save it to the class folder you are working from: 
# https://www.dropbox.com/s/r2ingd0l3zt8hxs/frozen_east_text_detection.tar.gz?dl=1 
# Also double check that your image file path is correct

MODEL_PATH = "frozen_east_text_detection.pb" 
IMAGE_FILE = "images/test_image.jpg" 
INPUT_SIZE = (320, 320)  # Must be a multiple of 32 for the EAST model
CONFIDENCE_THRESHOLD = 0.8  # Minimum confidence score for text detection (0-1)
NMS_THRESHOLD = 0.4  # Non-Maximum Suppression threshold to filter overlapping boxes
# Mean values used for training the EAST model (BGR format)
MEAN_VALUES = (123.68, 116.78, 103.94)
# Option to save results
SAVE_RESULTS = True

# Check if model file exists
if not os.path.exists(MODEL_PATH):
    print(f"Error: EAST model not found at {MODEL_PATH}")
    print("Please download the frozen_east_text_detection.pb file and update the MODEL_PATH variable.")
    exit()

# 1. Load the Image and Prepare Copies
image_orig = cv2.imread(IMAGE_FILE)
if image_orig is None:
    print(f"Error: Could not load image at {IMAGE_FILE}. Check the path/filename.")
    exit()

# Store original dimensions for potential resize back
orig_height, orig_width = image_orig.shape[:2]
print(f"[INFO] Original image size: {orig_width}x{orig_height}")

# Resize for consistent model input (EAST requires multiples of 32)
image = cv2.resize(image_orig, INPUT_SIZE)
annotated_image = image.copy()
inpaint_image = image.copy()

# 2. Load the Deep Learning EAST Text Detector
print("[INFO] Loading EAST Text Detector...")
print("[INFO] Note: EAST works best on clear text with good contrast.")
print("[INFO] It may struggle with handwritten, curved, or very small text.")
textDetector = cv2.dnn_TextDetectionModel_EAST(MODEL_PATH)

# Set the deep learning model's parameters
textDetector.setConfidenceThreshold(CONFIDENCE_THRESHOLD).setNMSThreshold(NMS_THRESHOLD)
textDetector.setInputParams(1.0, INPUT_SIZE, MEAN_VALUES, True)

# 3. Detect Text
# 'boxes' contains the four corner points (polygons) of detected text regions
boxes, _ = textDetector.detect(image)
print(f"[INFO] Detected {len(boxes)} text regions.")

# 4. Create the Inpainting Mask and Annotate
# Initialize the mask as a black image (0s)
inpaint_mask = np.zeros(image.shape[:2], dtype=np.uint8)

# Loop over all detected text boxes
for box in boxes:
    # Fill the polygon shape on the mask with white (255) to mark the area for removal
    cv2.fillPoly(inpaint_mask, [np.array(box, np.int32)], 255)
    
    # Draw the bounding box on the annotated image for visualization (Green)
    cv2.polylines(annotated_image, [np.array(box, np.int32)], 
                  isClosed=True, color=(0, 255, 0), thickness=1)

# 5. Perform Inpainting (The Removal Step)
# Compare both inpainting algorithms
print("[INFO] Performing text removal using two different algorithms...")

# INPAINT_NS: Navier-Stokes based method
# Good for: Larger regions, smoother results
inpainted_ns = cv2.inpaint(
    inpaint_image, 
    inpaint_mask, 
    inpaintRadius=5,  # Size of neighborhood around pixel to consider (larger = smoother but slower)
    flags=cv2.INPAINT_NS  # Navier-Stokes algorithm
)

# INPAINT_TELEA: Fast Marching Method
# Good for: Smaller regions, preserves edges better
inpainted_telea = cv2.inpaint(
    inpaint_image, 
    inpaint_mask, 
    inpaintRadius=5,  # Size of neighborhood around pixel to consider
    flags=cv2.INPAINT_TELEA  # Alexandru Telea's algorithm
)

# 6. Resize results back to original size
final_output_ns = cv2.resize(inpainted_ns, (orig_width, orig_height))
final_output_telea = cv2.resize(inpainted_telea, (orig_width, orig_height))
annotated_resized = cv2.resize(annotated_image, (orig_width, orig_height))
mask_resized = cv2.resize(inpaint_mask, (orig_width, orig_height))

# 7. Save results if requested
if SAVE_RESULTS:
    print("[INFO] Saving results...")
    cv2.imwrite('output_annotated.jpg', annotated_resized)
    cv2.imwrite('output_mask.jpg', mask_resized)
    cv2.imwrite('output_inpainted_ns.jpg', final_output_ns)
    cv2.imwrite('output_inpainted_telea.jpg', final_output_telea)
    print("[INFO] Results saved to current directory.")

# 8. Show Results
# Create comparison views
comparison_top = np.hstack([image, annotated_image])
comparison_bottom = np.hstack([inpainted_ns, inpainted_telea])

# Display windows
cv2.imshow('1. Original vs Annotated (Detection)', comparison_top)
cv2.imshow('2. Inpainting Mask', inpaint_mask)
cv2.imshow('3. Navier-Stokes vs TELEA Algorithm', comparison_bottom)
cv2.imshow('4. Final Output - Navier-Stokes (Full Size)', final_output_ns)
cv2.imshow('5. Final Output - TELEA (Full Size)', final_output_telea)

print("\n[INFO] Press any key to close all windows...")
print("[TIP] Navier-Stokes: Better for larger text regions, smoother results")
print("[TIP] TELEA: Better for smaller regions, preserves edges and textures")

cv2.waitKey(0)
cv2.destroyAllWindows()
