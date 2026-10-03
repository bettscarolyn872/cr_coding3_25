# references
# This code example and the image used in the code has been adapted from the Learn Open CV tutorial below
# https://learnopencv.com/blob-detection-using-opencv-python-c/
# https://medium.com/image-processing-in-robotics/blob-detection-309226a3ea5b

import cv2
import numpy as np

def nothing(x):
    pass

# Function to get blob information as a string
def get_blob_info(keypoints):
    info = f"Detected {len(keypoints)} blobs:\n"
    for i, keypoint in enumerate(keypoints):
        x, y = keypoint.pt
        s = keypoint.size
        info += f"Blob {i+1}: Position (x, y) = ({x:.2f}, {y:.2f}), Size = {s:.2f}\n"
    return info

# Create a window for the sliders
cv2.namedWindow('Blob Detection')

# Create trackbars for min and max thresholds
cv2.createTrackbar('Min Threshold', 'Blob Detection', 10, 255, nothing)
cv2.createTrackbar('Max Threshold', 'Blob Detection', 200, 255, nothing)

# Read image
image_path1 = "image/BlobTest.jpg"
im = cv2.imread(image_path1, cv2.IMREAD_GRAYSCALE)

# Variable to store the last blob information
last_blob_info = ""

while True:
    # Get current positions of the trackbars
    min_threshold = cv2.getTrackbarPos('Min Threshold', 'Blob Detection')
    max_threshold = cv2.getTrackbarPos('Max Threshold', 'Blob Detection')

    # Ensure min_threshold is less than max_threshold
    if min_threshold >= max_threshold:
        max_threshold = min(min_threshold + 1, 255)
        cv2.setTrackbarPos('Max Threshold', 'Blob Detection', max_threshold)

    # Setup SimpleBlobDetector parameters.
    params = cv2.SimpleBlobDetector_Params()

    # Change thresholds
    params.minThreshold = min_threshold
    params.maxThreshold = max_threshold

    # Set minRepeatability to 1 to avoid the warning
    params.minRepeatability = 1

    # Filter by Area.
    params.filterByArea = True
    params.minArea = 500

    # Filter by Circularity
    params.filterByCircularity = True
    params.minCircularity = 0.5

    # Filter by Convexity
    params.filterByConvexity = True
    params.minConvexity = 0.87

    # Filter by Inertia
    params.filterByInertia = True
    params.minInertiaRatio = 0.01

    # Create a detector with the parameters
    detector = cv2.SimpleBlobDetector_create(params)

    # Detect blobs.
    keypoints = detector.detect(im)

    # Draw detected blobs as red circles.
    im_with_keypoints = cv2.drawKeypoints(im, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    # Show blobs
    cv2.imshow("Blob Detection", im_with_keypoints)
    
    # Get current blob information
    current_blob_info = get_blob_info(keypoints)
    
    # Print blob information only if it has changed
    if current_blob_info != last_blob_info:
        print(current_blob_info)
        last_blob_info = current_blob_info

    # Break the loop if 'esc' is pressed
    if cv2.waitKey(30) == 27:
        break 

cv2.destroyAllWindows()