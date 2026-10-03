# references
# https://learnopencv.com/blob-detection-using-opencv-python-c/

# Standard imports
import cv2 as cv
import numpy as np

# ////// blob info code starts
# Function to get blob information as a string
def get_blob_info(keypoints):
    info = f"Detected {len(keypoints)} blobs:\n"
    for i, keypoint in enumerate(keypoints):
        x, y = keypoint.pt
        s = keypoint.size
        info += f"Blob {i+1}: Position (x, y) = ({x:.2f}, {y:.2f}), Size = {s:.2f}\n"
    return info

# Variable to store the last blob information
last_blob_info = ""
# ////// blob info code ends

# Setup SimpleBlobDetector parameters.
params = cv.SimpleBlobDetector_Params()

# Change thresholds
params.minThreshold = 10
params.maxThreshold = 200

# Filter by Area.
params.filterByArea = True
params.minArea = 1000
params.maxArea = 10000

# Filter by Circularity
params.filterByCircularity = True
params.minCircularity = 0.3

# Filter by Convexity
params.filterByConvexity = True
params.minConvexity = 0.8
    
# Filter by Inertia
params.filterByInertia = True
params.minInertiaRatio = 0.01

# Filter by colour - 0 for dark blob 255 for a light blob
params.filterByColor = True
# params.blobColor = 0
# params.blobColor = 255

# Create a detector with the parameters
detector = cv.SimpleBlobDetector_create(params)

# Initialize the webcam
cap = cv.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    if not ret:
        print("Failed to grab frame")
        break
    frame = cv.flip(frame, 1) 

    # Convert to grayscale
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)


    # https://docs.opencv.org/4.x/d9/d61/tutorial_py_morphological_ops.html
    # create a kernel to use erode/dilate w
    kernel = np.ones((7,7),np.uint8)
    # apply erosion or dilation
    modified_grey = cv.erode(gray,kernel,iterations = 2)
    # modified_grey = cv.dilate(gray,kernel,iterations = 1)

    input_feed = modified_grey   
    # input_feed = gray

    # Detect blobs
    keypoints = detector.detect(input_feed)

    # Draw detected blobs as red circles
    im_with_keypoints = cv.drawKeypoints(input_feed, keypoints, np.array([]), (0,0,255), cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    # Show blobs
    cv.imshow("Keypoints", im_with_keypoints)

    # ////// blob info code starts
    # Get current blob information
    current_blob_info = get_blob_info(keypoints)
    
    # Print blob information only if it has changed
    if current_blob_info != last_blob_info:
        print(current_blob_info)
        last_blob_info = current_blob_info
    # ////// blob info code ends

    # Break the loop if 'q' is pressed
    if cv.waitKey(1) == ord('q'):
        break

# Release the webcam and close all windows
cap.release()
cv.destroyAllWindows()