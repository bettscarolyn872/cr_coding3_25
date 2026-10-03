# https://docs.opencv.org/4.x/d1/dc5/tutorial_background_subtraction.html

import cv2 as cv
import numpy as np


# Create a background subtractor object: Background subtraction method - KNN or MOG2
backSub = cv.createBackgroundSubtractorMOG2()
# backSub = cv.createBackgroundSubtractorKNN()

capture = cv.VideoCapture(0)
if not capture.isOpened():
    print("Cannot open camera")
    exit()

while True:
    ret, frame = capture.read()
    frame = cv.flip(frame, 1) 
    # flip the camera horizontally 
    if frame is None:
        break

    #update the background model
    fgMask = backSub.apply(frame)

    # Apply 2 morphological operations: 'open' and 'close' to denoise the mask - by using the numpy generated kernel sized 5x5 
    kernel = np.ones((5,5), np.uint8)
    fgMask = cv.morphologyEx(fgMask, cv.MORPH_OPEN, kernel)
    fgMask = cv.morphologyEx(fgMask, cv.MORPH_CLOSE, kernel)
    # more about 'open' and 'close'
    # https://docs.opencv.org/4.x/d9/d61/tutorial_py_morphological_ops.html

    # Find contours in the mask
    contours, _ = cv.findContours(fgMask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    
    # Draw bounding boxes around detected objects
    for contour in contours:
        if cv.contourArea(contour) > 1000:  # Adjust this threshold as needed
            x, y, w, h = cv.boundingRect(contour)
            cv.rectangle(frame, (x, y), (x+w, y+h), (36, 255, 12), 8)

            # print(contour)

    #show the current frame and the fg masks
    cv.imshow('Frame', frame)
    cv.imshow('FG Mask', fgMask)

    keyboard = cv.waitKey(30)
    if keyboard == 27:
        break
capture.release()
cv.destroyAllWindows()