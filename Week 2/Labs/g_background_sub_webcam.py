# references
# https://docs.opencv.org/4.x/d1/dc5/tutorial_background_subtraction.html

import cv2 as cv

# Create a background subtractor object: Background subtraction method - KNN or MOG2
backSub = cv.createBackgroundSubtractorMOG2()
# backSub = cv.createBackgroundSubtractorKNN()

capture = cv.VideoCapture(0)
if not capture.isOpened():
    print("Cannot open camera")
    exit()

while True:
    ret, frame = capture.read()
    if frame is None:
        break
    frame = cv.flip(frame, 1) 

    #update the background model
    fgMask = backSub.apply(frame)

    #show the current frame and the fg masks
    cv.imshow('Frame', frame)
    cv.imshow('FG Mask', fgMask)

    keyboard = cv.waitKey(30)
    if keyboard == 27:
        break

capture.release()
cv.destroyAllWindows()