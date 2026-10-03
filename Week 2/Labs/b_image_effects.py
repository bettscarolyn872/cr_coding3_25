# References
# https://docs.opencv.org/4.x/da/d6a/tutorial_trackbar.html 
# https://docs.opencv.org/4.x/dc/da5/tutorial_py_drawing_functions.html

import cv2 as cv
import numpy as np

rect_size = 126
y_pos = 150

def update_rectangle(x):
    img = np.zeros((512,512,3), np.uint8)
    # Create a black image
    cv.rectangle(img,(x,0+y_pos),(x+rect_size,rect_size+y_pos),(0,255,0),3)
    cv.imshow('image', img)

# Create a window
cv.namedWindow('image')

# Create trackbar
cv.createTrackbar('X Position', 'image', 0, 385, update_rectangle)

# place the initial rectangle
update_rectangle(0)

cv.waitKey(0)

cv.destroyAllWindows()


