# References 
# https://docs.opencv.org/3.4/db/df6/tutorial_erosion_dilatation.html
# https://docs.opencv.org/4.x/d9/d61/tutorial_py_morphological_ops.html


import cv2 as cv
import numpy as np

img = cv.imread('image/tamagotchi1.jpeg', cv.IMREAD_GRAYSCALE)

# Create 5×5 kernel
kernel = np.ones((5,5), np.uint8)

# Apply operations
erosion = cv.erode(img, kernel, iterations=1)
dilation = cv.dilate(img, kernel, iterations=1)

erosion_multi = cv.erode(img, kernel, iterations=5)
dilation_multi = cv.dilate(img, kernel, iterations=5)

# Display results
cv.imshow('Original', img)
cv.imshow('Erosion (1 iteration)', erosion)
cv.imshow('Dilation (1 iteration)', dilation)
cv.imshow('Erosion (5 iterations)', erosion_multi)
cv.imshow('Dilation (5 iterations)', dilation_multi)

cv.waitKey(0)
cv.destroyAllWindows()
