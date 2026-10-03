# https://docs.opencv.org/4.x/d1/dee/tutorial_introduction_to_pca.html
# https://github.com/opencv/opencv/blob/4.x/samples/python/tutorial_code/ml/introduction_to_pca/introduction_to_pca.py

# https://docs.opencv.org/3.4/d4/d73/tutorial_py_contours_begin.html

from __future__ import print_function
from __future__ import division
import cv2 as cv
import numpy as np
from math import atan2, cos, sin, sqrt, pi

def drawAxis(img, p_, q_, colour, scale):
    p = list(p_)
    q = list(q_)
    ## [visualization1]
    angle = atan2(p[1] - q[1], p[0] - q[0]) # angle in radians
    hypotenuse = sqrt((p[1] - q[1]) * (p[1] - q[1]) + (p[0] - q[0]) * (p[0] - q[0]))

    # Here we lengthen the arrow by a factor of scale
    q[0] = p[0] - scale * hypotenuse * cos(angle)
    q[1] = p[1] - scale * hypotenuse * sin(angle)
    cv.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), colour, 1, cv.LINE_AA)

    # create the arrow hooks
    p[0] = q[0] + 9 * cos(angle + pi / 4)
    p[1] = q[1] + 9 * sin(angle + pi / 4)
    cv.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), colour, 1, cv.LINE_AA)

    p[0] = q[0] + 9 * cos(angle - pi / 4)
    p[1] = q[1] + 9 * sin(angle - pi / 4)
    cv.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), colour, 1, cv.LINE_AA)
    ## [visualization1]

def getOrientation(pts, img):
    ## [pca]
    # Construct a buffer used by the pca analysis
    sz = len(pts)
    data_pts = np.empty((sz, 2), dtype=np.float64)
    # print(data_pts)
    for i in range(data_pts.shape[0]):
        data_pts[i,0] = pts[i,0,0]
        data_pts[i,1] = pts[i,0,1]

    # Perform PCA analysis
    mean = np.empty((0))
    mean, eigenvectors, eigenvalues = cv.PCACompute2(data_pts, mean)

    # Store the center of the object
    centre = (int(mean[0,0]), int(mean[0,1]))
   

    ## [visualization]
    # Draw the principal components
    cv.circle(img, centre, 3, (255, 0, 255), 2)
    p1 = (centre[0] + 0.02 * eigenvectors[0,0] * eigenvalues[0,0], centre[1] + 0.02 * eigenvectors[0,1] * eigenvalues[0,0])
    p2 = (centre[0] - 0.02 * eigenvectors[1,0] * eigenvalues[1,0], centre[1] - 0.02 * eigenvectors[1,1] * eigenvalues[1,0])
    drawAxis(img, centre, p1, (0, 255, 0), 1)
    drawAxis(img, centre, p2, (255, 255, 0), 5)

    angle = atan2(eigenvectors[0,1], eigenvectors[0,0]) # orientation in radians
    angle_in_degrees = -int(np.rad2deg(angle)) * -1 - 90 
    # convert radians into degrees using numpy https://www.geeksforgeeks.org/numpy-degrees-rad2deg-python/
    print(angle_in_degrees)

    # Label with the rotation angle
    label = "  Rotation Angle: " + str(angle_in_degrees) + " degrees"
    textbox = cv.rectangle(img, (centre[0], centre[1]-25), (centre[0] + 250, centre[1] + 10), (255,255,255), -1)
    cv.putText(img, label, (centre[0], centre[1]), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv.LINE_AA)

    return angle

## [pre-process]
# Load image

src = cv.imread(cv.samples.findFile('image/input_img.jpg'))
# Check if image is loaded successfully
if src is None:
    print('Could not open or find the image')
    exit(0)

cv.imshow('src', src)

# Convert image to grayscale
gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)

# Convert image to binary
_, bw = cv.threshold(gray, 50, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
## [pre-process]

## [contours]
# Find all the contours in the thresholded image
contours, _ = cv.findContours(bw, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)

for i, c in enumerate(contours):
    # Calculate the area of each contour
    area = cv.contourArea(c)
    # Ignore contours that are too small or too large
    if area < 3700 or 100000 < area:
        continue

    # Draw each contour only for visualisation purposes
    cv.drawContours(src, contours, i, (0, 0, 255), 2)
    # Find the orientation of each shape
    getOrientation(c, src)
## [contours]

cv.imshow('output', src)
cv.waitKey()
