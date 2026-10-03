import cv2 as cv
import numpy as np

# Load an image
image = cv.imread('image/tamagotchi1.jpeg', cv.IMREAD_COLOR)

# What is it?
print(type(image))        # <class 'numpy.ndarray'>
print(image.shape)        # (height, width, 3), in this case, it's (480, 623, 3)
print(image.dtype)        # uint8 (0-255 for each channel)

# Example: (480, 623, 3) means:
# - 480 pixels tall
# - 623 pixels wide
# - 3 color channels (Blue, Green, Red)

# Total elements
print(image.size)         # 897,120 (480 × 623 × 3)