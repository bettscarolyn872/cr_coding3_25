# Reference:
# https://docs.opencv.org/4.x/db/deb/tutorial_display_image.html
# 
# image sources:
# Image: File:Tamagotchi 0124 ubt.jpeg
# https://commons.wikimedia.org/wiki/File:Tamagotchi_0124_ubt.jpeg 

# Image: File:C624ADHE0.jpg
# https://commons.wikimedia.org/wiki/File:C624ADHE0.jpg
# Die shot of the Epson 4-bit microcontroller C624ADHE0 (E0C62 family) used in the Tamagotchi Gen 2 (P2).
# ////////////////////////////////

import cv2 as cv

file_path = "image/tamagotchi1.jpeg"

img_loaded = cv.imread(file_path, cv.IMREAD_COLOR)

if img_loaded is None:
    exit("Could not read the image.")    
    # if the image is not found, it termintates the programme and print a message

cv.imshow("window", img_loaded)
# displaying the image loaded

close_with_a_key = cv.waitKey(0)
# pressing any key will close the window

if close_with_a_key == ord("s"):
    cv.imwrite("image/tamagotchi1.png", img_loaded)
    # if the key pressed is 's' it'll save the image as .png
