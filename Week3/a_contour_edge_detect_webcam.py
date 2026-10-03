# https://learnopencv.com/edge-detection-using-opencv/

import cv2 as cv

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

    # Convert to graycsale
    img_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    # Blur the image for better edge detection
    img_blur = cv.GaussianBlur(frame, (3,3), 0) 

    # Canny Edge Detection
    edges = cv.Canny(image=img_blur, threshold1=100, threshold2=200) # Canny Edge Detection
    # Find contours in the mask
    contours, _ = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    
    # Draw bounding boxes around detected objects
    for contour in contours:
        if cv.contourArea(contour) > 1000:  # Adjust this threshold as needed
            x, y, w, h = cv.boundingRect(contour)
            cv.rectangle(frame, (x, y), (x+w, y+h), (250, 229, 202), 2)

            # print(contour)

    #show the current frame and the fg masks
    cv.imshow('Frame', frame)
    cv.imshow('Edge detected', edges)

    keyboard = cv.waitKey(30)
    if keyboard == 'q' or keyboard == 27:
        break

# When everything is done, release the capture
capture.release()
cv.destroyAllWindows()