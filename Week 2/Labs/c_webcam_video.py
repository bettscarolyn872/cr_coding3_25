import cv2 as cv

# Initialize camera (0 = default camera)
capture = cv.VideoCapture(0)

if not capture.isOpened():
    print("Cannot open camera")
    exit()

# Create resizable window
cv.namedWindow('frame', cv.WINDOW_NORMAL)
cv.resizeWindow('frame', 800, 600)

while True:
    ret, frame = capture.read()
    
    if not ret:
        print("Can't receive frame. Exiting ...")
        break
    
    # Flip horizontally (mirror effect)
    frame = cv.flip(frame, 1)
    
    cv.imshow('frame', frame)
    
    # ESC key (27) to exit
    if cv.waitKey(30) == 27:
        break

capture.release()
cv.destroyAllWindows()