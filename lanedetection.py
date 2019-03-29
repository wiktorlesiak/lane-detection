import cv2
import cv2 as cv
import numpy as np


video = cv2.VideoCapture('StayingInLane.avi')
fgbg = cv2.createBackgroundSubtractorMOG2()
while (1):
    ret, orig_frame = video.read()
    if not ret:
        video = cv2.VideoCapture('StayingInLane.avi')
        continue
    fgmask = fgbg.apply(orig_frame)

    frame = cv2.GaussianBlur(orig_frame, (5, 5), 0)
    # grayscale video
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # hsv mask
    # TODO: FIX THE MASKING 
    lower = np.array([0, 0, 212])
    upper = np.array([131, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)
    res = cv2.bitwise_and(frame, frame, mask=mask)
    # canny edge detection
    edges = cv2.Canny(frame, 120, 200)
    # Hough Lines
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 50, maxLineGap=50)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 5)
    # output video
    cv2.imshow("edges", edges)
    cv2.imshow('lines', frame)
    cv2.imshow('mask', mask)
    cv2.imshow('Original', fgmask)


    k = cv2.waitKey(20) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()
video.release()