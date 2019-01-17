import numpy as np
import cv2
import signal
import sys
from imutils.video import VideoStream
import imutils

scale = 1
blur = 3
min_area = 100
max_area = 10000
main_window = 'overlay'

def signal_handler(sig, frame):
    print('\nRelease cap..')
    cap.release()
    cv2.destroyAllWindows()
    print('Exit..')
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)


cap = cv2.VideoCapture("rtsp://admin:admin@192.168.1.29:554/RVi/1/3")

ret, last_frame = cap.read()
last_frame = cv2.resize(last_frame, (0,0), fx=scale, fy=scale)
last_frame_gray = cv2.cvtColor(last_frame, cv2.COLOR_BGR2GRAY)
last_frame_gray = cv2.GaussianBlur(last_frame_gray, (blur, blur), 0)

while(1):
    ret, frame = cap.read()
    frame = cv2.resize(frame, (0,0), fx=scale, fy=scale)
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_gray = cv2.GaussianBlur(frame_gray, (blur, blur), 0)

    fgmask = cv2.absdiff(last_frame_gray, frame_gray)
    fgmask = cv2.threshold(fgmask, 25, 255, cv2.THRESH_BINARY)[1]
    fgmask = cv2.dilate(fgmask, None, iterations=2)


    cnts = cv2.findContours(fgmask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    last_frame_gray = frame_gray

    for c in cnts:
        if cv2.contourArea(c) < min_area:
            continue
        if cv2.contourArea(c) > max_area:
            (x, y, w, h) = cv2.boundingRect(c)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            continue

        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    fgmask_rgba = cv2.cvtColor(fgmask, cv2.COLOR_GRAY2RGBA)
    fgmask_rgba[np.where((fgmask_rgba == [255,255,255,255]).all(axis = 2))] = [0,0,225,0]
    fgmask_rgba[np.where((fgmask_rgba == [127,127,127,255]).all(axis = 2))] = [0,0,225,0]
    fgmask_rgba[np.where((fgmask_rgba == [0,0,0,255]).all(axis = 2))] = [0,0,0,0]



    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2RGBA)
    added_image = cv2.addWeighted(frame,1,fgmask_rgba,0.5,0)
    cv2.namedWindow(main_window)
    cv2.moveWindow(main_window, 40, 30)
    cv2.imshow(main_window, added_image)
    cv2.waitKey(1)


