import numpy as np
import cv2
import signal
import sys
from imutils.video import VideoStream
import argparse
import imutils
import json
from shapely.geometry import Polygon


detect_areas = []


ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help="video type")
ap.add_argument("-m", "--min-area", type=int, default=100, help="minimum area size")
ap.add_argument("-M", "--max-area", type=int, default=10000, help="maximum area size")
ap.add_argument("-a", "--areas", default="areas.json", help="areas file")
args = vars(ap.parse_args())

scale = 1
blur = 3
main_window = 'overlay'

def signal_handler(sig, frame):
    print('\nRelease cap..')
    cap.release()
    cv2.destroyAllWindows()
    print('Exit..')
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

if args.get("video", None) is None:
	cap = cv2.VideoCapture(0)
else:
	cap = cv2.VideoCapture(args["video"])


if args.get("areas", None) is None:
    detect_areas = []
else:
    with open('areas.json') as file:
        detect_areas = json.load(file)

if (len(detect_areas) > 0):
    print("Load %s areas" % len(detect_areas))

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


    countours = cv2.findContours(fgmask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    countours = imutils.grab_contours(countours)

    last_frame_gray = frame_gray

    for detect_area in detect_areas:
        cv2.polylines(frame, np.array([detect_area]), True, (110, 110, 110), 1)

    for countour in countours:
        if cv2.contourArea(countour) < args["min_area"]:
            continue

        (x, y, w, h) = cv2.boundingRect(countour)
        countour_area = [[x, y],[x+w, y],[x+w, y+h],[x, y+h]]
        cv2.polylines(frame, np.array([countour_area]), True, (127, 255, 127), 2)
        cv2.polylines(frame, np.array([countour]), True, (127, 255, 127), 2)

        for detect_area in detect_areas:
            detect_area_pl = Polygon(detect_area)
            countour_area_pl = Polygon(countour_area)
            #countour_typle = tuple([tuple(row) for row in countour])
            #countour_area_typle_pl = Polygon(countour_typle)
            #print(countour_area_typle_pl)
            intersect = detect_area_pl.intersects(countour_area_pl)
            if (intersect == True):
                cv2.polylines(frame, np.array([detect_area]), True, (0, 0, 255), 3)
                print("Intersect in armed area %s!" % detect_area)

    fgmask_rgba = cv2.cvtColor(fgmask, cv2.COLOR_GRAY2RGBA)
    fgmask_rgba[np.where((fgmask_rgba == [255,255,255,255]).all(axis = 2))] = [0,0,225,0]
    fgmask_rgba[np.where((fgmask_rgba == [127,127,127,255]).all(axis = 2))] = [0,0,225,0]
    fgmask_rgba[np.where((fgmask_rgba == [0,0,0,255]).all(axis = 2))] = [0,0,0,0]

    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2RGBA)
    added_image = cv2.addWeighted(frame,1,fgmask_rgba,0,0)
    cv2.namedWindow(main_window)
    cv2.moveWindow(main_window, 20, 20)
    cv2.imshow(main_window, added_image)
    cv2.waitKey(1)


