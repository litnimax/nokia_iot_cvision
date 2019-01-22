import time
from threading import Thread

import cv2
import numpy as np
from shapely.geometry import Polygon


class Frame(object):
    def __init__(self, source, width, height, threshold):
        print("Init capture object...")
        if (source is None):
            self.capture_o = cv2.VideoCapture(0)
        else:
            self.capture_o = cv2.VideoCapture(source)
        print("Init frame struct...")
        self.width = width
        self.height = height
        if threshold % 2 == 0:
            threshold += 1
        self.blur_core = threshold
        self.current_frame = np.zeros((height, width, 1), np.uint8)
        self.current_color_frame = np.zeros((height, width, 3), np.uint8)
        self.prev_frame = np.zeros((height, width, 1), np.uint8)
        self.start_time = time.time()
        self.frame_os_counter = 0
        self.capture()
        Thread(target=self.frames_clear).start()

    def __del__(self):
        print('\nRelease cap..')
        self.capture_o.release()
        cv2.destroyAllWindows()

    def window(self, name, x, y, image):
        cv2.namedWindow(name)
        cv2.moveWindow(name, x, y)
        cv2.imshow(name, image)
        cv2.waitKey(1)

    def frames_clear(self):
        while True:
            time.sleep(5)
            self.start_time = time.time()
            self.frame_os_counter = 0

    def capture(self):
        self.prev_frame = self.current_frame.copy()
        ret, frame = self.capture_o.read()
        self.frame_os_counter += 1
        resized_frame = cv2.resize(frame, (self.width, self.height))
        self.current_color_frame = resized_frame.copy()
        gray_resized_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)
        blur_gray_resized_frame = cv2.GaussianBlur(gray_resized_frame, (self.blur_core, self.blur_core), 0)
        self.current_frame = blur_gray_resized_frame

    def render_detect_areas(self, frame, areas):
        for area_key, detect_area in areas.items():
            cv2.polylines(frame, np.array([detect_area]), True, (110, 110, 110), 1)
            detect_area_pl = Polygon(detect_area)
            x = int(detect_area_pl.centroid.coords[0][0])
            y = int(detect_area_pl.centroid.coords[0][1])
            cv2.putText(frame, area_key[:13], (x-50, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 127), 1)
        return frame

    def get_fps(self):
        return self.frame_os_counter / (time.time() - self.start_time)

    def get_color_frame(self):
        return self.current_color_frame.copy()

    def get_current_frame(self):
        return self.current_frame.copy()

    def get_prev_frame(self):
        return self.prev_frame.copy()
