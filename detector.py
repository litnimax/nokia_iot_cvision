import argparse
import base64
import json
import signal
import sys
import time
from multiprocessing import Queue
from threading import Thread

import cv2
import gevent
import gevent.monkey
import gevent.socket
import imutils
import numpy as np
from shapely.geometry import Polygon

import http_api_server

if 'threading' in sys.modules:
    del sys.modules['threading']
gevent.monkey.patch_all()


def arg_init():
    print("Init argparse...")
    argumets = argparse.ArgumentParser()
    argumets.add_argument("-c", "--source", help="video source")
    argumets.add_argument("-m", "--min-area", type=int, default=100, help="minimum area size")
    argumets.add_argument("-H", "--height", type=int, default=360, help="height")
    argumets.add_argument("-W", "--width", type=int, default=640, help="width")
    argumets.add_argument("-t", "--threshold", type=int, default=5, help="threshold level(low - max sensetive)")
    argumets.add_argument("-a", "--areas", default="areas.json", help="areas file")
    argumets.add_argument("-p", "--port", type=int, default=9001, help="http api port")
    argumets.add_argument("-i", "--interface", action="store_true", help="interface")
    return vars(argumets.parse_args())


def read_areas(areas_file):
    with open(areas_file) as file:
        areas = json.load(file)
        if (len(areas) > 0):
            print("Load %s areas" % len(areas))
        return areas


def window(name, x, y, image):
    cv2.namedWindow(name)
    cv2.moveWindow(name, x, y)
    cv2.imshow(name, image)
    cv2.waitKey(1)


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

    def frames_clear(self):
        while True:
            time.sleep(5)
            self.start_time = time.time()
            self.frame_os_counter = 0

    def __del__(self):
        print('\nRelease cap..')
        self.capture_o.release()
        cv2.destroyAllWindows()



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


class Mask(object):
    def __init__(self, width, height):
        print("Init mask object...")
        self.fgmask = []
        self.width = width
        self.height = height
        self.accum_image = np.zeros((self.height, self.width), np.uint8)

    def get_countours(self, prev, current):
        mask = cv2.absdiff(prev, current)
        mask = cv2.threshold(mask, 25, 255, cv2.THRESH_BINARY)[1]
        self.fgmask = cv2.dilate(mask, None, iterations=2)
        countours = cv2.findContours(self.fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        countours = imutils.grab_contours(countours)
        return countours

    def get_mask(self):
        fgmask = self.fgmask.copy()
        fgmask = cv2.cvtColor(fgmask, cv2.COLOR_GRAY2RGB)
        fgmask[np.where((fgmask == [255, 255, 255]).all(axis=2))] = [0, 0, 225]
        fgmask[np.where((fgmask == [127, 127, 127]).all(axis=2))] = [0, 0, 225]
        return fgmask

    def clear_accum(self):
        self.accum_image = np.zeros((self.height, self.width), np.uint8)

    def update_accum(self):
        mask = self.fgmask.copy()
        mask[mask == 255] = 1
        self.accum_image = cv2.add(self.accum_image, mask)
        max_arr = self.accum_image.max()
        if (max_arr > 250):
            self.accum_image = np.divide(self.accum_image, 1.01)
            self.accum_image = self.accum_image.astype(np.uint8)
        #print(max_arr, self.accum_image.max())

    def get_heatmap(self):
        colormap = cv2.applyColorMap(self.accum_image, cv2.COLORMAP_JET)
        colormap_rgba = cv2.cvtColor(colormap, cv2.COLOR_RGB2RGBA)
        colormap_rgba[np.where((colormap_rgba == [128, 0, 0, 255]).all(axis=2))] = [0, 0, 0, 0]
        return colormap_rgba


class Http():
    def __init__(self):
        self.miso = Queue()
        self.mosi = Queue()

    def get_data_by_key(self, frametype):
        self.miso.put(frametype)
        try:
            return self.mosi.get(timeout=1)
        except Exception:
            return b""

    def send_data(self, data):
        self.mosi.put(data)

    def get_key(self):
        try:
            return self.miso.get_nowait()
        except Exception:
            return ""

    def get_user_image(self, path, body):
        frame = self.get_data_by_key("user_image")
        return base64.b64encode(cv2.imencode('.jpg', frame)[1])

    def get_heatmap_image(self, path, body):
        frame = self.get_data_by_key("heatmap_image")
        return base64.b64encode(cv2.imencode('.jpg', frame)[1])

    def get_real_image(self, path, body):
        frame = self.get_data_by_key("real_image")
        return base64.b64encode(cv2.imencode('.jpg', frame)[1])

    def get_areas(self, path, body):
        areas = self.get_data_by_key("areas")
        return str(areas).encode()

    def get_fps(self, path, body):
        areas = self.get_data_by_key("fps")
        return str(areas).encode()


class main():
    def __init__(self):
        signal.signal(signal.SIGINT, (lambda s, f: sys.exit(0)))
        self.args = arg_init()
        self.detect_areas = read_areas(self.args["areas"])
        self.frame_o = Frame(self.args.get("source", None), self.args["width"], self.args["height"], self.args["threshold"])
        self.mask_o = Mask(self.args["width"], self.args["height"])
        self.http = Http()
        http_api_server.server(self.args["port"], self.http)

        print("Start main cycle...")
        while True:
            self.cycle()


    def render_user_frame(self):
        countours = self.mask_o.get_countours(self.frame_o.get_prev_frame(), self.frame_o.get_current_frame())
        overlay_frame = self.mask_o.get_mask()
        overlay_frame = self.frame_o.render_detect_areas(overlay_frame, self.detect_areas)
        frame = cv2.cvtColor(self.frame_o.get_color_frame(), cv2.COLOR_RGB2RGBA)

        for countour in countours:
            if cv2.contourArea(countour) < self.args["min_area"]:
                continue

            (x, y, w, h) = cv2.boundingRect(countour)
            countour_rect = [[x, y], [x + w, y], [x + w, y + h], [x, y + h]]
            cv2.polylines(overlay_frame, np.array([countour_rect]), True, (127, 255, 127), 2)
            cv2.polylines(overlay_frame, np.array([countour]), True, (127, 255, 127), 1)

            for key, detect_area in self.detect_areas.items():
                detect_area_pl = Polygon(detect_area)
                countour_area_pl = Polygon(countour_rect)
                intersect = detect_area_pl.intersects(countour_area_pl)
                if (intersect):
                    cv2.polylines(overlay_frame, np.array([detect_area]), True, (0, 0, 255), 3)

        overlay_frame = cv2.cvtColor(overlay_frame, cv2.COLOR_RGB2RGBA)
        overlay_frame[np.where((overlay_frame == [0, 0, 0, 255]).all(axis=2))] = [0, 0, 0, 0]
        cv2.putText(frame, "FPS: %.1f" % self.frame_o.get_fps(), (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        user_image = cv2.addWeighted(frame, 1, overlay_frame, 0.5, 0)
        return user_image

    def render_heatmap_frame(self):
        frame = cv2.cvtColor(
            self.frame_o.get_color_frame(), cv2.COLOR_RGB2RGBA)
        heatmap = self.mask_o.get_heatmap()
        heatmap_user_image = cv2.addWeighted(frame, 0.7, heatmap, 0.5, 0)
        return heatmap_user_image

    def render_real_frame(self):
        countours = self.mask_o.get_countours(
            self.frame_o.get_prev_frame(), self.frame_o.get_current_frame())
        real_image = self.frame_o.get_current_frame()
        real_image = cv2.cvtColor(real_image, cv2.COLOR_GRAY2RGB)
        for countour in countours:
            if cv2.contourArea(countour) < self.args["min_area"]:
                continue
            cv2.polylines(real_image, np.array([countour]), True, (127, 255, 127), 1)
        return real_image

    def cycle(self):
        self.frame_o.capture()
        countours = self.mask_o.get_countours(
            self.frame_o.get_prev_frame(), self.frame_o.get_current_frame())
        self.mask_o.update_accum()

        for countour in countours:
            if cv2.contourArea(countour) > self.args["min_area"]:
                (x, y, w, h) = cv2.boundingRect(countour)
                countour_rect = [[x, y], [x + w, y], [x + w, y + h], [x, y + h]]
                for key, detect_area in self.detect_areas.items():
                    if (Polygon(detect_area).intersects(Polygon(countour_rect))):
                        print("Intersect in armed area %s!" % key)

        if self.args["interface"]:
            window('Motion detector', 20, 20, self.render_user_frame())
            window('Heatmap', 20, 20 + 450, self.render_heatmap_frame())
            window('Real image', 20 + 700, 20 + 450, self.render_real_frame())

        key = self.http.get_key()
        if key == "user_image":
            http_user_image = self.render_user_frame()
            self.http.send_data(http_user_image)
        elif key == "heatmap_image":
            http_heatmap_user_image = self.render_heatmap_frame()
            self.http.send_data(http_heatmap_user_image)
        elif key == "real_image":
            http_real_image = self.render_real_frame()
            self.http.send_data(http_real_image)
        elif key == "areas":
            self.http.send_data(self.detect_areas)
        elif key == "fps":
            self.http.send_data({'fps': "%.1f" % self.frame_o.get_fps()})


if __name__ == "__main__":
    main()
