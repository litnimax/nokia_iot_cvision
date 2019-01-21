import numpy as np
import cv2
import signal
import sys
from imutils.video import VideoStream
import argparse
import imutils
import json
import time
from shapely.geometry import Polygon
import http_api_server
import base64
from multiprocessing import Process, Queue
from threading import Thread

if 'threading' in sys.modules:
    del sys.modules['threading']
import gevent
import gevent.socket
import gevent.monkey
gevent.monkey.patch_all()

def arg_init():
    print("Init argparse...")
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--source", help="video source")
    ap.add_argument("-m", "--min-area", type=int, default=100, help="minimum area size")
    ap.add_argument("-H", "--height", type=int, default=360, help="height")
    ap.add_argument("-W", "--width", type=int, default=640, help="width")
    ap.add_argument("-b", "--blur", type=int, default=5, help="blur core")
    ap.add_argument("-a", "--areas", default="areas.json", help="areas file")
    ap.add_argument("-p", "--port", type=int, default=9001, help="http api port")
    return vars(ap.parse_args())


def read_areas():
    if args.get("areas", None) is None:
        return {}
    else:
        with open(args["areas"]) as file:
            detect_areas = json.load(file)
            if (len(detect_areas) > 0):
                print("Load %s areas" % len(detect_areas))
            return detect_areas


class Frame(object):
    def __init__(self, source, width, height, blur_core):
        print("Init capture object...")
        if (source is None):
            self.capture_object = cv2.VideoCapture(0)
        else:
            self.capture_object = cv2.VideoCapture(source)
        print("Init frame struct...")
        self.width = width
        self.height = height
        self.blur_core = blur_core
        self.current_frame = np.zeros((height,width,1), np.uint8)
        self.current_color_frame = np.zeros((height,width,3), np.uint8)
        self.prev_frame = self.current_frame.copy()
        self.start_time = time.time()
        self.frames_counter = 0
        Thread(target=self.frames_clear).start()

    def frames_clear(self):
        while True:
            time.sleep(5)
            self.start_time = time.time()
            self.frames_counter = 0

    def __del__(self):
        print('\nRelease cap..')
        self.capture_object.release()
        cv2.destroyAllWindows()

    def get_fps(self):
        fps = self.frames_counter / (time.time() - self.start_time)
        return fps

    def capture_frame(self):
        self.prev_frame = self.current_frame.copy()
        ret, frame = self.capture_object.read()
        self.frames_counter += 1
        resized_frame = cv2.resize(frame, (self.width, self.height))
        self.current_color_frame = resized_frame.copy()
        gray_resized_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)
        blur_gray_resized_frame = cv2.GaussianBlur(gray_resized_frame, (self.blur_core, self.blur_core), 0)
        self.current_frame = blur_gray_resized_frame

    def render_detect_areas(self, frame, areas):
        for key, detect_area in areas.items():
            cv2.polylines(frame, np.array([detect_area]), True, (110, 110, 110), 1)
            detect_area_pl = Polygon(detect_area)
            x = int(detect_area_pl.centroid.coords[0][0])
            y = int(detect_area_pl.centroid.coords[0][1])
            cv2.putText(frame, key[:13], (x-50,y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,127), 1)
        return frame

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
        self.accum_image = np.zeros((height,width), np.uint8)

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
        fgmask[np.where((fgmask == [255,255,255]).all(axis = 2))] = [0,0,225]
        fgmask[np.where((fgmask == [127,127,127]).all(axis = 2))] = [0,0,225]
        return fgmask

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
        colormap_rgba[np.where((colormap_rgba == [128,0,0,255]).all(axis = 2))] = [0,0,0,0]
        return colormap_rgba


class Http():
    def __init__(self):
        self.miso = Queue()
        self.mosi = Queue()

    def get_frame(self, frametype):
        self.miso.put(frametype)
        try:
            return self.mosi.get(timeout=1)
        except:
            return b""

    def send_frame(self, frame):
        self.mosi.put(frame)

    def get_cmd(self):
        try:
            return self.miso.get_nowait()
        except:
            return ""

    def callbacks_generate(self):
        callbacks = {   '/get_user_image':          self.get_user_image,
                        '/get_heatmap_image':       self.get_heatmap_image,
                        '/get_area':                self.get_area}
        return callbacks

    def get_user_image(self, path, body):
        frame = self.get_frame("user_frame")
        return base64.b64encode(cv2.imencode('.jpg', frame)[1])

    def get_heatmap_image(self, path, body):
        frame = self.get_frame("heatmap_frame")
        return base64.b64encode(cv2.imencode('.jpg', frame)[1])

    def get_area(self, path, body):
        return "http_callback_get_area"

def render_user_frame():
    countours = mask_object.get_countours(frame_object.get_prev_frame(), frame_object.get_current_frame())
    overlay_frame = mask_object.get_mask()
    overlay_frame = frame_object.render_detect_areas(overlay_frame, detect_areas)
    frame = cv2.cvtColor(frame_object.get_color_frame(), cv2.COLOR_RGB2RGBA)

    for countour in countours:
        if cv2.contourArea(countour) < args["min_area"]:
            continue

        (x, y, w, h) = cv2.boundingRect(countour)
        countour_rect = [[x, y], [x+w, y], [x+w, y+h], [x, y+h]]
        cv2.polylines(overlay_frame, np.array([countour_rect]), True, (127, 255, 127), 2)
        cv2.polylines(overlay_frame, np.array([countour]), True, (127, 255, 127), 1)

        for key, detect_area in detect_areas.items():
            detect_area_pl = Polygon(detect_area)
            countour_area_pl = Polygon(countour_rect)
            intersect = detect_area_pl.intersects(countour_area_pl)
            if (intersect == True):
                cv2.polylines(overlay_frame, np.array([detect_area]), True, (0, 0, 255), 3)

    overlay_frame = cv2.cvtColor(overlay_frame, cv2.COLOR_RGB2RGBA)
    overlay_frame[np.where((overlay_frame == [0,0,0,255]).all(axis = 2))] = [0,0,0,0]
    cv2.putText(frame, "FPS: %.1f" % frame_object.get_fps(), (5,15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 1)
    user_image = cv2.addWeighted(frame, 1, overlay_frame, 0.5, 0)
    return user_image

def render_real_frame():
    real_image = frame_object.get_current_frame()
    real_image = cv2.cvtColor(real_image, cv2.COLOR_GRAY2RGB)
    for countour in countours:
        if cv2.contourArea(countour) < args["min_area"]:
            continue
        cv2.polylines(real_image, np.array([countour]), True, (127, 255, 127), 1)

    return real_image

def render_heatmap_frame():
    frame = cv2.cvtColor(frame_object.get_color_frame(), cv2.COLOR_RGB2RGBA)
    http_heatmap = mask_object.get_heatmap()
    http_heatmap_user_image = cv2.addWeighted(frame, 0.7, http_heatmap, 0.5 , 0)
    return http_heatmap_user_image

def show_window(name, x, y, image):
    cv2.namedWindow(name)
    cv2.moveWindow(name, x, y)
    cv2.imshow(name, image)
    cv2.waitKey(1)


args = arg_init()
signal.signal(signal.SIGINT, (lambda sig, frame: sys.exit(0)))
detect_areas = read_areas()

frame_object = Frame(args.get("source", None), args["width"], args["height"], args["blur"])
mask_object = Mask(args["width"], args["height"])
http_api = Http()

server = http_api_server.server(args["port"], http_api.callbacks_generate())

print("Start main cycle...")
while(1):
    frame_object.capture_frame()
    countours = mask_object.get_countours(frame_object.get_prev_frame(), frame_object.get_current_frame())
    mask_object.update_accum()

    for countour in countours:
        if cv2.contourArea(countour) < args["min_area"]:
            continue

        (x, y, w, h) = cv2.boundingRect(countour)
        countour_rect = [[x, y], [x+w, y], [x+w, y+h], [x, y+h]]

        for key, detect_area in detect_areas.items():
            detect_area_pl = Polygon(detect_area)
            countour_area_pl = Polygon(countour_rect)
            intersect = detect_area_pl.intersects(countour_area_pl)
            if (intersect == True):
                print("Intersect in armed area %s!" % key)

    #show_window('Motion detector', 20, 20, render_user_frame())
    #show_window('Heatmap', 20, 20+450, render_heatmap_frame())
    #show_window('Real image', 20+700, 20+450, render_real_frame())

    cmd = http_api.get_cmd()
    if (cmd == "user_frame"):
        http_user_image = render_user_frame()
        http_api.send_frame(http_user_image)
    elif (cmd == "heatmap_frame"):
        http_heatmap_user_image = render_heatmap_frame()
        http_api.send_frame(http_heatmap_user_image)

