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

main_window = 'Motion detector'

def signal_handler(sig, frame):
    sys.exit(0)

def arg_init():
    print("Init argparse...")
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--source", help="video source")
    ap.add_argument("-m", "--min-area", type=int, default=100, help="minimum area size")
    ap.add_argument("-H", "--height", type=int, default=360, help="height")
    ap.add_argument("-W", "--width", type=int, default=640, help="width")
    ap.add_argument("-b", "--blur", type=int, default=5, help="blur core")
    ap.add_argument("-t", "--type", default="color", help="video type: color or processed")
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
        print("Create window...")
        cv2.namedWindow(main_window)
        cv2.moveWindow(main_window, 20, 20)
        cv2.imshow(main_window, self.get_color_frame())
        cv2.waitKey(1)

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

    def print_fps(self):
        while True:
            time.sleep(1)
            print("%.1f" % self.get_fps())
            #print(self.frames_counter)

    def get_frame(self):
        self.prev_frame = self.current_frame.copy()
        ret, frame = self.capture_object.read()
        self.frames_counter += 1
        resized_frame = cv2.resize(frame, (self.width, self.height))
        self.current_color_frame = resized_frame.copy()
        gray_resized_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)
        blur_gray_resized_frame = cv2.GaussianBlur(gray_resized_frame, (self.blur_core, self.blur_core), 0)
        self.current_frame = blur_gray_resized_frame
        return self.current_frame.copy()

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






class http_api():
    def get_image(path, body, miso, mosi):
        miso.put("user_frame")
        try:
            frame = mosi.get(timeout=1)
        except:
            return b""
        frame_jpg = cv2.imencode('.jpg', frame)[1]
        frame_jpg_encoded = base64.b64encode(frame_jpg)
        return frame_jpg_encoded

    def get_area(path, body, miso, mosi):
        return "http_callback_get_area"

miso = Queue()
mosi = Queue()
args = arg_init()
signal.signal(signal.SIGINT, signal_handler)
detect_areas = read_areas()

callbacks = {'/get_image': http_api.get_image, '/get_area': http_api.get_area}
server = http_api_server.server(args["port"], callbacks, miso, mosi)
frame_object = Frame(args.get("source", None), args["width"], args["height"], args["blur"])
mask_object = Mask(args["width"], args["height"])
Thread(target=frame_object.print_fps).start()

print("Start main cycle...")
while(1):
    current_frame = frame_object.get_frame()
    prev_frame = frame_object.get_prev_frame()
    if (args["type"] == "color"):
        frame = cv2.cvtColor(frame_object.get_color_frame(), cv2.COLOR_RGB2RGBA)
    else:
        frame = cv2.cvtColor(frame_object.get_frame(), cv2.COLOR_RGB2RGBA)

    countours = mask_object.get_countours(prev_frame, current_frame)
    overlay_frame = mask_object.get_mask()
    mask_object.update_accum()

    overlay_frame = frame_object.render_detect_areas(overlay_frame, detect_areas)

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
                #print("Intersect in armed area %s!" % key)

    overlay_frame = cv2.cvtColor(overlay_frame, cv2.COLOR_RGB2RGBA)
    overlay_frame[np.where((overlay_frame == [0,0,0,255]).all(axis = 2))] = [0,0,0,0]
    cv2.putText(frame, "FPS: %.1f" % frame_object.get_fps(), (5,15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 1)
    user_image = cv2.addWeighted(frame, 1, overlay_frame, 0.5, 0)
    cv2.namedWindow(main_window)
    cv2.moveWindow(main_window, 20, 20)
    cv2.imshow(main_window, user_image)

    #heatmap = mask_object.get_heatmap()
    #heatmap_user_image = cv2.addWeighted(frame, 0.7, heatmap, 0.5 , 0)
    #heatmap_windows = "Heatmap"
    #cv2.namedWindow(heatmap_windows)
    #cv2.moveWindow(heatmap_windows, 20, 20+450)
    #cv2.imshow(heatmap_windows, heatmap_user_image)

    #real_image = frame_object.get_frame()
    #real_image = cv2.cvtColor(real_image, cv2.COLOR_RGB2RGBA)
    #cv2.polylines(real_image, np.array([countour]), True, (127, 255, 127), 1)
    #real_image = cv2.addWeighted(real_image, 1, overlay_frame, 0 , 0)
    #real_windows = "Real image"
    #cv2.namedWindow(real_windows)
    #cv2.moveWindow(real_windows, 20+700, 20+450)
    #cv2.imshow(real_windows, real_image)

    cv2.waitKey(1)

    cmd = ""
    try:
        cmd = miso.get_nowait()
    except:
        continue

    if (cmd == "user_frame"):
        mosi.put(user_image)

