import argparse
import json
import signal
import sys

import cv2
import gevent
import gevent.monkey
import gevent.socket
from shapely.geometry import Polygon

from frame_module import Frame
from http_module import Http
from mask_module import Mask
from render_module import Render

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


class main():
    def __init__(self):
        signal.signal(signal.SIGINT, (lambda s, f: sys.exit(0)))
        self.args = arg_init()
        self.detect_areas = read_areas(self.args["areas"])
        self.frame_o = Frame(self.args.get("source", None), self.args["width"], self.args["height"], self.args["threshold"])
        self.mask_o = Mask(self.args["width"], self.args["height"], self.args["min_area"])
        self.http = Http(self.args["port"])
        self.render = Render(self.mask_o, self.frame_o, self.detect_areas)

        print("Start main cycle...")
        while True:
            self.cycle()

    def cycle(self):
        self.frame_o.capture()
        countours = self.mask_o.get_countours(self.frame_o.get_prev_frame(), self.frame_o.get_current_frame())
        self.mask_o.update_accum()

        for countour in countours:
            if cv2.contourArea(countour) > self.mask_o.get_min_area():
                (x, y, w, h) = cv2.boundingRect(countour)
                countour_rect = [[x, y], [x + w, y], [x + w, y + h], [x, y + h]]
                for key, detect_area in self.detect_areas.items():
                    if (Polygon(detect_area).intersects(Polygon(countour_rect))):
                        print("Intersect in armed area %s!" % key)

        if self.args["interface"]:
            self.frame_o.window('Motion detector', 20, 20, self.render.render_user_frame())
            self.frame_o.window('Heatmap', 20, 20 + 450, self.render.render_heatmap_frame())
            self.frame_o.window('Real image', 20 + 700, 20 + 450, self.render.render_real_frame())

        key = self.http.get_key()
        if key == "user_image":
            http_user_image = self.render.render_user_frame()
            self.http.send_data(http_user_image)
        elif key == "heatmap_image":
            http_heatmap_user_image = self.render.render_heatmap_frame()
            self.http.send_data(http_heatmap_user_image)
        elif key == "real_image":
            http_real_image = self.render.render_real_frame()
            self.http.send_data(http_real_image)
        elif key == "areas":
            self.http.send_data(self.detect_areas)
        elif key == "fps":
            self.http.send_data({'fps': "%.1f" % self.frame_o.get_fps()})


if __name__ == "__main__":
    main()
