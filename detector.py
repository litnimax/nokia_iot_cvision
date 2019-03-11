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
from settings_module import Settings

if 'threading' in sys.modules:
    del sys.modules['threading']
gevent.monkey.patch_all()


def arg_init():
    print("Init argparse...")
    argumets = argparse.ArgumentParser()
    argumets.add_argument("-c", "--source", help="video source")
    argumets.add_argument("-a", "--settings", default="settings.json", help="settings file")
    argumets.add_argument("-p", "--port", type=int, default=9001, help="http api port")
    argumets.add_argument("-i", "--interface", action="store_true", help="interface")
    argumets.add_argument("-w", "--warnings", action="store_true", help="warnings")
    return vars(argumets.parse_args())


class main():
    def __init__(self):
        signal.signal(signal.SIGINT, (lambda s, f: sys.exit(0)))
        self.args = arg_init()
        self.settings_o = Settings(self.args["settings"])
        self.frame_o = Frame(self.args.get("source", None), self.settings_o)
        self.mask_o = Mask(self.settings_o)
        self.http = Http(self.args["port"])
        self.render = Render(self.mask_o, self.frame_o, self.settings_o)
        self.cycle()

    def cycle(self):
        print("Start main cycle...")
        while True:
            self.detect()
            key = self.http.get_key()
            self.http_msg_catch(key)
            if self.args["interface"]:
                self.interface_show()

    def http_msg_catch(self, key):
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
            self.http.send_data(self.settings_o.get_areas())
        elif key == "fps":
            self.http.send_data({'fps': "%.1f" % self.frame_o.get_fps()})
        elif key == "set_areas":
            self.http.send_data("ok")
            data = self.http.get_data()
            data_decoded = json.loads(data.decode("utf-8"))
            if (type(data_decoded).__name__ == 'dict'):
                self.settings_o.set_areas(data_decoded)
        elif key == "set_size":
            self.http.send_data("ok")
            data = self.http.get_data()
            data_decoded = json.loads(data.decode("utf-8"))
            if type(data_decoded['width']).__name__ == 'int' and type(data_decoded['height']).__name__ == 'int':
                self.settings_o.set_size(data_decoded['width'], data_decoded['height'])
        elif key == "set_min_area":
            self.http.send_data("ok")
            data = self.http.get_data()
            data_decoded = json.loads(data.decode("utf-8"))
            if type(data_decoded['min_area']).__name__ == 'int':
                self.settings_o.set_min_area(data_decoded['min_area'])

    def detect(self):
        self.frame_o.capture()
        countours = self.mask_o.get_countours(self.frame_o.get_prev_frame(), self.frame_o.get_current_frame())
        self.mask_o.update_accum()

        for countour in countours:
            if cv2.contourArea(countour) > self.settings_o.get_min_area():
                (x, y, w, h) = cv2.boundingRect(countour)
                countour_rect = [[x, y], [x + w, y], [x + w, y + h], [x, y + h]]
                for key, detect_area in self.settings_o.get_areas().items():
                    if Polygon(detect_area).intersects(Polygon(countour_rect)):
                        if self.args["warnings"]:
                            print("Intersect in armed area %s!" % key)

    def interface_show(self):
        self.frame_o.window('Motion detector', 20, 20, self.render.render_user_frame())
        self.frame_o.window('Heatmap', 20, 20 + 450, self.render.render_heatmap_frame())
        self.frame_o.window('Real image', 20 + 700, 20 + 450, self.render.render_real_frame())


if __name__ == "__main__":
    main()
