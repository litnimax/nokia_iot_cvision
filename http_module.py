import base64
from http.server import BaseHTTPRequestHandler, HTTPServer
from multiprocessing import Process, Queue

import cv2


class Http():
    def __init__(self, port):
        self.miso = Queue()
        self.mosi = Queue()
        print("Init http api server...")

        def handler(*args):
            http_handler(self, *args)
        self.httpd = HTTPServer(('', port), handler)
        Process(target=self.httpd.serve_forever).start()

    def __del__(self):
        print("Close http server...")
        self.httpd.server_close()

    def get_data_by_key(self, frametype):
        self.miso.put(frametype)
        try:
            return self.mosi.get(timeout=1)
        except Exception:
            return b""

    def send_data_by_key(self, datatype, data):
        self.miso.put(datatype)
        self.mosi.get(timeout=1)
        self.miso.put(data)

    def send_data(self, data):
        self.mosi.put(data)

    def get_key(self):
        try:
            return self.miso.get_nowait()
        except Exception:
            return ""

    def get_data(self):
        try:
            return self.miso.get(timeout=1)
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

    def set_areas(self, path, body):
        self.send_data_by_key("set_areas", body)

    def set_size(self, path, body):
        self.send_data_by_key("set_size", body)

    def set_min_area(self, path, body):
        self.send_data_by_key("set_min_area", body)

    def get_fps(self, path, body):
        areas = self.get_data_by_key("fps")
        return str(areas).encode()


class http_handler(BaseHTTPRequestHandler):
    def __init__(self, http_user_object, *args):
        self.http_user_object = http_user_object
        BaseHTTPRequestHandler.__init__(self, *args)

    def do_POST(self):
        content_length = int(self.headers['Content-Length'])
        body = self.rfile.read(content_length)
        if (self.path[1:5] == "get_" or self.path[1:5] == "set_"):
            try:
                callback = getattr(self.http_user_object, self.path[1:])
                message = callback(self.path, body)
                self.send_response(200)
                self.send_header('Content-Type', 'application/json; charset=utf-8')
                self.end_headers()
                if (message is not None):
                    self.wfile.write(message)
            except TypeError as ex:
                self.send_response(500)
                self.end_headers()
                message = "TypeError in callback(not 'b', or bad args): {0} ({1})"
                self.wfile.write(message.format(type(ex).__name__, ex.args).encode())
            except Exception as ex:
                self.send_response(520)
                self.end_headers()
                message = "Another error in callback: {0} ({1})"
                self.wfile.write(message.format(type(ex).__name__, ex.args).encode())
        else:
            self.send_response(400)
            self.end_headers()
            self.wfile.write(b"No valid function name(not 'set' or 'get'?)")

        return
