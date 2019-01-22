from http.server import BaseHTTPRequestHandler, HTTPServer
from multiprocessing import Process, Queue

class http_handler(BaseHTTPRequestHandler):
    def __init__(self, http_user_object, *args):
        self.http_user_object = http_user_object
        BaseHTTPRequestHandler.__init__(self, *args)

    def do_POST(self):
        content_length = int(self.headers ['Content-Length'])
        body = self.rfile.read(content_length)
        if (self.path[1:5] == "get_" or self.path[1:5] == "set_"):
            try:
                callback = getattr(self.http_user_object, self.path[1:])
                message = callback(self.path, body)
                print("message", message)
                self.send_response(200)
                self.send_header('Content-Type', 'application/json; charset=utf-8')
                self.end_headers()
                if (message is not None):
                    self.wfile.write(message)
            except TypeError:
                self.send_response(500)
                self.end_headers()
                self.wfile.write(b"TypeError in callback(not 'b'?)")
        else:
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b"No valid function name(not 'set' or 'get'?)")

        return

class server:
    def __init__(self, port, http_user_object):
        print("Init http api server...")
        def handler(*args):
            http_handler(http_user_object, *args)
        self.httpd = HTTPServer(('', port), handler)
        Process(target=self.httpd.serve_forever).start()

    def __del__(self):
        print("Close http server...")
        self.httpd.server_close()

