from http.server import BaseHTTPRequestHandler, HTTPServer
from multiprocessing import Process, Queue

class http_handler(BaseHTTPRequestHandler):
    def __init__(self, callbacks, *args):
        self.callbacks = callbacks
        BaseHTTPRequestHandler.__init__(self, *args)

    def do_POST(self):
        content_length = int(self.headers ['Content-Length'])
        body = self.rfile.read(content_length)
        path = self.path
        callback = self.callbacks.get(path)
        message = callback(path, body)
        self.send_response(200)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.end_headers()
        self.wfile.write(message)
        return

class server:
    def __init__(self, port, callbacks):
        print("Init http api server...")
        def handler(*args):
            http_handler(callbacks, *args)
        self.httpd = HTTPServer(('', port), handler)
        Process(target=self.httpd.serve_forever).start()

    def __del__(self):
        print("Close http server...")
        self.httpd.server_close()

