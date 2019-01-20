from http.server import BaseHTTPRequestHandler, HTTPServer

class http_handler(BaseHTTPRequestHandler):
    def __init__(self, callback, *args):
        self.callback = callback
        BaseHTTPRequestHandler.__init__(self, *args)

    def do_POST(self):
        content_length = int(self.headers ['Content-Length'])
        body = self.rfile.read(content_length)
        path = self.path
        message = self.callback(path, body)
        self.send_response(200)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.end_headers()
        self.wfile.write(message.encode())
        return

class server:
    def __init__(self, port, callback):
        def handler(*args):
            http_handler(callback, *args)
        self.httpd = HTTPServer(('', port), handler)

    def start(self):
      self.httpd.serve_forever()

    def __del__(self):
      self.httpd.server_close()

