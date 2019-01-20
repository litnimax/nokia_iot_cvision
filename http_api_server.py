from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib import parse

def http_callback(path, body):
  return body

class http_handler(BaseHTTPRequestHandler):
    def do_POST(self):
        content_length = int(self.headers ['Content-Length'])
        body = self.rfile.read(content_length)
        path = self.path
        message = http_callback(path, body)
        self.send_response(200)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.end_headers()
        self.wfile.write(message)

class server():
  def __init__(self, port):
    self.port = port
    self.httpd = HTTPServer(('localhost', self.port), http_handler)

  def __del__(self):
    self.httpd.server_close()

  def start(self):
    try:
        self.httpd.serve_forever()
    except KeyboardInterrupt:
        pass
