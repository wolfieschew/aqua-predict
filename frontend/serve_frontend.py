import http.server
import socketserver

PORT = 5500
Handler = http.server.SimpleHTTPRequestHandler

with socketserver.TCPServer(("", PORT), Handler) as httpd:
    print(f"Serving frontend at http://localhost:{PORT}")
    httpd.serve_forever()