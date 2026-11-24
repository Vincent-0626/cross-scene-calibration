import secrets,hmac,urllib.request
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from pathlib import Path
root=Path('/root/cross_scene_20261120/dac_augmix')
keyfile=root/'mobile_access_key'
if not keyfile.exists():
 keyfile.write_text(secrets.token_urlsafe(24));keyfile.chmod(0o600)
key=keyfile.read_text().strip()
class Handler(BaseHTTPRequestHandler):
 def do_GET(self):
  path=self.path.split('?',1)[0]
  cookie=self.headers.get('Cookie','')
  if hmac.compare_digest(path,'/'+key):
   self.send_response(303);self.send_header('Location','/');self.send_header('Set-Cookie','dash_access='+key+'; HttpOnly; SameSite=Strict; Path=/');self.send_header('Cache-Control','no-store');self.end_headers();return
  supplied=next((x.strip().split('=',1)[1] for x in cookie.split(';') if x.strip().startswith('dash_access=')),'')
  if not hmac.compare_digest(supplied,key):self.send_error(403);return
  if path not in ('/','/api/status'):self.send_error(404);return
  try:
   with urllib.request.urlopen('http://127.0.0.1:8765'+path,timeout=8) as r:body=r.read();kind=r.headers.get('Content-Type','application/octet-stream')
   self.send_response(200);self.send_header('Content-Type',kind);self.send_header('Content-Length',str(len(body)));self.send_header('Cache-Control','no-store');self.end_headers();self.wfile.write(body)
  except Exception:self.send_error(502)
 def log_message(self,*args):pass
ThreadingHTTPServer(('127.0.0.1',8766),Handler).serve_forever()

