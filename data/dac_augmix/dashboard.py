import csv,json,os,subprocess,time,threading
from collections import deque
from pathlib import Path
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
ROOT=Path('/root/cross_scene_20261120/dac_augmix')
METHODS={'augmix':'AugMix','dac':'DAC','ls_entropy':'标签平滑＋熵正则化','alis':'ALiS'}
DOMAINS={'plant':'植物病害','office':'日常物品','visda':'VisDA'}
gpu_samples=deque(maxlen=10)
def sample_gpu():
 while True:
  try:
   raw=subprocess.check_output(['nvidia-smi','--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw','--format=csv,noheader,nounits'],text=True,timeout=4).splitlines()[0].split(', ')
   gpu_samples.append({'time':time.monotonic(),'name':raw[0],'utilization':float(raw[1]),'memory_used':float(raw[2]),'memory_total':float(raw[3]),'temperature':float(raw[4]),'power':float(raw[5])})
  except Exception:pass
  time.sleep(1)
threading.Thread(target=sample_gpu,daemon=True).start()
def read_text(path):
 try:return path.read_text(encoding='utf-8').strip()
 except (OSError,UnicodeError):return ''
def snapshot():
 p=subprocess.run(['ps','-eo','pid,args'],capture_output=True,text=True,timeout=5).stdout
 samples=[x for x in list(gpu_samples) if time.monotonic()-x['time']<15]
 if samples:
  gpu=dict(samples[-1]);gpu['instant_utilization']=gpu['utilization'];gpu['utilization']=sum(x['utilization'] for x in samples)/len(samples);gpu['minimum']=min(x['utilization'] for x in samples);gpu['maximum']=max(x['utilization'] for x in samples);gpu['samples']=len(samples);gpu.pop('time',None)
 else:gpu=None
 queue_alive={method:(method+'_queue.sh') in p for method in METHODS}
 stages=[]
 for method,method_name in METHODS.items():
  status=read_text(ROOT/(method+'.status'))
  for domain,domain_name in DOMAINS.items():
   folder=ROOT/domain/method
   log=ROOT/(domain+'_'+method+'.log')
   active=any((method+'_experiment.py') in line and ('--domain '+domain) in line and 'python' in line for line in p.splitlines())
   complete=(folder/'COMPLETE').is_file()
   failed=domain in status and 'FAILED' in status and not active
   state='complete' if complete else 'running' if active else 'failed' if failed else 'waiting' if queue_alive[method] else 'stopped'
   rows=[]
   try:
    with (folder/'epochs.csv').open(newline='',encoding='utf-8') as f:rows=list(csv.DictReader(f))
   except (FileNotFoundError,OSError,csv.Error):pass
   latest=rows[-1] if rows else {}
   epoch=int(latest.get('epoch') or 0)
   result={}
   try:result=json.loads((folder/'result.json').read_text(encoding='utf-8'))
   except (OSError,ValueError):pass
   target=result.get('target',{})
   stage={'method':method_name,'domain':domain_name,'state':state,'epoch':epoch,'max_epoch':30 if method!='dac' else None,'train_accuracy':latest.get('train_accuracy'),'val_accuracy':latest.get('val_accuracy'),'best_val':latest.get('best_val'),'stale':latest.get('stale'),'target_accuracy':target.get('accuracy'),'target_ece':target.get('ece_10bins'),'log_tail':read_text(log)[-800:] if failed else ''}
   stages.append(stage)
 repeat_root=ROOT/'seed_repeats'
 repeat_status=read_text(repeat_root/'queue.status')
 repeat_history=read_text(repeat_root/'queue.history').splitlines()
 repeat_alive='seed_repeats/queue.sh' in p
 repeat={'status':repeat_status if repeat_alive or repeat_status=='ALL COMPLETE' else ('队列未运行：'+repeat_status),'completed':len(repeat_history),'total':24,'alive':repeat_alive,'latest_epoch':None}
 parts=repeat_status.split()
 if len(parts)>=4 and parts[3]=='RUNNING':
  domain,method,seed=parts[0],parts[1],parts[2].split('=',1)[-1]
  try:
   with (repeat_root/domain/method/('seed_'+seed)/'epochs.csv').open(newline='',encoding='utf8') as f: rows=list(csv.DictReader(f))
   if rows:repeat['latest_epoch']=int(rows[-1]['epoch'])
  except (OSError,ValueError,KeyError):pass
 return {'repeat':repeat,'time':time.strftime('%Y-%m-%d %H:%M:%S',time.localtime()),'gpu':gpu,'stages':stages,'completed':sum(s['state']=='complete' for s in stages),'total':len(stages),'note':'训练上限进度，早停可能提前结束。DAC 为特征提取和校准，不按轮数计算。ALiS 尚未加入队列。'}
class Handler(BaseHTTPRequestHandler):
 def do_GET(self):
  path=self.path.split('?',1)[0]
  if path=='/api/status':
   try:body=json.dumps(snapshot(),ensure_ascii=False).encode('utf-8');code=200;content='application/json; charset=utf-8'
   except Exception as e:body=json.dumps({'error':str(e)},ensure_ascii=False).encode();code=500;content='application/json; charset=utf-8'
  elif path=='/':body=(ROOT/'dashboard.html').read_bytes();code=200;content='text/html; charset=utf-8'
  else:body=b'Not found';code=404;content='text/plain'
  self.send_response(code);self.send_header('Content-Type',content);self.send_header('Content-Length',str(len(body)));self.send_header('Cache-Control','no-store');self.end_headers();self.wfile.write(body)
 def log_message(self,*args):pass
ThreadingHTTPServer(('127.0.0.1',8765),Handler).serve_forever()

