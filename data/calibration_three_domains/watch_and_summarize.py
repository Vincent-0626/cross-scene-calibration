import os,time,json,csv,subprocess
from pathlib import Path
R=Path('/root/cross_scene_20261120/calibration_three_domains')
domains={'plant':'\u690d\u7269\u75c5\u5bb3','office':'\u65e5\u5e38\u7269\u54c1','visda':'VisDA'}
methods={'baseline':'\u539f\u6a21\u578b','temperature':'\u6e29\u5ea6\u7f29\u653e','label_smoothing':'\u6807\u7b7e\u5e73\u6ed1','confidence_penalty':'\u7f6e\u4fe1\u5ea6\u60e9\u7f5a'}
expected=[(d,m) for d in domains for m in ('temperature','label_smoothing','confidence_penalty')]
def queue_alive():
 try:
  st=Path('/proc/12608/status').read_text()
  cmd=Path('/proc/12608/cmdline').read_bytes().replace(b'\x00',b' ')
  return 'State:\tZ' not in st and b'run_queue.sh' in cmd
 except OSError:return False
def pct(v):return '\u2014' if v is None else f'{100*v:.2f}%'
def snapshot():
 rows=[]
 for domain in domains:
  for method in methods:
   p=R/domain/('temperature' if method=='baseline' else method)
   if not (p/'COMPLETE').exists():continue
   try:d=json.loads((p/'result.json').read_text(encoding='utf-8'))
   except (OSError,json.JSONDecodeError):continue
   x=d['target_baseline' if method=='baseline' else 'target_temperature' if method=='temperature' else 'target']
   rows.append({'domain':domain,'method':method,'n':x['n'],'accuracy':x['accuracy'],'ece_10bins':x['ece_10bins'],'mean_confidence':x['mean_confidence'],'nll':x['nll'],'high_conf_count':x['high_conf_count'],'high_conf_accuracy':x['high_conf_accuracy'],'wrong_confidence_ge_90':x['wrong_confidence_ge_90'],'best_epoch':d.get('baseline_best_epoch') if method in ('baseline','temperature') else d.get('best_epoch'),'temperature':d.get('temperature') if method=='temperature' else None})
 completed=[f'{d}/{m}' for d,m in expected if (R/d/m/'COMPLETE').exists()]
 alive=queue_alive()
 try:q=(R/'queue.status').read_text().strip()
 except OSError:q='unknown'
 state={'updated_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'queue_running':alive,'queue_status':q,'completed':completed,'expected_method_runs':len(expected),'all_completed':len(completed)==len(expected)}
 (R/'status.tmp').write_text(json.dumps(state,ensure_ascii=False,indent=2),encoding='utf-8');os.replace(R/'status.tmp',R/'status.json')
 cols=['domain','method','n','accuracy','ece_10bins','mean_confidence','nll','high_conf_count','high_conf_accuracy','wrong_confidence_ge_90','best_epoch','temperature']
 with (R/'comparison.tmp').open('w',newline='',encoding='utf-8-sig') as f:
  w=csv.DictWriter(f,fieldnames=cols);w.writeheader();w.writerows(rows)
 os.replace(R/'comparison.tmp',R/'comparison.csv')
 lines=['# \u4e09\u9886\u57df\u7f6e\u4fe1\u5ea6\u5b9e\u9a8c\u81ea\u52a8\u6c47\u603b','','\u66f4\u65b0\u65f6\u95f4\uff08UTC\uff09\uff1a'+state['updated_utc'],'','\u961f\u5217\u72b6\u6001\uff1a'+q,'\u5df2\u5b8c\u6210\u65b9\u6cd5\u4efb\u52a1\uff1a'+str(len(completed))+'/'+str(len(expected)),'','| \u9886\u57df | \u65b9\u6cd5 | \u6837\u672c\u6570 | \u51c6\u786e\u7387 | ECE | \u5e73\u5747\u7f6e\u4fe1\u5ea6 | \u9ad8\u7f6e\u4fe1\u56fe\u6570 | \u9ad8\u7f6e\u4fe1\u51c6\u786e\u7387 | \u9ad8\u7f6e\u4fe1\u9519\u8bef\u6570 |','|---|---|---:|---:|---:|---:|---:|---:|---:|']
 for r in rows:
  lines.append('| '+domains[r['domain']]+' | '+methods[r['method']]+' | '+str(r['n'])+' | '+pct(r['accuracy'])+' | '+pct(r['ece_10bins'])+' | '+pct(r['mean_confidence'])+' | '+str(r['high_conf_count'])+' | '+pct(r['high_conf_accuracy'])+' | '+str(r['wrong_confidence_ge_90'])+' |')
 lines += ['', '\u9ad8\u7f6e\u4fe1\u6307\u6700\u9ad8\u9884\u6d4b\u6982\u7387\u81f3\u5c11 90%\uff1b\u987b\u540c\u65f6\u770b\u9ad8\u7f6e\u4fe1\u56fe\u6570\uff0c\u4e0d\u80fd\u53ea\u770b\u9519\u8bef\u6570\u91cf\u3002','\u6e29\u5ea6\u53ea\u7528\u6e90\u57df\u9a8c\u8bc1\u96c6\u62df\u5408\uff1b\u6807\u7b7e\u5e73\u6ed1\u7cfb\u6570 0.1\uff0c\u7f6e\u4fe1\u5ea6\u60e9\u7f5a\u7cfb\u6570 0.1\uff1b\u4ec5\u4e00\u6b21\u968f\u673a\u79cd\u5b50 42\u3002', '\u4e09\u4e2a\u76ee\u6807\u96c6\u7684\u7ed3\u679c\u6b64\u524d\u5df2\u7ecf\u88ab\u67e5\u770b\uff0c\u672c\u6b21\u6bd4\u8f83\u662f\u63a2\u7d22\u6027\u7684\uff0c\u4e0d\u80fd\u4f5c\u4e3a\u5b8c\u5168\u72ec\u7acb\u76f2\u6d4b\uff1b\u690d\u7269\u7c7b\u522b\u6620\u5c04\u4ecd\u5b58\u5728\u4e0d\u786e\u5b9a\u6027\u3002', '\u8bad\u7ec3\uff1aResNet-50 ImageNet1K V1\uff0c\u6700\u591a 30 \u8f6e\uff0c\u6e90\u57df\u9a8c\u8bc1\u51c6\u786e\u7387\u8fde\u7eed 5 \u8f6e\u65e0\u63d0\u5347\u65e9\u505c\u3002', '\u6b64\u6587\u4ef6\u81ea\u52a8\u751f\u6210\uff1b\u8bb0\u5f55\u7684\u6bcf\u4e2a\u7ed3\u679c\u4ee5\u5404\u81ea result.json \u548c logits.npz \u4e3a\u51c6\u3002']
 (R/'\u6c47\u603b\u62a5\u544a.tmp').write_text('\n'.join(lines)+'\n',encoding='utf-8');os.replace(R/'\u6c47\u603b\u62a5\u544a.tmp',R/'\u4e09\u9886\u57df_\u5b9e\u9a8c\u81ea\u52a8\u6c47\u603b.md')
 return state
while True:
 try:
  status=snapshot()
  print(json.dumps(status,ensure_ascii=False),flush=True)
  if not status['queue_running']:break
 except Exception as e:print('SNAPSHOT_ERROR',type(e).__name__,str(e),flush=True)
 time.sleep(30)
