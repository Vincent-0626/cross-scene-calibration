import json,gzip,hashlib
from pathlib import Path
import numpy as np
from scipy.special import softmax
from sklearn.metrics import roc_auc_score
root=Path('/root/cross_scene_20261120')
base=root/'calibration_three_domains'
work=root/'dac_augmix'
out=work/'independent_audit'
out.mkdir(exist_ok=True)
leaves=json.load(gzip.open('/root/plantdoc_experiment/artifacts/leaves.json.gz','rt'))
plant=[r for r in leaves if r['partition']=='test']
rng=np.random.default_rng(20261122)
result={}
for domain in ['plant','office','visda']:
 data={}
 for name,path,key in [
 ('baseline',base/domain/'temperature/logits.npz','target_logits'),
 ('temperature',base/domain/'temperature/logits.npz','target_logits'),
 ('smoothing',base/domain/'label_smoothing/logits.npz','target_logits'),
 ('penalty',base/domain/'confidence_penalty/logits.npz','target_logits'),
 ('augmix',work/domain/'augmix/logits.npz','target_logits'),
 ('dac',work/domain/'dac/logits.npz','target_calibrated_logits')]:
  if not path.exists(): continue
  f=np.load(path); y=f['target_labels']; z=f[key]
  if data and not np.array_equal(y,next(iter(data.values()))['y']):raise ValueError(domain+' labels differ')
  T=1
  if name=='temperature':T=json.load(open(base/domain/'temperature/result.json'))['temperature']
  p=softmax(z.astype('float64')/T,axis=1); pred=p.argmax(1);conf=p.max(1);ok=pred==y
  n20=int(np.ceil(len(y)*.2));idx=np.argsort(-conf,kind='stable')[:n20]
  high=conf>=.9
  ece=0.
  for k in range(10):
   m=(conf>=k/10)&((conf<(k+1)/10) if k<9 else (conf<=1))
   if m.any():ece+=m.mean()*abs(ok[m].mean()-conf[m].mean())
  entry={'n':len(y),'accuracy':float(ok.mean()),'ece_10':float(ece),'mean_confidence':float(conf.mean()),
  'high_conf_wrong':int((high&~ok).sum()),'high_conf_total':int(high.sum()),
  'top20_n':n20,'top20_wrong':int((~ok[idx]).sum()),'top20_accuracy':float(ok[idx].mean()),
  'error_detection_auroc':float(roc_auc_score(~ok,1-conf)) if len(np.unique(ok))==2 else None,
  'sha256_logits':hashlib.sha256(path.read_bytes()).hexdigest()}
  data[name]={'y':y,'correct':ok,'summary':entry}
 groups=np.array([r['image_id'] for r in plant]) if domain=='plant' else np.arange(len(y))
 if domain=='plant':assert np.array_equal(np.array([r['class_index'] for r in plant]),y)
 unique,inverse=np.unique(groups,return_inverse=True)
 counts=np.bincount(inverse)
 baseline=data['baseline']['correct'].astype(int)
 for name,d in data.items():
  if name=='baseline':continue
  delta=d['correct'].astype(int)-baseline
  sums=np.bincount(inverse,weights=delta)
  v=[]
  for _ in range(1000):
   sample=rng.integers(0,len(unique),len(unique))
   v.append(float(sums[sample].sum()/counts[sample].sum()))
  d['summary']['paired_accuracy_difference_pp']=float(delta.mean()*100)
  d['summary']['paired_bootstrap_95_pp']=[float(x*100) for x in np.quantile(v,[.025,.975])]
 result[domain]={'groups':len(unique),'group_definition':'parent image_id' if domain=='plant' else 'individual files; same-scene correlations unresolved','methods':{k:v['summary'] for k,v in data.items()}}
(out/'metrics.json').write_text(json.dumps(result,indent=2,ensure_ascii=False))
print(json.dumps(result,ensure_ascii=False))

