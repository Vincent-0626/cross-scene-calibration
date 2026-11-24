import json
from pathlib import Path
import numpy as np
from scipy.special import logsumexp
from scipy.optimize import minimize_scalar
root=Path('/root/cross_scene_20261120/target_domain_baselines_20261122')
for domain in ['plant','office','visda']:
 inp=root/domain/'smoothing'/'logits.npz'
 if not inp.exists():raise RuntimeError('missing '+str(inp))
 d=np.load(inp);zv=d['val_logits'];yv=d['val_labels'];zt=d['test_logits'];yt=d['test_labels']
 def metric(z,y,T):
  x=z.astype('float64')/T;lp=x-logsumexp(x,axis=1,keepdims=True);p=np.exp(lp);c=p.max(1);ok=p.argmax(1)==y;hi=c>=.9;ece=0.
  for i in range(10):
   m=(c>=i/10)&((c<(i+1)/10) if i<9 else (c<=1))
   if m.any():ece+=m.mean()*abs(ok[m].mean()-c[m].mean())
  return {'n':len(y),'correct':int(ok.sum()),'accuracy':float(ok.mean()),'mean_confidence':float(c.mean()),'ece':float(ece),'nll':float(-lp[np.arange(len(y)),y].mean()),'high_confidence':int(hi.sum()),'high_confidence_wrong':int((hi&~ok).sum())}
 opt=minimize_scalar(lambda t:metric(zv,yv,np.exp(t))['nll'],bounds=(np.log(.02),np.log(50)),method='bounded');assert opt.success
 t=float(np.exp(opt.x));r={'domain':domain,'method':'smoothing_plus_temperature','temperature':t,'fit':'target-domain validation NLL only','validation':metric(zv,yv,t),'test':metric(zt,yt,t),'warning':'exploratory; target test previously observed; validation also used for model selection'}
 assert r['test']['correct']==metric(zt,yt,1)['correct']
 out=root/domain/'smoothing_plus_temperature';out.mkdir(exist_ok=True);(out/'result.json').write_text(json.dumps(r,indent=2,ensure_ascii=False));(out/'COMPLETE').write_text('complete\n');print(json.dumps(r),flush=True)
