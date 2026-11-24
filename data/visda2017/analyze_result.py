import csv,json,numpy as np
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path('/root/cross_scene_20261120/visda2017/runs/source_only_30')
cfg=json.loads((root/'config.json').read_text()); res=json.loads((root/'result.json').read_text())
z=np.load(root/'target_probabilities.npz'); prob=z['probabilities']; ys=z['labels']; pred=prob.argmax(1);conf=prob.max(1);wrong=pred!=ys
per=[]
for i,n in enumerate(cfg['classes']):
 m=ys==i
 per.append({'class':n,'n':int(m.sum()),'correct':int((pred[m]==i).sum()),'accuracy':float((pred[m]==i).mean()),'mean_confidence':float(conf[m].mean()),'predicted_count':int((pred==i).sum())})
bins=[]
for i in range(10):
 m=(conf>=i/10)&((conf<(i+1)/10) if i<9 else (conf<=1))
 bins.append({'lower':i/10,'upper':(i+1)/10,'n':int(m.sum()),'accuracy':float((pred[m]==ys[m]).mean()) if m.any() else None,'mean_confidence':float(conf[m].mean()) if m.any() else None})
res['nll']=float(-np.log(np.clip(prob[np.arange(len(ys)),ys],1e-12,1)).mean())
res['brier_multiclass']=float(np.square(prob-np.eye(12)[ys]).sum(1).mean())
res['wrong_total']=int(wrong.sum());res['wrong_conf_ge_90_share_of_errors']=float((wrong&(conf>=.9)).sum()/wrong.sum())
(root/'analysis.json').write_text(json.dumps({'summary':res,'per_class':per,'calibration_bins':bins},indent=2))
with (root/'epochs.csv').open() as f: rows=list(csv.DictReader(f))
epochs=[int(x['epoch']) for x in rows]
fig,ax=plt.subplots(1,2,figsize=(11,4),layout='constrained')
for key,label in [('train_accuracy','Source train (augmented)'),('val_accuracy','Source validation')]:
 ax[0].plot(epochs,[100*float(r[key]) for r in rows],marker='o',label=label)
ax[0].set(xlabel='Epoch',ylabel='Accuracy (%)',title='Synthetic source');ax[0].grid(alpha=.25);ax[0].legend()
xs=[b['mean_confidence'] for b in bins if b['n']];y=[b['accuracy'] for b in bins if b['n']]
ax[1].plot([0,1],[0,1],linestyle='--',color='gray',label='Perfect calibration')
ax[1].plot(xs,y,marker='o',label='Target real photos');ax[1].set(xlim=(0,1),ylim=(0,1),xlabel='Mean confidence',ylabel='Accuracy',title='Real-domain calibration (10 bins)');ax[1].grid(alpha=.25);ax[1].legend()
fig.suptitle('VisDA-2017: source-only, seed 42')
fig.savefig(root/'training_and_calibration.png',dpi=180)
print(json.dumps({'nll':res['nll'],'brier':res['brier_multiclass'],'wrong':res['wrong_total'],'wrong90_share':res['wrong_conf_ge_90_share_of_errors'],'bins':bins}))
