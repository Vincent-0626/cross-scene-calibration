import os,json,gzip,hashlib,random,csv,time,sys
from pathlib import Path
from collections import defaultdict
import numpy as np
from PIL import Image
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp
from sklearn.model_selection import StratifiedGroupKFold
import torch
from torch import nn
from torch.utils.data import Dataset,DataLoader
from torchvision import models,transforms
R=Path('/root/cross_scene_20261120'); O=R/'target_domain_baselines_20261122'; O.mkdir(exist_ok=True)
domain,method=sys.argv[1:3]; out=O/domain/method;out.mkdir(parents=True,exist_ok=True)
if (out/'COMPLETE').exists(): print('already complete');sys.exit(0)
random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42)
assert torch.cuda.is_available()
base=R/'calibration_three_domains'; splitfile=O/domain/'splits.json'
if not splitfile.exists():
 if domain=='plant':
  p=Path('/root/plantdoc_experiment'); cfg=json.load(gzip.open(p/'artifacts/config.json.gz','rt'));classes=[c['plantdoc_folder'] for c in cfg['class_order']]; rows=json.load(gzip.open(p/'artifacts/leaves.json.gz','rt'));splits={}
  for k in ('train','val','test'):
   splits[k]=[]
   for row in rows:
    if row['partition']==k:
     row['path']=str(p/'data/PlantDoc-Original'/row['image_path'].split('/PlantDoc-Original/',1)[1]);row['label']=row['class_index'];splits[k].append(row)
 else:
  if domain=='office':
   p=R/'runs/office_product_to_real'; classes=json.load(open(p/'audit.json'))['classes'];rows=json.load(open(p/'test.json'))
  else:
   p=R/'visda2017';classes=json.load(open(p/'runs/source_only_30/config.json'))['classes'];rows=json.load(open(p/'runs/source_only_30/splits.json'))['target_test']
   for row in rows:row['path']=str(p/'dataset'/row['path'])
  # Keep exact duplicates together; remove groups with contradictory labels.
  labels_by_hash=defaultdict(set)
  for i,row in enumerate(rows):
   row['original_index']=i;row['hash']=hashlib.sha256(Path(row['path']).read_bytes()).hexdigest();labels_by_hash[row['hash']].add(row['label'])
  rows=[r for r in rows if len(labels_by_hash[r['hash']])==1]; y=np.array([r['label'] for r in rows]);g=np.array([r['hash'] for r in rows]);idx=np.arange(len(rows))
  a,b=next(StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=42).split(idx,y,g)); trainval=idx[a]; test=idx[b]
  c,d=next(StratifiedGroupKFold(n_splits=4,shuffle=True,random_state=43).split(trainval,y[trainval],g[trainval])); train=trainval[c];val=trainval[d]
  splits={k:[rows[i] for i in v] for k,v in [('train',train),('val',val),('test',test)]}
  assert not (set(g[train])&set(g[test]) or set(g[train])&set(g[val]) or set(g[val])&set(g[test]))
 splitfile.parent.mkdir(parents=True,exist_ok=True);splitfile.write_text(json.dumps({'classes':classes,'splits':splits},ensure_ascii=False))
data=json.loads(splitfile.read_text());classes=data['classes'];s=data['splits']; nclass=len(classes)
print('COUNTS',domain,method,{k:len(v) for k,v in s.items()},flush=True)
norm=transforms.Normalize([.485,.456,.406],[.229,.224,.225]); ev=transforms.Compose([transforms.Resize(256),transforms.CenterCrop(224),transforms.ToTensor(),norm]);aug=transforms.Compose([transforms.RandomResizedCrop(224,scale=(.8,1.)),transforms.RandomHorizontalFlip(),transforms.ToTensor(),norm])
if domain=='plant':aug=transforms.Compose([transforms.Resize((224,224)),transforms.RandomResizedCrop(224,scale=(.8,1)),transforms.RandomHorizontalFlip(),transforms.RandomVerticalFlip(),transforms.ColorJitter(.2,.2,.2,.05),transforms.ToTensor(),norm])
class Images(Dataset):
 def __init__(self,rows,tf):self.rows=rows;self.tf=tf
 def __len__(self):return len(self.rows)
 def __getitem__(self,i):
  r=self.rows[i]
  with Image.open(r['path']) as im:
   im=im.convert('RGB')
   if 'box' in r:im=im.crop(tuple(r['box']))
   return self.tf(im),r['label']
def ld(rows,tf,shuffle=False):return DataLoader(Images(rows,tf),batch_size=32,shuffle=shuffle,num_workers=6 if domain=='visda' else 4,pin_memory=True)
val=ld(s['val'],ev);test=ld(s['test'],ev);net=models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1 if method in ('baseline','smoothing') and not (domain=='plant' and method=='baseline') else None);net.fc=nn.Linear(net.fc.in_features,nclass);net.cuda()
def logits(loader):
 net.eval();zs=[];ys=[]
 with torch.inference_mode():
  for x,y in loader:zs.append(net(x.cuda(non_blocking=True)).cpu().numpy());ys.append(y.numpy())
 return np.concatenate(zs),np.concatenate(ys)
def stats(z,y,T=1):
 x=z.astype('float64')/T;lp=x-logsumexp(x,axis=1,keepdims=True);p=np.exp(lp);c=p.max(1);ok=p.argmax(1)==y;hi=c>=.9;ece=0
 for i in range(10):
  m=(c>=i/10)&((c<(i+1)/10) if i<9 else (c<=1))
  if m.any():ece+=m.mean()*abs(ok[m].mean()-c[m].mean())
 return dict(n=len(y),correct=int(ok.sum()),accuracy=float(ok.mean()),mean_confidence=float(c.mean()),ece=float(ece),nll=float(-lp[np.arange(len(y)),y].mean()),high_confidence=int(hi.sum()),high_confidence_wrong=int((hi&~ok).sum()))
if method=='temperature':
 f=O/domain/'baseline'/'logits.npz'; dat=np.load(f);zv=dat['val_logits'];yv=dat['val_labels'];zt=dat['test_logits'];yt=dat['test_labels'];fit=minimize_scalar(lambda t:stats(zv,yv,np.exp(t))['nll'],bounds=(np.log(.02),np.log(50)),method='bounded');assert fit.success;T=float(np.exp(fit.x));result={'temperature':T,'validation':stats(zv,yv,T),'test':stats(zt,yt,T),'fit':'target-domain validation NLL only'}
else:
 ckpt=Path('/root/plantdoc_experiment/plantdoc_unified30_v1/best.pt') if domain=='plant' and method=='baseline' else None
 if ckpt is None:
  train=ld(s['train'],aug,True);opt=torch.optim.AdamW(net.parameters(),lr=.0001,weight_decay=.01);sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=30);criterion=nn.CrossEntropyLoss(label_smoothing=.1 if method=='smoothing' else 0);best=-1;stale=0
  with (out/'epochs.csv').open('w',newline='') as f:
   w=csv.DictWriter(f,fieldnames=['epoch','train_accuracy','train_loss','val_accuracy','val_nll','stale','seconds']);w.writeheader()
   for epoch in range(1,31):
    t=time.time();net.train();correct=total=0;loss_sum=0
    for x,y in train:
     x=x.cuda(non_blocking=True);y=y.cuda(non_blocking=True);opt.zero_grad(set_to_none=True);z=net(x);loss=criterion(z,y);loss.backward();opt.step();correct+=(z.argmax(1)==y).sum().item();total+=len(y);loss_sum+=loss.item()*len(y)
    sch.step();zv,yv=logits(val);v=stats(zv,yv)
    if v['accuracy']>best:best=v['accuracy'];stale=0;torch.save({'model':net.state_dict(),'epoch':epoch},out/'best.pt')
    else:stale+=1
    row=dict(epoch=epoch,train_accuracy=correct/total,train_loss=loss_sum/total,val_accuracy=v['accuracy'],val_nll=v['nll'],stale=stale,seconds=round(time.time()-t,1));w.writerow(row);f.flush();print('EPOCH',row,flush=True)
    if stale>=5:break
  ckpt=out/'best.pt'
 checkpoint=torch.load(ckpt,map_location='cpu',weights_only=False);net.load_state_dict(checkpoint['model']);net.cuda();zv,yv=logits(val);zt,yt=logits(test);result={'best_epoch':checkpoint.get('epoch'),'validation':stats(zv,yv),'test':stats(zt,yt),'checkpoint':str(ckpt)};np.savez_compressed(out/'logits.npz',val_logits=zv,val_labels=yv,test_logits=zt,test_labels=yt)
 if method=='baseline' and domain!='plant':
  old=np.load(base/domain/'temperature/logits.npz');ix=np.array([r['original_index'] for r in s['test']]);assert np.array_equal(old['target_labels'][ix],yt);result['cross_scene_same_heldout']=stats(old['target_logits'][ix],yt)
result.update(domain=domain,method=method,counts={k:len(v) for k,v in s.items()},seed=42,early_stop='validation accuracy patience 5 max 30',warning='exploratory; target previously observed; exact duplicates grouped; near duplicates or shared scenes can remain');(out/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2));(out/'COMPLETE').write_text('complete\n');print('RESULT',json.dumps(result),flush=True)
