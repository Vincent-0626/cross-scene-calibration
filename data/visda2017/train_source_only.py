import os,csv,json,random,time,hashlib
from collections import defaultdict
from pathlib import Path
import numpy as np
from PIL import Image
from sklearn.metrics import f1_score,confusion_matrix
import torch
from torch import nn
from torch.utils.data import Dataset,DataLoader
from torchvision import models,transforms

ROOT=Path('/root/cross_scene_20261120/visda2017'); DATA=ROOT/'dataset'; RUN=ROOT/'runs/source_only_30'; RUN.mkdir(parents=True,exist_ok=True)
audit=json.loads((ROOT/'audit.json').read_text())
classes=sorted(audit['domains']['train']['per_class'])
assert len(classes)==12 and set(classes)==set(audit['domains']['validation']['per_class'])
assert torch.cuda.is_available(),'GPU required'
random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42)
torch.backends.cudnn.benchmark=True
mapping={s:i for i,s in enumerate(classes)}
paths={}
for split in ('train','validation'):
 rows=[]
 for cl in classes:
  files=sorted((DATA/split/cl).glob('*'))
  rows.extend({'path':str(p.relative_to(DATA)),'label':mapping[cl],'class':cl} for p in files if p.is_file() and p.suffix.lower() in ('.jpg','.jpeg','.png'))
 paths[split]=rows
assert len(paths['train'])==152397 and len(paths['validation'])==55388
manifest=RUN/'splits.json'
if manifest.exists():
 splits=json.loads(manifest.read_text())
else:
 rng=random.Random(42); splits={'source_train':[],'source_val':[],'target_test':paths['validation']}
 for cl in classes:
  groups=defaultdict(list)
  for r in paths['train']:
   if r['class']!=cl:continue
   name=Path(r['path']).name
   if '__' not in name: raise RuntimeError('unrecognized source model filename '+name)
   groups[name.split('__')[0]].append(r)
  ids=sorted(groups);rng.shuffle(ids)
  if len(ids)<2:raise RuntimeError('insufficient 3D model groups '+cl)
  n=min(len(ids)-1,max(1,round(len(ids)*.2)))
  for key in ids[:n]:splits['source_val'].extend(groups[key])
  for key in ids[n:]:splits['source_train'].extend(groups[key])
 manifest.write_text(json.dumps(splits,indent=1),encoding='utf8')
config={'model':'ResNet50 ImageNet1K_V1','task':'VisDA synthetic to real; source only supervised classification','seed':42,'batch_size':32,'max_epochs':30,'patience':5,'early_stop':'source validation accuracy, strict improvement','optimizer':'AdamW','lr':0.0001,'weight_decay':0.01,'schedule':'CosineAnnealingLR T_max=30','classes':classes,'split':'stratified 80/20 source by original 3D model filename prefix before __; target official validation images held out for one-time final exploratory test','counts':{k:len(v) for k,v in splits.items()},'source_transform':'RandomResizedCrop 224 scale .8..1, horizontal flip','eval_transform':'resize 256 center crop 224','initialization':'ImageNet pretrained; source renderings not independent 3D models'}
(RUN/'config.json').write_text(json.dumps(config,indent=2),encoding='utf8')
print('CONFIG',json.dumps(config),flush=True)
norm=transforms.Normalize((.485,.456,.406),(.229,.224,.225))
traintf=transforms.Compose([transforms.RandomResizedCrop(224,scale=(.8,1)),transforms.RandomHorizontalFlip(),transforms.ToTensor(),norm])
evaltf=transforms.Compose([transforms.Resize(256),transforms.CenterCrop(224),transforms.ToTensor(),norm])
class Images(Dataset):
 def __init__(self,items,tf):self.rows=items;self.tf=tf
 def __len__(self):return len(self.rows)
 def __getitem__(self,i):
  r=self.rows[i]
  with Image.open(DATA/r['path']) as img:x=self.tf(img.convert('RGB'))
  return x,r['label'],i
def make(key,tf,shuffle=False):return DataLoader(Images(splits[key],tf),batch_size=32,shuffle=shuffle,num_workers=6,pin_memory=True,persistent_workers=True)
tr=make('source_train',traintf,True); val=make('source_val',evaltf)
device=torch.device('cuda'); net=models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1);net.fc=nn.Linear(net.fc.in_features,12);net.to(device)
opt=torch.optim.AdamW(net.parameters(),lr=1e-4,weight_decay=.01); sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=30); criterion=nn.CrossEntropyLoss(); best=-1;stale=0
with (RUN/'epochs.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=['epoch','lr','train_loss','train_accuracy','val_loss','val_accuracy','seconds']);w.writeheader();f.flush()
 for epoch in range(1,31):
  t=time.time();net.train();tot=correct=0;loss_sum=0.
  for x,y,_ in tr:
   x=x.to(device,non_blocking=True);y=y.to(device,non_blocking=True);opt.zero_grad(set_to_none=True);logits=net(x);loss=criterion(logits,y);loss.backward();opt.step()
   n=y.size(0);tot+=n;loss_sum+=loss.item()*n;correct+=(logits.argmax(1)==y).sum().item()
  net.eval();vt=vc=0;vl=0.
  with torch.inference_mode():
   for x,y,_ in val:
    x=x.to(device,non_blocking=True);y=y.to(device,non_blocking=True);logits=net(x);n=y.size(0);vt+=n;vc+=(logits.argmax(1)==y).sum().item();vl+=criterion(logits,y).item()*n
  score=vc/vt; row={'epoch':epoch,'lr':opt.param_groups[0]['lr'],'train_loss':loss_sum/tot,'train_accuracy':correct/tot,'val_loss':vl/vt,'val_accuracy':score,'seconds':round(time.time()-t,2)}
  w.writerow(row);f.flush();print('EPOCH',json.dumps(row),flush=True)
  if score>best:
   best=score;stale=0;torch.save({'epoch':epoch,'model':net.state_dict(),'source_val_accuracy':score,'config':config},RUN/'best.pt')
  else:stale+=1
  sched.step()
  torch.save({'epoch':epoch,'model':net.state_dict(),'optimizer':opt.state_dict(),'scheduler':sched.state_dict(),'best':best,'stale':stale,'config':config},RUN/'last.pt')
  if stale>=5:print('EARLY_STOP',flush=True);break
ckpt=torch.load(RUN/'best.pt',map_location=device,weights_only=False);net.load_state_dict(ckpt['model']);net.eval();test=make('target_test',evaltf)
ys=[];preds=[];confs=[];all_probs=[]
with torch.inference_mode():
 for x,y,_ in test:
  prob=net(x.to(device,non_blocking=True)).softmax(1).cpu().numpy()
  ys.extend(y.tolist());preds.extend(prob.argmax(1).tolist());confs.extend(prob.max(1).tolist());all_probs.append(prob)
probs=np.concatenate(all_probs);ys=np.array(ys);preds=np.array(preds);confs=np.array(confs)
ece=0.
for b in range(10):
 mask=(confs>=b/10)&((confs<(b+1)/10) if b<9 else (confs<=1))
 if mask.any():ece+=float(mask.mean()*abs((preds[mask]==ys[mask]).mean()-confs[mask].mean()))
res={'best_epoch':ckpt['epoch'],'source_val_accuracy':ckpt['source_val_accuracy'],'target_n':len(ys),'target_correct':int((preds==ys).sum()),'target_accuracy':float((preds==ys).mean()),'target_macro_f1':float(f1_score(ys,preds,average='macro',labels=list(range(12)),zero_division=0)),'target_ece_10bins':ece,'target_mean_confidence':float(confs.mean()),'target_wrong_mean_confidence':float(confs[preds!=ys].mean()) if (preds!=ys).any() else None,'wrong_conf_ge_90':int(((preds!=ys)&(confs>=.9)).sum()),'confusion_matrix':confusion_matrix(ys,preds,labels=list(range(12))).tolist()}
np.savez_compressed(RUN/'target_probabilities.npz',probabilities=probs,labels=ys)
(RUN/'result.json').write_text(json.dumps(res,indent=2))
print('RESULT',json.dumps(res),flush=True)
