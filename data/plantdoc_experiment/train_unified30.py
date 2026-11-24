import argparse,csv,gzip,hashlib,json,random,time
from pathlib import Path
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
import torch
from torch import nn
from torch.utils.data import Dataset,DataLoader
from torchvision import models,transforms
R=Path('/root/plantdoc_experiment');A=R/'artifacts';D=R/'data/PlantDoc-Original';S=R/'data/PlantVillage/raw/color'
p=argparse.ArgumentParser();p.add_argument('--mode',choices=['plantdoc','mixed'],required=True);args=p.parse_args();O=R/(args.mode+'_unified30_v1');O.mkdir(exist_ok=True)
if (O/'history.csv').exists():raise RuntimeError('run exists; refuse overwrite')
def read(n):
 with gzip.open(A/(n+'.json.gz'),'rt',encoding='utf-8') as f:return json.load(f)
cfg=read('config');classes=[c['plantdoc_folder'] for c in cfg['class_order']];leaves=read('leaves');verified={}
for r in leaves:
 if r['partition'] not in ('train','val','test'):continue
 marker='/PlantDoc-Original/';path=D/r['image_path'].split(marker,1)[1]
 if not path.is_file():raise FileNotFoundError(path)
 if str(path) not in verified:verified[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
 if verified[str(path)]!=r['sha256']:raise ValueError('hash mismatch '+str(path))
 assert classes[r['class_index']]==r['class'];r['path']=str(path);r['label']=r['class_index']
field={k:[r for r in leaves if r['partition']==k] for k in ('train','val','test')}
assert {k:len(v) for k,v in field.items()}=={'train':7162,'val':1232,'test':452}
source=[]
for i,c in enumerate(cfg['class_order']):
 folder=S/c['plantvillage_folder']
 if not folder.is_dir():raise FileNotFoundError(folder)
 source.extend({'path':str(path),'label':i} for path in sorted(x for x in folder.rglob('*') if x.is_file() and x.suffix.lower() in ('.jpg','.jpeg','.png')))
pvtrain,rest=train_test_split(source,test_size=.3,stratify=[r['label'] for r in source],random_state=42)
lines=sorted(str(Path(x['path']).relative_to(S))+'\t'+str(x['label']) for x in pvtrain)
if hashlib.sha256(('\n'.join(lines)+'\n').encode()).hexdigest()!='96827fe44267077491a34aec1273819acd5085428bed67cd58cf6b962c8a32d2':raise RuntimeError('PlantVillage training split mismatch')
train=field['train']+(pvtrain if args.mode=='mixed' else []);field_subset=random.Random(42).sample(field['train'],4096)
random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42)
norm=transforms.Normalize([.485,.456,.406],[.229,.224,.225]);aug=transforms.Compose([transforms.Resize((224,224)),transforms.RandomResizedCrop(224,scale=(.8,1.)),transforms.RandomHorizontalFlip(),transforms.RandomVerticalFlip(),transforms.ColorJitter(.2,.2,.2,.05),transforms.ToTensor(),norm]);ev=transforms.Compose([transforms.Resize(256),transforms.CenterCrop(224),transforms.ToTensor(),norm])
class Images(Dataset):
 def __init__(self,rows,t):self.rows=rows;self.t=t
 def __len__(self):return len(self.rows)
 def __getitem__(self,i):
  r=self.rows[i]
  with Image.open(r['path']) as im:
   im=im.convert('RGB');im=im.crop(tuple(r['box'])) if 'box' in r else im;return self.t(im),r['label'],i
loaders={'train':DataLoader(Images(train,aug),batch_size=32,shuffle=True,num_workers=4,pin_memory=True),'train_eval_field':DataLoader(Images(field_subset,ev),batch_size=32,num_workers=4,pin_memory=True),'val':DataLoader(Images(field['val'],ev),batch_size=32,num_workers=4,pin_memory=True),'test':DataLoader(Images(field['test'],ev),batch_size=32,num_workers=4,pin_memory=True)}
net=models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1);net.fc=nn.Linear(net.fc.in_features,28);net.cuda();opt=torch.optim.AdamW(net.parameters(),lr=.0001,weight_decay=.01);scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=30);lossfn=nn.CrossEntropyLoss();best=-1;stale=0
config={'mode':args.mode,'seed':42,'classes':classes,'field_counts':{k:len(v) for k,v in field.items()},'source_train_count':len(pvtrain) if args.mode=='mixed' else 0,'train_count':len(train),'max_epochs':30,'patience':5,'selection':'PlantDoc validation accuracy','target_each_epoch':'previously observed 452 leaves, exploratory only, not used for training, early stopping or model selection','train_eval':'fixed random 4096 PlantDoc training leaves; deterministic eval transform','init':'ImageNet ResNet50 V1','lr':0.0001,'weight_decay':0.01,'batch_size':32,'scheduler':'CosineAnnealingLR T_max=30','caveat':'three models share init/optimizer/epoch and early stopping rule but use different validation domains and data volumes; not equal gradient updates, candidate map uncertain, test not blind'}
(O/'config.json').write_text(json.dumps(config,indent=2,ensure_ascii=False),encoding='utf-8')
def evaluate(loader,return_rows=False):
 net.eval();total=0;correct=0;n=0;result=[]
 with torch.no_grad():
  for x,y,idx in loader:
   x=x.cuda(non_blocking=True);yg=y.cuda(non_blocking=True);z=net(x);proba=z.softmax(1);confidence,pred=proba.max(1);total+=lossfn(z,yg).item()*len(y);correct+=(pred==yg).sum().item();n+=len(y)
   if return_rows:
    for i,a,b,c in zip(idx.tolist(),y.tolist(),pred.cpu().tolist(),confidence.cpu().tolist()):result.append((i,a,b,c))
 return {'loss':total/n,'accuracy':correct/n,'n':n},result
print('hashes verified; mode',args.mode,'train',len(train),'val',len(field['val']),'test',len(field['test']),flush=True)
for epoch in range(1,31):
 start=time.time();net.train();loss_sum=0;correct=0;n=0
 for x,y,_ in loaders['train']:
  x=x.cuda(non_blocking=True);y=y.cuda(non_blocking=True);opt.zero_grad(set_to_none=True);z=net(x);loss=lossfn(z,y);loss.backward();opt.step();loss_sum+=loss.item()*len(y);correct+=(z.argmax(1)==y).sum().item();n+=len(y)
 scheduler.step();tr,_=evaluate(loaders['train_eval_field']);val,_=evaluate(loaders['val']);test,_=evaluate(loaders['test']);acc=val['accuracy']
 if acc>best:best=acc;stale=0;torch.save({'model':net.state_dict(),'epoch':epoch,'val_accuracy':acc,'config':config},O/'best.pt')
 else:stale+=1
 row={'epoch':epoch,'train_aug_loss':loss_sum/n,'train_aug_accuracy':correct/n,'train_field_eval_loss':tr['loss'],'train_field_eval_accuracy':tr['accuracy'],'val_loss':val['loss'],'val_accuracy':acc,'best_val_accuracy':best,'stale':stale,'test_accuracy_exploratory':test['accuracy'],'test_loss_exploratory':test['loss'],'seconds':time.time()-start}
 with (O/'history.csv').open('a',newline='') as f:
  w=csv.DictWriter(f,fieldnames=row.keys());
  if epoch==1:w.writeheader()
  w.writerow(row)
 print(json.dumps(row),flush=True)
 if stale>=5:print('early stop',flush=True);break
net.load_state_dict(torch.load(O/'best.pt',map_location='cuda',weights_only=False)['model']);metrics,preds=evaluate(loaders['test'],True);(O/'test_metrics.json').write_text(json.dumps(metrics,indent=2),encoding='utf-8')
with (O/'test_predictions.csv').open('w',newline='',encoding='utf-8') as f:
 w=csv.writer(f);w.writerow(['index','label','prediction','confidence']);w.writerows(preds)
(O/'COMPLETE').write_text('completed\n');print('BEST test',json.dumps(metrics),flush=True)
