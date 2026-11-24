import argparse,csv,gzip,hashlib,json,random,time
from pathlib import Path
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
import torch
from torch import nn
from torch.utils.data import Dataset,DataLoader
from torchvision import models,transforms
R=Path('/root/plantdoc_experiment');A=R/'artifacts'; S=R/'data/PlantVillage/raw/color';D=R/'data/PlantDoc-Original';O=R/'plantvillage_30epoch_v1'
p=argparse.ArgumentParser();p.add_argument('--monitor-target',action='store_true');args=p.parse_args()
with gzip.open(A/'config.json.gz','rt',encoding='utf-8') as f:cfg=json.load(f)
classes=[x['plantdoc_folder'] for x in cfg['class_order']]
rows=[]
for i,c in enumerate(cfg['class_order']):
 folder=S/c['plantvillage_folder']
 if not folder.is_dir():raise FileNotFoundError(folder)
 paths=sorted(x for x in folder.rglob('*') if x.is_file() and x.suffix.lower() in {'.jpg','.jpeg','.png'})
 rows.extend([{'path':str(path),'label':i} for path in paths])
train,rest=train_test_split(rows,test_size=.3,stratify=[r['label'] for r in rows],random_state=42)
val,test=train_test_split(rest,test_size=.5,stratify=[r['label'] for r in rest],random_state=42)
groups={'train':train,'val':val,'test':test}; hashes={'train':'96827fe44267077491a34aec1273819acd5085428bed67cd58cf6b962c8a32d2','val':'9b624e47da8e115aeb4fe3f097895610600af4204ac245f74a6621ca2931aa0a','test':'2cabd02f512626c0a7d35a49ec0e42cc8a55b544667b62b45098a2c936872b4c'}
for k,v in groups.items():
 lines=sorted(str(Path(x['path']).relative_to(S))+'\t'+str(x['label']) for x in v);h=hashlib.sha256(('\n'.join(lines)+'\n').encode()).hexdigest()
 if h!=hashes[k]:raise RuntimeError(f'{k} old manifest mismatch {h}')
with gzip.open(A/'leaves.json.gz','rt',encoding='utf-8') as f: target=json.load(f)
target=[x for x in target if x['partition']=='test']
for r in target:
 marker='/PlantDoc-Original/';r['path']=str(D/r['image_path'].split(marker,1)[1]);r['label']=r['class_index']
assert len(target)==452
for r in target:
 if hashlib.sha256(Path(r['path']).read_bytes()).hexdigest()!=r['sha256']:raise ValueError('target hash mismatch '+r['path'])
if (O/'history.csv').exists():raise RuntimeError('run exists; do not overwrite')
O.mkdir(exist_ok=True);random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42)
norm=transforms.Normalize([.485,.456,.406],[.229,.224,.225]); aug=transforms.Compose([transforms.Resize((224,224)),transforms.RandomResizedCrop(224,scale=(.8,1)),transforms.RandomHorizontalFlip(),transforms.RandomVerticalFlip(),transforms.ColorJitter(.2,.2,.2,.05),transforms.ToTensor(),norm]);ev=transforms.Compose([transforms.Resize(256),transforms.CenterCrop(224),transforms.ToTensor(),norm])
class Images(Dataset):
 def __init__(self,rs,t):self.rows=rs;self.t=t
 def __len__(self):return len(self.rows)
 def __getitem__(self,i):
  r=self.rows[i]
  with Image.open(r['path']) as im:
   im=im.convert('RGB');im=im.crop(tuple(r['box'])) if 'box' in r else im;return self.t(im),r['label'],i
train_eval_rows=random.Random(42).sample(train,4096)
loaders={'train_eval':DataLoader(Images(train_eval_rows,ev),batch_size=32,num_workers=4,pin_memory=True),'train':DataLoader(Images(train,aug),batch_size=32,shuffle=True,num_workers=4,pin_memory=True),'val':DataLoader(Images(val,ev),batch_size=32,num_workers=4,pin_memory=True),'test':DataLoader(Images(test,ev),batch_size=32,num_workers=4,pin_memory=True),'target':DataLoader(Images(target,ev),batch_size=32,num_workers=4,pin_memory=True)}
net=models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1);net.fc=nn.Linear(net.fc.in_features,28);net.cuda();opt=torch.optim.AdamW(net.parameters(),lr=.0001,weight_decay=.01);scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=30);lossfn=nn.CrossEntropyLoss();best=-1;stale=0
config={'seed':42,'classes':classes,'counts':{k:len(v) for k,v in groups.items()},'target_count':len(target),'max_epochs':30,'patience':5,'selection':'source validation accuracy','target_per_epoch':args.monitor_target,'target_status':'previously observed, exploratory, not blind','source_data': '28 candidate classes, old manifests verified by list digest','training_metrics':'train_aug metrics are from augmented online training; train_eval is fixed seed random subset 4096 using deterministic eval transforms, directly comparable with val','init':'ImageNet ResNet50 V1','lr':0.0001,'weight_decay':0.01,'batch_size':32,'scheduler':'CosineAnnealingLR T_max=30'}
(O/'config.json').write_text(json.dumps(config,indent=2,ensure_ascii=False),encoding='utf-8')
def evaluate(loader):
 net.eval();total=0;correct=0;n=0;predictions=[]
 with torch.no_grad():
  for x,y,idx in loader:
   x=x.cuda(non_blocking=True);yg=y.cuda(non_blocking=True);z=net(x);p=z.softmax(1);conf,pred=p.max(1);loss=lossfn(z,yg);total+=loss.item()*len(y);correct+=(pred==yg).sum().item();n+=len(y)
   for i,a,b,c in zip(idx.tolist(),y.tolist(),pred.cpu().tolist(),conf.cpu().tolist()):predictions.append((i,a,b,c))
 return {'loss':total/n,'accuracy':correct/n,'n':n},predictions
print('validated original splits; counts',config['counts'],'target monitor',args.monitor_target,flush=True)
for epoch in range(1,31):
 start=time.time();net.train();loss_sum=0;correct=0;n=0
 for x,y,_ in loaders['train']:
  x=x.cuda(non_blocking=True);y=y.cuda(non_blocking=True);opt.zero_grad(set_to_none=True);z=net(x);loss=lossfn(z,y);loss.backward();opt.step();loss_sum+=loss.item()*len(y);correct+=(z.argmax(1)==y).sum().item();n+=len(y)
 scheduler.step();tr_eval,_=evaluate(loaders['train_eval']);v,_=evaluate(loaders['val']);acc=v['accuracy']
 if acc>best:best=acc;stale=0;torch.save({'model':net.state_dict(),'epoch':epoch,'val_accuracy':acc,'config':config},O/'best.pt')
 else:stale+=1
 row={'epoch':epoch,'train_aug_loss':loss_sum/n,'train_aug_accuracy':correct/n,'train_eval_loss':tr_eval['loss'],'train_eval_accuracy':tr_eval['accuracy'],'val_loss':v['loss'],'val_accuracy':acc,'best_val_accuracy':best,'stale':stale,'seconds':time.time()-start}
 if args.monitor_target:
  t,_=evaluate(loaders['target']);row['target_accuracy_exploratory']=t['accuracy'];row['target_loss_exploratory']=t['loss']
 with (O/'history.csv').open('a',newline='') as f:
  w=csv.DictWriter(f,fieldnames=row.keys());
  if epoch==1:w.writeheader()
  w.writerow(row)
 print(json.dumps(row),flush=True)
 if stale>=5:print('early stop',flush=True);break
net.load_state_dict(torch.load(O/'best.pt',map_location='cuda',weights_only=False)['model']);net.eval()
for name in ('test','target'):
 metrics,pred=evaluate(loaders[name]);(O/(name+'_metrics.json')).write_text(json.dumps(metrics,indent=2),encoding='utf-8')
 with (O/(name+'_predictions.csv')).open('w',newline='',encoding='utf-8') as f:
  w=csv.writer(f);w.writerow(['index','label','prediction','confidence']);w.writerows(pred)
 print('BEST',name,json.dumps(metrics),flush=True)
(O/'COMPLETE').write_text('completed\n')
