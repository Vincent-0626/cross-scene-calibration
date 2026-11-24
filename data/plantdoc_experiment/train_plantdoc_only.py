import csv,gzip,hashlib,json,random,time
from pathlib import Path
from collections import Counter
import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.utils.data import Dataset,DataLoader
from torchvision import models,transforms
R=Path('/root/plantdoc_experiment'); A=R/'artifacts'; D=R/'data/PlantDoc-Original'; O=R/'plantdoc_only_v1'; O.mkdir(exist_ok=True)
if (O/'history.csv').exists(): raise RuntimeError('existing run: refuse overwrite')
def load(n):
 with gzip.open(A/(n+'.json.gz'),'rt',encoding='utf-8') as f:return json.load(f)
leaves=load('leaves'); oldconfig=load('config'); classes=[x['plantdoc_folder'] for x in oldconfig['class_order']]
verified={}
for x in leaves:
 if x['partition'] not in ('train','val','test'):continue
 marker='/PlantDoc-Original/'
 if marker not in x['image_path']:raise ValueError('unexpected original path')
 relative=x['image_path'].split(marker,1)[1];path=D/relative
 if not path.is_file():raise FileNotFoundError(path)
 if str(path) not in verified:verified[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
 if verified[str(path)]!=x['sha256']:raise ValueError('hash mismatch: '+str(path))
 x['path']=str(path)
 assert classes[x['class_index']]==x['class']
groups={k:[r for r in leaves if r['partition']==k] for k in ('train','val','test')}
assert {k:len(v) for k,v in groups.items()}=={'train':7162,'val':1232,'test':452}
print('Verified hashes for source images, counts:',{k:len(v) for k,v in groups.items()},flush=True)
random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42)
norm=transforms.Normalize([.485,.456,.406],[.229,.224,.225]); aug=transforms.Compose([transforms.Resize((224,224)),transforms.RandomResizedCrop(224,scale=(.8,1.)),transforms.RandomHorizontalFlip(),transforms.RandomVerticalFlip(),transforms.ColorJitter(.2,.2,.2,.05),transforms.ToTensor(),norm]);ev=transforms.Compose([transforms.Resize(256),transforms.CenterCrop(224),transforms.ToTensor(),norm])
class CropData(Dataset):
 def __init__(self,rows,t):self.rows=rows;self.t=t
 def __len__(self):return len(self.rows)
 def __getitem__(self,i):
  x=self.rows[i]
  with Image.open(x['path']) as im:
   return self.t(im.convert('RGB').crop(tuple(x['box']))),x['class_index'],i
tr=DataLoader(CropData(groups['train'],aug),batch_size=32,shuffle=True,num_workers=4,pin_memory=True)
val=DataLoader(CropData(groups['val'],ev),batch_size=32,num_workers=4,pin_memory=True)
net=models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1);net.fc=nn.Linear(net.fc.in_features,28);net.cuda()
opt=torch.optim.AdamW(net.parameters(),lr=.0001,weight_decay=.01);scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=12); lossfn=nn.CrossEntropyLoss();best=-1; stale=0
config={'purpose':'PlantDoc only; comparison exploratory; test previously inspected','source':'original train boxes and held-out test boxes as frozen in previous run','counts':{k:len(v) for k,v in groups.items()},'seed':42,'model':'ResNet50 ImageNet IMAGENET1K_V1','class_names':classes,'batch_size':32,'max_epochs':12,'patience':4,'lr':0.0001,'optimizer':'AdamW','weight_decay':0.01,'scheduler':'cosine 12 epochs','transforms':'match prior mixed training','caveat':'mixed comparator starts from PlantVillage-finetuned checkpoint and has additional PlantVillage training; not equal training budget; source test previously inspected; candidate mapping uncertain'}
(O/'config.json').write_text(json.dumps(config,ensure_ascii=False,indent=2),encoding='utf-8')
for epoch in range(1,13):
 start=time.time();net.train();total=0;n=0
 for x,y,_ in tr:
  x=x.cuda(non_blocking=True);y=y.cuda(non_blocking=True);opt.zero_grad(set_to_none=True);loss=lossfn(net(x),y);loss.backward();opt.step();total+=loss.item()*len(y);n+=len(y)
 scheduler.step();net.eval();correct=0;seen=0
 with torch.no_grad():
  for x,y,_ in val:
   y=y.cuda(non_blocking=True);correct+=(net(x.cuda(non_blocking=True)).argmax(1)==y).sum().item();seen+=len(y)
 acc=correct/seen
 if acc>best:best=acc;stale=0;torch.save({'model':net.state_dict(),'epoch':epoch,'val_accuracy':acc,'config':config},O/'best.pt')
 else:stale+=1
 row={'epoch':epoch,'train_loss':total/n,'val_accuracy':acc,'best':best,'seconds':time.time()-start,'stale':stale}
 with (O/'history.csv').open('a',newline='') as f:
  w=csv.DictWriter(f,fieldnames=row.keys());
  if epoch==1:w.writeheader()
  w.writerow(row)
 print(json.dumps(row),flush=True)
 if stale>=4:break
(O/'TRAIN_COMPLETE').write_text('completed\n')
net.load_state_dict(torch.load(O/'best.pt',map_location='cuda',weights_only=False)['model']);net.eval(); test=DataLoader(CropData(groups['test'],ev),batch_size=32,num_workers=4,pin_memory=True); predictions=[]
with torch.no_grad():
 for x,y,idx in test:
  proba=net(x.cuda(non_blocking=True)).softmax(1).cpu();confidence,pred=proba.max(1)
  for label,p,c,i in zip(y.tolist(),pred.tolist(),confidence.tolist(),idx.tolist()):
   r=groups['test'][i];predictions.append({'image_id':r['image_id'],'box':json.dumps(r['box']),'label':classes[label],'predicted':classes[p],'correct':int(label==p),'confidence':c})
with (O/'test_predictions.csv').open('w',newline='',encoding='utf-8') as f:
 w=csv.DictWriter(f,fieldnames=predictions[0].keys());w.writeheader();w.writerows(predictions)
accuracy=sum(x['correct'] for x in predictions)/len(predictions);report={'count':len(predictions),'accuracy':accuracy,'correct':sum(x['correct'] for x in predictions),'mean_confidence':sum(x['confidence'] for x in predictions)/len(predictions),'warning':config['caveat']};(O/'test_metrics.json').write_text(json.dumps(report,indent=2),encoding='utf-8');(O/'EVALUATION_COMPLETE').write_text('completed\n');print('TEST',json.dumps(report),flush=True)
