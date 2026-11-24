import csv,gzip,hashlib,json,time
from pathlib import Path
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score,classification_report,confusion_matrix
import torch
from torch import nn
from torch.utils.data import Dataset,DataLoader
from torchvision import models,transforms
R=Path('/root/plantdoc_experiment');S=R/'data/PlantVillage/raw/color';A=R/'artifacts';O=R/'plantdoc_to_plantvillage_20261120';O.mkdir(exist_ok=True)
if (O/'metrics.json').exists():raise RuntimeError('Existing results; refusing overwrite')
with gzip.open(A/'config.json.gz','rt',encoding='utf-8') as f:old=json.load(f)
classes=[r['plantdoc_folder'] for r in old['class_order']];rows=[]
for i,r in enumerate(old['class_order']):
 folder=S/r['plantvillage_folder']
 if not folder.is_dir():raise FileNotFoundError(folder)
 rows.extend({'path':str(p),'label':i} for p in sorted(p for p in folder.rglob('*') if p.is_file() and p.suffix.lower() in {'.jpg','.jpeg','.png'}))
train,rest=train_test_split(rows,test_size=.3,stratify=[r['label'] for r in rows],random_state=42)
val,test=train_test_split(rest,test_size=.5,stratify=[r['label'] for r in rest],random_state=42)
lines=sorted(str(Path(x['path']).relative_to(S))+'\t'+str(x['label']) for x in test)
if hashlib.sha256(('\n'.join(lines)+'\n').encode()).hexdigest()!='2cabd02f512626c0a7d35a49ec0e42cc8a55b544667b62b45098a2c936872b4c':raise RuntimeError('PlantVillage test manifest differs')
checkpoint=torch.load(R/'plantdoc_unified30_v1/best.pt',map_location='cpu',weights_only=False)
if checkpoint['config']['classes']!=classes:raise RuntimeError('Class order differs')
net=models.resnet50(weights=None);net.fc=nn.Linear(net.fc.in_features,len(classes));net.load_state_dict(checkpoint['model']);net.cuda().eval()
tf=transforms.Compose([transforms.Resize(256),transforms.CenterCrop(224),transforms.ToTensor(),transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
class Images(Dataset):
 def __len__(self):return len(test)
 def __getitem__(self,i):
  r=test[i]
  with Image.open(r['path']) as im:return tf(im.convert('RGB')),r['label'],i
loader=DataLoader(Images(),batch_size=48,num_workers=4,pin_memory=True)
output=[];total_loss=0.;lossfn=nn.CrossEntropyLoss();start=time.time()
with torch.inference_mode():
 for x,y,idx in loader:
  yg=y.cuda(non_blocking=True);z=net(x.cuda(non_blocking=True));probs=z.softmax(1);conf,pred=probs.max(1);total_loss+=lossfn(z,yg).item()*len(y)
  for i,t,q,c in zip(idx.tolist(),y.tolist(),pred.cpu().tolist(),conf.cpu().tolist()):output.append({'path':test[i]['path'],'class_index':t,'label':classes[t],'prediction':classes[q],'correct':int(t==q),'confidence':c})
y=[r['label'] for r in output];pred=[r['prediction'] for r in output];conf=np.array([r['confidence'] for r in output]);correct=np.array([r['correct'] for r in output]);ece=0.
for lo,hi in zip(np.linspace(0,1,16)[:-1],np.linspace(0,1,16)[1:]):
 m=(conf>=lo)&((conf<hi) if hi<1 else (conf<=hi))
 if m.any():ece+=m.mean()*abs(correct[m].mean()-conf[m].mean())
result={'n':len(y),'correct':int(correct.sum()),'accuracy':float(correct.mean()),'macro_f1':float(f1_score(y,pred,labels=classes,average='macro',zero_division=0)),'mean_confidence':float(conf.mean()),'ece_equal_width_15_bins':float(ece),'cross_entropy':total_loss/len(y),'model_selected_epoch':checkpoint['epoch'],'data':'PlantVillage same fixed source test split','classes':classes,'class_report':classification_report(y,pred,labels=classes,output_dict=True,zero_division=0),'limitations':'exploratory; semantic class map unverified; source test previously inspected; possible near-duplicates not exhaustively excluded','seconds':time.time()-start}
with (O/'predictions.csv').open('w',encoding='utf-8',newline='') as f:w=csv.DictWriter(f,fieldnames=output[0].keys());w.writeheader();w.writerows(output)
(O/'metrics.json').write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf-8');print(json.dumps({k:v for k,v in result.items() if k not in ('classes','class_report')},ensure_ascii=False,indent=2),flush=True)
