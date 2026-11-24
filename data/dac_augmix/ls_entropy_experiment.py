import argparse,csv,gzip,hashlib,json,random,time
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
import torch
from torch import nn
from torch.utils.data import Dataset,DataLoader
from torchvision import models,transforms

ROOT=Path('/root/cross_scene_20261120/dac_augmix')
p=argparse.ArgumentParser();p.add_argument('--domain',required=True,choices=['plant','office','visda']);p.add_argument('--method',required=True,choices=['ls_entropy']);args=p.parse_args()
OUT=ROOT/args.domain/args.method; OUT.mkdir(parents=True,exist_ok=True)
if (OUT/'COMPLETE').exists():raise SystemExit('This run already completed; refusing to overwrite.')
assert torch.cuda.is_available()
random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42);torch.backends.cudnn.benchmark=True
norm=transforms.Normalize([.485,.456,.406],[.229,.224,.225])
ev=transforms.Compose([transforms.Resize(256),transforms.CenterCrop(224),transforms.ToTensor(),norm])
aug=transforms.Compose([transforms.RandomResizedCrop(224,scale=(.8,1.)),transforms.RandomHorizontalFlip(),transforms.ToTensor(),norm])
extra_train_eval=None
if args.domain=='plant':
 R=Path('/root/plantdoc_experiment');A=R/'artifacts';S=R/'data/PlantVillage/raw/color';D=R/'data/PlantDoc-Original'
 with gzip.open(A/'config.json.gz','rt',encoding='utf8') as f: old=json.load(f)
 classes=[x['plantdoc_folder'] for x in old['class_order']]; rows=[]
 for i,c in enumerate(old['class_order']):
  folder=S/c['plantvillage_folder']
  if not folder.is_dir():raise RuntimeError('missing folder '+str(folder))
  files=sorted(x for x in folder.rglob('*') if x.is_file() and x.suffix.lower() in ('.jpg','.jpeg','.png'))
  rows.extend([{'path':str(x),'label':i} for x in files])
 tr,rest=train_test_split(rows,test_size=.3,stratify=[r['label'] for r in rows],random_state=42)
 va,te=train_test_split(rest,test_size=.5,stratify=[r['label'] for r in rest],random_state=42)
 expected={'train':'96827fe44267077491a34aec1273819acd5085428bed67cd58cf6b962c8a32d2','val':'9b624e47da8e115aeb4fe3f097895610600af4204ac245f74a6621ca2931aa0a','test':'2cabd02f512626c0a7d35a49ec0e42cc8a55b544667b62b45098a2c936872b4c'}
 for k,v in [('train',tr),('val',va),('test',te)]:
  lines=sorted(str(Path(x['path']).relative_to(S))+'\t'+str(x['label']) for x in v)
  assert hashlib.sha256(('\n'.join(lines)+'\n').encode()).hexdigest()==expected[k],k
 with gzip.open(A/'leaves.json.gz','rt',encoding='utf8') as f:t=json.load(f)
 target=[x for x in t if x['partition']=='test']
 for r in target:r['path']=str(D/r['image_path'].split('/PlantDoc-Original/',1)[1]);r['label']=r['class_index']
 assert len(target)==452
 for r in target:
  if hashlib.sha256(Path(r['path']).read_bytes()).hexdigest()!=r['sha256']:raise RuntimeError('target hash mismatch')
 splits={'train':tr,'val':va,'target':target}
 aug=transforms.Compose([transforms.Resize((224,224)),transforms.RandomResizedCrop(224,scale=(.8,1)),transforms.RandomHorizontalFlip(),transforms.RandomVerticalFlip(),transforms.ColorJitter(.2,.2,.2,.05),transforms.ToTensor(),norm])
 extra_train_eval=random.Random(42).sample(tr,4096)
 baseline=R/'plantvillage_30epoch_v1/best.pt'; expected_correct=125
elif args.domain=='office':
 P=Path('/root/cross_scene_20261120/runs/office_product_to_real')
 audit=json.loads((P/'audit.json').read_text());classes=audit['classes']
 splits={k:json.loads((P/(f+'.json')).read_text()) for k,f in [('train','train'),('val','val'),('target','test')]}
 baseline=P/'best.pt';expected_correct=2684
else:
 V=Path('/root/cross_scene_20261120/visda2017'); P=V/'runs/source_only_30'
 config=json.loads((P/'config.json').read_text());classes=config['classes'];old=json.loads((P/'splits.json').read_text())
 splits={k:old[f] for k,f in [('train','source_train'),('val','source_val'),('target','target_test')]}
 for items in splits.values():
  for row in items:row['path']=str(V/'dataset'/row['path'])
 baseline=P/'best.pt';expected_correct=11581
num_classes=len(classes)
assert all(0<=r['label']<num_classes for items in splits.values() for r in items)
class Images(Dataset):
 def __init__(self,rows,tf):self.rows=rows;self.tf=tf
 def __len__(self):return len(self.rows)
 def __getitem__(self,i):
  r=self.rows[i]
  with Image.open(r['path']) as im:
   im=im.convert('RGB')
   if 'box' in r:im=im.crop(tuple(r['box']))
   return self.tf(im),int(r['label']),i
workers=6 if args.domain=='visda' else 4
def loader(rows,tf,shuffle=False):
 return DataLoader(Images(rows,tf),batch_size=32,shuffle=shuffle,num_workers=workers,pin_memory=True,persistent_workers=args.domain=='visda')
val=loader(splits['val'],ev);target=loader(splits['target'],ev)
device=torch.device('cuda')
net=models.resnet50(weights=None if args.method=='temperature' else models.ResNet50_Weights.IMAGENET1K_V1)
net.fc=nn.Linear(net.fc.in_features,num_classes);net.to(device)
def eval_logits(ld):
 net.eval();zs=[];ys=[]
 with torch.inference_mode():
  for x,y,_ in ld:
   z=net(x.to(device,non_blocking=True));zs.append(z.cpu().numpy());ys.append(y.numpy())
 return np.concatenate(zs),np.concatenate(ys)
def summarize(z,y,T=1):
 v=z.astype(np.float64)/T; lp=v-logsumexp(v,axis=1,keepdims=True);prob=np.exp(lp);pred=prob.argmax(1);cf=prob.max(1);correct=pred==y
 bins=[];ece=0.
 for i in range(10):
  sel=(cf>=i/10)&((cf<(i+1)/10) if i<9 else (cf<=1))
  if sel.any():
   a=float(correct[sel].mean());c=float(cf[sel].mean());ece+=float(sel.mean()*abs(a-c))
   bins.append({'bin':i,'n':int(sel.sum()),'accuracy':a,'confidence':c})
 wrong=~correct; high=cf>=.9
 return {'n':len(y),'correct':int(correct.sum()),'accuracy':float(correct.mean()),'macro_f1':float(f1_score(y,pred,labels=list(range(num_classes)),average='macro',zero_division=0)),'mean_confidence':float(cf.mean()),'ece_10bins':ece,'nll':float(-lp[np.arange(len(y)),y].mean()),'brier':float(((prob-np.eye(num_classes)[y])**2).sum(1).mean()),'wrong_count':int(wrong.sum()),'wrong_mean_confidence':float(cf[wrong].mean()) if wrong.any() else None,'wrong_confidence_ge_90':int((wrong&high).sum()),'high_conf_count':int(high.sum()),'high_conf_accuracy':float(correct[high].mean()) if high.any() else None,'bins':bins}
cfg={'domain':args.domain,'method':args.method,'source_val_selection':'source validation accuracy, patience 5','temperature_fit':'source validation only; bounded 0.02..50 minimizing NLL' if args.method=='temperature' else None,'label_smoothing':.05 if args.method=='ls_entropy' else 0,'confidence_penalty_lambda':.1 if args.method=='ls_entropy' else 0,'source_only':True,'target_previously_observed':'yes; exploratory comparison, not final held-out','seed':42,'classes':classes,'counts':{k:len(v) for k,v in splits.items()},'max_epochs':30,'patience':5,'optimizer':'AdamW lr=0.0001 weight_decay=0.01','scheduler':'cosine T_max=30','batch_size':32,'baseline_checkpoint':str(baseline)}
(OUT/'config.json').write_text(json.dumps(cfg,indent=2,ensure_ascii=False))
print('CONFIG',json.dumps(cfg,ensure_ascii=False),flush=True)
if args.method=='temperature':
 checkpoint=torch.load(baseline,map_location='cpu',weights_only=False);net.load_state_dict(checkpoint['model']);net.to(device)
 zv,yv=eval_logits(val);zt,yt=eval_logits(target)
 before=summarize(zt,yt)
 if abs(before['correct']-expected_correct)>max(2,round(.002*len(yt))):raise RuntimeError('baseline target reproduction mismatch '+str(before['correct'])+' expected '+str(expected_correct))
 def objective(logt):
  x=zv.astype(np.float64)/np.exp(logt)
  return float((logsumexp(x,axis=1)-x[np.arange(len(yv)),yv]).mean())
 fit=minimize_scalar(objective,bounds=(np.log(.02),np.log(50)),method='bounded')
 if not fit.success:raise RuntimeError('temperature fit failed')
 T=float(np.exp(fit.x));after=summarize(zt,yt,T)
 assert before['correct']==after['correct']
 result={'method':'temperature','temperature':T,'baseline_best_epoch':checkpoint.get('epoch'),'source_validation_before':summarize(zv,yv),'source_validation_after':summarize(zv,yv,T),'target_baseline':before,'target_temperature':after,'selection':'Temperature minimized source-validation NLL, no target labels used in fitting'}
 np.savez_compressed(OUT/'logits.npz',source_validation_logits=zv,source_validation_labels=yv,target_logits=zt,target_labels=yt)
else:
 train=loader(splits['train'],aug,True); train_eval=loader(extra_train_eval,ev) if extra_train_eval is not None else None
 opt=torch.optim.AdamW(net.parameters(),lr=.0001,weight_decay=.01); sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=30)
 ce=nn.CrossEntropyLoss(); smooth=nn.CrossEntropyLoss(label_smoothing=.05);best=-1;stale=0
 with (OUT/'epochs.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=['epoch','learning_rate','train_objective','train_accuracy','train_eval_accuracy','val_nll','val_accuracy','best_val','stale','seconds']);w.writeheader();f.flush()
  for epoch in range(1,31):
   t0=time.time();net.train();total=correct=0;loss_sum=0.
   for x,y,_ in train:
    x=x.to(device,non_blocking=True);y=y.to(device,non_blocking=True);opt.zero_grad(set_to_none=True);z=net(x)
    if args.method=='label_smoothing':loss=smooth(z,y)
    else:
     logp=torch.log_softmax(z,dim=1);prob=logp.exp();loss=smooth(z,y)+.1*(prob*logp).sum(dim=1).mean()
    loss.backward();opt.step();n=len(y);total+=n;loss_sum+=loss.item()*n;correct+=(z.argmax(1)==y).sum().item()
   if args.domain=='plant':sched.step()
   teacc=None
   if train_eval is not None:
    ze,ye=eval_logits(train_eval);teacc=float((ze.argmax(1)==ye).mean())
   zv,yv=eval_logits(val);v=summarize(zv,yv);acc=v['accuracy']
   if acc>best:best=acc;stale=0;torch.save({'epoch':epoch,'model':net.state_dict(),'val_accuracy':acc,'config':cfg},OUT/'best.pt')
   else:stale+=1
   row={'epoch':epoch,'learning_rate':opt.param_groups[0]['lr'],'train_objective':loss_sum/total,'train_accuracy':correct/total,'train_eval_accuracy':teacc,'val_nll':v['nll'],'val_accuracy':acc,'best_val':best,'stale':stale,'seconds':round(time.time()-t0,2)}
   w.writerow(row);f.flush();print('EPOCH',json.dumps(row),flush=True)
   if args.domain!='plant':sched.step()
   if stale>=5:print('EARLY_STOP',flush=True);break
 checkpoint=torch.load(OUT/'best.pt',map_location='cpu',weights_only=False);net.load_state_dict(checkpoint['model']);net.to(device)
 zv,yv=eval_logits(val);zt,yt=eval_logits(target)
 result={'method':args.method,'best_epoch':checkpoint['epoch'],'source_validation':summarize(zv,yv),'target':summarize(zt,yt)}
 np.savez_compressed(OUT/'logits.npz',source_validation_logits=zv,source_validation_labels=yv,target_logits=zt,target_labels=yt)
(OUT/'result.json').write_text(json.dumps(result,indent=2,ensure_ascii=False))
(OUT/'COMPLETE').write_text('complete\n')
print('RESULT',json.dumps(result,ensure_ascii=False),flush=True)
