import csv, json, random, time
from pathlib import Path
import numpy as np
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms

root=Path('/root/cross_scene_20261120/runs/office_product_to_real')
audit=json.loads((root/'audit.json').read_text())
random.seed(42); np.random.seed(42); torch.manual_seed(42); torch.cuda.manual_seed_all(42)
torch.backends.cudnn.benchmark=True
assert torch.cuda.is_available(), 'GPU is required for training'
device=torch.device('cuda')
mean=(.485,.456,.406); std=(.229,.224,.225)
train_tf=transforms.Compose([transforms.RandomResizedCrop(224,scale=(.8,1.)),transforms.RandomHorizontalFlip(),transforms.ToTensor(),transforms.Normalize(mean,std)])
eval_tf=transforms.Compose([transforms.Resize(256),transforms.CenterCrop(224),transforms.ToTensor(),transforms.Normalize(mean,std)])
class Images(Dataset):
    def __init__(self, split, tf): self.rows=json.loads((root/(split+'.json')).read_text()); self.tf=tf
    def __len__(self): return len(self.rows)
    def __getitem__(self,i):
        r=self.rows[i]
        with Image.open(r['path']) as img: return self.tf(img.convert('RGB')),r['label'],i
def loader(split,tf,shuffle=False): return DataLoader(Images(split,tf),batch_size=32,shuffle=shuffle,num_workers=4,pin_memory=True)
train=loader('train',train_tf,True); val=loader('val',eval_tf)
model=models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
model.fc=nn.Linear(model.fc.in_features,len(audit['classes'])); model.to(device)
optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=.01)
scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=30)
criterion=nn.CrossEntropyLoss()
config={'model':'torchvision ResNet50, ImageNet1K_V1 initialization','source':'Office-Home Product','target':'Office-Home Real World','classes':len(audit['classes']),'seed':42,'batch_size':32,'max_epochs':30,'early_stopping_patience':5,'early_stopping_metric':'source validation accuracy','optimizer':'AdamW','learning_rate':1e-4,'weight_decay':.01,'lr_scheduler':'cosine T_max=30','input':'224x224; training RandomResizedCrop scale 0.8-1, horizontal flip; evaluation Resize256 CenterCrop224; ImageNet normalization','target_policy':audit['test_policy'],'counts':audit['counts']}
(root/'config.json').write_text(json.dumps(config,indent=2))
best=-1.; bad=0
with (root/'epochs.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=['epoch','lr','train_loss','train_accuracy','val_loss','val_accuracy','seconds']); w.writeheader(); f.flush()
    for epoch in range(1,31):
        start=time.time(); model.train(); total=correct=0; loss_sum=0.
        for x,y,_ in train:
            x=x.to(device,non_blocking=True); y=y.to(device,non_blocking=True); optimizer.zero_grad(set_to_none=True)
            logits=model(x); loss=criterion(logits,y); loss.backward(); optimizer.step()
            n=y.size(0); total+=n; loss_sum+=loss.item()*n; correct+=(logits.argmax(1)==y).sum().item()
        model.eval(); vtotal=vc=0; vl=0.
        with torch.inference_mode():
            for x,y,_ in val:
                x=x.to(device,non_blocking=True); y=y.to(device,non_blocking=True); logits=model(x); n=y.size(0)
                vtotal+=n; vl+=criterion(logits,y).item()*n; vc+=(logits.argmax(1)==y).sum().item()
        score=vc/vtotal
        row={'epoch':epoch,'lr':optimizer.param_groups[0]['lr'],'train_loss':loss_sum/total,'train_accuracy':correct/total,'val_loss':vl/vtotal,'val_accuracy':score,'seconds':round(time.time()-start,2)}
        w.writerow(row); f.flush(); print(json.dumps(row),flush=True)
        if score>best:
            best=score; bad=0; torch.save({'epoch':epoch,'model':model.state_dict(),'val_accuracy':score,'config':config},root/'best.pt')
        else: bad+=1
        scheduler.step()
        if bad>=5: print('early stopping',flush=True); break
ckpt=torch.load(root/'best.pt',map_location=device,weights_only=False); model.load_state_dict(ckpt['model']); model.eval()
test=loader('test',eval_tf); ys=[]; ps=[]; confs=[]; indices=[]
with torch.inference_mode():
    for x,y,i in test:
        p=model(x.to(device,non_blocking=True)).softmax(1); conf,pred=p.max(1)
        ys.extend(y.tolist()); ps.extend(pred.cpu().tolist()); confs.extend(conf.cpu().tolist()); indices.extend(i.tolist())
ece=0.
for lo in np.linspace(0,1,11)[:-1]:
    ix=[i for i,c in enumerate(confs) if c>=lo and (c<lo+.1 or (lo>.89 and c<=1))]
    if ix: ece+=len(ix)/len(ys)*abs(sum(ys[i]==ps[i] for i in ix)/len(ix)-sum(confs[i] for i in ix)/len(ix))
metrics={'best_epoch':ckpt['epoch'],'source_validation_accuracy':ckpt['val_accuracy'],'target_test_n':len(ys),'target_accuracy':accuracy_score(ys,ps),'target_macro_f1':f1_score(ys,ps,labels=list(range(len(audit['classes']))),average='macro',zero_division=0),'target_ece_10_bins':ece,'correct':sum(a==b for a,b in zip(ys,ps))}
(root/'result.json').write_text(json.dumps(metrics,indent=2))
with (root/'target_predictions.csv').open('w',newline='') as f:
    w=csv.writer(f); w.writerow(['index_in_test_manifest','true_label','predicted_label','confidence']); w.writerows(zip(indices,ys,ps,confs))
print('RESULT '+json.dumps(metrics),flush=True)
