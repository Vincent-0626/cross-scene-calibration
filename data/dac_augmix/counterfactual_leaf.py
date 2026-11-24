import csv,gzip,json,random,hashlib
from pathlib import Path
import cv2,numpy as np,torch
from PIL import Image,ImageDraw
from torchvision import models,transforms
from scipy.special import softmax
from sklearn.metrics import f1_score
R=Path('/root/plantdoc_experiment');O=Path('/root/cross_scene_20261120/dac_augmix/counterfactual_leaf');O.mkdir(exist_ok=True)
assert torch.cuda.is_available();np.random.seed(42);torch.manual_seed(42)
rows=[r for r in json.load(gzip.open(R/'artifacts/leaves.json.gz','rt')) if r['partition']=='test'];assert len(rows)==452
net=models.resnet50(weights=None);net.fc=torch.nn.Linear(net.fc.in_features,28);ck=torch.load(R/'plantvillage_30epoch_v1/best.pt',map_location='cpu',weights_only=False);net.load_state_dict(ck['model']);net.cuda().eval()
tf=transforms.Compose([transforms.Resize(256),transforms.CenterCrop(224),transforms.ToTensor(),transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
records=[];images=[];z_original=[];z_grey=[];z_green=[];ys=[]
for i,r in enumerate(rows):
 path=R/'data/PlantDoc-Original'/r['image_path'].split('/PlantDoc-Original/',1)[1]
 if hashlib.sha256(path.read_bytes()).hexdigest()!=r['sha256']:raise ValueError('image hash mismatch')
 im=Image.open(path).convert('RGB').crop(tuple(r['box'])).resize((256,256),Image.Resampling.LANCZOS);rgb=np.array(im)
 mask=np.zeros((256,256),np.uint8);bg=np.zeros((1,65),np.float64);fg=np.zeros((1,65),np.float64);error=None
 try:cv2.grabCut(cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR),mask,(8,8,240,240),bg,fg,5,cv2.GC_INIT_WITH_RECT)
 except cv2.error as e:error=str(e);mask[:]=cv2.GC_FGD
 binary=np.isin(mask,[cv2.GC_FGD,cv2.GC_PR_FGD]);fraction=float(binary.mean())
 grey=np.where(binary[:,:,None],rgb,np.array([180,180,180],dtype=np.uint8));green=np.where(binary[:,:,None],rgb,np.array([92,130,82],dtype=np.uint8))
 batch=torch.stack([tf(im),tf(Image.fromarray(grey)),tf(Image.fromarray(green))]).cuda()
 with torch.inference_mode():z=net(batch).cpu().numpy()
 z_original.append(z[0]);z_grey.append(z[1]);z_green.append(z[2]);ys.append(r['class_index'])
 records.append({'index':i,'image_id':r['image_id'],'class':r['class'],'label':r['class_index'],'mask_fraction':fraction,'flag_mask':bool(error or fraction<.1 or fraction>.9),'segmentation_error':error})
 if i in set(np.linspace(0,len(rows)-1,36,dtype=int)):
  images.append((i,im,Image.fromarray(binary.astype('uint8')*255).convert('RGB'),Image.fromarray(grey),Image.fromarray(green)))
 if i%50==0:print('PROGRESS',i,flush=True)
y=np.array(ys);result={}
for name,z in [('original',z_original),('grey',z_grey),('green',z_green)]:
 z=np.array(z);p=softmax(z,axis=1);pred=p.argmax(1);conf=p.max(1);ok=pred==y;hi=conf>=.9
 ece=sum(((conf>=j/10)&(conf<((j+1)/10 if j<9 else 1.00001))).mean()*abs(ok[(conf>=j/10)&(conf<((j+1)/10 if j<9 else 1.00001))].mean()-conf[(conf>=j/10)&(conf<((j+1)/10 if j<9 else 1.00001))].mean()) for j in range(10) if ((conf>=j/10)&(conf<((j+1)/10 if j<9 else 1.00001))).any())
 result[name]={'accuracy':float(ok.mean()),'ece':float(ece),'high_confidence':int(hi.sum()),'high_confidence_wrong':int((hi&~ok).sum()),'mean_confidence':float(conf.mean())}
for alt,z in [('grey',z_grey),('green',z_green)]:result[alt]['pred_changed']=int((np.argmax(z,axis=1)!=np.argmax(z_original,axis=1)).sum())
result.update({'n':len(y),'checkpoint_epoch':ck['epoch'],'flagged_masks':sum(x['flag_mask'] for x in records),'mask_method':'GrabCut on bounding-box crop, 8px margin; automated imperfect segmentation, requires manual inspection','caveat':'exploratory paired background manipulation, solid colours only, not causal proof; resized original control; ground truth boxes used'})
np.savez_compressed(O/'logits.npz',original=z_original,grey=z_grey,green=z_green,labels=y)
(O/'records.json').write_text(json.dumps(records,ensure_ascii=False,indent=2));(O/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2));
for start in range(0,len(images),12):
 selected=images[start:start+12];sheet=Image.new('RGB',(4*192,len(selected)*216),'white');draw=ImageDraw.Draw(sheet)
 for line,(idx,*ims) in enumerate(selected):
  for j,im in enumerate(ims):sheet.paste(im.resize((192,192)),(j*192,line*216+20))
  draw.text((0,line*216),str(idx)+' original / mask / grey / green',fill='black')
 sheet.save(O/('mask_review_%02d.jpg'%(start//12+1)),quality=90)
(O/'COMPLETE').write_text('complete\n');print('RESULT',json.dumps(result),flush=True)
