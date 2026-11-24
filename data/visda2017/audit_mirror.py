import zipfile,hashlib,collections,json,os,stat
from pathlib import Path,PurePosixPath
root=Path('/root/cross_scene_20261120/visda2017'); path=root/'downloads/kaggle_visda.zip'; dest=root/'dataset'
expected_size=8569190903
if path.stat().st_size!=expected_size:raise RuntimeError('Mirror archive incomplete')
h=hashlib.sha256()
with path.open('rb') as f:
 for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
counts={'train':collections.Counter(),'validation':collections.Counter()}; modelsets=collections.defaultdict(set); seen=set(); n=0; other=[]
with zipfile.ZipFile(path) as z:
 for info in z.infolist():
  p=PurePosixPath(info.filename); parts=p.parts
  if p.is_absolute() or '..' in parts or stat.S_ISLNK(info.external_attr >> 16):raise RuntimeError('unsafe member '+info.filename)
  if len(parts)<4 or parts[:2]!=('dataset','VisDA') or parts[2] not in counts:other.append(info.filename);continue
  rel=Path(*parts[2:]);target=dest/rel
  if str(rel) in seen:raise RuntimeError('duplicate member '+str(rel))
  seen.add(str(rel))
  if info.is_dir():target.mkdir(parents=True,exist_ok=True);continue
  target.parent.mkdir(parents=True,exist_ok=True)
  with z.open(info) as src,target.open('wb') as out:
   while True:
    b=src.read(1024*1024)
    if not b:break
    out.write(b)
  if len(parts)>=5 and target.suffix.lower() in ('.jpg','.jpeg','.png'):
   domain,cls=parts[2],parts[3];counts[domain][cls]+=1;n+=1
   if domain=='train':modelsets[cls].add(target.name.split('__')[0])
  if n and n%25000==0:print('processed_images',n,flush=True);n+=1;n-=1
train_total=sum(counts['train'].values());val_total=sum(counts['validation'].values())
summary={'mirror':'https://www.kaggle.com/datasets/xman619/visda-2017-train-validation','mirror_sha256':h.hexdigest(),'checksum_note':'locally calculated, no independently verified official hash','domains':{k:{'count':sum(v.values()),'per_class':dict(sorted(v.items()))} for k,v in counts.items()},'training_model_prefix_counts':{k:len(v) for k,v in sorted(modelsets.items())},'other_entries_first_20':other[:20], 'other_entries_count':len(other)}
(root/'audit.json').write_text(json.dumps(summary,indent=2))
print('SUMMARY',json.dumps(summary),flush=True)
if train_total!=152397 or val_total!=55388 or len(counts['train'])!=12 or set(counts['train'])!=set(counts['validation']):raise RuntimeError('counts/class discrepancy; audit saved')
