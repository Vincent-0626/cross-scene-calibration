import json,collections
from pathlib import Path
root=Path('/root/cross_scene_20261120/visda2017'); dataset=root/'dataset'; audit=json.loads((root/'audit.json').read_text())
classes=sorted(audit['domains']['train']['per_class']);mapping={c:i for i,c in enumerate(classes)}
out={}
for domain in ('train','validation'):
 f=dataset/domain/'image_list.txt'
 if not f.exists():raise RuntimeError('missing image_list '+str(f))
 seen=set(); errors=[]; n=0
 for line in f.read_text().splitlines():
  if not line.strip():continue
  path,label=line.rsplit(None,1); parts=Path(path).parts
  cls=next((v for v in parts if v in mapping),None)
  if cls is None or int(label)!=mapping[cls] or not (dataset/domain/path).is_file():
   if len(errors)<10:errors.append([path,label,cls])
  if path in seen:
   if len(errors)<10:errors.append(['duplicate',path])
  seen.add(path);n+=1
 expected=audit['domains'][domain]['count']
 out[domain]={'lines':n,'expected':expected,'examples_of_errors':errors}
 if errors or n!=expected:raise RuntimeError('label verification failed '+repr(out[domain]))
(root/'label_audit.json').write_text(json.dumps(out,indent=2))
print('labels verified',json.dumps(out))
