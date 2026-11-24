import collections, hashlib, json, random, zipfile
from pathlib import Path
from PIL import Image

base=Path('/root/cross_scene_20261120')
dest=base/'datasets'/'officehome'
dest.mkdir(parents=True,exist_ok=True)
with zipfile.ZipFile(base/'downloads'/'officehome_mirror.zip') as z:
    members=[x for x in z.infolist() if any('/'+s+'/' in x.filename for s in ('Product','Real World')) and not x.is_dir() and not x.filename.startswith('__MACOSX/')]
    for x in members:
        p=Path(x.filename)
        if p.is_absolute() or '..' in p.parts: raise ValueError(x.filename)
        z.extract(x,dest)
roots=list(dest.rglob('Product'))
if len(roots)!=1: raise RuntimeError(f'Product roots: {roots}')
root=roots[0].parent
domains={d:root/d for d in ('Product','Real World')}
classes={d:sorted(p.name for p in path.iterdir() if p.is_dir()) for d,path in domains.items()}
if classes['Product']!=classes['Real World']: raise RuntimeError('Domain labels differ: '+str(classes))
rows=[]; errors=[]; digests=collections.defaultdict(list)
for domain,dp in domains.items():
    for ci,cl in enumerate(classes[domain]):
        for p in sorted((dp/cl).iterdir()):
            if not p.is_file() or p.suffix.lower() not in ('.jpg','.jpeg','.png','.bmp'): continue
            try:
                with Image.open(p) as im: im.verify()
                h=hashlib.sha256(p.read_bytes()).hexdigest()
                row={'path':str(p),'domain':domain,'class':cl,'label':ci,'sha256':h}
                rows.append(row); digests[h].append(row)
            except Exception as e: errors.append({'path':str(p),'error':str(e)})
conflicts={h:v for h,v in digests.items() if len({x['label'] for x in v})>1}
if conflicts:
    print('Conflicting labels for identical files:',len(conflicts),flush=True)
    for h,v in list(conflicts.items())[:20]: print(h,[(x['domain'],x['class'],Path(x['path']).name) for x in v],flush=True)
    out=base/'runs'/'office_product_to_real'; out.mkdir(parents=True,exist_ok=True)
    (out/'label_conflicts.json').write_text(json.dumps(conflicts,ensure_ascii=False,indent=2),encoding='utf8')
    rows=[r for r in rows if r['sha256'] not in conflicts]
    for h in conflicts: del digests[h]
cross=[h for h,v in digests.items() if len({r['domain'] for r in v})>1]
# Exclude target exact duplicates of source, and preserve groups when splitting source.
rng=random.Random(42); splits={'train':[],'val':[],'test':[],'excluded_cross_domain_duplicates':[]}
for ci,cl in enumerate(classes['Product']):
    groups=collections.defaultdict(list)
    for r in rows:
        if r['label']==ci and r['domain']=='Product': groups[r['sha256']].append(r)
        elif r['label']==ci and r['domain']=='Real World': splits['excluded_cross_domain_duplicates' if r['sha256'] in cross else 'test'].append(r)
    keys=sorted(groups); rng.shuffle(keys)
    if len(keys)<2: raise RuntimeError('Insufficient independent source images in '+cl)
    n=max(1,round(len(keys)*.2)); n=min(n,len(keys)-1)
    for k in keys[:n]: splits['val'].extend(groups[k])
    for k in keys[n:]: splits['train'].extend(groups[k])
out=base/'runs'/'office_product_to_real'; out.mkdir(parents=True,exist_ok=True)
for name,data in splits.items():
    (out/(name+'.json')).write_text(json.dumps(data,ensure_ascii=False,indent=1),encoding='utf8')
archive_hash='4b3ee939c181e508109064c484d59bcff0386bdf15960e8d11305ee5c4bb81ff'  # verified separately with sha256sum
report={'archive_sha256':archive_hash,'source':'https://hf-mirror.com/huangyuyang11/officehome/resolve/main/OfficeHomeDataset_10072016.zip','source_note':'third-party mirror; not verified byte-for-byte against official distribution','classes':classes['Product'],'counts':{n:len(v) for n,v in splits.items()},'per_class':{n:dict(collections.Counter(r['class'] for r in v)) for n,v in splits.items()},'bad_images':errors,'cross_domain_duplicate_hashes':len(cross),'conflicting_label_hashes':len(conflicts),'conflicting_label_images_excluded':sum(map(len,conflicts.values())),'seed':42,'validation':'20% of distinct source file hashes per class, seed 42','test_policy':'all Real World except exact source duplicates and files with conflicting labels; no target labels or results used to select checkpoint'}
(out/'audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
print(json.dumps({'classes':len(classes['Product']),'counts':report['counts'],'bad_images':len(errors),'cross_duplicate_hashes':len(cross)},ensure_ascii=False))
