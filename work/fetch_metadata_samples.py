import hashlib
import json
import sys
from pathlib import Path
from fetch_evidence import BASE, connect

REMOTE_CODE = r'''
import gzip,json
from pathlib import Path
b=Path('/root/cross_scene_20261120')
a=Path('/root/plantdoc_experiment/artifacts')
with gzip.open(a/'config.json.gz','rt',encoding='utf8') as f: cfg=json.load(f)
with gzip.open(a/'leaves.json.gz','rt',encoding='utf8') as f: leaves=json.load(f)
target=[x for x in leaves if x['partition']=='test']
for r in target:
 r['path']='/root/plantdoc_experiment/data/PlantDoc-Original/'+r['image_path'].split('/PlantDoc-Original/',1)[1]
 r['label']=r['class_index']
samples=[]
c=cfg['class_order'][0]
p=Path('/root/plantdoc_experiment/data/PlantVillage/raw/color')/c['plantvillage_folder']
src=next(x for x in sorted(p.rglob('*')) if x.suffix.lower() in ('.jpg','.jpeg','.png'))
tgt=sorted((x for x in target if x['label']==0),key=lambda x:(x['path'],x.get('box',[])))[0]
samples.extend([{'domain':'plant','side':'source','path':str(src),'class':c['plantdoc_folder']},
                {'domain':'plant','side':'target','path':tgt['path'],'box':tgt['box'],'class':c['plantdoc_folder']}])
p=b/'runs/office_product_to_real'
train=json.loads((p/'train.json').read_text());test=json.loads((p/'test.json').read_text())
for side,rows in [('source',train),('target',test)]:
 r=sorted((x for x in rows if x['label']==0),key=lambda x:x['path'])[0]
 samples.append({'domain':'office','side':side,'path':r['path'],'class':'Alarm_Clock'})
p=b/'visda2017/runs/source_only_30'
splits=json.loads((p/'splits.json').read_text()); vc=json.loads((p/'config.json').read_text())
for side,key in [('source','source_train'),('target','target_test')]:
 r=sorted((x for x in splits[key] if x['label']==0),key=lambda x:x['path'])[0]
 samples.append({'domain':'visda','side':side,'path':str(b/'visda2017/dataset'/r['path']),'class':vc['classes'][0]})
print(json.dumps({'plant_class_order':cfg['class_order'],'plant_target_manifest':target,'samples':samples},ensure_ascii=False))
'''

def main():
    c=connect()
    stdin,stdout,stderr=c.exec_command('python3 -')
    stdin.write(REMOTE_CODE)
    stdin.channel.shutdown_write()
    payload=stdout.read().decode('utf8')
    errors=stderr.read().decode('utf8')
    if stdout.channel.recv_exit_status()!=0:
        raise RuntimeError(errors)
    data=json.loads(payload)
    sftp=c.open_sftp()
    folder=BASE/'data'/'metadata'
    folder.mkdir(parents=True,exist_ok=True)
    for r in data['samples']:
        dest=folder/(r['domain']+'_'+r['side']+Path(r['path']).suffix.lower())
        sftp.get(r['path'],str(dest))
        r['local_path']=str(dest)
        r['sha256']=hashlib.sha256(dest.read_bytes()).hexdigest()
    (folder/'metadata_samples.json').write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf8')
    sftp.close();c.close()
    print('Fetched class order, 452 target identities, and six original sample images.')

if __name__=='__main__':main()
