import csv,json
from pathlib import Path
R=Path('/root/cross_scene_20261120/calibration_three_domains')
cols=['domain','method','target_n','accuracy','macro_f1','mean_confidence','ece_10bins','nll','wrong_count','wrong_confidence_ge_90','high_conf_count','high_conf_accuracy','best_epoch','temperature']
rows=[]
for domain in ('plant','office','visda'):
 for method in ('baseline','temperature','label_smoothing','confidence_penalty'):
  fn=R/domain/('temperature' if method=='baseline' else method)/'result.json'
  if not fn.exists():continue
  d=json.loads(fn.read_text());x=d['target_baseline' if method=='baseline' else 'target_temperature' if method=='temperature' else 'target']
  row={'domain':domain,'method':method,'target_n':x['n'],'best_epoch':d.get('baseline_best_epoch') if method in ('baseline','temperature') else d.get('best_epoch'),'temperature':d.get('temperature') if method=='temperature' else None}
  for k in cols:
   if k in x:row[k]=x[k]
  rows.append(row)
with (R/'comparison.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=cols);w.writeheader();w.writerows(rows)
(R/'comparison.json').write_text(json.dumps(rows,indent=2))
print(json.dumps(rows))
