from pathlib import Path

root = Path('/root/cross_scene_20261120/dac_augmix')
source = (root / 'augmix_experiment.py').read_text()
source = source[:source.index("if args.method=='temperature':\n checkpoint")]
source = source.replace("choices=['augmix']", "choices=['alis']")
source = source.replace("num_classes=len(classes)", """num_classes=len(classes)
manifest = {
 k: [{'path': r['path'], 'label': r['label'], **({'box':r['box']} if 'box' in r else {})} for r in rows]
 for k, rows in splits.items()
}
manifest_json = json.dumps(manifest,sort_keys=True,separators=(',',':'))
(OUT/'split_manifest.json.gz').write_bytes(gzip.compress(manifest_json.encode(),mtime=0))
""")
source = source[:source.index('class AugMixImages')] + source[source.index('workers=6 if'):]
source = source.replace("print('CONFIG'", """cfg.update(alis={'paper':'10.1016/j.asoc.2025.114410','lambda':1.0,'bins':15,
 'temperature_initial':1.0,'temperature_parameterization':'exp(log_temperature)',
 'bin_score':'unscaled max softmax','targets':'previous source validation, original and augmented separately',
 'empty_bins':'nearest occupied bin, ties lower','initial_bins':'ImageNet initialization with new source head, pre-epoch 1 validation',
 'augmentation':'torchvision AugMix severity 3, width 3, random depth, alpha 1, single additional view',
 'adaptation':'ours: original split/AdamW/30 epochs; paper uses SGD and 200 epochs'})
cfg['manifest_sha256']=hashlib.sha256(manifest_json.encode()).hexdigest()
(OUT/'config.json').write_text(json.dumps(cfg,indent=2,ensure_ascii=False))
print('CONFIG'""")
(root/'alis_experiment.py').write_text(source)

