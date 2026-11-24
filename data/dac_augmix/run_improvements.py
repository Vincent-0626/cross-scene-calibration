import gzip, hashlib, json, random, sys, time
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
import faiss

sys.path.insert(0, '/root/cross_scene_20261120/dac_augmix/official_dac')
from density_aware_calib import DAC

faiss.omp_set_num_threads(8)
assert torch.cuda.is_available()

OUT_ROOT = Path('/root/cross_scene_20261120/dac_augmix/improvements_20261124')
OUT_ROOT.mkdir(parents=True, exist_ok=True)

norm = transforms.Normalize([.485, .456, .406], [.229, .224, .225])
ev = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), norm])

class Images(Dataset):
    def __init__(self, rows, tf):
        self.rows = rows
        self.tf = tf
    def __len__(self):
        return len(self.rows)
    def __getitem__(self, i):
        r = self.rows[i]
        with Image.open(r['path']) as im:
            im = im.convert('RGB')
            if 'box' in r:
                im = im.crop(tuple(r['box']))
            return self.tf(im), int(r['label']), i

def load_splits(domain):
    if domain == 'plant':
        R = Path('/root/plantdoc_experiment'); A = R / 'artifacts'; S = R / 'data/PlantVillage/raw/color'; D = R / 'data/PlantDoc-Original'
        with gzip.open(A / 'config.json.gz', 'rt', encoding='utf8') as f:
            old = json.load(f)
        classes = [x['plantdoc_folder'] for x in old['class_order']]
        rows = []
        for i, c in enumerate(old['class_order']):
            folder = S / c['plantvillage_folder']
            files = sorted(x for x in folder.rglob('*') if x.is_file() and x.suffix.lower() in ('.jpg', '.jpeg', '.png'))
            rows.extend([{'path': str(x), 'label': i} for x in files])
        tr, rest = train_test_split(rows, test_size=.3, stratify=[r['label'] for r in rows], random_state=42)
        va, te = train_test_split(rest, test_size=.5, stratify=[r['label'] for r in rest], random_state=42)
        with gzip.open(A / 'leaves.json.gz', 'rt', encoding='utf8') as f:
            t = json.load(f)
        target = [x for x in t if x['partition'] == 'test']
        for r in target:
            r['path'] = str(D / r['image_path'].split('/PlantDoc-Original/', 1)[1])
            r['label'] = r['class_index']
        return classes, {'train': tr, 'val': va, 'target': target}
    elif domain == 'office':
        P = Path('/root/cross_scene_20261120/runs/office_product_to_real')
        audit = json.loads((P / 'audit.json').read_text())
        classes = audit['classes']
        splits = {k: json.loads((P / (f + '.json')).read_text()) for k, f in [('train', 'train'), ('val', 'val'), ('target', 'test')]}
        return classes, splits
    else:
        V = Path('/root/cross_scene_20261120/visda2017'); P = V / 'runs/source_only_30'
        config = json.loads((P / 'config.json').read_text())
        classes = config['classes']
        old = json.loads((P / 'splits.json').read_text())
        splits = {k: old[f] for k, f in [('train', 'source_train'), ('val', 'source_val'), ('target', 'target_test')]}
        for items in splits.values():
            for row in items:
                row['path'] = str(V / 'dataset' / row['path'])
        return classes, splits

def summarize(z, y, num_classes, T=1.0):
    v = z.astype(np.float64) / T
    lp = v - logsumexp(v, axis=1, keepdims=True)
    prob = np.exp(lp)
    pred = prob.argmax(1)
    cf = prob.max(1)
    correct = pred == y
    ece = 0.0
    for i in range(10):
        sel = (cf >= i / 10) & ((cf < (i + 1) / 10) if i < 9 else (cf <= 1))
        if sel.any():
            ece += float(sel.mean() * abs(float(correct[sel].mean()) - float(cf[sel].mean())))
    wrong = ~correct
    high = cf >= 0.9
    return {
        'n': int(len(y)),
        'correct': int(correct.sum()),
        'accuracy': float(correct.mean()),
        'macro_f1': float(f1_score(y, pred, labels=list(range(num_classes)), average='macro', zero_division=0)),
        'mean_confidence': float(cf.mean()),
        'ece_10bins': float(ece),
        'nll': float(-lp[np.arange(len(y)), y].mean()),
        'brier': float(((prob - np.eye(num_classes)[y]) ** 2).sum(1).mean()),
        'wrong_count': int(wrong.sum()),
        'wrong_confidence_ge_90': int((wrong & high).sum()),
        'high_conf_count': int(high.sum()),
        'high_conf_accuracy': float(correct[high].mean()) if high.any() else None,
    }

def fit_temp(zv, yv, low=0.02, high=50.0):
    def obj(T):
        v = zv.astype(np.float64) / T
        lp = v - logsumexp(v, axis=1, keepdims=True)
        return float(-lp[np.arange(len(yv)), yv].mean())
    res = minimize_scalar(obj, bounds=(low, high), method='bounded')
    return float(res.x)

results = {'clamped_ts_on_label_smoothing': {}, 'combined_improvements': {}}

# Part 1: Clamped TS (T >= 1) on existing label_smoothing logits.npz
for domain in ('plant', 'office', 'visda'):
    npz_path = Path(f'/root/cross_scene_20261120/calibration_three_domains/{domain}/label_smoothing/logits.npz')
    d = np.load(npz_path)
    zv = d['source_validation_logits'] if 'source_validation_logits' in d else d['val_logits']
    yv = d['source_validation_labels'] if 'source_validation_labels' in d else d['val_labels']
    zt, yt = d['target_logits'], d['target_labels']
    nc = zv.shape[1]
    t_unconstrained = fit_temp(zv, yv, 0.02, 50.0)
    t_clamped = fit_temp(zv, yv, 1.0, 50.0)
    results['clamped_ts_on_label_smoothing'][domain] = {
        'T_unconstrained': t_unconstrained,
        'T_clamped': t_clamped,
        'uncalibrated': summarize(zt, yt, nc, 1.0),
        'ts_unconstrained': summarize(zt, yt, nc, t_unconstrained),
        'ts_clamped': summarize(zt, yt, nc, t_clamped),
    }
    print('PART1', domain, 'T_unc=', round(t_unconstrained, 4), 'T_clamp=', round(t_clamped, 4), flush=True)

# Part 2: Combine best base models (ls_entropy & augmix) with Clamped TS (T>=1) and DAC
for domain in ('plant', 'office', 'visda'):
    classes, splits = load_splits(domain)
    num_classes = len(classes)
    workers = 6 if domain == 'visda' else 4
    def make_loader(rows):
        return DataLoader(Images(rows, ev), batch_size=32, shuffle=False, num_workers=workers, pin_memory=True, persistent_workers=(domain == 'visda'))
    ref = random.Random(42).sample(splits['train'], min(20000, len(splits['train'])))

    for base_method in ('ls_entropy', 'augmix'):
        t0 = time.time()
        ckpt_path = Path(f'/root/cross_scene_20261120/dac_augmix/{domain}/{base_method}/best.pt')
        net = models.resnet50(weights=None)
        net.fc = nn.Linear(net.fc.in_features, num_classes)
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        net.load_state_dict(ckpt['model'])
        net.cuda(); net.eval()

        features = {}; hooks = []
        names = ['layer1', 'layer2', 'layer3', 'layer4']
        for name in names:
            def hook(module, inputs, output, name=name):
                features[name] = torch.nn.functional.adaptive_avg_pool2d(output, 1).flatten(1).detach().cpu().numpy()
            hooks.append(getattr(net, name).register_forward_hook(hook))

        def extract(rows):
            allf = {name: [] for name in names}; zs = []; ys = []
            for x, y, _ in make_loader(rows):
                with torch.inference_mode():
                    z = net(x.cuda(non_blocking=True))
                zs.append(z.cpu().numpy()); ys.append(y.numpy())
                for name in names:
                    allf[name].append(features[name])
            return {name: np.concatenate(v).astype('float32') for name, v in allf.items()}, np.concatenate(zs), np.concatenate(ys)

        trainf, _, _ = extract(ref)
        valf, zv, yv = extract(splits['val'])
        targetf, zt, yt = extract(splits['target'])
        for h in hooks:
            h.remove()

        vs = []; ts = []
        for name in names:
            def normalize(x):
                return np.ascontiguousarray(x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-10), dtype='float32')
            a = normalize(trainf[name]); v = normalize(valf[name]); t = normalize(targetf[name])
            dim = a.shape[1]
            nlist = 64 if len(a) >= 2500 else 1
            if nlist > 1:
                index = faiss.IndexIVFFlat(faiss.IndexFlatL2(dim), dim, nlist, faiss.METRIC_L2)
                index.train(a); index.nprobe = 16; index.add(a)
            else:
                index = faiss.IndexFlatL2(dim); index.add(a)
            vs.append(index.search(v, 50)[0][:, -1])
            ts.append(index.search(t, 50)[0][:, -1])

        cal = DAC(ood_values_num=4)
        cal.optimize(zv, np.eye(num_classes)[yv], vs, loss='ce')
        zcal = cal.calibrate_before_softmax(zt, ts)
        vcal = cal.calibrate_before_softmax(zv, vs)
        assert np.array_equal(zcal.argmax(1), zt.argmax(1))

        t_unc = fit_temp(zv, yv, 0.02, 50.0)
        t_clamp = fit_temp(zv, yv, 1.0, 50.0)

        key = f'{domain}_{base_method}'
        results['combined_improvements'][key] = {
            'domain': domain,
            'base_method': base_method,
            'best_epoch': ckpt.get('epoch'),
            'dac_weights': cal.w.tolist(),
            'T_unconstrained': t_unc,
            'T_clamped': t_clamp,
            'source_val_base': summarize(zv, yv, num_classes, 1.0),
            'source_val_dac': summarize(vcal, yv, num_classes, 1.0),
            'target_base': summarize(zt, yt, num_classes, 1.0),
            'target_ts_unconstrained': summarize(zt, yt, num_classes, t_unc),
            'target_ts_clamped': summarize(zt, yt, num_classes, t_clamp),
            'target_dac': summarize(zcal, yt, num_classes, 1.0),
            'seconds': round(time.time() - t0, 1),
        }
        r = results['combined_improvements'][key]
        print(
            f"DONE {key}: base_acc={r['target_base']['accuracy']*100:.2f}% "
            f"base_ece={r['target_base']['ece_10bins']*100:.2f}% | "
            f"T_unc={t_unc:.3f} ece={r['target_ts_unconstrained']['ece_10bins']*100:.2f}% | "
            f"T_clamp={t_clamp:.3f} ece={r['target_ts_clamped']['ece_10bins']*100:.2f}% | "
            f"DAC ece={r['target_dac']['ece_10bins']*100:.2f}% ge90={r['target_dac']['wrong_confidence_ge_90']}/{r['target_dac']['high_conf_count']} "
            f"({r['seconds']}s)",
            flush=True
        )

(OUT_ROOT / 'results.json').write_text(json.dumps(results, indent=2, ensure_ascii=False))
(OUT_ROOT / 'COMPLETE').write_text('complete\n')
print('ALL IMPROVEMENTS COMPLETE', flush=True)
