"""Recompute manuscript results from saved predictions, without loading a model."""
import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE / 'vendor'))
import numpy as np

DATA = BASE / 'data'
ANALYSIS = BASE / 'analysis'
DOMAINS = ['plant', 'office', 'visda']
REPEAT_METHODS = ['baseline', 'augmix', 'ls_entropy', 'alis']
SEEDS = [42, 7, 2026]
EXCLUDED = {
    'Apple rust leaf': 'Rust label does not identify cedar apple rust pathogen.',
    'Bell_pepper leaf spot': 'Leaf spot label does not establish bacterial etiology.',
    'Corn leaf blight': 'Generic blight label does not establish northern leaf blight.',
    'Corn rust leaf': 'Generic rust label does not establish common rust.',
    'Tomato leaf yellow virus': 'Yellow virus label does not establish yellow leaf curl virus.',
}


def read_json(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))


def write_json(p, value):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def write_csv(p, rows):
    if not rows:
        return
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def probabilities(z, temperature=1.0):
    z = np.asarray(z, dtype=np.float64) / temperature
    assert np.isfinite(z).all() and np.all(np.asarray(temperature) > 0)
    shifted = z - z.max(axis=1, keepdims=True)
    logp = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))
    return np.exp(logp), logp


def metric_values(prob, logp, y, n_bins=10):
    y = np.asarray(y, dtype=np.int64)
    n, k = prob.shape
    assert n == len(y) and n > 0
    pred, confidence = prob.argmax(axis=1), prob.max(axis=1)
    correct = pred == y
    bin_index = np.minimum((confidence * n_bins).astype(int), n_bins - 1)
    bins, ece = [], 0.0
    for i in range(n_bins):
        mask = bin_index == i
        count = int(mask.sum())
        acc = float(correct[mask].mean()) if count else None
        conf = float(confidence[mask].mean()) if count else None
        if count:
            ece += count / n * abs(acc - conf)
        bins.append({'bin': i, 'left': i/n_bins, 'right': (i+1)/n_bins,
                     'n': count, 'accuracy': acc, 'confidence': conf})
    high = confidence >= 0.9
    order = np.argsort(-confidence, kind='stable')
    count20 = int(np.ceil(n * 0.2))
    risks = np.cumsum(~correct[order]) / np.arange(1, n + 1)
    per_class = [float(correct[y == c].mean()) for c in np.unique(y)]
    out = {'n': n, 'classes_observed': len(per_class), 'correct': int(correct.sum()),
           'accuracy': float(correct.mean()), 'mean_class_accuracy_observed': float(np.mean(per_class)),
           'mean_confidence': float(confidence.mean()), 'ece': float(ece),
           'nll': float(-logp[np.arange(n), y].mean()),
           'brier': float((np.square(prob).sum(axis=1) - 2*prob[np.arange(n), y] + 1).mean()),
           'high_conf_count': int(high.sum()), 'high_conf_wrong': int((~correct & high).sum()),
           'high_conf_coverage': float(high.mean()),
           'high_conf_risk': float((~correct[high]).mean()) if high.any() else None,
           'top20_count': count20, 'top20_wrong': int((~correct[order[:count20]]).sum()),
           'top20_risk': float((~correct[order[:count20]]).mean()),
           'aurc': float(risks.mean()), 'bins': bins}
    return out


@dataclass
class Run:
    domain: str
    method: str
    seed: int
    path: Path
    z: np.ndarray
    y: np.ndarray
    temperature: float
    result: dict

    def calc(self, mask=None, n_bins=10):
        p, lp = probabilities(self.z, self.temperature)
        if mask is not None:
            p, lp, y = p[mask], lp[mask], self.y[mask]
        else:
            y = self.y
        return metric_values(p, lp, y, n_bins)


def load_run(domain, method, seed, directory):
    p = directory/'logits.npz'
    if not p.exists() or not (directory/'COMPLETE').exists():
        return None
    r = read_json(directory/'result.json')
    with np.load(p, allow_pickle=False) as a:
        key = 'target_calibrated_logits' if method == 'dac' else 'target_logits'
        z = a[key].copy()
        y = a['target_labels'].astype(np.int64)
        t = float(a['temperature']) if method == 'alis' else 1.0
        if method == 'temperature':
            t = float(r['temperature'])
    return Run(domain, method, seed, p, z, y, t, r)


def load_runs():
    runs = []
    for d in DOMAINS:
        cal = DATA/'calibration_three_domains'/d
        baseline = load_run(d, 'baseline', 42, cal/'temperature')
        if baseline:
            runs.append(baseline)
        for m in ['temperature', 'label_smoothing', 'confidence_penalty']:
            run = load_run(d, m, 42, cal/m)
            if run:
                runs.append(run)
        for m in ['augmix', 'dac', 'ls_entropy', 'alis']:
            run = load_run(d, m, 42, DATA/'dac_augmix'/d/m)
            if run:
                runs.append(run)
        ls = next((x for x in runs if x.domain == d and x.method == 'label_smoothing'), None)
        combo_path = cal/'smoothing_plus_temperature'/'result.json'
        if ls and combo_path.exists():
            r = read_json(combo_path)
            runs.append(Run(d, 'smoothing_plus_temperature', 42, ls.path,
                            ls.z, ls.y, float(r['temperature']), r))
        for seed in [7, 2026]:
            for m in REPEAT_METHODS:
                run = load_run(d, m, seed, DATA/'dac_augmix/seed_repeats'/d/m/f'seed_{seed}')
                if run:
                    runs.append(run)
    for d in DOMAINS:
        subset = [x for x in runs if x.domain == d]
        if subset:
            for run in subset:
                assert np.array_equal(run.y, subset[0].y), (d, run.method, run.seed, 'target label order')
    return runs


def verify_saved_metrics(run, measured):
    if run.method == 'baseline' and 'target_baseline' in run.result:
        expected = run.result['target_baseline']
    elif run.method == 'temperature':
        expected = run.result['target_temperature']
    elif run.method == 'smoothing_plus_temperature':
        expected = run.result['test']
    else:
        expected = run.result['target']
    assert measured['n'] == expected['n']
    for a, b in [('accuracy', 'accuracy'), ('ece', 'ece_10bins' if 'ece_10bins' in expected else 'ece'), ('nll', 'nll')]:
        assert abs(measured[a] - expected[b]) < 5e-7, (run.domain, run.method, run.seed, a, measured[a], expected[b])


def grouped_bootstrap(run, groups, mask, draws):
    p, lp = probabilities(run.z, run.temperature)
    confidence, correct = p.max(axis=1), p.argmax(axis=1) == run.y
    gi, count = groups, draws.shape[1]
    weights = mask.astype(float)
    sizes = np.bincount(gi, weights=weights, minlength=count)
    corrects = np.bincount(gi, weights=correct * weights, minlength=count)
    total = draws @ sizes
    accuracies = (draws @ corrects) / total
    bin_index = np.minimum((confidence * 10).astype(int), 9)
    ece = np.zeros(len(draws))
    for b in range(10):
        bw = weights * (bin_index == b)
        residual = np.bincount(gi, weights=(correct-confidence)*bw, minlength=count)
        ece += np.abs(draws @ residual) / total
    return accuracies, ece


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--require-three-seeds', action='store_true')
    ap.add_argument('--bootstrap', type=int, default=2000)
    args = ap.parse_args()
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    runs = load_runs()
    if not runs:
        raise RuntimeError('No complete prediction files available.')
    results, flat, bins_csv, curve_csv, bin_sensitivity = [], [], [], [], []
    metrics_keys = ['n','classes_observed','correct','accuracy','mean_class_accuracy_observed','mean_confidence','ece','nll','brier','high_conf_count','high_conf_wrong','high_conf_coverage','high_conf_risk','top20_count','top20_wrong','top20_risk','aurc']
    for run in runs:
        measured = run.calc()
        verify_saved_metrics(run, measured)
        row = {'domain': run.domain, 'method': run.method, 'seed': run.seed,
               **{k: measured[k] for k in metrics_keys}}
        flat.append(row)
        results.append({**row, 'temperature': run.temperature, 'bins': measured['bins'],
                        'prediction_source': str(run.path)})
        for b in measured['bins']:
            bins_csv.append({'domain': run.domain, 'method': run.method, 'seed': run.seed, **b})
        for n_bins in [5, 10, 15, 20]:
            bin_sensitivity.append({'domain':run.domain,'method':run.method,'seed':run.seed,
                                    'n_bins':n_bins,'ece':run.calc(n_bins=n_bins)['ece']})
        p, _ = probabilities(run.z, run.temperature)
        order = np.argsort(-p.max(axis=1), kind='stable')
        errors = np.cumsum(p.argmax(axis=1)[order] != run.y[order])
        n = len(run.y)
        for coverage in np.arange(1, 101)/100:
            take = int(np.ceil(n * coverage))
            curve_csv.append({'domain':run.domain,'method':run.method,'seed':run.seed,
                              'requested_coverage':float(coverage),'n_retained':take,
                              'actual_coverage':take/n,'wrong':int(errors[take-1]),
                              'risk':float(errors[take-1]/take)})
    write_csv(ANALYSIS/'all_run_metrics.csv', flat)
    write_csv(ANALYSIS/'reliability_bins.csv', bins_csv)
    write_csv(ANALYSIS/'ece_bin_sensitivity.csv', bin_sensitivity)
    write_csv(ANALYSIS/'risk_coverage.csv', curve_csv)
    write_json(ANALYSIS/'all_run_metrics.json', results)
    means, missing = [], []
    for d in DOMAINS:
        for m in REPEAT_METHODS:
            rows = [x for x in flat if x['domain']==d and x['method']==m and x['seed'] in SEEDS]
            seeds = sorted(x['seed'] for x in rows)
            if sorted(SEEDS) != seeds:
                missing.append({'domain':d,'method':m,'available_seeds':seeds})
                continue
            out = {'domain':d,'method':m,'n_seeds':3,'seeds':'42,7,2026'}
            for key in ['accuracy','ece','nll','brier','aurc']:
                values = np.array([x[key] for x in rows])
                out[key+'_mean'] = float(values.mean())
                out[key+'_sd'] = float(values.std(ddof=1))
            means.append(out)
    write_csv(ANALYSIS/'three_seed_summary.csv', means)
    write_json(ANALYSIS/'seed_completion.json', {'complete_comparisons':len(means),'total_comparisons':12,'missing':missing})
    metadata = read_json(DATA/'metadata/metadata_samples.json')
    class_order = metadata['plant_class_order']
    names = [x['plantdoc_folder'] for x in class_order]
    excluded_indices = [names.index(x) for x in EXCLUDED]
    plant_rows = metadata['plant_target_manifest']
    py = np.array([r['label'] for r in plant_rows])
    assert all(np.array_equal(x.y, py) for x in runs if x.domain=='plant')
    strict = ~np.isin(py, excluded_indices)
    identities = [r['image_path'] for r in plant_rows]
    group_names, gi = np.unique(identities, return_inverse=True)
    assert len(group_names)==236 and len(py)==452
    mapping = []
    for i, c in enumerate(class_order):
        mapping.append({'class_index':i,'plantdoc_class':c['plantdoc_folder'],
                        'plantvillage_class':c['plantvillage_folder'],
                        'retained_in_23':i not in excluded_indices,'target_boxes':int((py==i).sum()),
                        'exclusion_reason':EXCLUDED.get(c['plantdoc_folder'], '')})
    write_csv(ANALYSIS/'plant_class_mapping_and_exclusions.csv', mapping)
    write_json(ANALYSIS/'plant_sensitivity_protocol.json',{
        'full_candidate_classes':28,'full_observed_classes':len(np.unique(py)),
        'full_boxes':len(py),'full_parent_images':len(group_names),
        'retained_mapping_classes':23,'retained_observed_classes':len(np.unique(py[strict])),
        'retained_boxes':int(strict.sum()),'retained_parent_images':len(np.unique(np.array(identities)[strict])),
        'excluded_indices':excluded_indices,'excluded_names':list(EXCLUDED),
        'filter_rule':'Filter on ground-truth labels only. Preserve original 28 output probabilities and all predictions; predictions into excluded classes remain incorrect. No retraining, class-column removal, renormalization, or temperature refit.',
        'interpretation':'Names are more specific under this mapping; pathogen identities have not been independently established. Different subsets have different class/sample composition.',
        'bootstrap_replicates':args.bootstrap,'bootstrap_seed':20261123,
        'bootstrap_unit':'Original parent image; sample 236 images with replacement, include all associated boxes, then apply fixed subset filter.',
        'interval':'Paired parent-image cluster bootstrap, percentile 95%; conditional on trained model and observed dataset, not a training-seed interval.'})
    rng = np.random.default_rng(20261123)
    draws = rng.multinomial(len(group_names), np.repeat(1/len(group_names),len(group_names)), size=args.bootstrap).astype(float)
    sensitivities, intervals, boot = [], [], {}
    for run in [x for x in runs if x.domain=='plant']:
        for label, mask in [('28_candidate',np.ones(len(py),dtype=bool)),('23_clearer',strict)]:
            measured = run.calc(mask)
            sensitivities.append({'method':run.method,'seed':run.seed,'subset':label,
                                  **{k:measured[k] for k in metrics_keys}})
            ba, be = grouped_bootstrap(run, gi, mask, draws)
            boot[(run.method,run.seed,label)] = (ba,be)
            intervals.append({'method':run.method,'seed':run.seed,'subset':label,'n':measured['n'],
                              'accuracy':measured['accuracy'],'accuracy_ci_low':float(np.quantile(ba,.025)),
                              'accuracy_ci_high':float(np.quantile(ba,.975)),
                              'ece':measured['ece'],'ece_ci_low':float(np.quantile(be,.025)),
                              'ece_ci_high':float(np.quantile(be,.975))})
    paired = []
    for run in [x for x in runs if x.domain=='plant' and x.method!='baseline']:
        for subset in ['28_candidate','23_clearer']:
            key=(run.method,run.seed,subset)
            ref=('baseline',run.seed,subset)
            if ref not in boot:
                continue
            for index,metric in enumerate(['accuracy','ece']):
                diff=boot[key][index]-boot[ref][index]
                values=[x for x in sensitivities if x['method']==run.method and x['seed']==run.seed and x['subset']==subset][0]
                base=[x for x in sensitivities if x['method']=='baseline' and x['seed']==run.seed and x['subset']==subset][0]
                paired.append({'method':run.method,'seed':run.seed,'subset':subset,'metric':metric,
                               'difference':values[metric]-base[metric],
                               'ci_low':float(np.quantile(diff,.025)),'ci_high':float(np.quantile(diff,.975))})
    write_csv(ANALYSIS/'plant_23_vs_28.csv', sensitivities)
    write_csv(ANALYSIS/'plant_cluster_bootstrap.csv', intervals)
    write_csv(ANALYSIS/'plant_paired_differences.csv', paired)
    write_json(ANALYSIS/'analysis_protocol.json',{
        'units':'Accuracy, ECE, risk and coverage saved as proportions; multiply by 100 for percentages. NLL and Brier are unitless scores, lower is better.',
        'binning':'Equal-width bins [i/M,(i+1)/M), with last bin including confidence 1. Empty bins contribute zero and are retained as empty.',
        'ece_bin_counts':[5,10,15,20],'three_seed_sd':'Sample standard deviation, ddof=1; compute from unrounded per-run metrics.',
        'top20':'ceil(n*0.20); stable descending confidence order; each model selects its own subset.',
        'risk_coverage':'All ranking cutoffs used for AURC; points at 1%-100% coverage exported for plotting; stable order resolves exact ties.',
        'data_reuse':'Previously observed target sets, exploratory results; no new independent test set.',
        'prediction_verification':'Recomputed accuracy, ECE and NLL must match saved run results within 5e-7.',
        'local_compute':'Only saved logits and metadata are analyzed. No model, GPU inference or retraining is used locally.'})
    print(json.dumps({'runs_checked':len(runs),'three_seed_comparisons':len(means),
                      'plant_23_boxes':int(strict.sum()),'plant_23_observed_classes':len(np.unique(py[strict])),
                      'missing':missing},ensure_ascii=False))
    if args.require_three_seeds and missing:
        raise RuntimeError('Three-seed manuscript gate not yet satisfied.')


if __name__=='__main__':
    main()
