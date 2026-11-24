import argparse
import hashlib
import json
import stat
import sys
from pathlib import Path, PurePosixPath

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, r'F:\desktop\毕设\.remote_access\plantdoc_ssh_libs')
import paramiko

REMOTE = '/root/cross_scene_20261120'
ROOTS = ['calibration_three_domains', 'dac_augmix',
         'visda2017/runs/source_only_30', 'runs/office_product_to_real',
         'target_domain_baselines_20261122']
SKIP = {'official_dac', 'paper_2025', '__pycache__', '.git', 'datasets',
        'interrupted_seed_7_20261123'}
NAMES = {'result.json', 'config.json', 'logits.npz', 'epochs.csv', 'COMPLETE',
         'splits.json', 'comparison.csv', 'queue.status', 'queue.history',
         'target_predictions.csv', 'target_probabilities.npz',
         'visda_train_test_examples_sources.json', 'office_samples_comparison_sources.json'}


def connect():
    c = paramiko.SSHClient()
    c.load_host_keys(r'F:\desktop\毕设\.remote_access\plantdoc_20261119_known_hosts')
    c.set_missing_host_key_policy(paramiko.RejectPolicy())
    c.connect('8.130.9.186', port=25039, username='root',
              key_filename=r'F:\desktop\毕设\.remote_access\plantdoc_20261105_ed25519',
              timeout=15, allow_agent=False, look_for_keys=False)
    return c


def walk(sftp, path):
    for entry in sftp.listdir_attr(path):
        p = path + '/' + entry.filename
        if stat.S_ISDIR(entry.st_mode):
            if entry.filename not in SKIP:
                yield from walk(sftp, p)
        else:
            yield p, entry


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--repeats-only', action='store_true')
    args = ap.parse_args()
    client = connect()
    sftp = client.open_sftp()
    paths = ['dac_augmix/seed_repeats'] if args.repeats_only else ROOTS
    downloaded, total = 0, 0
    records = []
    for root in paths:
        for remote, info in walk(sftp, REMOTE + '/' + root):
            name = PurePosixPath(remote).name
            if not (name in NAMES or name.endswith('.py')):
                continue
            if info.st_size > 200_000_000:
                continue
            rel = PurePosixPath(remote).relative_to(REMOTE)
            dest = BASE / 'data' / str(rel)
            dest.parent.mkdir(parents=True, exist_ok=True)
            valid = dest.exists() and dest.stat().st_size == info.st_size and int(dest.stat().st_mtime) == info.st_mtime
            if not valid:
                tmp = dest.with_suffix(dest.suffix + '.part')
                sftp.get(remote, str(tmp))
                if tmp.stat().st_size != info.st_size:
                    raise RuntimeError('Changed remote file: ' + remote)
                tmp.replace(dest)
                import os
                os.utime(dest, (info.st_mtime, info.st_mtime))
                downloaded += 1
                total += info.st_size
            records.append({'remote': remote, 'local': str(dest), 'bytes': info.st_size,
                            'mtime': info.st_mtime, 'sha256': hashlib.sha256(dest.read_bytes()).hexdigest()})
    sftp.close()
    client.close()
    (BASE / 'data' / ('fetch_repeats_manifest.json' if args.repeats_only else 'fetch_manifest.json')).write_text(
        json.dumps(records, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'downloaded_files': downloaded, 'downloaded_MB': round(total / 1e6, 2),
                      'total_files': len(records)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
