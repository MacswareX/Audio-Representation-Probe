"""Check transferred summaries; never fit models or replay FSD predictions."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


def require(condition, message):
    if not condition:
        raise ValueError(message)


def check(root: Path) -> dict:
    manifest = json.loads((root / 'manifest.json').read_text())
    for row in manifest['files']:
        path = root / row['path']
        require(path.resolve().is_relative_to(root.resolve()), 'Manifest path escapes root')
        data = path.read_bytes()
        require(len(data) == row['bytes'], f'Size mismatch: {row["path"]}')
        require(hashlib.sha256(data).hexdigest() == row['sha256'],
                f'Hash mismatch: {row["path"]}')

    protocol = json.loads((root / 'protocol_extract.json').read_text())
    require(protocol['counts'] == {'train': 36796, 'val': 4170, 'eval': 10231},
            'Unexpected split sizes')
    require(protocol['classes'] == 200 and protocol['final_models'] == 36,
            'Unexpected protocol coverage')
    require(len(protocol['representations']) == 9, 'Expected nine representations')
    with (root / 'results_reported.csv').open(newline='') as f:
        rows = list(csv.DictReader(f))
    expected = {(r.replace(' ', '_').lower(), p, a)
                for r in ['logmel', 'E1 z', 'E1 q2', 'E1 q4', 'CNN6',
                          'EnCodec z', 'EnCodec q2', 'EnCodec q4', 'BEATs']
                for p in ['G', 'T4'] for a in ['full', 'dedup']}
    indexed = {(r['representation'], r['pool'], r['train_arm']): r for r in rows}
    require(len(rows) == 36 and set(indexed) == expected, 'Incomplete/duplicate table')
    for row in rows:
        value = float(row['macro_ap_rounded'])
        require(math.isfinite(value) and 0 <= value <= 1, 'Invalid reported AP')

    detailed = {}
    for stem in ['e1_q2_G_full', 'e1_q4_G_full', 'encodec_q2_G_full', 'encodec_q4_G_full']:
        data = json.loads((root / 'metrics' / (stem + '_METRICS.json')).read_text())
        full = data['official_full']
        values = full['per_class_ap']
        require(len(values) == 200 and full['defined_classes'] == 200,
                f'Incomplete classes: {stem}')
        require(all(math.isfinite(v) and 0 <= v <= 1 for v in values),
                f'Invalid class AP: {stem}')
        mean = math.fsum(values) / 200
        require(abs(mean - full['macro_200_ap']) < 1e-14, f'Mean mismatch: {stem}')
        positive, negative = full['positive_support'], full['negative_support']
        require(len(positive) == len(negative) == 200, f'Support shape: {stem}')
        require(all(p + n == 10231 and p > 0 and n > 0 for p, n in zip(positive, negative)),
                f'Official denominator mismatch: {stem}')
        rep = stem.removesuffix('_G_full')
        rounded = float(indexed[(rep, 'G', 'full')]['macro_ap_rounded'])
        require(abs(mean - rounded) <= 0.000000500001, f'Rounded-table mismatch: {stem}')
        selection = json.loads((root / 'metrics' / (stem + '_SELECTION.json')).read_text())
        require(selection['lambda_'] == 0.0001 and selection['val_refit'] is False
                and selection['all_validation_artifacts_complete'] is True,
                f'Selection metadata mismatch: {stem}')
        bootstrap = json.loads((root / 'metrics' / (stem + '_BOOTSTRAP.json')).read_text())
        require(bootstrap['identity']['draws'] == 2000
                and bootstrap['identity']['seed'] == 17170911
                and bootstrap['groups'] == 2297 and bootstrap['refits'] == 0,
                f'Bootstrap metadata mismatch: {stem}')
        detailed[stem] = mean

    paired = json.loads((root / 'paired_contrasts.json').read_text())
    require(paired['draws'] == 2000 and paired['seed'] == 17170911
            and paired['conditional_on_saved_models'] is True, 'Paired metadata mismatch')
    require(len(paired['contrasts']) == 76, 'Expected 76 paired interval records')
    conditions = {f'{r}_{p}_{a}' for r, p, a in expected}
    for contrast in paired['contrasts']:
        lo, hi = contrast['paired_percentile_95']
        require(contrast['a'] in conditions and contrast['b'] in conditions,
                'Unknown contrast condition')
        require(math.isfinite(lo) and math.isfinite(hi) and -1 <= lo <= hi <= 1,
                'Invalid interval')
        require(contrast['defined_draws'] + contrast['undefined_draws'] == 2000,
                'Invalid draw accounting')
    return {'status': 'PASS', 'hashed_files': len(manifest['files']),
            'historically_reported_conditions': len(rows),
            'detailed_saved_macro_means_checked': detailed,
            'paired_interval_summaries': len(paired['contrasts']),
            'predictions_replayed': False, 'models_trained': False,
            'scope': 'File integrity and saved-summary consistency only'}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--evidence-dir', type=Path,
                    default=Path(__file__).resolve().parents[1] / 'evidence/fsd50k')
    args = ap.parse_args()
    print(json.dumps(check(args.evidence_dir), indent=2))


if __name__ == '__main__':
    main()
