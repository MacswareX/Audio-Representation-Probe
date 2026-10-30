"""Fit/select a frozen-feature ridge head, or score its held-out recordings.

This portable entry uses existing NPY arrays and a recording-order manifest.
It does not extract audio features, download models, or invoke the historical
Windows runtime. It is an adaptation, not a rerun of the archived 36 conditions.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.fsd50k import Statistics, coefficient, predict, MultilabelAP, LAMBDAS


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def inputs(features, labels, manifest):
    rows = [json.loads(line) for line in Path(manifest).read_text(encoding='utf-8-sig').splitlines()]
    x = np.load(features, mmap_mode='r', allow_pickle=False)
    y = np.load(labels, mmap_mode='r', allow_pickle=False)
    if x.ndim != 2 or y.shape != (len(x), 200) or len(rows) != len(x):
        raise ValueError('Expected aligned X[N,d], Y[N,200], and N manifest rows')
    if x.shape[1] == 0 or not np.issubdtype(x.dtype, np.number) or np.iscomplexobj(x):
        raise ValueError('Features must be a finite numeric matrix with at least one column')
    for j in range(0, len(x), 128):
        if not np.isfinite(x[j:j + 128]).all():
            raise ValueError('Features must be finite')
    if not np.isin(y, [0, 1]).all():
        raise ValueError('Labels must be binary')
    if [r['index'] for r in rows] != list(range(len(rows))) or len({str(r['id']) for r in rows}) != len(rows):
        raise ValueError('Manifest indices must match array order and recording IDs must be unique')
    if any(r['official_split'] not in ('train', 'val', 'eval') for r in rows):
        raise ValueError('Unknown split in manifest')
    ix = {s: np.array([i for i, r in enumerate(rows) if r['official_split'] == s], dtype=np.int64)
          for s in ('train', 'val', 'eval')}
    if not all(len(v) for v in ix.values()):
        raise ValueError('All three recording splits must be nonempty')
    # Optional per-row label indices bind a historical manifest to the labels.
    for i, row in enumerate(rows):
        if 'label_indices' in row and sorted(row['label_indices']) != np.flatnonzero(y[i]).tolist():
            raise ValueError(f'Manifest/label mismatch at index {i}')
    identity = {k: sha(p) for k, p in [('features', features), ('labels', labels), ('manifest', manifest)]}
    return x, y, rows, ix, identity


def score_rows(x, indices, factors, w, batch_rows=128):
    scores = np.empty((len(indices), w.shape[1]), dtype=np.float64)
    for j in range(0, len(indices), batch_rows):
        scores[j:j + batch_rows] = predict(x[indices[j:j + batch_rows]], factors, w)
    return scores


def fit(args):
    if args.out.exists():
        raise ValueError('Output exists; select a new directory')
    x, y, rows, ix, identity = inputs(args.features, args.labels, args.manifest)
    train = ix['train']
    if args.arm == 'dedup':
        if any('dedup_train_keep' not in rows[i] or not isinstance(rows[i]['dedup_train_keep'], bool) for i in train):
            raise ValueError('Dedup arm requires a boolean dedup_train_keep on every training row')
        train = np.array([i for i in train if rows[i]['dedup_train_keep']], dtype=np.int64)
    if len(train) < 2:
        raise ValueError('At least two training recordings are required')
    stats = Statistics(x.shape[1], 200)
    for j in range(0, len(train), args.batch_rows):
        idx = train[j:j + args.batch_rows]
        stats.add(x[idx], y[idx])
    factors = stats.factors()
    candidates = []
    for lam in LAMBDAS:
        w = coefficient(factors, lam)
        scores = score_rows(x, ix['val'], factors, w, args.batch_rows)
        metrics = MultilabelAP(y[ix['val']], scores).report(scores)
        if metrics['macro_200_ap'] is None:
            raise ValueError('Validation must have positive and negative examples for all 200 classes')
        candidates.append({'lambda_': float(lam), 'alpha': float(len(train) * lam), 'metric': metrics})
    selected = max(range(len(LAMBDAS)), key=lambda j: (candidates[j]['metric']['macro_200_ap'], LAMBDAS[j]))
    args.out.mkdir(parents=True)
    w = coefficient(factors, LAMBDAS[selected])
    model_path = args.out / 'selected_model.npz'
    np.savez(model_path, mean=factors['mean'], std=factors['std'], target_mean=factors['target_mean'],
             coefficients=w, lambda_=LAMBDAS[selected], n=len(train))
    report = {'kind': 'portable_frozen_feature_readout', 'input_identity': identity,
              'arm': args.arm, 'split_counts': {k: len(v) for k, v in ix.items()},
              'training_rows_used': len(train), 'feature_dimension': x.shape[1],
              'official_split_counts_match': {k: len(v) for k, v in ix.items()} == {'train': 36796, 'val': 4170, 'eval': 10231},
              'candidates': candidates, 'selected_index': selected, 'lambda_': float(LAMBDAS[selected]),
              'val_refit': False, 'evaluation_scores_computed': False,
              'model_sha256': sha(model_path),
              'scope': 'One supplied frozen-feature condition; not the historical all-36 selection gate'}
    write_json(args.out / 'selection.json', report)
    return {'output': str(args.out), 'lambda_': report['lambda_'], 'evaluation_scores_computed': False}


def evaluate(args):
    if args.out.exists():
        raise ValueError('Output exists; select a new directory')
    report = json.loads((args.model_dir / 'selection.json').read_text())
    model_path = args.model_dir / 'selected_model.npz'
    if sha(model_path) != report['model_sha256']:
        raise ValueError('Selected model bytes changed')
    x, y, rows, ix, identity = inputs(args.features, args.labels, args.manifest)
    if identity != report['input_identity']:
        raise ValueError('Feature, label or manifest identity differs from fitted inputs')
    with np.load(model_path, allow_pickle=False) as model:
        f = {k: model[k] for k in ('mean', 'std', 'target_mean')}
        scores = score_rows(x, ix['eval'], f, model['coefficients'])
    metrics = MultilabelAP(y[ix['eval']], scores).report(scores)
    args.out.mkdir(parents=True)
    np.savez(args.out / 'scores.npz', scores=scores, indices=ix['eval'])
    write_json(args.out / 'metrics.json', {'official_full': metrics, 'recordings': len(scores),
               'input_identity': identity, 'selected_model_sha256': report['model_sha256'],
               'scope': 'Scores from supplied arrays; no waveform-to-feature verification or retraining'})
    return {'output': str(args.out), 'recordings': len(scores), 'macro_200_ap': metrics['macro_200_ap']}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest='command', required=True)
    for name in ('fit', 'evaluate'):
        p = sub.add_parser(name)
        p.add_argument('--features', required=True, type=Path)
        p.add_argument('--labels', required=True, type=Path)
        p.add_argument('--manifest', required=True, type=Path)
        p.add_argument('--out', required=True, type=Path)
        if name == 'fit':
            p.add_argument('--arm', choices=('full', 'dedup'), default='full')
            p.add_argument('--batch-rows', type=int, default=128)
        else:
            p.add_argument('--model-dir', required=True, type=Path)
    args = ap.parse_args()
    if args.command == 'fit' and args.batch_rows <= 0:
        ap.error('--batch-rows must be positive')
    try:
        print(json.dumps(fit(args) if args.command == 'fit' else evaluate(args), indent=2))
    except (OSError, ValueError, KeyError) as exc:
        ap.exit(1, f'Readout not completed: {exc}\n')


if __name__ == '__main__':
    main()
