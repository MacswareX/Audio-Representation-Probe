# FSD50K implementation: original source and a portable readout

## What was recovered

The October 30 engineering handoff contains the complete text source/configuration/
record scope from the original Windows work area, not the large binary assets.
All 1,445 manifested members were checked against their bytes here with no
hash mismatch. Its `MAIN_PACKET.json` has the same SHA-256 cited by the September
13 review. All 118 available text members bound by that packet match; its
36 bound bootstrap arrays were not transferred and were not revalidated.

This repository publishes selected relevant artifacts rather than that private
work area. It excludes correspondence, internal state/logs, raw audio, source
recordings, labels, feature/score matrices, checkpoints and vendor code.

## Resolve source versions before importing

Five source files differ from the initial `PROTOCOL_LOCK`: common runtime,
stage gate and feature/readout/evaluation drivers. The later source-override
and execution locks bind those five updated hashes. All 17 current locked
source files match the repaired execution lock. Numerical statistics and
pooling files match both locks.

[Source lineage](../evidence/fsd50k/source_lineage.json) records the input ZIP,
packet/lock/review hashes, all 17 source hashes and the five before/after pairs.
The byte-preserved files are in [source_snapshot](../experiments/fsd50k/source_snapshot/).
Their `.py.txt` suffix prevents accidental import/execution as a ready-to-run
pipeline: historical drivers require the original runtime, grants and assets.
These snapshots are readable provenance, not a portable extraction release.

The original author-acceptance fields remain false; the later AI-assisted review
separately records acceptance under its scope. We do not rewrite historical
flags or imply human peer review. Context30 remains incomplete.

## Portable code derived from those sources

| Public implementation | Original member | Adaptation |
|---|---|---|
| `src/fsd50k/statistics.py` | `fsd/statistics_core.py` | Replace runtime wildcard import by NumPy and small array-I/O helpers; numerical body retained |
| `src/fsd50k/pooling.py` | `fsd/time_coverage.py` | Replace runtime import; omit waveform I/O/resampling functions; partition/pooling body retained |
| `src/fsd50k/_io.py` | Helper functions formerly supplied by common runtime | Local array persistence, no private runtime state or resource sampling |
| `scripts/fsd50k_readout.py` | Integration around the extracted numerical core | New portable fit/select and held-out-score entry points on existing arrays |

This is a source-derived adaptation. It is not the original Windows native
suite running under a shim. The historical gate requires all 36 selections
before evaluation; the portable CLI operates on one supplied condition and
does not claim that all-36 gate. It does not download or instantiate encoders.

## Inputs and commands

Use Python 3.11+; the tested environment is Python 3.12.14 with the versions in
`requirements-fsd.txt`. The numerical core needs NumPy/SciPy; scikit-learn is
used for independent AP checks in tests. No PyTorch installation is needed for
this feature-to-score interface.

Inputs are existing arrays and an aligned manifest:

- `features.npy`: finite numeric `[N,d]` frozen features for one representation
  and pooling scheme, in manifest row order.
- `labels.npy`: binary `[N,200]` labels in the same row/class order.
- `manifest.jsonl`: one row per recording with contiguous `index`, unique `id`
  and `official_split` equal to `train`, `val` or `eval`. `dedup_train_keep`
  must be boolean on training rows for the dedup arm. If `label_indices` is
  present, it is checked against each label row.

Run from the repository root. Paths below stand for locally available existing
files; they are not downloads or files bundled with this repository.

```bash
python scripts/fsd50k_readout.py fit \
  --features /path/to/e1_q2_G.npy --labels /path/to/labels.npy \
  --manifest /path/to/manifest.jsonl --arm full --out runs/fsd50k/e1_q2_G_full

python scripts/fsd50k_readout.py evaluate \
  --features /path/to/e1_q2_G.npy --labels /path/to/labels.npy \
  --manifest /path/to/manifest.jsonl \
  --model-dir runs/fsd50k/e1_q2_G_full --out runs/fsd50k/e1_q2_G_full/evaluation
```

`fit` estimates normalization/target means from its training arm, selects one
of five lambdas on validation AP and saves the head without computing eval
scores. `evaluate` verifies input/model hashes, scores held-out rows and saves
scores/AP. Output directories must be new. Nonstandard recording counts are
reported honestly; matching the official sizes is not proof of data provenance.
The feature hash is bound to the selected head, so changing even an evaluation
feature requires a separately identified input rather than silent reuse.

No automatic bootstrap or threshold optimization occurs. The API exposes the
original paired group bootstrap for deliberate comparisons with fixed models.
Raw ridge scores are ranks, not calibrated class probabilities.

## Validation actually performed

```bash
python scripts/check_fsd_evidence.py
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 python -m unittest discover -s tests -v
```

Six constructed-data tests exercise:

- Streamed/removed-row covariance and ridge versus a direct linear solve,
  including constant and duplicate feature columns.
- Weighted/tied AP versus scikit-learn and undefined classes.
- Transitive uploader/content grouping and paired bootstrap draws.
- Unequal time-support pooling and original/resampled clock partition.
- Fit→save→held-out-score serialization, train-only normalization, dedup,
  changed-input rejection and output no-overwrite behavior.
- Larger-lambda selection on exact validation ties.

These are tests of the portable feature/readout implementation, not measurements
of FSD accuracy. Separately, the private manifest reconstructed the recorded
label and evaluation-group arrays with the original hashes, confirming 51,197
unique IDs, 106,158 windows, the official splits and 2,297 groups. No raw scores
were available to recompute historical AP or bootstrap intervals here.

## One concrete path through the code

For a single recording vector, `fit` has already saved training means/stds and
the selected W. `predict` computes `((x - mean) / std) @ W + target_mean`, giving
200 scores. `MultilabelAP` ranks that class's scores across recordings. Changing
one recording's score changes its rank; it does not create an event timestamp.

This gives an inspectable implementation for technical discussion without
claiming that the author independently developed the whole pipeline. An
upstream feature-extraction replay needs the original checkpoints, audio and
native dependencies and remains separate from these verified operations.
