# Evidence status: October 30 source integration

Public base: `4d47b78608b0a95d1f92e6b0c5b2c5abe6c554c1`.
The original ESC-50 execution code is unchanged.

## Transferred files

- `evidence/fsd50k/historical_review.md`: unchanged September 13 completion
  review from the September 14 contact archive. Earlier numerical checks are
  reported there; they are not repeated by this renovation.
- `results.csv`: full-precision AP and selected lambda from all 36 original
  metric/selection records. `results_reported.csv` retains the earlier rounded
  review table, separately labeled.
- `metrics/`: original METRICS, SELECTION and BOOTSTRAP JSONs for all 36 conditions.
- `validation/`: all 180 original candidate-validation JSON records.
- `paired_contrasts.json`: 76 archived paired-interval summaries.
- `protocol_extract.json`: curated extract omitting absolute dependency paths
  and obsolete phase fields; source hash preserved. This is not the original lock.
- `manifest.json`: archive identity and hashes of transferred/derived files.
- `source_lineage.json`: original/repaired lock lineage and 17 byte-preserved
  source snapshots. See [implementation](fsd50k_implementation.md).

The earlier contact archive's 45 entries and the engineering handoff's 1,445
entries were checked with no hash mismatch. Relevant non-contact records were
selected. No private correspondence, raw audio, TUT annotations or private
absolute account paths were copied into this repository.

## Checks and limits

`python scripts/check_fsd_evidence.py` uses the standard library to check hashes,
condition coverage, all 36 200-class means/support totals, both result tables,
all 180 validation selections, 17 source snapshots and interval metadata. It cannot recompute AP from
labels/scores, generate bootstrap draws, fit heads, extract features, retrain
E1, validate native full-context equivalence or establish personal ability.

## What is now available and what still requires binary assets

Original text source, protocol/repair locks and result records have been
transferred from the Windows work area. Five source changes versus the first
protocol lock are fully accounted for by the later repaired execution lock.
Current source bytes must not be called the initial protocol-locked versions.

| Purpose | Original material | Availability here |
|---|---|---|
| Read execution code | Original feature/readout/evaluation drivers, imports and source locks | Available; selected snapshots preserved; portable numerical adaptation tested |
| Resolve run identity | MAIN_PACKET, terminal receipt, review and repair records | Text bytes verified; public lineage preserves hashes |
| Verify saved results | All 36 metrics/selections, 180 validation candidates, original RESULT_TABLE | Available and checked for consistency, not re-scored |
| Recompute predictions/AP | Features, selected-model arrays and scores | Not transferred |
| Recompute intervals | Uploader/content groups and saved prediction/draw arrays | Groups reconstructed privately with matching hash; scores/draws not transferred |
| Check feature path | Native adapters, checkpoints and authorized audio | Source available; checkpoints/audio not transferred; no native forward run |

The private manifest reconstructed the 200-column labels and evaluation groups
with hashes matching the historical bindings. That verifies label/group
construction from supplied metadata, not model predictions. Transfer heavy
arrays only for a named verification; this source release is not a complete
waveform-to-result reproduction package.

## Publication status

The original-source handoff is resolved. This update can be reviewed as a branch
and pull request; publication or merging does not establish fresh experimental
results. Source-derived numerical code and saved experimental records remain
separately labeled.
