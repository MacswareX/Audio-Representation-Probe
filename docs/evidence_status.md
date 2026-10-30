# Evidence status: October 30 renovation draft

Public base: `4d47b78608b0a95d1f92e6b0c5b2c5abe6c554c1`.
The original ESC-50 execution code is unchanged.

## Transferred files

- `evidence/fsd50k/historical_review.md`: unchanged September 13 completion
  review from the September 14 contact archive. Earlier numerical checks are
  reported there; they are not repeated by this renovation.
- `results_reported.csv`: 36 six-decimal values transcribed from that review,
  not extracted from 36 original prediction matrices.
- `metrics/`: METRICS, SELECTION and BOOTSTRAP JSONs for four conditions:
  E1/EnCodec q2/q4, G/full. Other conditions are summary-only here.
- `paired_contrasts.json`: 76 archived paired-interval summaries.
- `protocol_extract.json`: curated extract omitting absolute dependency paths
  and obsolete phase fields; source hash preserved. This is not the original lock.
- `manifest.json`: archive identity and hashes of transferred/derived files.

The source archive's 45 manifest entries were checked with no hash mismatch.
Only relevant non-contact excerpts were selected. No private correspondence,
raw audio, TUT annotations or local account paths were copied into this draft.

## Checks and limits

`python scripts/check_fsd_evidence.py` uses the standard library to check hashes,
condition coverage, four 200-class means/support totals, rounded-table
consistency, selections and interval metadata. It cannot recompute AP from
labels/scores, generate bootstrap draws, fit heads, extract features, retrain
E1, validate native full-context equivalence or establish personal ability.

## Needed originals

The historical source index points to a Windows project containing
`36_IDENTIFIABLE_PRESSURE_REPAIR/workflow_r9/full_fsd_readout_01`.
Its current contents must be checked on that machine.

| Purpose | Original material | Availability here |
|---|---|---|
| Import execution code | `full_features.py`, `full_readouts.py`, imports, exact dependencies/models | Missing |
| Resolve run identity | `execution_04/MAIN_PACKET.json`, completion/terminal receipts, recovery/version records | Historical hashes only |
| Verify all conditions | All 36 METRICS/SELECTION records, candidate validation artifacts, RESULT_TABLE | Four conditions plus rounded review table |
| Recompute predictions/AP | Recording/class order, labels, selected predictions, normalization/head arrays | Missing |
| Recompute intervals | Uploader/content groups and draws or exact resampling implementation | Summaries only |
| Check feature path | Native adapters, checkpoint identities, source manifests and authorized inputs | Not executed here |

First transfer source and compact metadata; transfer heavy arrays separately
for a named verification. A partial handoff must not be labeled complete
reproduction. Supplying existing files requires no new training.

## Publication status

This local review candidate awaits the original-source handoff requested by
the author. It has not been pushed/merged and is not a complete FSD release.
Resolve source gaps and inspect the final diff before publishing.
