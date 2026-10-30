# Audio Representation Probe

ARP studies what sound-event information a specified downstream readout can
access from an audio representation. Its development spans exploratory ESC-50
group classification, FSD50K frozen-representation comparisons, and separate
training and event-detection diagnostics.

> **Renovation draft — October 30, 2026.** FSD50K has a historical completion
> and review record, but its original execution source and complete predictions
> have not yet been transferred into this checkout. The following files expose
> the available evidence; they do not make the entire FSD experiment reproducible.

## Start with a question

| Question | Entry point | Available here |
|---|---|---|
| What did the initial compact interfaces reveal? | [Historical ESC-50 benchmark](docs/esc50_exploratory.md) | Existing code, archived results and figures |
| How do frozen representations compare on 200-label recordings? | [FSD50K readout](docs/fsd50k_readout.md) | Protocol extract, 36 reported rounded scores, four detailed metric/selection excerpts and paired intervals |
| How do the branches connect? | [Project evolution](docs/project_evolution.md) | Stage boundaries, later evidence and source gaps |
| What can this checkout verify? | [Evidence status](docs/evidence_status.md) | Artifact integrity and summary consistency; no FSD prediction replay |

## FSD50K: the main technical conversation interface

The recorded protocol freezes nine representations and trains only a ridge
readout: a linear map fitted by regularized least squares. Two pooling schemes
and two training arms yield 36 conditions. Official train/validation/evaluation
sizes are 36,796 / 4,170 / 10,231 recordings. Targets have 200 recording labels;
the task does not locate event boundaries.

| Frozen representation | G / full macro AP |
|---|---:|
| BEATs | 0.601386 |
| CNN6 | 0.441266 |
| EnCodec continuous latent | 0.144143 |
| E1 continuous latent | 0.110619 |
| E1 cumulative q2 | 0.087100 |
| E1 cumulative q4 | 0.095509 |

These are rounded values from the [historical AI-assisted review](evidence/fsd50k/historical_review.md),
not a new run. [All 36 reported values](evidence/fsd50k/results_reported.csv)
retain that provenance. Average precision (AP) evaluates the ranking of
recording scores; macro AP averages it across the 200 classes.

The useful comparison is bounded: cumulative q4 outperforms q2 under this
protocol for E1 and EnCodec. This does not establish an intrinsic semantic
ordering of codebooks, reconstruction necessity, or waveform quality.
T4 preserves coarse temporal bins but also quadruples feature dimension and
readout capacity. Pretraining exposure and five-second context resets matter.

## Check transferred evidence

From the repository root, with Python 3.10+ and its standard library:

```bash
python scripts/check_fsd_evidence.py
```

This checks file hashes, four per-class-to-macro means, the reported table and
paired-interval records. It does not recompute AP from raw predictions, fit a
readout, extract features or train a model. Newly written code is not substituted
for the missing original FSD implementation.

## Historical implementation and authorship

Existing numbered scripts, `src/`, configuration and environment files belong
to the exploratory ESC-50 implementation. Its six event groups, logistic head
and accuracy/macro-F1 differ from FSD200 ridge/AP. See the [archived commands](docs/esc50_exploratory.md).
Exact custom-model retraining is not guaranteed by the historical checkout.

The historical 0.4420 log-mel accuracy remains an unchanged archived result.
A recovered later official-fold log-mel evaluation reports 0.4415; see
[project evolution](docs/project_evolution.md). These must not be merged with
FSD AP into one improvement curve.

ARP does not establish a semantic tokenizer, production codec, audio foundation
model, reliable event detector or embodied policy. Later TUT/CRNN/R12 work is a
separate system and does not show that RVQ caused a detection failure.

Implementation, coordination and review used substantial AI assistance.
Artifacts do not prove the author's independent coding ability or external
human validation. Code is MIT licensed; third-party data/code/weights retain
their terms. Raw audio, TUT annotations and contact drafts are not included.
