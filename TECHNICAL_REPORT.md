> Historical exploratory ESC-50 report, retained without numerical revision.
> Later FSD records and source-transfer status: [overview](README.md) and
> [project evolution](docs/project_evolution.md).

# Audio Representation Probe: A Diagnostic Study of Event-Group Accessibility

Technical report v0.1
Author: MacswareX
Date: 2026-07-15

## Abstract

This report evaluates how much event-group-discriminative information is
linearly accessible from three compact audio representation interfaces under a
common downstream protocol. The benchmark reorganizes 50 ESC-50 categories
into six designed event groups and compares pooled log-mel statistics,
first-two-codebook histograms from pretrained EnCodec, and
first-two-codebook histograms from a small custom mel residual-vector-quantized
representation model (mel-RVQ). Across ESC-50's five official folds, EnCodec
obtains 52.15% accuracy and 0.4927 macro-F1, compared with 44.20% and 0.3699
for pooled log-mel and 37.75% and 0.3450 for custom mel-RVQ. These observations
compare complete representation/readout interfaces; they do not establish an
inherent semantic hierarchy across RVQ stages or identify where task-facing
accessibility changes. Nominal discrete index rates and local encoding timings
are implementation diagnostics, not matched audio-codec comparisons. The
project is not a semantic tokenizer, sound-event-detection system, audio
foundation model, embodied-agent evaluation, or audio world model.

## 1. Motivation and Research Question

Environmental audio can carry evidence of contact, pouring, human activity, and
mechanical activity. Compact continuous features and discrete RVQ code IDs are
both possible interfaces to downstream models, but a representation's training
objective and codebook usage do not by themselves determine its measured
performance on a specific downstream task.

The narrow empirical question is:

> How much event-group information is accessible from different compact audio
> representation interfaces under a common downstream probe?

A secondary motivating question asks which representation stages might limit
that accessibility. The archived experiments do not perform the controlled
stage-wise interventions needed to answer this causally.

The first-K prefix is tested as an experimental interface; semantic superiority
of early RVQ stages is not assumed.

## 2. Relation to ObsCodec

ObsCodec studied bottlenecks for low-dimensional multi-agent observations using
PCA, autoencoders, a quantized autoencoder, beta-VAE, and single-codebook
VQ-VAE. This project carries forward three practices:

- Compare learned representations against simple references.
- Inspect bottleneck health and report negative findings.
- Record implementation cost alongside task-facing probe performance.

The domain changes from low-dimensional observation vectors to environmental
audio, while the primary reported task-facing outcome becomes event-group
linear-probe performance. The relationship is conceptual, not an integrated
embodied system or a measurement of downstream policy utility.

## 3. Benchmark Setup

### 3.1 Dataset

ESC-50 contains 2,000 five-second mono recordings in 50 categories, with 40 clips
per category and five official source-aware folds. Audio is read from the
official WAV distribution. The dataset is licensed for non-commercial use
under CC BY-NC.

### 3.2 Designed Event Groups

The original categories are mapped into six designed groups:

| Group | Clips | Intended interpretation |
|---|---:|---|
| `human_activity` | 360 | Human presence and activity evidence |
| `water_liquid` | 240 | Flow, pouring, drinking, flushing, rain, and waves |
| `impact_contact` | 240 | Sharp contact and transient interaction |
| `object_interaction` | 120 | Door and container manipulation |
| `appliance_mechanical` | 440 | Machine, appliance, vehicle, and alarm activity |
| `ambient_natural` | 600 | Natural and animal background sounds |

This mapping is a project-specific experimental ontology, not a canonical
embodied-audio or sound-event-detection benchmark. Its majority-class baseline
accuracy is 0.30, so macro-F1 accompanies accuracy.

### 3.3 Audio Frontend

The pooled log-mel reference and custom mel-RVQ representation path use:

| Parameter | Value |
|---|---:|
| Sample rate | 16 kHz |
| FFT size | 1,024 |
| Hop length | 320 samples (50 frames/s) |
| Mel bins | 64 |
| Frequency range | 20 Hz to 8 kHz |
| Fixed custom-model input | 256 mel frames |

The archived experiments ran under Python 3.10, PyTorch/Torchaudio 2.5.1 with
CUDA 12.4, and an NVIDIA GeForce RTX 3050 with 8 GB VRAM.

### 3.4 Frozen Probe Protocol

Every frozen representation is converted into one clip-level feature vector.
A `StandardScaler` and multinomial logistic-regression classifier (`C=1.0`,
`max_iter=2000`) are fitted on four official folds and evaluated on the fifth.
The process is repeated for all five held-out folds with the same probe
implementation and seed.

```text
frozen representation -> clip-level readout -> StandardScaler
                      -> multinomial logistic regression
                      -> held-out fold metrics
```

Held-out accuracy and macro-F1 are the primary outcomes. Macro-F1 is computed
over pooled out-of-fold predictions; reported fold standard deviations are
population standard deviations, not confidence intervals.

The secondary relative score is:

```text
relative probe accuracy vs. log-mel
    = representation probe accuracy / pooled log-mel probe accuracy
```

This ratio may exceed 1.0. It measures neither mutual information nor a literal
fraction of information retained. A frozen linear probe measures accessibility
under the chosen representation and readout, not all information potentially
present in the underlying signal.

## 4. Representation Interfaces

### 4.1 Pooled Log-Mel Reference

Each log-mel bin is summarized by its temporal mean and standard deviation,
producing a 128-dimensional vector. This is a compact classical reference, not
a theoretical upper bound on audio information.

### 4.2 Pretrained EnCodec First-K Prefix

`facebook/encodec_24khz` encodes resampled 24 kHz audio at the configured
6 kbps bandwidth. The probe uses normalized discrete-code histograms from the
first two RVQ codebooks, producing 2,048 features. A two-codebook prefix has an
approximate nominal index rate of 1.5 kbps, although extraction in the archived
experiment uses the full 6 kbps encoding configuration.

The selection of an early-codebook prefix is an experimental interface choice.
It does not establish that early stages contain superior event semantics.

### 4.3 Custom Mel-RVQ Representation

The custom path is:

```text
waveform -> log-mel -> convolutional encoder -> continuous latent
         -> four-stage EMA RVQ -> quantized latent / discrete code IDs
         -> decoder -> temporally averaged 64-dimensional mel target
```

The model uses four EMA-updated RVQ codebooks of size 256, data-dependent
initialization, inactive-code replacement, and a lightweight decoder. It has
321,280 trainable parameters. The downstream probe uses normalized histograms
from the first two codebooks, producing 512 features.

The full four-codebook nominal discrete index stream is approximately 0.4096
kbps; the two-codebook prefix is approximately 0.2048 kbps. These are computed
index-stream rates, not measured end-to-end waveform-codec rates. The decoder
reconstructs a temporally averaged 64-dimensional mel target; it does not
reconstruct waveform audio.

The archived model was trained for 20 epochs using pooled-mel reconstruction
plus VQ commitment loss. Its final full-dataset codebook-0 utilization is 96.9%
with perplexity 185.9. This observation rules out a fully inactive first
codebook, but it does not establish event-group discriminability, causal
utility, or semantic alignment.

The custom representation model is trained without labels on all ESC-50 clips
before fold-wise probe evaluation. Its current result is therefore a
transductive representation diagnostic. A strict generalization benchmark would
train a separate representation model on the four training folds for each
held-out fold, or pretrain it on a separate non-evaluation corpus.

## 5. Archived Results

All numerical values below are taken from the existing archived artifacts. No
model was retrained and no archived result was regenerated for this report.

### 5.1 Probe Performance

| Representation | Accuracy | Fold std | Macro-F1 | Relative vs. log-mel |
|---|---:|---:|---:|---:|
| Pooled log-mel | 0.4420 | 0.0089 | 0.3699 | 1.00 |
| EnCodec, first 2 codebooks | **0.5215** | 0.0197 | **0.4927** | **1.18** |
| Custom mel-RVQ, first 2 codebooks | 0.3775 | 0.0292 | 0.3450 | 0.85 |

Per-fold accuracies are:

| Representation | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 |
|---|---:|---:|---:|---:|---:|
| Pooled log-mel | 0.4400 | 0.4350 | 0.4450 | 0.4575 | 0.4325 |
| EnCodec | 0.4975 | 0.5375 | 0.5400 | 0.4975 | 0.5350 |
| Custom mel-RVQ | 0.3400 | 0.3675 | 0.3650 | 0.3875 | 0.4275 |

**[OBSERVED]** EnCodec exceeds pooled log-mel by 7.95 percentage points and
custom mel-RVQ by 14.4 points under the selected prefix-histogram readout.

**[INFERENCE]** The selected EnCodec representation/readout combination exposes
more linearly accessible event-group-discriminative information to this
particular probe than the two other tested combinations.

**[HYPOTHESIS]** Different codebook prefixes, sequence-preserving readouts, or
controlled feature dimensions could change the ordering. No archived
experiment tests those alternatives.

![Held-out event-group probe performance](assets/figures/probe_performance_esc50.png)

### 5.2 Nominal Index Rates and Codebook Diagnostics

| Representation | Nominal index stream | Params | Local encode timing | Codebook-0 utilization | Perplexity |
|---|---:|---:|---:|---:|---:|
| EnCodec | 1.5 kbps prefix; 6 kbps configured | 14.85M | 38.17 ms/clip | 98.8% | 261.6 |
| Custom mel-RVQ | 0.20 kbps prefix; 0.41 kbps full | 0.32M trainable | 1.43 ms/clip | 96.9% | 185.9 |

For the custom model, the reported stream is a nominal count of discrete index
bits. It excludes framing, entropy-coding overhead, and waveform
reconstruction. EnCodec timing starts from waveform input and includes
resampling; custom timing starts from precomputed mel tensors. The values
therefore do not support an apples-to-apples codec-efficiency or deployment
comparison.

### 5.3 Group-Level Behavior

For pooled log-mel, diagonal recalls are highest for
`appliance_mechanical` (0.58), `ambient_natural` (0.56), and
`human_activity` (0.45). `impact_contact` reaches 0.30, while
`object_interaction` (0.17) and `water_liquid` (0.15) are difficult.

**[INFERENCE]** Frequent confusion into `ambient_natural` may reflect acoustic
overlap, class imbalance, or the design of the six-group ontology. The
available confusion matrices do not distinguish those mechanisms.

Representation-specific normalized confusion matrices are provided in:

- `assets/figures/confusion_mel_probe_esc50.png`
- `assets/figures/confusion_encodec_esc50.png`
- `assets/figures/confusion_melrvq_esc50.png`

## 6. Diagnostic Findings and Claim Boundaries

### 6.1 Active Codebooks Do Not Establish Event-Group Accessibility

**[OBSERVED]** The custom model uses 248 of 256 first-codebook entries and
obtains 0.3775 event-group probe accuracy.

**[INFERENCE]** Active codebook usage excludes a trivial explanation in which
the first codebook never receives assignments. It does not establish that the
selected assignments align with the six designed event groups.

**[HYPOTHESIS]** Later RVQ stages or a different readout may expose
complementary event-group-discriminative information.

### 6.2 Reconstruction Loss and Probe Performance Are Distinct Observations

**[OBSERVED]** Archived custom-model training reconstruction loss decreases
from 0.0204 to approximately 0.0025, while the resulting first-two-codebook
histogram obtains 0.3775 event-group probe accuracy.

**[INFERENCE]** In this particular training run, improvement on the pooled-mel
reconstruction objective coincides with comparatively weak performance of the
selected downstream readout.

**[HYPOTHESIS]** Objective mismatch, frontend pooling, continuous-latent
structure, quantization, prefix selection, or histogram pooling could
contribute. The experiment does not prove independence between reconstruction
and event information or identify quantization as the primary loss point.

### 6.3 Designed Group Difficulty Is Uneven

**[OBSERVED]** Water/liquid and object-interaction groups have lower recall for
the pooled log-mel interface than several other designed groups.

**[HYPOTHESIS]** Within-group acoustic heterogeneity and the removal of
temporal ordering may contribute. Controlled group remapping and
sequence-preserving readouts are needed to test those explanations.

### 6.4 What the Benchmark Does Not Establish

- A semantic or event-aware tokenizer.
- An inherent semantic hierarchy across early and later RVQ codebooks.
- Statistical or causal independence of reconstruction and event-related task
  information.
- Quantization as the dominant source of task-facing information loss.
- Embodied-world-state estimation, planning, or robot-policy utility.
- Matched waveform-codec efficiency or comparable end-to-end latency.

## 7. Limitations

1. The evaluation is a frozen linear probe, not policy return, planning, or
   closed-loop robot behavior.
2. ESC-50 is small, curated, and not recorded from an embodied platform.
3. The six-group mapping is designed and imbalanced; it is not a canonical
   benchmark ontology.
4. Feature dimensionalities differ (128, 512, and 2,048), so results compare
   complete representation/readout interfaces rather than equal-dimensional
   representations.
5. Histogram and mean/std pooling discard temporal order.
6. The custom result is transductive because unsupervised representation-model
   training sees all clips before held-out-fold probing.
7. EnCodec is externally pretrained on much broader data, while the custom
   model is trained only on ESC-50; training data and compute are not matched.
8. Local latency is measured on one Windows/CUDA workstation with different
   preprocessing boundaries across the two representation paths.
9. Relative probe accuracy can exceed one and is not an information-theoretic
   quantity.
10. A single custom-model training run and five evaluation folds do not
    establish statistical superiority.

## 8. Next Discriminating Experiments

The next experiments should be ordered as follows:

1. Build a representation-stage ladder spanning frontend features, continuous
   encoder latents, quantized latents, and discrete-code readouts.
2. Run frozen probes for individual RVQ stages, later stages, and first-K
   prefixes without assuming that earlier stages are inherently semantic.
3. Compare sequence-preserving and Bag-of-Codes readouts while controlling
   feature dimensionality.
4. Separate representation-model training strictly from held-out clips and
   report multiple random seeds.
5. Only after these diagnostics, test objective interventions with explicit
   controls and the same downstream probe.

These are future hypotheses and proposed controls. None has been performed in
the archived benchmark.

## 9. Historical Artifact Provenance

The original `assets/results/event_retention_summary.json` and the other
archived ESC-50 JSON files remain unchanged. Historical keys such as
`chance_majority` and custom-model `bitrate_kbps` reflect the naming used when
those artifacts were created; they mean majority-class baseline accuracy and
nominal discrete index rate, respectively. New summaries use corrected field
names and a source-specific `event_group_probe_summary_<source>.json` filename.

The former `scripts/4_eval_event_retention.py` path remains a compatibility
entry point. Its canonical replacement is
`scripts/4_eval_event_group_probe.py`.

## 10. Conclusion

**[OBSERVED]** Under the archived five-fold ESC-50 protocol, the pretrained
EnCodec first-two-codebook histogram obtains the highest event-group probe
accuracy among three tested representation/readout interfaces.

**[INFERENCE]** The custom mel-RVQ model provides a smaller discrete
representation interface with an active first codebook, but its selected
prefix-histogram readout has lower event-group accessibility under this probe.

**[HYPOTHESIS]** Stage-wise, codebook-wise, readout-controlled, and strictly
separated experiments may reveal where that accessibility differs. Establishing
any mechanism requires additional experiments rather than stronger terminology.

## References

- K. J. Piczak, "ESC: Dataset for Environmental Sound Classification," ACM
  Multimedia, 2015.
- N. Zeghidour et al., "SoundStream: An End-to-End Neural Audio Codec," 2021.
- A. Defossez et al., "High Fidelity Neural Audio Compression," 2022.
- H. Siuzdak, "SNAC: Multi-Scale Neural Audio Codec," 2024.
- A. Defossez et al., "Moshi: a speech-text foundation model for real-time
  dialogue," 2024.
- F. Zhang and M. Gienger, "Learning Robot Manipulation from Audio World
  Models," arXiv:2512.08405, 2025.
