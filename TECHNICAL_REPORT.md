# Audio Event Tokenizer: An Event-Retention Benchmark

Technical report v0.1
Author: MacswareX
Date: 2026-07-15

## Abstract

This report evaluates whether compact audio representations preserve event/state
information relevant to embodied agents. The benchmark reorganizes the 50
ESC-50 classes into six designed embodied groups and applies the same frozen
logistic-regression probe to three interfaces: pooled log-mel statistics,
histograms from the first two codebooks of pretrained EnCodec, and histograms
from the first two codebooks of a small custom mel residual-vector-quantized
autoencoder (mel-RVQ). Across ESC-50's five official folds, EnCodec obtains
52.15% accuracy and 0.4927 macro-F1, compared with 44.20% and 0.3699 for pooled
log-mel and 37.75% and 0.3450 for the custom mel-RVQ. EnCodec therefore exposes
more linearly decodable group information than this particular pooled log-mel
reference. The custom model has fewer trainable parameters and lower nominal
rate, but its coarse tokens retain less event information. Local encoding
timings are reported as implementation diagnostics rather than an end-to-end
comparison. The result is a representation diagnostic, not an audio foundation
model or a closed-loop world-model evaluation.

## 1. Motivation

Audio can reveal physical events that vision misses because of occlusion,
lighting, viewpoint, or limited field of view. Contact, pouring, appliance
state, footsteps, and object interaction are not merely sound labels: they can
be evidence of changes in an environment.

Modern audio generation and speech systems frequently expose discrete neural
codec tokens to sequence models. SoundStream introduced an influential
convolutional encoder/decoder with residual vector quantization (RVQ), and
EnCodec made a practical pretrained general-audio codec available. Later
systems such as Mimi and SpeechTokenizer distinguish coarse/semantic and
fine/acoustic code streams. These developments motivate a narrower empirical
question:

> How much embodied event-group information is linearly decodable from the
> coarse part of an audio-token representation?

The benchmark intentionally isolates this representation question before any
temporal dynamics, action conditioning, or robot-policy integration.

## 2. Relation to ObsCodec

ObsCodec studied bottlenecks for low-dimensional multi-agent observations using
PCA, autoencoders, a quantized autoencoder, beta-VAE, and single-codebook VQ-VAE.
This project carries forward three practices:

- compare learned representations against simple references;
- measure bottleneck health and preserve collapse/negative results;
- report cost together with task-facing quality.

The domain changes from low-dimensional PettingZoo observation vectors to
environmental audio.
The quantizer changes from a single VQ bottleneck to RVQ, and the primary
task-facing metric changes from reconstruction to event-group probe
performance. This is an evaluation progression from reconstruction toward a
task-facing semantic proxy, not evidence of a completed embodied system.

## 3. Benchmark Setup

### 3.1 Dataset

ESC-50 contains 2,000 five-second mono recordings in 50 classes, with 40 clips
per class and five official source-aware folds. Audio is read from the official
WAV distribution. The dataset is licensed for non-commercial use under CC
BY-NC.

### 3.2 Embodied grouping

The original classes are mapped into six designed groups:

| Group | Clips | Intended interpretation |
|---|---:|---|
| `human_activity` | 360 | Human presence/activity evidence |
| `water_liquid` | 240 | Flow, pouring, drinking, flushing, rain/waves |
| `impact_contact` | 240 | Sharp contact and transient interaction |
| `object_interaction` | 120 | Door/container manipulation |
| `appliance_mechanical` | 440 | Machine, appliance, vehicle, alarm state |
| `ambient_natural` | 600 | Natural/animal background and distractors |

The grouping is deliberately not balanced. Majority-class accuracy is 0.30,
so macro-F1 is reported with accuracy.

### 3.3 Audio frontend

The log-mel and custom-tokenizer path uses:

| Parameter | Value |
|---|---:|
| Sample rate | 16 kHz |
| FFT size | 1,024 |
| Hop length | 320 samples (50 frames/s) |
| Mel bins | 64 |
| Frequency range | 20 Hz to 8 kHz |
| Fixed tokenizer input | 256 mel frames |

The final experiments ran under Python 3.10, PyTorch/Torchaudio 2.5.1 with CUDA
12.4, and an NVIDIA GeForce RTX 3050 with 8 GB VRAM.

### 3.4 Probe protocol

Every representation is converted into one clip-level feature vector. A
`StandardScaler` and multinomial logistic-regression classifier (`C=1.0`,
`max_iter=2000`) are fitted on four folds and evaluated on the fifth, repeated
for all five official folds. The same probe implementation and seed are used
for every representation.

Absolute held-out accuracy and macro-F1 are primary. A secondary relative probe
ratio is defined as:

```text
relative probe accuracy = representation accuracy / pooled log-mel accuracy
```

This ratio may exceed 1.0 and is not a literal percentage of retained mutual
information.

## 4. Representations

### 4.1 Pooled log-mel reference

Each log-mel bin is summarized by its temporal mean and standard deviation,
producing a 128-dimensional vector. This is a compact classical reference, not
a theoretical upper bound on audio information.

### 4.2 Pretrained EnCodec

`facebook/encodec_24khz` encodes resampled 24 kHz audio at the configured 6
kbps bandwidth. The probe uses normalized token histograms from the first two
of the resulting codebooks, producing 2,048 features. A two-codebook prefix has
an approximate nominal rate of 1.5 kbps, although extraction in this experiment
uses the full 6 kbps encoding configuration.

### 4.3 Custom mel-RVQ

The custom model contains a convolutional mel encoder, four EMA-updated RVQ
codebooks of size 256 with data-dependent initialization and inactive-code
replacement, and a lightweight decoder. It has 321,280 trainable parameters.
The probe uses normalized histograms from the first two codebooks (512
features). The full four-codebook stream is approximately 0.4096 kbps; the
two-codebook prefix is approximately 0.2048 kbps.

The model is trained for 20 epochs using pooled-mel reconstruction plus VQ
commitment loss. Its final full-dataset codebook-0 utilization is 96.9% with
perplexity 185.9, showing that the codebook is active rather than collapsed.

Important protocol caveat: the custom tokenizer is trained without labels on
all ESC-50 clips before fold-wise probe evaluation. Its current result is a
transductive representation diagnostic. A strict generalization benchmark must
train a separate tokenizer on the four training folds for each held-out fold,
or pretrain it on a separate corpus.

## 5. Results

### 5.1 Probe quality

| Representation | Accuracy | Fold std | Macro-F1 | Relative vs log-mel |
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

EnCodec exceeds pooled log-mel by 7.95 percentage points and the custom model
by 14.4 points. The result means that the selected EnCodec code-histogram
interface exposes more linearly decodable group information than the two other
interfaces; it does not prove that compression creates information or that
EnCodec is optimal for an embodied policy.

![Held-out probe performance](assets/figures/probe_performance_esc50.png)

### 5.2 Cost and codebook health

| Representation | Nominal stream | Params | Encode latency | Codebook-0 utilization | Perplexity |
|---|---:|---:|---:|---:|---:|
| EnCodec | 1.5 kbps prefix; 6 kbps configured | 14.85M | 38.17 ms/clip | 98.8% | 261.6 |
| Custom mel-RVQ | 0.20 kbps prefix; 0.41 kbps full | 0.32M trainable | 1.43 ms/clip | 96.9% | 185.9 |

Parameter counts and nominal rates indicate a smaller custom representation,
but the latency values are not directly comparable: EnCodec timing starts from
waveforms and includes resampling, whereas custom timing starts from precomputed
mel tensors. The custom result is therefore a diagnostic quality-cost point,
not evidence of a deployment advantage.

### 5.3 Group-level behavior

For pooled log-mel, diagonal recalls are highest for
`appliance_mechanical` (0.58), `ambient_natural` (0.56), and
`human_activity` (0.45). `impact_contact` reaches 0.30, while
`object_interaction` (0.17) and `water_liquid` (0.15) are difficult. Frequent
confusion into `ambient_natural` partly reflects both acoustic overlap and the
larger number of classes assigned to that group.

Representation-specific normalized confusion matrices are provided in:

- `assets/figures/confusion_mel_probe_esc50.png`
- `assets/figures/confusion_encodec_esc50.png`
- `assets/figures/confusion_melrvq_esc50.png`

## 6. Negative and Diagnostic Findings

### 6.1 Codebook health is not semantic sufficiency

The custom tokenizer uses 248 of 256 first-codebook entries, yet underperforms
both EnCodec and pooled log-mel. High utilization prevents a trivial collapse
explanation but does not guarantee that token partitions align with event/state
semantics.

### 6.2 Reconstruction optimization is insufficient

The custom training reconstruction loss drops from 0.0204 to approximately
0.0025, while event-group probe accuracy remains 0.3775. Improving the pooled
reconstruction objective alone does not ensure event-preserving coarse tokens.

### 6.3 Designed group difficulty is uneven

Water/liquid and object-interaction groups are poorly separated by the compact
log-mel reference. The taxonomy may combine acoustically heterogeneous classes,
and clip-level histogram/statistical pooling removes temporal ordering that can
distinguish actions such as pouring, knocking, and opening.

## 7. Limitations

1. The evaluation is a frozen linear probe, not policy return, planning, or
   closed-loop robot behavior.
2. ESC-50 is small, curated, and not recorded from an embodied platform.
3. The six-group mapping is designed and imbalanced; it is not a canonical
   benchmark ontology.
4. Feature dimensionalities differ (128, 512, and 2,048), so results compare
   complete representation/pooling interfaces rather than pure codec content.
5. Histogram and mean/std pooling discard temporal order.
6. The custom tokenizer result is transductive because unsupervised tokenizer
   training sees all clips before fold probing.
7. EnCodec is externally pretrained on much broader data, while the custom
   tokenizer is trained only on ESC-50; the comparison is deliberately
   practical rather than compute/data matched.
8. Latency is measured on one Windows/CUDA workstation and is not a deployment
   guarantee; preprocessing differs between the two measured paths.
9. Relative probe accuracy can exceed one and must not be interpreted as a
   bounded information-retention percentage.

## 8. Next Work

The next experiments should be ordered as follows:

1. Make custom-tokenizer evaluation strict by training fold-wise or pretraining
   on an external non-evaluation corpus.
2. Preserve token timing using sequence probes instead of only histograms.
3. Add noise, mixing, cropping, and second-source generalization tests.
4. Distill event embeddings from PANNs or CLAP into codebook 0, analogous to
   semantic distillation in speech tokenizers.
5. Train a small causal transformer on coarse token sequences and compare
   next-token loss with downstream event retention.
6. Build a controlled action-conditioned transition dataset and evaluate
   future event-token prediction against no-action, shuffled-action, random,
   and persistence baselines.

Only step 6 introduces the minimum temporal and interventional structure needed
for an audio-world-model claim.

## 9. Conclusion

The benchmark demonstrates a working path from audio preprocessing through
coarse codec tokens to task-facing and cost-aware evaluation. Pretrained
EnCodec's first two codebooks are the strongest tested representation for
six-group ESC-50 probing. The custom mel-RVQ establishes an efficient,
independently implemented alternative with healthy codebooks, but its lower
event score shows that reconstruction and utilization are not enough: coarse
tokens need an explicit semantic/event objective or richer training data.

## References

- K. J. Piczak, "ESC: Dataset for Environmental Sound Classification," ACM
  Multimedia, 2015.
- N. Zeghidour et al., "SoundStream: An End-to-End Neural Audio Codec," 2021.
- A. Defossez et al., "High Fidelity Neural Audio Compression," 2022.
- H. Siuzdak, "SNAC: Multi-Scale Neural Audio Codec," 2024.
- A. Défossez et al., "Moshi: a speech-text foundation model for real-time
  dialogue," 2024.
- F. Zhang and M. Gienger, "Learning Robot Manipulation from Audio World
  Models," arXiv:2512.08405, 2025.
