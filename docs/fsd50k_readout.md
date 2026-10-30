# FSD50K: trace the readout before interpreting the score

This explains the archived protocol and available evidence. It is not the
original implementation. See [evidence status](evidence_status.md).

## Trace one recording

1. Read a complete recording, official split and 200-dimensional binary label.
   Multiple sound classes may be present at once.
2. Cover its original time span with balanced, nonoverlapping windows of at most
   five seconds. The record reports 106,158 windows over 51,197 recordings;
   it does not retain only each recording's first five seconds.
3. Apply a frozen encoder at its native sample rate. Recover feature time
   anchors and valid support; padding is not counted as extra audio.
4. Pool to one recording vector. G uses support-weighted mean/population std;
   T4 computes them in four equal-duration recording bins and concatenates them.
5. Fit centering/scaling from the training arm only. Fit five ridge candidates,
   then select regularization on validation macro AP without validation refit.
6. Score the selected head on 10,231 evaluation recordings. Rank scores within
   each class, calculate AP, then average all 200 classes.

For a constructed 13-second recording, there are three balanced windows of
about 4.33 seconds. T4's bins each span 3.25 seconds over the whole recording.
Window boundaries and pooling-bin boundaries need not coincide. Complete
coverage does not mean the encoder attends across all 13 seconds: its context
resets at each input window.

## The interface changed after ESC-50

`z` is the continuous prequantization latent; `q2`/`q4` are cumulative latent
vectors reconstructed from the first two/four residual-vector-quantization
(RVQ) stages. They are not the original ESC-50 code-ID histograms.
E1 G features have 128 dimensions for z/q2/q4; EnCodec has 256. T4 quadruples
each dimension. RVQ stage count is neither time resolution nor INT8 weight
quantization.

q4-minus-q2 changes the frozen representation exposed to the readout.
ALL-minus-EARLY in the separate training branch changes supervision placement.
These are different interventions.

| Frozen output before pooling | Input sample rate | Per-time feature channels | G / T4 dimensions |
|---|---:|---:|---:|
| log-mel | 16 kHz | 64 | 128 / 512 |
| E1 z/q2/q4 | 16 kHz | 64 | 128 / 512 |
| CNN6 block4 | 32 kHz | 512 | 1,024 / 4,096 |
| EnCodec z/q2/q4 | 24 kHz | 128 | 256 / 1,024 |
| BEATs | 16 kHz | 768 | 1,536 / 6,144 |

BEATs patch tokens must first be restored to time × frequency layout and
averaged over frequency; flattening the frequency patches does not create more
time steps. The recorded FSD EnCodec setting activates four stages at 3 kbps,
unlike the August extraction configured at 6 kbps and read through two code-ID
histograms. This draft has not independently executed either adapter.

## Fit the head

X contains standardized training vectors; Y contains 200 binary labels per row.
The archived objective is

$$\min_{W,b}\frac{1}{N}\|XW+\mathbf{1}b^T-Y\|_F^2+\lambda\|W\|_F^2.$$

W has shape d by 200; the 200-entry intercept b is not penalized. With centered
X and Y, the normal equations use `X.T @ X + N * lambda * I`.
The recorded implementation streams float64 sufficient statistics and uses an
eigendecomposition. Its source-derived portable implementation is now in
`src/fsd50k/statistics.py`; the byte-preserved original is separately recorded.
This is not Adam or
binary-cross-entropy training.

Lambda candidates are 0.0001, 0.001, 0.01, 0.1 and 1.0. Validation 200-class
macro AP selects one, with larger lambda winning exact ties; no validation refit.
The full/dedup training arms have 36,796/36,794 rows. Dedup removes two previously
identified training duplicates while validation/evaluation stay fixed.
There are 180 candidate fits and 36 selected models.

Ridge outputs are ranking scores, not calibrated probabilities. A 0.5 decision
threshold is not part of this evaluation. Sigmoid is unnecessary for AP and
can introduce numerical ties even though it preserves mathematical ordering.

## What the metric measures

AP accumulates precision at positive-recall increments, grouping equal scores
at a common threshold. In a constructed single-class example, labels
`[1, 0, 1, 0]` in descending score order give AP `(1 + 2/3)/2 = 5/6`.
This ranking is not the temporal order of sounds inside a recording. AP does
not supply event onset/offset times.

The archived definition leaves classes with no positive or no negative examples
undefined. Official macro AP requires all 200 defined classes; a conditional
defined-class mean must be identified separately.

## Read each contrast as a bounded decision

| Contrast | Historical observation | Supported interpretation | Still unresolved |
|---|---|---|---|
| E1 q4−q2, G/full | +0.008409; paired interval [0.006843, 0.010357] | More useful signal is accessible at depth four under this readout | Semantic hierarchy, reconstruction necessity, other heads/tasks |
| EnCodec q4−q2, G/full | +0.008547; [0.006218, 0.010891] | Same bounded depth comparison | Waveform quality or effect of model training history |
| T4−G | Positive for log-mel/E1/EnCodec, negative for CNN6/BEATs | Pooling interacts with representation | Temporal information alone: dimensions/capacity also change |
| dedup−full | Maximum absolute change 0.00003676; intervals cross zero | No resolved effect from removing these two IDs | Equivalence or removal of all exposure |
| BEATs versus CNN6 | G/full 0.601386 versus 0.441266 | Complete frozen-interface comparison | Architecture-only effect: pretraining/dimensions differ |

Intervals use 2,000 paired uploader/content-group bootstrap draws, 2,297 groups
and seed 17170911. The same group draw is used for both conditions, predictions
stay fixed, and heads are not refitted. These are descriptive pointwise
intervals conditional on saved models, excluding training randomness and
multiple-comparison guarantees. This checkout has interval summaries, not
the score matrices/draws required to regenerate them.

## Boundaries that change the interpretation

- EnCodec's reported FSD pretraining, AudioSet exposure of CNN6/BEATs and
  E1's ESC/FSD shared sources limit unseen-source generalization claims.
- E1's secondary panel excludes 119 known source-overlap evaluation files and
  retains 10,112. The official main denominator is still 10,231; the panel
  does not prove that every unknown exposure was removed.
- Complete coverage is distinct from uninterrupted native encoder context.
  Context30 sensitivity remains open, not silently completed or imposed as
  another prerequisite for an honest technical conversation.
- Current checks validate stored summaries, not the waveform-to-feature path,
  checkpoint retraining or human peer review.

## Develop ownership on the same object

Stay with one recording, pooled vector, selected head and score. Trace its
time support, normalization and ranking before adding another model. Explain
which training statistic changes when a training row changes, and why an
evaluation row must not change it. This is a learning operation, not a new
experiment required before contact.

An external discussion can begin with the q4/q2 comparison and ask which
input, representation or evaluation decision the recipient needs to make.
Only a concrete need should determine another implementation. These results
alone do not establish demand for a new tokenizer or benchmark wrapper.
