# Project evolution and evidence boundaries

The branches share an interest in task-facing audio representations. They are
not one model's continuous performance-improvement curve.

| Stage | Question | Located evidence | Checkout relationship |
|---|---|---|---|
| August ESC-50 exploration | Six groups; statistics versus code histograms | Public code, archived metrics/figures | Original code retained; [historical overview](esc50_exploratory.md) |
| September ESC-50 repair | Provenance, out-of-fold probabilities, source-aware comparison | Exported patches, evaluator modules, log-mel cache and predictions | Located privately; not automatically applied |
| September E1 intervention | Matched-opportunity NONE/EARLY/ALL and fresh heads | Report and bounded negative/inconclusive results | Separate training branch; complete source/weights needed |
| September FSD50K main | Nine representations, G/T4, full/dedup, 200 labels | Completion review, four detailed conditions, 76 intervals | Evidence excerpts in draft; original execution code absent |
| September TUT diagnostics | CRNN/R12, boundary auxiliary, separate BEATs comparison | Saved-prediction replay pack, partial model code and summaries | Separate detection lineage; no TUT annotations copied |
| October Physical Audio S1 | Digital rotation/history fixture | October 23 retrospective | Separate learning fixture, not ARP/robotics integration |

## Preserve original experiments, update their status

The August report/source/results remain historical. They use six-class logistic
regression, not the FSD 200-output ridge head. Initial log-mel accuracy was
0.4420. A recovered later official-fold evaluation reports 0.4415 accuracy and
0.369545703 macro-F1; its source-purged variant reports 0.4450 and 0.371873605.
These are later artifacts, not a silent correction of the August three-way
table. They have not been rerun here.

Repair exports contain `evaluation_protocol.py`, `feature_artifacts.py`,
probability-aware probe changes, tests and patches. Import requires checking
their base/dependencies and native behavior. Blindly applying an exported
patch to the August checkout would erase that distinction. Private Windows
paths and administrative records are not the public implementation.

## Keep three questions separate

1. **Frozen evaluation:** which fixed representation/readout exposes recording
   labels? FSD addresses this under a specified protocol. A fresh head does
   not remove pretraining exposure.
2. **Training intervention:** how does supervision placement change E1 with
   matched opportunities? Corrected pressure experiments have bounded negative
   or inconclusive outcomes; the earlier head-opportunity confound must not be
   applied to every later version. They do not isolate removing reconstruction.
3. **Detection measurement:** do better activity rankings yield matched event
   instances? TUT tests label/time/decoder interfaces. CRNN/R12 has no RVQ and
   cannot identify quantization as the cause of a detection failure.

In TUT, boundary assistance raises validation duration-weighted AP for all three
seeds while event true positives stay zero under the fixed decoder. The separate
native-CRNN/BEATs comparison raises mean AP 0.188152→0.511611 while event F1
changes 0.004505→0.003848. Its baseline differs and feature/fusion capacity also
change. These are contact-package reports, not new reproduction results here.

## Apply the October 23 retrospective

Use an existing comparison to discuss a real recipient's need rather than
expanding the preparation syllabus. A useful match requires the actual problem,
demonstrated operations on a concrete object, eligibility/resources, and a
review/feedback route.

For this renovation: explain and expose FSD without inventing a replacement
baseline, industrial adoption of E1, or automatic context30/C1-C5 obligations.
Missing files are a transfer problem, not a reason to rerun accepted science.
AI-assisted explanations and reviews do not establish personal coding ownership;
that develops by operating and modifying the same pipeline.
