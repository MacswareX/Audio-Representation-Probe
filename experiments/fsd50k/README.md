# Historical FSD50K source snapshot

`source_snapshot/*.py.txt` preserves the 17 repaired-execution-locked source
files from the private October 30 text handoff. Bytes are unchanged. File names
use `.txt` to prevent accidental execution of historical resource/control-state
drivers. Hashes and original member names are in
[`source_lineage.json`](../../evidence/fsd50k/source_lineage.json).

Use [`src/fsd50k`](../../src/fsd50k/) and the documented
[portable entry](../../docs/fsd50k_implementation.md) for the feature/readout
interface. It is a separately identified adaptation, not an invocation of the
original Windows runtime. Native waveform feature extraction is not replayed
or claimed to be portable by this release.
