"""Array persistence only; no archived runtime/control-state imports."""
from pathlib import Path
import os
import numpy as np

def load(path):
    with np.load(path, allow_pickle=False) as data:
        return {k: data[k] for k in data.files}

def npz(path, **arrays):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix('.npz.tmp')
    with temporary.open('wb') as stream:
        np.savez_compressed(stream, **arrays)
    os.replace(temporary,path)
