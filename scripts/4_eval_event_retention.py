"""Historical compatibility wrapper for the former event-retention entry point.

Use ``scripts/4_eval_event_group_probe.py`` for new commands. The historical
path remains available so existing local scripts and automation keep working.
"""
from pathlib import Path
import runpy


if __name__ == "__main__":
    script = Path(__file__).with_name("4_eval_event_group_probe.py")
    runpy.run_path(str(script), run_name="__main__")
