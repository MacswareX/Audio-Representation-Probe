# Import the prepared integration without replacing local work

The hosted integration produced a Git bundle because the connected GitHub
write API returned `403 Resource not accessible by integration` and the hosted
Git CLI had no HTTPS credentials. No remote branch, PR or merge was created.
This is an authentication limitation, not a failed numerical check.

Download `ARP_FSD_Integration_20261030.bundle` to the Windows machine with the
existing ARP Git repository. The bundle requires the public base commit
`4d47b78608b0a95d1f92e6b0c5b2c5abe6c554c1`, and contains the prepared branch's
commits and file objects. No audio, features, scores or checkpoints are included.

From the existing repository, inspect `git status --short` and
`git remote get-url origin`. Origin must identify
`MacswareX/Audio-Representation-Probe`. Then use an isolated worktree so current
local modifications and the local main branch stay untouched:

```powershell
git bundle verify 'C:\path\to\ARP_FSD_Integration_20261030.bundle'
git fetch 'C:\path\to\ARP_FSD_Integration_20261030.bundle' refs/heads/renovate/fsd-evidence-20261030
git worktree add -b review/fsd-integrated-20261030 ..\ARP-FSD-Review FETCH_HEAD
Set-Location ..\ARP-FSD-Review
python scripts/check_fsd_evidence.py
python -m unittest discover -s tests -v
git push origin HEAD:refs/heads/renovate/fsd-evidence-20261030
```

Replace the bundle path. If the branch/worktree name already exists, choose a
new name; do not delete or force-reset it. Tests need the numerical dependencies
in `requirements-fsd.txt`; use a matching existing environment or an isolated
one rather than changing the historical model-training environment.

The push creates an independent review branch and does not merge main.
If authentication or repository access is unavailable locally too, restore the
GitHub connection's repository write access; do not paste a token into chat.

Suggested PR title: **Expose FSD50K frozen-readout implementation and saved evidence**

Suggested description:

> The public repository previously described only the exploratory ESC-50 stage.
> This change preserves it as history and adds the original FSD source lineage,
> 36 saved evaluation conditions, 180 validation candidate records and a portable
> feature-to-ridge-to-AP interface. It separates frozen evaluation, training
> interventions and TUT detection diagnostics rather than presenting one curve.
>
> Validation: six constructed-data tests; integrity/consistency checks for 294
> evidence files, all 36 conditions, 180 candidates, 17 source snapshots and 76
> paired intervals. Full historical feature extraction, prediction replay,
> bootstrap regeneration and model retraining were not performed; the binary
> assets required for them are not included.
