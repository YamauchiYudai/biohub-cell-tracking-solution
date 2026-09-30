# Release checklist

The repository is **currently private**, reconfirmed on 2026-10-01 (JST) after the security PR was merged.
It was public during the initial review and was subsequently made private. No visibility change is made
by this security update. Remaining leaderboard and model-availability tasks below concern results and
reproducibility; the security results and execution limits are recorded separately.

## Security checks (2026-10-01 JST)

- [x] Secret/history scanning enforced by the repository CI and local release checker
- [x] Dependabot vulnerability alerts/security updates enabled
- [ ] When public again, verify GitHub secret scanning, push protection and private vulnerability reporting
      are enabled (they were enabled during the public review; these APIs are currently unavailable for
      this private repository)
- [x] Current files and all 9 reachable commits scanned: no detected secrets beyond reviewed SHA fingerprints
- [x] No raw competition data or model checkpoints found among tracked files/history
- [x] Model loaders reject outdated PyTorch, unrestricted pickle and paths outside the model bundle
- [x] Restricted-loading notebook generator added; original submitted code-cell fingerprint retained
- [x] Known-vulnerability audit of the resolved environment passes; see coverage limits in [SECURITY.md](SECURITY.md)
- [x] Static analysis: no medium/high findings; security regression tests pass
- [x] Repeatable release scanner, read-only CI workflow and dependency-update configuration added
- [ ] Restricted-loading notebook verified on Kaggle GPU with real inputs and output parity
- [x] Security fix merged into `main` via [PR #1](https://github.com/YamauchiYudai/biohub-cell-tracking-solution/pull/1);
      both pre-merge GitHub CI runs passed (107 tests plus security scans)

The historical notebook contains unsafe legacy model-loading paths and is provided for archival inspection.
For new runs use `python scripts/prepare_safe_notebook.py`, with reviewed inputs in an isolated Kaggle
session. Do not describe the archive as safe for arbitrary downloaded models. See [SECURITY.md](SECURITY.md)
for execution boundaries and private reporting.

## Results, attribution and reproducibility

- [x] **Competition finished** (deadline 2026-09-30 08:59 JST passed)
- [ ] Private LB of both selected submissions and the final rank filled in: README "Competition Result",
      `result.scoring_submission` in `configs/final.yaml`, the header and section 7 of
      `notebooks/solution_writeup.ipynb`, and `docs/results.md` "Final result". README records 79 / 0.930;
      the per-submission scores and scoring candidate still need confirmation.
- [ ] If `fc` (not `fc_f03`) turns out to be the scoring submission: `python scripts/prepare_notebook.py --candidate fc`,
      set `notebook.candidate: fc`, then `python scripts/prepare_notebook.py --check`
- [x] Final submissions identified: `fc_f03` (56663011, Public 0.966) and `fc` (56663004, Public 0.964) in
      `result.selected_submissions`
- [x] Final config matches submission: `prepare_notebook.py --check` OK for the committed `fc_f03`
      (code-cell SHA-256 equals the submitted kernel); `fc` is regenerated and hash-checked by the tests
- [x] Final notebook matches submission (same check)
- [x] Components used by the selected submissions are all in `src/` (b1c, T3 + V5a, J2, frozen-frame consensus,
      F03 head); none needs to be removed
- [x] Solution write-up filled in (selection, all public scores); only the private LB and rank remain TBD
- [x] Tests pass: `pytest -q` (all extras + matplotlib: all pass; `.[dev]` only: optional-extra tests skip)
- [x] Verified on Kaggle (2026-09-30, private kernels, nothing submitted):
      `notebooks/final_submission.ipynb` as committed, GPU T4 x2 with the submission's inputs: completed, and its
      `submission.csv` for the 4 visible movies is byte-identical to the output of submission 56663011
      (SHA-256 e5bd8b01...), deadline not degraded, only timing statistics differ;
      the package on Kaggle's standard image (Python 3.12): install with all extras, 97 tests pass, CLIs run on
      real train movies (`read_geff`, `evaluate.py` with the official metric, `train.py t3-crops` / `t3`), the real T3
      weights load and score real crops, the real F03 head matches its SHA-256;
      `notebooks/solution_writeup.ipynb`: runs end to end with the same outputs as locally
- [x] Third-party licenses reviewed (`THIRD_PARTY_NOTICES.md`, `LICENSES/Apache-2.0.txt`)
- [ ] README reviewed by the owner
- [x] Teammate approved publication and attribution
- [ ] Kaggle weight datasets made public after the deadline (planned by the owner):
      `yudaiyamauchi/biomed-x138-division-t3-full-weights`, `yudaiyamauchi/biomed-x138-v1284-head-f03`

## Before each publication update

```bash
python -m pip install --upgrade 'pip>=26.2'
python -m pip install -e '.[dev,torch,metric,security]' matplotlib
python scripts/check_release_security.py --history
bandit -r src scripts -ll
pip-audit --skip-editable
pytest -q
python scripts/prepare_notebook.py --check
python scripts/prepare_safe_notebook.py
git diff --check
git status --short
```

Review new figures for private data, license notices and anonymous access to linked model datasets. A
previous green scan or clean working tree does not describe newly edited files. Leaderboard verification,
owner review and dataset publication remain explicit pending tasks; do not mark them complete by inference.
