# Release checklist

The GitHub repository is **private**. Pushing to it is allowed (owner approval, 2026-09-30). Making it **public**
is not allowed before the competition has ended (final submission deadline 2026-09-29 23:59 UTC = 2026-09-30
08:59 JST) and the private results are filled in.

Status as of 2026-09-30 09:36 JST (`main` pushed to the private repository):

- [x] **Competition finished** (deadline 2026-09-30 08:59 JST passed); the repository stays private until the
      private results are filled in
- [ ] Private LB of both selected submissions and the final rank filled in: README "Competition Result",
      `result.scoring_submission` in `configs/final.yaml`, the header and section 7 of
      `notebooks/solution_writeup.ipynb`, and `docs/results.md` "Final result" (all currently TBD)
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
- [x] Secret scan passes (detect-secrets: only SHA-256 code / weight fingerprints flagged; targeted pattern
      scan: only the scan patterns themselves and .gitignore entries) - rerun right before pushing
- [x] No competition data committed (no `.zarr`, `.geff`, images or GT tables)
- [x] No checkpoints accidentally committed (no `.pt` / `.pth` / `.ckpt` / `.npz`; largest file is the notebook)
- [x] Third-party licenses reviewed (`THIRD_PARTY_NOTICES.md`, `LICENSES/Apache-2.0.txt`)
- [ ] README reviewed by the owner
- [x] Teammate approved publication and attribution
- [ ] Kaggle weight datasets made public after the deadline (planned by the owner):
      `yudaiyamauchi/biomed-x138-division-t3-full-weights`, `yudaiyamauchi/biomed-x138-v1284-head-f03`
- [x] git status clean
- [x] User approved git push to the private repository (2026-09-30)
- [ ] **Repository visibility switched to public** (after the competition ends, with the private results filled in)

## Commands to run before every push and before switching the repository to public

```bash
python -m compileall -q src scripts
pytest -q
python scripts/prepare_notebook.py --check
detect-secrets scan --all-files --exclude-files '(^\.git/|LICENSES/)'
git ls-files | xargs grep -nIE 'ghp_|github_pat_|sk-[A-Za-z0-9]{20}|KAGGLE_KEY|CLEARML_API_(ACCESS|SECRET)_KEY|/Users/|@gmail\.com|CloudStorage' ; echo "exit $? (1 = no match)"
git ls-files | grep -iE '(^|/)(kaggle\.json|\.env)$|\.(pt|pth|ckpt|onnx|npz|zarr|geff)$' ; echo "exit $? (1 = none)"
git status --short
```
