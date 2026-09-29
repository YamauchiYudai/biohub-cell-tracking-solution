# Release checklist

This repository must not be pushed to GitHub before the competition has ended
(final submission deadline 2026-09-29 23:59 UTC = 2026-09-30 08:59 JST) and the owner has approved the push.

Status as of 2026-09-29 (local branch `public-release-prep`, not pushed):

- [ ] **Competition finished** - not yet: **push is not allowed.**
- [ ] Final rank updated (README "Competition Result": currently TBD)
- [ ] Private LB updated (README "Competition Result": currently TBD)
- [ ] Solution write-up updated before publishing it on Kaggle: header result line and section 7 "Where we draw
      the line" in `notebooks/solution_writeup.ipynb` (final selection and private results are TBD there)
- [ ] Final submission identified: set `result.final_submission` in `configs/final.yaml`
- [ ] Final config matches submission: `python scripts/prepare_notebook.py --candidate <final>` then
      `python scripts/prepare_notebook.py --check` (code-cell SHA-256 equals the submitted kernel)
- [ ] Final notebook matches submission (same command; for `v_add`, pass the teammate's kernel notebook with
      `--source` and record its `code_sha256`)
- [ ] Components not used by the final submission removed or marked (if the final submission uses neither
      the F03 head nor the frozen-frame consensus, delete `src/biohub_tracking/detection/`,
      `postprocessing/consensus.py` and their tests, or keep them labelled as final-day candidates)
- [x] Tests pass: `pytest -q` (96 passed with all extras + matplotlib; 90 passed + 6 skipped with `.[dev]` only)
- [x] Secret scan passes (detect-secrets: only SHA-256 code / weight fingerprints flagged; targeted pattern
      scan: only the scan patterns themselves and .gitignore entries) - rerun right before pushing
- [x] No competition data committed (no `.zarr`, `.geff`, images or GT tables)
- [x] No checkpoints accidentally committed (no `.pt` / `.pth` / `.ckpt` / `.npz`; largest file is the notebook)
- [x] Third-party licenses reviewed (`THIRD_PARTY_NOTICES.md`, `LICENSES/Apache-2.0.txt`)
- [ ] README reviewed by the owner
- [x] Teammate approved publication and attribution
- [ ] Kaggle weight datasets made public, or README notes they are private
      (`yudaiyamauchi/biomed-x138-division-t3-full-weights`, `yudaiyamauchi/biomed-x138-v1284-head-f03`
      are private as of 2026-09-29; without them the notebook needs retrained weights from `scripts/train.py`)
- [ ] git status clean
- [ ] **User approved git push**

## Commands to run right before pushing

```bash
python -m compileall -q src scripts
pytest -q
python scripts/prepare_notebook.py --check
detect-secrets scan --all-files --exclude-files '(^\.git/|LICENSES/)'
git ls-files | xargs grep -nIE 'ghp_|github_pat_|sk-[A-Za-z0-9]{20}|KAGGLE_KEY|CLEARML_API_(ACCESS|SECRET)_KEY|/Users/|@gmail\.com|CloudStorage' ; echo "exit $? (1 = no match)"
git ls-files | grep -iE '(^|/)(kaggle\.json|\.env)$|\.(pt|pth|ckpt|onnx|npz|zarr|geff)$' ; echo "exit $? (1 = none)"
git status --short
```
