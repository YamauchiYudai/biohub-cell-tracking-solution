# Security policy

This repository contains offline research code, not a network service. Public source availability does not
make arbitrary checkpoints, notebooks or input archives safe to execute.

## Supported execution

Use the current source revision with updated dependencies. The maintained package requires PyTorch 2.14.0+
for model loading, always passes `weights_only=True`, and confines manifest paths to the model directory
(including symlink resolution). No unrestricted-pickle fallback is provided. Scikit-learn requires 1.5+
(the earlier minimum included [CVE-2024-5206](https://github.com/advisories/GHSA-jw8x-6495-233v)).
Older PyTorch allowed code execution even with `weights_only=True`
([upstream advisory](https://github.com/pytorch/pytorch/security/advisories/GHSA-53q9-r3pm-6pq6)); the new floor
also avoids the later advisories listed by PyPI as of this review. Re-audit regularly; a version floor is
not a permanent guarantee.

Generate `outputs/safe_submission.ipynb` with `python scripts/prepare_safe_notebook.py` for new inference runs.
Its first cell rejects old PyTorch, enforces weights-only loading through the environment inherited by child
processes, and the generator disables the legacy Plan-B pickle loader. It is a derivative, not a
byte-identical competition submission. Unsupported checkpoints fail closed. Its full GPU run and prediction
parity still require verification; do not remove its protections to force a legacy checkpoint to load.

`notebooks/final_submission.ipynb` is an **archival, trusted-input-only** artifact. Its code-cell fingerprint
is preserved for reproducibility. It contains upstream `weights_only=False` and an inactive legacy pickle
loader. The archive is not the recommended execution entry point. Executable source from Kaggle support
packs is outside the loader's protection, as are malicious tensor sizes and resource exhaustion. Run only
reviewed inputs in an isolated session with no credentials or unrelated private files. Never run arbitrary
notebooks/models from others merely because their filenames or dataset names match this documentation.

The official metric is pinned to a source commit, but its transitive `tracksdata` Git dependency follows an
upstream branch under ordinary pip resolution. Review and record that resolved commit for reproducible
metric environments. Neither the package audit nor the notebook fingerprint audits the external support
pack code, offline wheels, model contents, or the Kaggle runtime.

## Publication checks

```bash
python -m pip install -e '.[dev,torch,metric,security]' matplotlib
python scripts/check_release_security.py --history
bandit -r src scripts -ll
pip-audit --skip-editable
pytest -q
python scripts/prepare_notebook.py --check
python scripts/prepare_safe_notebook.py
```

The release scanner includes tracked and non-ignored new files, and optionally every blob reachable from
local Git refs. Fetch current refs before reviewing history. Its findings redact secret values. The
`.secrets.baseline` exceptions are only reviewed code/model SHA-256 fingerprints, matched by file, detector
and hashed value. Review changes to that file manually; never regenerate it just to make CI pass.
Notebooks with code outputs or execution counts, private-data directories, common model formats and new
secrets fail the check. File-based scans do not prove absence of all private information; inspect new images,
datasets and notebook metadata before publication. Raw competition data and annotations must stay out of Git.

CI repeats these checks on pushes, pull requests and weekly. Actions use pinned commits and read-only
repository permissions. Dependabot checks Python dependencies and Actions updates.

## Review on 2026-10-01 (JST)

- During the initial public review, secret scanning, push protection, Dependabot alerts/security
  updates and private vulnerability reporting were enabled; no open alerts were returned. After PR #1
  was merged, the repository was reconfirmed **private**. GitHub secret-scanning/private-reporting APIs
  are currently unavailable for this private repository; local and CI scans remain active. Recheck the
  GitHub protections before any future visibility change. Dependabot security updates remain enabled.
- All 9 reachable commits (94 distinct file/blob pairs at the reviewed base `3430042`) scanned; findings
  were only the reviewed SHA-256 fingerprints. Current publication files pass the same scanner.
- Bandit: no medium/high findings in `src/` and `scripts/`. Three low findings are the scanner's fixed Git
  subprocess calls; they use argument arrays, no shell, and do not interpolate candidate file contents.
- The resolved Python 3.12 environment with dev, torch, metric, matplotlib and security tooling had no
  reported known dependency vulnerabilities. The local package and `tracking-cellmot` are not on PyPI and
  cannot be covered by pip-audit; this is not a whole-program safety certification.
- Security regression tests cover malicious pickle refusal, old PyTorch rejection, manifest traversal /
  symlink rejection, and restricted-notebook generation. The historical submission fingerprint is unchanged.
- The older working environment's pip 24.3.1 was updated to 26.2.1 after its audit reported vulnerabilities.

## Reporting a vulnerability

When available, use [GitHub private vulnerability reporting](https://github.com/YamauchiYudai/biohub-cell-tracking-solution/security/advisories/new).
While the repository is private, report findings directly to its owner through an existing private channel.
Do not put tokens, private microscopy data or exploit checkpoints in a public issue. If a real credential is
found in history, revoke/rotate it first; deleting the visible file is not sufficient.
