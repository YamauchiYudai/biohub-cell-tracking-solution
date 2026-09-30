#!/usr/bin/env python3
"""Generate a restricted-loading derivative; never overwrite the historical submission."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import prepare_notebook

from biohub_tracking.kaggle import notebook as nbk
from biohub_tracking.security import MIN_TORCH

GUARD = f'''# Security preflight: run before every other cell, including subprocess launches.
import os
import torch
from packaging.version import Version
if Version(torch.__version__) < Version({MIN_TORCH!r}):
    raise RuntimeError("Upgrade to torch>={MIN_TORCH} before running this notebook")
os.environ.pop("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", None)
os.environ["TORCH_FORCE_WEIGHTS_ONLY_LOAD"] = "1"
'''


def hardened_notebook(nb, cfg):
    cells = nbk.code_cells(nb)
    problems = prepare_notebook.check_cells(cfg["notebook"]["candidate"], cells, cfg)
    if problems:
        raise ValueError("Source notebook failed submission parity: " + "; ".join(problems))
    # This occurs inside the embedded legacy Plan-B scorer, which selected submissions do not use.
    legacy = "artifact = pickle.loads(Path(path).read_bytes())"
    if sum(c.count(legacy) for c in cells) != 1:
        raise ValueError("Legacy pickle anchor changed; review the notebook before hardening")
    cells = [c.replace(legacy, "raise RuntimeError(\"Legacy pickle models are disabled\")")
             .replace("weights_only=False", "weights_only=True") for c in cells]
    # Use the same first cell rather than a separate optional cell; child processes inherit this setting.
    cells[0] = GUARD + "\n" + cells[0]
    notes = {0: "# Restricted-loading inference notebook\n\n"
             "Generated from the hash-verified submission. This derivative changes model loading and is "
             "not byte-identical to the competition submission. Use only reviewed input datasets and "
             "support-pack code in an isolated Kaggle session without credentials. Run all cells in order. "
             "Unsupported legacy checkpoints fail closed; never disable the restrictions to load them. "
             "GPU/output parity of this derivative must be verified separately. See SECURITY.md.\n"}
    result = nbk.publishable(nb, cells, notes)
    for i, source in enumerate(cells):
        compile(source, f"safe_cell_{i}", "exec")
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("outputs/safe_submission.ipynb"))
    args = ap.parse_args()
    cfg = prepare_notebook.load_config()
    source = prepare_notebook.ROOT / cfg["notebook"]["path"]
    if args.out.resolve() == source.resolve():
        ap.error("The historical submission cannot be overwritten")
    result = hardened_notebook(nbk.load_notebook(source), cfg)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1) + "\n")
    print(f"Wrote {args.out}; model-loading restrictions enabled, GPU parity not yet verified")


if __name__ == "__main__":
    main()
