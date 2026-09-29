#!/usr/bin/env python3
"""Write or check notebooks/final_submission.ipynb for one of the final-day submission candidates.

The submission is a Kaggle notebook; inference runs there (GPU, offline, competition test set attached).
This script only prepares the publishable copy: code cells are kept byte for byte (their SHA-256 must match
configs/final.yaml), outputs and run metadata are dropped and explanatory markdown is added.

  # check the committed notebook against configs/final.yaml
  python scripts/prepare_notebook.py --check

  # convert the committed notebook into another candidate (no download needed)
  python scripts/prepare_notebook.py --candidate fc_f03

  # a candidate that is not an edit of the core (e.g. the teammate's v_add): pass its kernel notebook
  kaggle kernels pull <owner>/<kernel> -p /tmp/final
  python scripts/prepare_notebook.py --candidate v_add --source /tmp/final/<kernel>.ipynb
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from biohub_tracking.kaggle import notebook as nbk  # noqa: E402
from biohub_tracking.kaggle import variants  # noqa: E402

CONFIG = ROOT / "configs" / "final.yaml"
DEFAULTS = ROOT / "configs" / "default.yaml"

CELL_NOTES = {
    1: "## Configuration drift guard (upstream)\nStops the run if a core upstream setting was changed by mistake.",
    2: "## Paths, artifacts and inference settings (upstream)\nResolves the competition test set, the "
       "attached model datasets and the prediction options.",
    3: "## Offline dependencies (upstream)\nInstalls tracksdata, ILP solvers, geff, zarr ... from the attached "
       "support pack (the notebook runs without internet).",
    4: "## GPU inference (upstream, head source per candidate)\nDual-seed TemporalUNet3D detection with 8-view "
       "TTA, V1284 coordinate refinement of every detection, association Transformer edge probabilities "
       "(forward + reverse time, harmonic fusion) and ILP linking. Writes one predicted graph per test movie "
       "plus the edge cache the division gate reads.",
    5: "## Post-processing (upstream + team blocks)\nUpstream: edge filter, motion relink, re-admission, gap "
       "closing, low-score gap filling, rule-based safe division with the DeepCenter veto, short-track filter, "
       "line-fit smoothing, submission writer.\n\nTeam blocks inside this cell, in execution order:\n"
       "1. `X138_RELINK_HELPERS` - **b1c** jump-aware relink (`biohub_tracking.postprocessing.relink`).\n"
       "2. `_DV_RT` - the division runtime modules written to disk and imported "
       "(`biohub_tracking.division`).\n"
       "3. Division scorer and hook - the `prob_angle_t2` gate + **T3** CNN; runs right after safe-div "
       "(`biohub_tracking.division.scoring` / `processor`).\n"
       "4. `J2` - jump-aware line fit (`biohub_tracking.postprocessing.linefit`).{fc}",
    6: "## Retention guard (upstream)\nChecks the submission schema and the per-frame candidate retention "
       "diagnostics.",
    7: "## Built-in validator (upstream, disabled)\n`BIOHUB_VALIDATOR_ENABLE=0` in submissions: cells 7-10 "
       "do not run any scoring on the hidden test set.",
    11: "## Pipeline manifest (upstream)\nPrints the resolved state (ensemble, fusion, DeepCenter).",
    12: "## Relink manifest (team)\nRecords relink statistics and whether the repair deadline degraded any "
        "movie.",
    13: "## Division manifest and deadline assert (team)\nRecords the division settings and per-movie logs, and "
        "fails the run if the post-processing deadline degraded any movie (a degraded run must not be "
        "submitted).",
}


def load_config() -> dict:
    return yaml.safe_load(CONFIG.read_text())


def markdown_for(name: str, cand: dict, cfg: dict, cells: list[str]) -> dict[int, str]:
    lb = cand.get("public_lb")
    edits = cand.get("edits")
    head = cfg["inputs"].get(cand.get("head") or "", "see cell 4")
    intro = (
        "# Biohub Cell Tracking - final submission notebook\n\n"
        f"Candidate **`{name}`** (Kaggle submission {cand['submission_id']}, Public LB {lb}): {cand['summary']}.\n\n"
        "Built on the public notebook \"biohub x138\" by Anvith Pothula (Apache-2.0), which extends Teddy Tennant's "
        "\"frontier947 readmit v1\" and Reyhan Ksatria's 0.947 pipeline on pilkwang's pretrained models (all "
        "Apache-2.0 / CC0). Upstream cells are unchanged; the team's "
        "additions are the environment overrides at the end of this cell, the blocks marked in cell 5 and "
        "cells 12-13. See README.md and THIRD_PARTY_NOTICES.md.\n\n"
        "**Inputs to attach:** the competition data, "
        + ", ".join(f"`{d}`" for d in cfg["inputs"]["common"]) + f", `{head}`.\n\n"
        "Accelerator: GPU T4 x2, internet off. The notebook writes `/kaggle/working/submission.csv`.\n\n"
        f"Team edits on top of the core: {', '.join(edits) if edits else 'none'}.\n\n"
        "## Configuration (cell 0)\nUpstream settings (listed in `configs/default.yaml`), then the team's "
        "overrides (`configs/final.yaml`)."
    )
    if len(cells) != 14:
        return {0: intro + "\n\n(This candidate's cell layout differs from the core; cell notes omitted.)"}
    fc = ("\n5. `FC` - frozen-frame coordinate consensus (`biohub_tracking.postprocessing.consensus`)."
          if nbk.FC_BEGIN in cells[nbk.POSTPROC_CELL] else "")
    notes = {i: text.format(fc=fc) for i, text in CELL_NOTES.items()}
    return {0: intro, **notes}


def expected_env(cfg: dict, cand: dict) -> dict[str, str]:
    env = dict(yaml.safe_load(DEFAULTS.read_text())["upstream_env"])
    env.update(cfg["production_env"])
    for edit in cand.get("edits") or []:
        env.update(cfg.get("edit_env", {}).get(edit, {}))
    return env


def check_cells(name: str, cells: list[str], cfg: dict) -> list[str]:
    """Problems of ``cells`` as candidate ``name`` (empty = parity holds)."""
    cand = cfg["candidates"][name]
    problems = []
    sha = nbk.code_sha256(cells)
    if cand.get("code_sha256") and sha != cand["code_sha256"]:
        problems.append(f"code sha256 {sha} != configs/final.yaml {cand['code_sha256']}")
    if cand.get("edits") is not None:
        env = nbk.env_overrides(cells)
        for key, value in expected_env(cfg, cand).items():
            if env.get(key) != value:
                problems.append(f"cell 0 {key}={env.get(key)!r}, config says {value!r}")
    return problems


def cells_for(name: str, cfg: dict, source: Path | None) -> list[str]:
    cand = cfg["candidates"][name]
    if source is not None:
        return nbk.code_cells(nbk.load_notebook(source))
    if cand.get("edits") is None:
        raise SystemExit(f"{name} is not an edit of the core notebook: pass its kernel notebook with --source")
    committed = nbk.code_cells(nbk.load_notebook(ROOT / cfg["notebook"]["path"]))
    core = variants.to_core(committed)
    core_sha = cfg["candidates"]["core"]["code_sha256"]
    if nbk.code_sha256(core) != core_sha:
        raise SystemExit("the committed notebook does not reduce to the recorded core; refusing to build")
    return variants.build(core, cand["edits"])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--candidate", help="a key of `candidates` in configs/final.yaml")
    ap.add_argument("--source", type=Path, help="kernel notebook for a candidate that is not an edit of the core")
    ap.add_argument("--check", action="store_true", help="only check the committed notebook")
    ap.add_argument("--out", type=Path, help="write somewhere else instead of the committed notebook")
    a = ap.parse_args()
    cfg = load_config()
    path = ROOT / cfg["notebook"]["path"]

    if a.check:
        name = cfg["notebook"]["candidate"]
        nb = nbk.load_notebook(path)
        problems = check_cells(name, nbk.code_cells(nb), cfg)
        code = [c for c in nb["cells"] if c["cell_type"] == "code"]
        problems += [f"code cell {i} has outputs" for i, c in enumerate(code) if c.get("outputs")]
        final = cfg["result"]["final_submission"]
        if final not in ("TBD", None) and final != name:
            problems.append(f"result.final_submission is {final} but the notebook holds {name}")
        print(f"{path.relative_to(ROOT)}: candidate {name}, code sha256 {nbk.code_sha256(nbk.code_cells(nb))}")
        for p in problems:
            print("  PROBLEM:", p)
        print("OK" if not problems else "FAILED")
        return 0 if not problems else 1

    if not a.candidate:
        ap.error("--candidate or --check is required")
    if a.candidate not in cfg["candidates"]:
        ap.error(f"unknown candidate {a.candidate}; known: {', '.join(cfg['candidates'])}")
    cells = cells_for(a.candidate, cfg, a.source)
    problems = check_cells(a.candidate, cells, cfg)
    if problems:
        for p in problems:
            print("PROBLEM:", p)
        return 1
    template = nbk.load_notebook(a.source) if a.source else nbk.load_notebook(path)
    out = nbk.publishable(template, cells, markdown_for(a.candidate, cfg["candidates"][a.candidate], cfg, cells))
    target = a.out or path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")
    print(f"wrote {target} (candidate {a.candidate}, code sha256 {nbk.code_sha256(cells)})")
    if target == path and cfg["notebook"]["candidate"] != a.candidate:
        print(f"now set `notebook.candidate: {a.candidate}` in configs/final.yaml")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
