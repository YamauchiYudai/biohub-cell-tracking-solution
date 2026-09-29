#!/usr/bin/env python3
"""Train the team's learned components (needs the ``torch`` extra; a GPU for T3 in reasonable time).

T3 division CNN (production):
  python scripts/train.py t3-crops --train-dir ${DATA_ROOT}/train --out work/t3_crops
  python scripts/train.py t3 --mode holdout --crops work/t3_crops --out work/t3_holdout   # leak-free estimate
  python scripts/train.py t3 --mode inner   --crops work/t3_crops --out work/t3_inner     # scores for tau selection
  python scripts/train.py t3 --mode full    --crops work/t3_crops --out work/t3_full      # submission weights

V1284 coordinate head, recipe F03 (used by the selected submission fc_f03):
  python scripts/train.py coordinate-head --pairs work/v1284_pairs.npz --public-head v1284_head.pt \
      --out work/head_f03 [--evaluate]

The pairs file holds, per matched detection, the frozen U-Net features at the fused detection (``features``,
N x 224), the detection and its GT partner on the model grid (``det``, ``gt``, N x 3, isotropic 1.625 um) and
the movie stem (``stem``). It is produced by a capture run of the notebook's inference stage.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def cmd_t3_crops(a) -> int:
    from biohub_tracking.division import crops

    manifest = crops.export(a.train_dir, a.out, a.stems, a.max_neg)
    print(json.dumps(manifest["totals"], indent=1))
    return 0


def cmd_t3(a) -> int:
    from biohub_tracking.division import t3

    seeds = [int(s) for s in a.seeds.split(",")]
    if a.mode == "holdout":
        report = t3.run_holdout(a.crops, a.out, a.epochs, seeds, a.width, a.device)["folds"]
    elif a.mode == "inner":
        report = t3.run_inner(a.crops, a.out, a.epochs, a.width, a.device, folds=a.folds)["metrics"]
    else:
        report = t3.run_full(a.crops, a.out, a.epochs, seeds, a.width, a.device)
    print(json.dumps(report, indent=1))
    return 0


def cmd_coordinate_head(a) -> int:
    import numpy as np

    from biohub_tracking.detection import coordinate_head as ch

    d = np.load(a.pairs)
    features = d["features"].astype(np.float32)
    target = ((d["gt"] - d["det"]) * ch.GRID_UM).astype(np.float32)
    movies = np.asarray(d["stem"])
    keep = ~np.isin(movies, [s for s in a.exclude.split(",") if s])
    public = ch.load_head(a.public_head)
    report: dict = {"recipe": f"public-init, distill lam={a.lam}, AdamW lr 1e-3 wd 0, {ch.EPOCHS} full-batch epochs",
                    "n_pairs": int(keep.sum()), "n_movies": int(len(np.unique(movies[keep])))}
    if a.evaluate:
        report["evaluation"] = ch.evaluate(public, features[keep], target[keep], movies[keep], lam=a.lam)
    head = ch.finetune(public, features[keep], target[keep], lam=a.lam)
    a.out.mkdir(parents=True, exist_ok=True)
    path = a.out / "v1284_head.pt"
    ch.save_head(path, *head)
    report["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    (a.out / "v1284_head.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("t3-crops", help="export T3 training crops from annotated train movies")
    c.add_argument("--train-dir", type=Path, required=True)
    c.add_argument("--out", type=Path, required=True)
    c.add_argument("--stems", nargs="*")
    c.add_argument("--max-neg", type=int, default=30)
    t = sub.add_parser("t3", help="train the T3 division CNN")
    t.add_argument("--mode", choices=("holdout", "inner", "full"), required=True)
    t.add_argument("--crops", type=Path, required=True)
    t.add_argument("--out", type=Path, required=True)
    t.add_argument("--epochs", type=int, default=30)
    t.add_argument("--seeds", default="0,1,2")
    t.add_argument("--width", type=int, default=16)
    t.add_argument("--folds", type=int, default=3)
    t.add_argument("--device", default="cuda")
    h = sub.add_parser("coordinate-head", help="fine-tune the V1284 coordinate head (recipe F03)")
    h.add_argument("--pairs", type=Path, required=True)
    h.add_argument("--public-head", type=Path, required=True)
    h.add_argument("--out", type=Path, required=True)
    h.add_argument("--lam", type=float, default=0.3)
    h.add_argument("--exclude", default="", help="comma-separated movie stems to leave out")
    h.add_argument("--evaluate", action="store_true", help="also report movie-fold / cross-embryo error vs public")
    a = ap.parse_args()
    return {"t3-crops": cmd_t3_crops, "t3": cmd_t3, "coordinate-head": cmd_coordinate_head}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
