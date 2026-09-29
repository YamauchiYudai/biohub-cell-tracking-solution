#!/usr/bin/env python3
"""Check a submission CSV and score it locally against train ground truth with the official metric.

  # schema + graph checks only (no extra dependencies)
  python scripts/evaluate.py --submission submission.csv --check-only

  # official score per movie / per embryo (needs: pip install -e ".[metric]")
  python scripts/evaluate.py --submission submission.csv --gt-dir ${DATA_ROOT}/train --out report.json

Scoring integerizes predictions through the submission schema first, exactly as a Kaggle submission is read.
Movies of the train set are in-sample for every public pretrained model used by the pipeline, so a local
score on train movies is a diagnostic, not an unbiased estimate (see docs/validation.md).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from biohub_tracking import VOXEL_SCALE_UM  # noqa: E402
from biohub_tracking.io.graph import validate_forest  # noqa: E402
from biohub_tracking.io.submission import read_submission  # noqa: E402


def check(graphs: dict) -> list[str]:
    problems = []
    for name, g in graphs.items():
        v = validate_forest(g)
        if not v.is_valid:
            problems.append(f"{name}: merges={len(v.merge_node_ids)} >2-children={len(v.multi_branch_node_ids)} "
                            f"dt!=1={len(v.dt_violations)} self-loops={len(v.self_loop_node_ids)} "
                            f"dangling={len(v.dangling_edge_ids)}")
        if g.nodes.node_id.duplicated().any():
            problems.append(f"{name}: duplicate node ids")
    return problems


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--submission", type=Path, required=True)
    ap.add_argument("--gt-dir", type=Path, help="directory with <movie>.geff ground truth (competition train)")
    ap.add_argument("--datasets", nargs="*", help="score only these movies")
    ap.add_argument("--check-only", action="store_true")
    ap.add_argument("--out", type=Path, help="write per-movie rows and the summary as JSON")
    a = ap.parse_args()

    graphs = read_submission(a.submission)
    if a.datasets:
        graphs = {k: v for k, v in graphs.items() if k in set(a.datasets)}
    problems = check(graphs)
    n_nodes = sum(g.num_nodes for g in graphs.values())
    n_edges = sum(g.num_edges for g in graphs.values())
    print(f"{len(graphs)} datasets, {n_nodes} nodes, {n_edges} edges; graph checks: "
          f"{'OK' if not problems else f'{len(problems)} problems'}")
    for p in problems:
        print("  ", p)
    if a.check_only:
        return 0 if not problems else 1
    if a.gt_dir is None:
        ap.error("--gt-dir is required unless --check-only")

    from biohub_tracking.io.geff import read_geff
    from biohub_tracking.metrics.official import score_graphs

    gts = {name: read_geff(a.gt_dir / f"{name}.geff") for name in graphs if (a.gt_dir / f"{name}.geff").exists()}
    report = score_graphs(graphs, gts, VOXEL_SCALE_UM)
    print(report.summary_line())
    for eid, s in sorted(report.per_embryo.items()):
        print(f"  {eid}: {s.n_datasets} movies, score {s.score:.5f}, adj edge J {s.adj_edge_jaccard:.5f}, "
              f"division TP/FP/FN {s.division_tp}/{s.division_fp}/{s.division_fn}")
    if a.out:
        a.out.write_text(json.dumps({"summary": {"official_score": report.official_score,
                                                 "adj_edge_jaccard": report.official_adj_edge_jaccard,
                                                 "division_jaccard": report.official_division_jaccard,
                                                 "per_embryo": {k: vars(v) for k, v in report.per_embryo.items()}},
                                     "per_dataset": report.per_dataset}, indent=1, default=float))
    return 0 if not problems else 1


if __name__ == "__main__":
    raise SystemExit(main())
