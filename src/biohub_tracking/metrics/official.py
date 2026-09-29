"""Local scoring with the organisers' official metric (`tracking_cellmot`, royerlab/kaggle-cell-tracking-competition).

This module implements *no* metric math of its own - no TP/FP/FN counting, no 7 um bipartite matching, no
adjusted-Jaccard node-count penalty. All of that comes from `tracking_cellmot.metrics`:

  - `evaluate(pred, gt, scale, max_distance)` matches `pred` against `gt` and returns edge / division
    TP / FP / FN counts;
  - `per_sample_metrics(result, n_total, node_recall)` derives edge Jaccard and the adjusted edge Jaccard of
    one dataset;
  - `summarise(rows)` aggregates: the adjusted edge Jaccard is a (TP+FP+FN)-weighted mean over datasets,
    the division Jaccard is micro-averaged over all datasets. Because the two terms aggregate differently,
    we always call `summarise` instead of re-implementing it.

    score = adjusted edge Jaccard + 0.1 x division Jaccard
    adjusted edge Jaccard (one movie) = edge Jaccard x (1 - 0.1 x (n_pred - n_est) / n_est)

What this module owns: converting our `TrackGraph` to the tracksdata graph the scorer expects, integerizing
predictions through the submission schema before scoring (so local numbers use the same voxel grid as a real
submission), and reporting per-embryo scores next to the overall one. Train movies come from two embryos
(44b6, 6bba) and 6bba carries most of the edges, so an overall number alone hides the weaker embryo.
"""

from __future__ import annotations

from dataclasses import dataclass

from .. import MATCH_RADIUS_UM
from ..io.graph import TrackGraph
from ..io.submission import roundtrip_integer_graph


def to_tracksdata_graph(graph: TrackGraph):
    """Convert a TrackGraph into the tracksdata graph the official metric expects.

    Mirrors `build_graph_from_rows` in the organisers' `csv_to_geffs.py` (down to the -999999.0 fill value),
    which is the path a real Kaggle submission goes through.
    """
    import polars as pl
    import tracksdata as td

    g = td.graph.InMemoryGraph()
    for key in ("z", "y", "x"):
        g.add_node_attr_key(key, pl.Float64, -999999.0)
    assigned = g.bulk_add_nodes([
        {"t": int(row.t), "z": float(row.z), "y": float(row.y), "x": float(row.x)}
        for row in graph.nodes.itertuples(index=False)
    ])
    id_map = dict(zip(graph.nodes["node_id"].to_numpy().tolist(), assigned, strict=True))
    if len(graph.edges):
        g.bulk_add_edges([
            {"source_id": id_map[int(s)], "target_id": id_map[int(t)]}
            for s, t in zip(graph.edges["source_id"], graph.edges["target_id"], strict=True)
        ])
    return g


def evaluate_dataset(pred: TrackGraph, gt: TrackGraph, voxel_scale: tuple[float, float, float],
                     max_distance: float = MATCH_RADIUS_UM, *, roundtrip_submission: bool = True) -> dict:
    """Score one dataset; returns a `per_sample_metrics` row ready for `summarise`."""
    from tracking_cellmot.metrics import evaluate, node_recall, per_sample_metrics

    if roundtrip_submission:
        pred = roundtrip_integer_graph(pred)
    pred_graph = to_tracksdata_graph(pred)
    gt_graph = to_tracksdata_graph(gt)
    result = evaluate(graph=pred_graph, gt_graph=gt_graph, scale=voxel_scale, max_distance=max_distance)
    recall = (node_recall(pred_graph, gt_graph)
              if pred_graph.num_edges() > 0 and pred_graph.num_nodes() > 0 else 0.0)
    n_total = gt.estimated_number_of_nodes
    n_total = float(n_total) if n_total is not None else float("nan")
    return per_sample_metrics(result, n_total, recall)


@dataclass
class EmbryoScore:
    embryo_id: str
    n_datasets: int
    score: float
    edge_jaccard: float
    adj_edge_jaccard: float
    division_jaccard: float
    division_tp: int
    division_fp: int
    division_fn: int


@dataclass
class ScoreReport:
    per_dataset: list[dict]
    per_embryo: dict[str, EmbryoScore]
    official_score: float
    official_adj_edge_jaccard: float
    official_division_jaccard: float
    score_worst: float
    worst_embryo: str
    skipped: list[str]

    def summary_line(self) -> str:
        parts = ", ".join(f"{eid}={s.score:.4f}" for eid, s in sorted(self.per_embryo.items()))
        line = (f"official_score={self.official_score:.5f} "
                f"(adj_edge_J={self.official_adj_edge_jaccard:.5f}, div_J={self.official_division_jaccard:.4f}) "
                f"| per embryo: {parts} | worst={self.worst_embryo}")
        if self.skipped:
            line += f" | SKIPPED (no prediction): {', '.join(self.skipped)}"
        return line


def aggregate(rows: list[dict], skipped: list[str] | None = None) -> ScoreReport:
    """Aggregate per-dataset rows (each carrying `dataset` and `embryo_id`) with the official `summarise`."""
    from tracking_cellmot.metrics import summarise

    if not rows:
        raise ValueError("no rows to aggregate")
    per_embryo: dict[str, EmbryoScore] = {}
    for embryo_id in sorted({r["embryo_id"] for r in rows}):
        s = summarise([r for r in rows if r["embryo_id"] == embryo_id])
        per_embryo[embryo_id] = EmbryoScore(
            embryo_id=embryo_id, n_datasets=sum(r["embryo_id"] == embryo_id for r in rows),
            score=s["score"], edge_jaccard=s["edge_jaccard"], adj_edge_jaccard=s["adj_edge_jaccard"],
            division_jaccard=s["division_jaccard"], division_tp=s["division_tp"],
            division_fp=s["division_fp"], division_fn=s["division_fn"])
    worst = min(per_embryo, key=lambda k: per_embryo[k].score)
    overall = summarise(rows)
    return ScoreReport(per_dataset=rows, per_embryo=per_embryo, official_score=overall["score"],
                       official_adj_edge_jaccard=overall["adj_edge_jaccard"],
                       official_division_jaccard=overall["division_jaccard"],
                       score_worst=per_embryo[worst].score, worst_embryo=worst, skipped=list(skipped or []))


def score_graphs(predictions: dict[str, TrackGraph], ground_truths: dict[str, TrackGraph],
                 voxel_scale: tuple[float, float, float], max_distance: float = MATCH_RADIUS_UM) -> ScoreReport:
    """Score every dataset present in both mappings. A dataset with GT but no prediction is reported as
    skipped rather than silently dropped (pass `TrackGraph.empty()` to score it as a full miss)."""
    names = sorted(set(predictions) & set(ground_truths))
    skipped = sorted(set(ground_truths) - set(predictions))
    rows = []
    for name in names:
        row = evaluate_dataset(predictions[name], ground_truths[name], voxel_scale, max_distance)
        row["dataset"] = name
        row["embryo_id"] = name.split("_")[0]
        rows.append(row)
    if not rows:
        raise ValueError("no overlapping dataset names between predictions and ground truths")
    return aggregate(rows, skipped)


def node_ratio(pred: TrackGraph, gt: TrackGraph) -> float:
    """(n_pred - n_est) / n_est, the quantity the node-count factor penalises (nan without an estimate)."""
    n_est = gt.estimated_number_of_nodes
    return float("nan") if not n_est else float((pred.num_nodes - n_est) / n_est)


__all__ = ["EmbryoScore", "ScoreReport", "aggregate", "evaluate_dataset", "node_ratio", "score_graphs",
           "to_tracksdata_graph"]
