"""The division hook that runs once per movie right after the rule-based safe-div stage.

The notebook wraps the upstream ``add_safe_divisions_postlink``: it keeps the graph entering safe-div
(``pre_edges``), runs safe-div, then calls :meth:`DivisionProcessor.process` on safe-div's output. The
processor enumerates candidate forks, lets a scorer add score columns, chooses adds / removes and edits the
edge list. It never creates or deletes nodes for shape-a candidates (the only shape used in production).
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

from . import candidates as division_candidates
from . import select as division_select

# safe-div's division budget in the upstream notebook (BIOHUB_SAFE_DIV_FRAME_FRAC_CAP / _GLOBAL_FRAC_CAP)
FRAME_FRACTION = 0.0076
EDGE_FRACTION = 0.00375


def edge_pairs(edges: list[dict]) -> np.ndarray:
    return np.asarray([(int(e["source_id"]), int(e["target_id"])) for e in edges], dtype=np.int64).reshape(-1, 2)


def node_table(nodes_by_id: dict) -> pd.DataFrame:
    return pd.DataFrame([{"node_id": int(i), "t": int(n["t"]), "z": float(n["z"]),
                          "y": float(n["y"]), "x": float(n["x"])} for i, n in nodes_by_id.items()])


def selected_pairs(pre_edges: list[dict], safe_edges: list[dict]) -> np.ndarray:
    """The forks safe-div added: edges present after safe-div but not before."""
    before = set(map(tuple, edge_pairs(pre_edges).tolist()))
    return np.asarray([pair for pair in edge_pairs(safe_edges).tolist() if tuple(pair) not in before],
                      dtype=np.int64).reshape(-1, 2)


def geometry_filter(cands: pd.DataFrame, *, max_mirror_um: float | None = None,
                    min_p_d2_um: float | None = None, max_p_d2_um: float | None = None,
                    min_angle_deg: float | None = None) -> pd.DataFrame:
    """Filter only add candidates. Existing safe-div forks always remain available for removal."""
    if cands.empty:
        return cands.copy()
    keep = cands.in_safe_div.astype(bool).to_numpy()
    eligible = cands["shape"].eq("a").to_numpy(copy=True)
    for col, bound, op in (("mirror_um", max_mirror_um, np.less_equal),
                           ("p_d2_um", min_p_d2_um, np.greater_equal),
                           ("p_d2_um", max_p_d2_um, np.less_equal),
                           ("angle_deg", min_angle_deg, np.greater_equal)):
        if bound is not None:
            eligible &= op(pd.to_numeric(cands[col], errors="coerce").to_numpy(float), bound)
    return cands.loc[keep | eligible].copy()


def cap_additions(adds: pd.DataFrame, nodes: pd.DataFrame, pre_edges: list[dict], safe_edges: list[dict],
                  removes: pd.DataFrame, score: str, *, frame_fraction: float = FRAME_FRACTION,
                  edge_fraction: float = EDGE_FRACTION) -> pd.DataFrame:
    """Apply safe-div's per-frame and global division limits to the learned adds (highest score first)."""
    if adds.empty:
        return adds
    frame_counts = nodes.t.value_counts().to_dict()
    removed = {(int(r.p_id), int(r.d2_id)) for r in removes.itertuples()}
    already = np.asarray([pair for pair in selected_pairs(pre_edges, safe_edges).tolist()
                          if tuple(pair) not in removed], dtype=np.int64).reshape(-1, 2)
    by_id = nodes.set_index("node_id").t.to_dict()
    existing_by_frame: dict[int, int] = {}
    for parent, _ in already:
        t = int(by_id[int(parent)])
        existing_by_frame[t] = existing_by_frame.get(t, 0) + 1
    global_cap = max(1, round(max(1, len(pre_edges)) * edge_fraction)) - len(already)
    if global_cap <= 0:
        return adds.iloc[:0]
    ranked = adds.sort_values(score, ascending=False, kind="stable")
    taken: dict[int, int] = {}
    keep = []
    for r in ranked.itertuples():
        t = int(r.p_t)
        cap = max(1, round(frame_counts.get(t, 0) * frame_fraction)) - existing_by_frame.get(t, 0)
        if len(keep) >= global_cap:
            break
        if taken.get(t, 0) < cap:
            keep.append(r.Index)
            taken[t] = taken.get(t, 0) + 1
    return adds.loc[keep]


class DivisionProcessor:
    """One decision path on a movie's post-safe-div graph.

    ``score_fn(cands, nodes, pre_edge_pairs, dataset, cache)`` returns the candidate table with a
    ``score_column`` added; it must not use labels. ``add_cap_scale`` scales the division budget the learned
    adds share with safe-div's forks (1.0: safe-div's own limits; ``None``: no limit on the learned adds,
    the production setting "V5a"). Safe-div's own forks keep their cap either way.
    """

    def __init__(self, *, score_fn: Callable, score_column: str = "score",
                 tau_add: float | None = None, tau_del: float | None = None,
                 geometry: dict | None = None, cache_fn: Callable | None = None,
                 add_cap_scale: float | None = 1.0):
        if add_cap_scale is not None and add_cap_scale <= 0:
            raise ValueError(f"add_cap_scale must be positive or None, got {add_cap_scale}")
        self.score_fn = score_fn
        self.score_column = score_column
        self.tau_add = tau_add
        self.tau_del = tau_del
        self.geometry = geometry or {}
        self.cache_fn = cache_fn
        self.add_cap_scale = add_cap_scale
        self.log: dict[str, dict] = {}

    def process(self, nodes_by_id: dict, safe_edges: list[dict], stats: dict, dataset: str,
                pre_edges: list[dict]) -> list[dict]:
        if not dataset:
            return safe_edges
        nodes = node_table(nodes_by_id)
        pre = edge_pairs(pre_edges)
        selected = selected_pairs(pre_edges, safe_edges)
        cache = self.cache_fn(dataset) if self.cache_fn is not None else None
        cands = division_candidates.enumerate_candidates(nodes, pre, selected, division_candidates.low_peaks(cache))
        cands = geometry_filter(cands, **self.geometry)
        if cands.empty:
            self.log[dataset] = {"candidates": 0, "adds": 0, "removes": 0}
            return safe_edges
        cands = self.score_fn(cands, nodes, pre, dataset, cache)
        if self.score_column not in cands:
            raise ValueError(f"scorer did not provide {self.score_column}")
        adds, removes = division_select.choose(cands, self.score_column, self.tau_add, self.tau_del, shapes=("a",))
        n_chosen = len(adds)
        if self.add_cap_scale is not None:
            adds = cap_additions(adds, nodes, pre_edges, safe_edges, removes, self.score_column,
                                 frame_fraction=FRAME_FRACTION * self.add_cap_scale,
                                 edge_fraction=EDGE_FRACTION * self.add_cap_scale)
        if {"c", "d"} & set(adds["shape"]):
            raise ValueError("only existing shape-a daughters may be added")
        safe_pairs = {(int(e["source_id"]), int(e["target_id"])) for e in safe_edges if e.get("safe_division")}
        missing = {(int(r.p_id), int(r.d2_id)) for r in removes.itertuples()} - safe_pairs
        if missing:
            raise ValueError(f"{dataset}: safe-div removals absent: {sorted(missing)[:5]}")
        n0 = len(nodes_by_id)
        out = division_select.apply_forks(nodes_by_id, safe_edges, adds, removes, stats)
        if len(nodes_by_id) != n0:
            raise ValueError(f"{dataset}: division postprocess changed node count")
        self.log[dataset] = {"candidates": int(len(cands)), "adds": int(len(adds)), "removes": int(len(removes)),
                             "nodes": n0, "edges_before": len(safe_edges), "edges_after": len(out),
                             "adds_before_cap": int(n_chosen)}
        return out
