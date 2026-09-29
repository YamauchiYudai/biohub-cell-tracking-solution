"""Keep the rule-based safe-div forks, then add and remove forks by a learned division score.

choose (per movie)
  remove  a safe-div fork whose parent scores < tau_del (tau_del None: remove nothing)
  add     for each parent P scoring >= tau_add that has no fork: the candidate D2 closest to the mirror of D1
          through P (2P - D1), shape d only when no shape-a D2 is free; a D2 already used by a kept fork or an
          earlier add is skipped. Parents are taken in descending score, and an add is dropped when a kept or
          added fork lies within +-1 frame and 10 um of P (two forks on one division: the second is an FP).
apply_forks
  removes the chosen safe-div edges and adds P -> D2 edges (and for shape d a new node at the dump peak) on
  the graph returned by safe-div. Nothing else changes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

SCALE = np.array([1.625, 0.40625, 0.40625])
DEDUP_FRAMES = 1
DEDUP_UM = 10.0


def choose(cands: pd.DataFrame, score: str, tau_add: float | None, tau_del: float | None,
           shapes: tuple[str, ...] = ("a", "d")) -> tuple[pd.DataFrame, pd.DataFrame]:
    """``cands``: :func:`~biohub_tracking.division.candidates.enumerate_candidates` rows of ONE movie plus a
    per-parent ``score`` column. Returns (adds, removes) as candidate rows."""
    if cands.empty:
        return cands.iloc[:0], cands.iloc[:0]
    safe = cands[cands.in_safe_div == 1].drop_duplicates(["p_id", "d2_id"])
    removes = safe[safe[score] < tau_del] if tau_del is not None else safe.iloc[:0]
    kept = safe.drop(removes.index)
    used_d2 = set(kept.d2_id.astype(int))
    forked = set(kept.p_id.astype(int))
    anchors = [(int(r.p_t), np.array([r.p_z, r.p_y, r.p_x], float)) for r in kept.itertuples()]
    adds = []
    if tau_add is not None:
        pool = cands[(cands.in_safe_div == 0) & cands["shape"].isin(shapes) & (cands[score] >= tau_add)]
        pool = pool[~pool.p_id.isin(forked)]                     # a kept safe-div fork at P: P already divides
        pool = pool.assign(_pen=pool.mirror_um + np.where(pool["shape"] == "d", 1e3, 0.0))
        by_p = {p: g.sort_values("_pen") for p, g in pool.groupby("p_id", sort=False)}
        order = pool.groupby("p_id")[score].max().sort_values(ascending=False, kind="stable")
        for p in order.index:
            for r in by_p[p].itertuples():
                if int(r.d2_id) in used_d2:
                    continue
                P = np.array([r.p_z, r.p_y, r.p_x], float)
                if any(abs(int(r.p_t) - t) <= DEDUP_FRAMES and np.linalg.norm((P - q) * SCALE) <= DEDUP_UM
                       for t, q in anchors):
                    break
                adds.append(r.Index)
                used_d2.add(int(r.d2_id))
                anchors.append((int(r.p_t), P))
                break
    return cands.loc[adds], removes


def apply_forks(nodes_by_id: dict, edges: list[dict], adds: pd.DataFrame, removes: pd.DataFrame,
                stats: dict | None = None) -> list[dict]:
    """Edit the post-safe-div graph (the node dict in place for shape d); returns the new edge list."""
    drop = {(int(r.p_id), int(r.d2_id)) for r in removes.itertuples()}
    available = {(int(e["source_id"]), int(e["target_id"])) for e in edges if e.get("safe_division")}
    if not drop <= available:
        raise ValueError(f"safe-div removals absent: {drop - available}")
    out = [e for e in edges if not (e.get("safe_division") and (int(e["source_id"]), int(e["target_id"])) in drop)]
    present = {(int(e["source_id"]), int(e["target_id"])) for e in out}
    parented = {int(e["target_id"]) for e in out}
    outdegree: dict[int, int] = {}
    for e in out:
        p = int(e["source_id"])
        outdegree[p] = outdegree.get(p, 0) + 1
    next_id = max(int(n) for n in nodes_by_id) + 1 if nodes_by_id else 0
    added_nodes = 0
    for r in adds.itertuples():
        target = int(r.d2_id)
        if r.shape == "d":
            target = next_id
            next_id += 1
            nodes_by_id[target] = {"node_id": target, "t": int(r.d2_t), "z": float(r.d2_z), "y": float(r.d2_y),
                                   "x": float(r.d2_x), "learned_division_node": 1}
            added_nodes += 1
        elif target not in nodes_by_id:
            raise ValueError(f"division daughter {target} absent")
        if target in parented or (int(r.p_id), target) in present:
            raise ValueError(f"division daughter {target} already has a parent")
        if outdegree.get(int(r.p_id), 0) != 1:
            raise ValueError(f"division parent {r.p_id} must have one existing child")
        dist = float(np.linalg.norm((np.array([r.d2_z, r.d2_y, r.d2_x], float)
                                     - np.array([r.p_z, r.p_y, r.p_x], float)) * SCALE))
        out.append({"source_id": int(r.p_id), "target_id": target, "edge_prob": None, "distance_um": dist,
                    "safe_division": 1, "learned_division": 1})
        parented.add(target)
        present.add((int(r.p_id), target))
        outdegree[int(r.p_id)] = 2
    if stats is not None:
        stats["learned_div_added"] = stats.get("learned_div_added", 0) + len(adds)
        stats["learned_div_added_nodes"] = stats.get("learned_div_added_nodes", 0) + added_nodes
        stats["learned_div_removed"] = stats.get("learned_div_removed", 0) + len(removes)
    return out
