"""Candidate features read from the per-movie edge cache written during inference.

The edge cache (``edge_cache/<movie>.npz``) stores every detection handed to the ILP (``coords``; ILP node
ids are its row numbers), the association Transformer's candidate edges above the cache threshold
(``edge_src``, ``edge_tgt``, ``edge_prob``) and the low-score detector peaks (``low_coords``, ``low_score``).
Post-processing can add nodes (re-admitted / gap-filled detections) whose ids are past the cached rows.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def cache_alignment(nodes: pd.DataFrame, edge_cache: dict | None) -> bool:
    """True when node ids index the cached ILP coordinates (checked on the first 200 nodes)."""
    if edge_cache is None or "coords" not in edge_cache:
        return False
    coords = np.asarray(edge_cache["coords"], float)
    index = nodes.set_index("node_id")[["t", "z", "y", "x"]]
    return all(int(n) >= len(coords) or np.allclose(coords[int(n)], row.to_numpy(float))
               for n, row in index.iloc[:200].iterrows() if int(n) >= 0)


def add_edge_probabilities(cands: pd.DataFrame, nodes: pd.DataFrame, edge_cache: dict | None) -> pd.DataFrame:
    """Add ``prob_p_d1``, ``prob_p_d2`` (0 when the pair was not cached) and their ``*_observed`` flags.

    Probabilities are used only when the cache is aligned with the node ids; the maximum is taken over
    duplicate cached pairs (the dual-seed ensemble can emit a pair twice).
    """
    if cands.empty:
        return cands
    prob: dict[tuple[int, int], float] = {}
    if cache_alignment(nodes, edge_cache) and all(k in edge_cache for k in ("edge_src", "edge_tgt", "edge_prob")):
        for s, d, p in zip(edge_cache["edge_src"], edge_cache["edge_tgt"], edge_cache["edge_prob"], strict=True):
            key = (int(s), int(d))
            prob[key] = max(prob.get(key, 0.0), float(p))
    rows = list(cands.itertuples())
    cands["prob_p_d1"] = [prob.get((int(r.p_id), int(r.d1_id)), 0.0) for r in rows]
    cands["prob_p_d2"] = [prob.get((int(r.p_id), int(r.d2_id)), 0.0) if r.shape != "d" else 0.0 for r in rows]
    cands["prob_p_d1_observed"] = [int((int(r.p_id), int(r.d1_id)) in prob) for r in rows]
    cands["prob_p_d2_observed"] = [int((int(r.p_id), int(r.d2_id)) in prob) if r.shape != "d" else 0 for r in rows]
    return cands
