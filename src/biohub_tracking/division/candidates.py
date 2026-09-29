"""Candidate forks P -> {D1, D2} on the graph that enters the rule-based safe-division stage.

A candidate is a possible division: P at frame t already continues to D1 at t+1 in the graph entering
safe-div (``pre_safe``), and D2 at t+1 would become the second daughter.

shape a  D2 is an existing node at t+1 with no parent (a track start). The geometry window is wider than
         safe-div's (P->D1 <= 12 um, P->D2 <= 12 um, D1-D2 <= 16 um; safe-div uses 10 / 9 / 14 plus mutual
         nearest neighbours, t+2 divergence, symmetry, a DeepCenter veto and per-frame / global caps).
shape d  D2 is not a node: a peak of the low-score detection dump (score >= 0.3) at t+1, >= 3 um from every
         node, P->D2 <= 9 um, angle D1-P-D2 >= 90 deg, D1-D2 <= 16 um.

The production submission uses shape a only. Enumeration never looks at ground truth; labels under the
official division rules live in :mod:`biohub_tracking.division.labels`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

SCALE = np.array([1.625, 0.40625, 0.40625])
A_P_D1_UM, A_P_D2_UM, A_SISTER_UM = 12.0, 12.0, 16.0
D_P_D2_UM, D_MIN_ANGLE, D_FREE_UM, LOW_MIN_SCORE = 9.0, 90.0, 3.0, 0.3
TRACK_LOOK = 10

COLUMNS = {
    "shape": "a = D2 is an existing track start; d = D2 is a low-score dump peak (a new node)",
    "t": "frame of P (D1, D2 are at t+1)",
    "p_id / d1_id / d2_id": "node ids (d2_id < 0 for shape d: -(row of the low dump + 1))",
    "p_t, p_z, p_y, p_x / d1_* / d2_*": "coordinates in native voxels (z 1.625 um, y/x 0.40625 um)",
    "p_d1_um, p_d2_um, sister_um": "distances P-D1, P-D2, D1-D2 in um",
    "angle_deg": "angle D1-P-D2 in um space (GT divisions: median 138 deg)",
    "len_ratio": "min(P-D1, P-D2) / max(P-D1, P-D2)",
    "mirror_um": "distance from D2 to the mirror of D1 through P (2P - D1): 0 for a perfectly opposite sister",
    "mirror_rank": "rank of this D2 among P's candidates by mirror_um (0 = best)",
    "t2_sister_um": "distance between the t+2 children of D1 and D2 (NaN if either has no single child)",
    "t2_gain_um": "t2_sister_um - sister_um (sisters separating at t+2 is positive)",
    "p_back_len, d1_fwd_len, d2_fwd_len": f"single-link track length behind P / after D1 / after D2 (cap {TRACK_LOOK})",
    "low_score_d2": "detector score of the dump peak (shape d)",
    "in_safe_div": "safe-div already adds exactly this fork (P -> D2)",
    "p_in_safe_div": "safe-div adds a fork at P with another D2",
    "d2_in_safe_div": "safe-div gives D2 to another P",
}


def um_distance(a, b) -> float:
    return float(np.linalg.norm((np.asarray(a, float) - np.asarray(b, float)) * SCALE))


def angle_deg(p, a, b) -> float:
    """Angle a-p-b in micrometre space (NaN when a or b coincides with p)."""
    u, v = (np.asarray(a, float) - p) * SCALE, (np.asarray(b, float) - p) * SCALE
    den = float(np.linalg.norm(u) * np.linalg.norm(v))
    return float(np.degrees(np.arccos(np.clip(np.dot(u, v) / den, -1, 1)))) if den > 1e-8 else float("nan")


def low_peaks(cache: dict | None):
    """(dump row, (t, z, y, x) native voxels, score) of the low-score detection dump at score >= 0.3."""
    if cache is None or "low_coords" not in cache:
        return None
    lc, ls = np.asarray(cache["low_coords"], float), np.asarray(cache["low_score"], float)
    keep = ls >= LOW_MIN_SCORE
    return np.where(keep)[0], lc[keep], ls[keep]


def enumerate_candidates(nodes: pd.DataFrame, edges: np.ndarray, selected: np.ndarray,
                         low=None) -> pd.DataFrame:
    """Shapes a / d with geometry and graph features, no ground truth.

    ``nodes``: node_id, t, z, y, x of the graph entering safe-div; ``edges``: its (source, target) pairs;
    ``selected``: the forks safe-div added (P, D2); ``low``: :func:`low_peaks` of the edge cache.
    """
    pos = nodes.set_index("node_id")[["t", "z", "y", "x"]]
    arr = pos[["z", "y", "x"]].to_numpy(float)
    xyz = dict(zip(pos.index.astype(int), arr, strict=True))
    tt = dict(zip(pos.index.astype(int), pos.t.astype(int), strict=True))
    out_pre: dict[int, list[int]] = {}
    inc_pre: dict[int, list[int]] = {}
    for s, t in edges:
        out_pre.setdefault(int(s), []).append(int(t))
        inc_pre.setdefault(int(t), []).append(int(s))
    sel = {(int(s), int(t)) for s, t in selected}
    sel_by_p = {s: t for s, t in sel}
    sel_by_d = {t: s for s, t in sel}
    by_t = {int(t): g.node_id.astype(int).to_numpy() for t, g in nodes.groupby("t")}

    def fwd(n):
        k = 0
        while k < TRACK_LOOK and len(out_pre.get(n, [])) == 1:
            n = out_pre[n][0]
            k += 1
        return k

    def back(n):
        k = 0
        while k < TRACK_LOOK and len(inc_pre.get(n, [])) == 1:
            n = inc_pre[n][0]
            k += 1
        return k

    rows = []
    for t in sorted(by_t):
        nxt = by_t.get(t + 1)
        if nxt is None:
            continue
        orphans = np.asarray([n for n in nxt if n not in inc_pre], dtype=np.int64)
        otree = cKDTree(np.stack([xyz[n] for n in orphans]) * SCALE) if len(orphans) else None
        free = None
        if low is not None:
            m = low[1][:, 0] == t + 1
            if m.any():
                pts = low[1][m][:, 1:]
                dist, _ = cKDTree(np.stack([xyz[n] for n in nxt]) * SCALE).query(pts * SCALE)
                f = dist >= D_FREE_UM
                if f.any():
                    free = (low[0][m][f], pts[f], low[2][m][f])
        ftree = cKDTree(free[1] * SCALE) if free is not None else None
        for p in by_t[t]:
            p = int(p)
            ch = out_pre.get(p, [])
            if len(ch) != 1 or tt[ch[0]] != t + 1:
                continue
            d1 = ch[0]
            P, D1 = xyz[p], xyz[d1]
            p_d1 = um_distance(P, D1)
            if p_d1 > A_P_D1_UM:
                continue
            cands = []
            if otree is not None:
                for i in otree.query_ball_point(P * SCALE, r=A_P_D2_UM):
                    d2 = int(orphans[i])
                    if um_distance(D1, xyz[d2]) <= A_SISTER_UM:
                        cands.append(("a", d2, xyz[d2], np.nan))
            if ftree is not None:
                for i in ftree.query_ball_point(P * SCALE, r=D_P_D2_UM):
                    L = free[1][i]
                    if um_distance(D1, L) <= A_SISTER_UM and angle_deg(P, D1, L) >= D_MIN_ANGLE:
                        cands.append(("d", -int(free[0][i]) - 1, L, float(free[2][i])))
            for shape, d2, D2, low_score in cands:
                p_d2, sis = um_distance(P, D2), um_distance(D1, D2)
                c1 = out_pre.get(d1, [])
                c2 = out_pre.get(d2, []) if shape == "a" else []
                t2 = um_distance(xyz[c1[0]], xyz[c2[0]]) if len(c1) == 1 and len(c2) == 1 else np.nan
                rows.append({
                    "shape": shape, "t": t, "p_id": p, "d1_id": d1, "d2_id": d2,
                    "p_t": t, "p_z": P[0], "p_y": P[1], "p_x": P[2],
                    "d1_t": t + 1, "d1_z": D1[0], "d1_y": D1[1], "d1_x": D1[2],
                    "d2_t": t + 1, "d2_z": D2[0], "d2_y": D2[1], "d2_x": D2[2],
                    "p_d1_um": p_d1, "p_d2_um": p_d2, "sister_um": sis, "angle_deg": angle_deg(P, D1, D2),
                    "len_ratio": min(p_d1, p_d2) / max(p_d1, p_d2, 1e-6),
                    "mirror_um": um_distance(D2, 2.0 * P - D1),
                    "t2_sister_um": t2, "t2_gain_um": t2 - sis if np.isfinite(t2) else np.nan,
                    "p_back_len": back(p), "d1_fwd_len": fwd(d1), "d2_fwd_len": fwd(d2) if shape == "a" else 0,
                    "low_score_d2": low_score,
                    "in_safe_div": int((p, d2) in sel),
                    "p_in_safe_div": int(p in sel_by_p and sel_by_p[p] != d2),
                    "d2_in_safe_div": int(d2 in sel_by_d and sel_by_d[d2] != p),
                })
    df = pd.DataFrame(rows)
    if len(df):
        df["mirror_rank"] = df.groupby("p_id").mirror_um.rank(method="first").astype(int) - 1
    return df
