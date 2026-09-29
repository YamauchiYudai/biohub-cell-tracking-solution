"""Label candidate forks with the official division scorer's rules (validation only; needs the ``metric`` extra).

A candidate's label is what the official scorer (`tracking_cellmot.division_metrics`) would do with this one
fork added to the post-safe-div graph (pre-safe edges + safe-div's forks), replacing any safe-div fork that
uses the same P or the same D2:

  tp       the fork would be paired with a GT division (P or P's parent matched to the GT divider or its
           parent; the D1 and D2 lineages matched to the two GT daughter lineages; the division not yet paired)
  fp       a false positive: P matched to an annotated cell that continues (annotated_continuing), P near a GT
           division without pairing (near_gt_division), or the branches matched to two GT tracks (cross_gt_tracks)
  ignored  anything else (P on an unannotated cell): the scorer does not count it

GT is sparse, so most candidates are ``ignored``; only tp / fp rows can move the division term. Node matching
(7 um) is computed once per movie; the fork only changes edges. :func:`verify_labels` re-scores sampled forks
with the real ``score_divisions`` on the edited graph to check the labeller.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .. import MATCH_RADIUS_UM
from ..io.graph import TrackGraph
from .candidates import SCALE, enumerate_candidates, low_peaks, um_distance


class Adj:
    """Adjacency of a base graph with a few edges removed / added (no copy of the base dicts)."""

    def __init__(self, base_out: dict, base_in: dict, removed=(), added=()):
        self.bo, self.bi, self.rm, self.add = base_out, base_in, set(removed), list(added)

    def out(self, n: int) -> list[int]:
        return [c for c in self.bo.get(n, []) if (n, c) not in self.rm] + [d for s, d in self.add if s == n]

    def inc(self, n: int) -> list[int]:
        return [s for s in self.bi.get(n, []) if (s, n) not in self.rm] + [s for s, d in self.add if d == n]


def adjacency(edges: np.ndarray) -> tuple[dict, dict]:
    out: dict[int, list[int]] = {}
    inc: dict[int, list[int]] = {}
    for s, t in edges:
        out.setdefault(int(s), []).append(int(t))
        inc.setdefault(int(t), []).append(int(s))
    return out, inc


class ForkLabeler:
    """Official division rules for one added fork on a fixed node set."""

    def __init__(self, nodes: pd.DataFrame, edges: np.ndarray, gt: TrackGraph, base_adj: Adj | None = None):
        import tracksdata as td
        from tracking_cellmot.division_metrics import (
            _gt_weak_component_ids,
            _match_full,
            _matched_division_nodes,
            _matched_node_attrs,
            extract_divisions,
            match_divisions,
            score_divisions,
        )

        from ..metrics.official import to_tracksdata_graph

        self.nodes = nodes
        self.t_of = dict(zip(nodes.node_id.astype(int), nodes.t.astype(int), strict=True))
        pg = to_tracksdata_graph(TrackGraph(nodes, pd.DataFrame(edges, columns=["source_id", "target_id"])))
        gg = to_tracksdata_graph(gt)
        self.p2o = dict(zip(pg.node_ids(), nodes.node_id.astype(int), strict=True))
        self.g2o = dict(zip(gg.node_ids(), gt.nodes.node_id.astype(int), strict=True))
        keys = td.DEFAULT_ATTR_KEYS
        full = _matched_node_attrs(_match_full(pg, gg, tuple(SCALE), MATCH_RADIUS_UM))
        self.to_gt = {self.p2o[int(a)]: self.g2o[int(b)]
                      for a, b in zip(full[keys.NODE_ID].to_list(), full[keys.MATCHED_NODE_ID].to_list(), strict=True)}
        self.comp = {self.g2o[k]: v for k, v in _gt_weak_component_ids(gg).items()}
        self.gt_out = {int(k): [int(x) for x in g.target_id] for k, g in gt.edges.groupby("source_id")}
        self.gt_pos = gt.nodes.set_index("node_id")[["t", "z", "y", "x"]]
        windows = extract_divisions(gg)
        self.div = {}
        for g_td, matched in match_divisions(pg, gg, tuple(SCALE), MATCH_RADIUS_UM).items():
            roles = _matched_division_nodes(_matched_node_attrs(matched), windows[g_td], g_td)
            gid = self.g2o[int(g_td)]
            gt_children = [self.g2o[int(c)] for c in windows[g_td].successors(g_td)]
            if roles is None:
                self.div[gid] = {"parents": set(), "daughters": [set(), set()], "children": gt_children}
                continue
            parents, daughters = roles
            self.div[gid] = {"parents": {self.p2o[int(n)] for n in parents},
                             "daughters": [{self.p2o[int(n)] for n in s} for s in daughters],
                             "children": gt_children}
        base = score_divisions(pg, gg, tuple(SCALE), MATCH_RADIUS_UM)
        self.paired = {self.g2o[int(g)] for g, v in base.scores.items() if v}
        self.base_tp = {self.p2o[int(n)] for n in base.tp_forks}
        self.base_fp = {self.p2o[int(n)] for n in base.fp_forks}
        # which GT division each existing TP fork holds: a candidate that replaces that fork frees it
        self.fork_div: dict[int, int] = {}
        if base_adj is not None:
            for f in self.base_tp:
                ch = base_adj.out(f)
                for gid in self.div:
                    if len(ch) == 2 and f in self._local(gid, base_adj) and self._valid(
                            gid, f, ch[0], ch[1], base_adj, None):
                        self.fork_div[f] = gid
                        break

    def _local(self, gid, adj: Adj) -> set[int]:
        parents = self.div[gid]["parents"]
        return parents | {c for p in parents for c in adj.out(p)}

    def _valid(self, gid: int, p: int, d1: int, d2: int, adj: Adj, new_gt: int | None) -> bool:
        """The scorer's local-topology test for fork P -> {D1, D2}."""
        info = self.div[gid]
        if {p, *adj.inc(p)}.isdisjoint(info["parents"]):
            return False
        lineages = [{d1, *adj.out(d1)}, {d2, *adj.out(d2)}]
        daughters = [set(x) for x in info["daughters"]]
        if new_gt is not None:
            for k, child in enumerate(info["children"]):
                if new_gt == child or new_gt in self.gt_out.get(child, []):
                    daughters[k].add(d2)
        hit = [[k for k, lin in enumerate(lineages) if not lin.isdisjoint(x)] for x in daughters]
        return bool(hit[0] and hit[1]) and not (len(hit[0]) == 1 and hit[0] == hit[1])

    def label(self, p: int, d1: int, d2: int, adj: Adj, new_d2: tuple | None = None) -> dict:
        """``adj``: the graph WITH the fork. ``new_d2`` = (t, z, y, x) when D2 is a new node (shape d)."""
        out, inc = adj.out, adj.inc
        res = {"label": "ignored", "fp_reason": "", "gt_divider_id": -1, "timing_vs_gt": np.nan,
               "gt_p_id": self.to_gt.get(p, -1),
               "gt_p_outdeg": len(self.gt_out.get(self.to_gt[p], [])) if p in self.to_gt else -1}
        to_gt = self.to_gt
        new_gt = None
        if new_d2 is not None:
            held = set(to_gt.values())
            best = None
            for gid, pos in self.gt_pos[self.gt_pos.t == int(new_d2[0])].iterrows():
                dist = um_distance(pos[["z", "y", "x"]].to_numpy(float), new_d2[1:])
                if dist <= MATCH_RADIUS_UM and gid not in held and (best is None or dist < best[0]):
                    best = (dist, int(gid))
            new_gt = best[1] if best else None
            if new_gt is not None:
                to_gt = {**to_gt, d2: new_gt}
        branch = []
        for c in (d1, d2):
            if inc(c) != [p]:
                res.update(label="fp", fp_reason="malformed")
                return res
            if c in to_gt:
                branch.append(self.comp[to_gt[c]])
                continue
            gcs = out(c)
            if any(inc(g) != [c] for g in gcs):
                res.update(label="fp", fp_reason="malformed")
                return res
            comps = {self.comp[to_gt[g]] for g in gcs if g in to_gt}
            if len(comps) == 1:
                branch.append(next(iter(comps)))
        cross = len(set(branch)) >= 2
        freed = {self.fork_div[s] for s in {p, *(s for s, _ in adj.rm)} if s in self.fork_div}
        paired = self.paired - freed
        near = []
        for gid in self.div:
            if p not in self._local(gid, adj):
                continue
            near.append(gid)
            if not cross and self._valid(gid, p, d1, d2, adj, new_gt) and gid not in paired:
                res.update(label="tp", gt_divider_id=gid,
                           timing_vs_gt=int(self.t_of[p] - self.gt_pos.at[gid, "t"]))
                return res
        if cross:
            res.update(label="fp", fp_reason="cross_gt_tracks")
        elif near:
            res.update(label="fp", fp_reason="near_gt_division", gt_divider_id=near[0],
                       timing_vs_gt=int(self.t_of[p] - self.gt_pos.at[near[0], "t"]))
        elif p in self.to_gt and len(self.gt_out.get(self.to_gt[p], [])) >= 1:
            res.update(label="fp", fp_reason="annotated_continuing")
        return res


def label_movie(movie: str, nodes: pd.DataFrame, edges: np.ndarray, selected: np.ndarray, gt: TrackGraph,
                cache: dict | None = None) -> tuple[pd.DataFrame, dict]:
    """Enumerate one movie's candidates on its pre-safe graph and label them. Returns (table, summary)."""
    df = enumerate_candidates(nodes, edges, selected, low_peaks(cache))
    out_pre, inc_pre = adjacency(edges)
    sel = {(int(s), int(t)) for s, t in selected}
    sel_by_p = {s: t for s, t in sel}
    out_post = {k: list(v) for k, v in out_pre.items()}
    inc_post = {k: list(v) for k, v in inc_pre.items()}
    for s, d in sel:
        out_post.setdefault(s, []).append(d)
        inc_post.setdefault(d, []).append(s)
    post_edges = np.concatenate([edges, selected]) if len(selected) else edges
    lab = ForkLabeler(nodes, post_edges, gt, Adj(out_post, inc_post))
    labels = []
    for r in df.itertuples():
        removed = [(s, d) for s, d in sel if (s == r.p_id or d == r.d2_id) and (s, d) != (r.p_id, r.d2_id)]
        adj = Adj(out_post, inc_post, removed, [] if (r.p_id, r.d2_id) in sel else [(r.p_id, r.d2_id)])
        labels.append(lab.label(r.p_id, r.d1_id, r.d2_id, adj,
                                new_d2=(r.d2_t, r.d2_z, r.d2_y, r.d2_x) if r.shape == "d" else None))
    if len(df):
        df.insert(0, "embryo", movie.split("_")[0])
        df.insert(0, "movie", movie)
        df = pd.concat([df, pd.DataFrame(labels, index=df.index)], axis=1)
    summary = {"movie": movie, "n_candidates": len(df), "safe_div_forks": len(sel),
               "safe_div_tp": len(lab.base_tp & set(sel_by_p)), "safe_div_fp": len(lab.base_fp & set(sel_by_p)),
               "gt_divisions": len(lab.div), "gt_divisions_paired_post_safe": len(lab.paired),
               "post_safe_tp": len(lab.base_tp), "post_safe_fp": len(lab.base_fp)}
    return df, summary


def verify_labels(sample: pd.DataFrame, nodes: pd.DataFrame, edges: np.ndarray, selected: np.ndarray,
                  gt: TrackGraph) -> list[dict]:
    """Re-score each sampled fork with the real ``score_divisions`` on the edited post-safe graph."""
    from tracking_cellmot.division_metrics import score_divisions

    from ..metrics.official import to_tracksdata_graph

    def score(nd, ed):
        pg = to_tracksdata_graph(TrackGraph(nd, pd.DataFrame(ed, columns=["source_id", "target_id"])))
        r = score_divisions(pg, to_tracksdata_graph(gt), tuple(SCALE), MATCH_RADIUS_UM)
        return sum(r.scores.values()), len(r.fp_forks)

    out = []
    for r in sample.itertuples():
        keep = [e for e in map(tuple, selected) if e[0] != r.p_id and e[1] != r.d2_id]
        nd = nodes
        if r.shape == "d":
            nd = pd.concat([nodes, pd.DataFrame([{"node_id": r.d2_id, "t": r.d2_t, "z": r.d2_z, "y": r.d2_y,
                                                  "x": r.d2_x}])], ignore_index=True)
        base_edges = np.concatenate([edges, np.asarray(keep, dtype=np.int64).reshape(-1, 2)])
        tp0, fp0 = score(nd, base_edges)
        tp1, fp1 = score(nd, np.concatenate([base_edges, [[r.p_id, r.d2_id]]]))
        real = "tp" if tp1 > tp0 else "fp" if fp1 > fp0 else "ignored"
        out.append({"p_id": int(r.p_id), "d2_id": int(r.d2_id), "shape": r.shape, "predicted": r.label,
                    "official": real, "agree": real == r.label})
    return out
