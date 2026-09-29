"""Frozen-frame coordinate consensus ("case A"), part of both selected submissions (fc, fc_f03).

A frozen transition t (frames t and t+1 bit-identical) shows the same nuclei twice, and the ground truth does
not move across it (6bba, 88 train movies: 99.5 % of 4,420 GT edges across frozen transitions are < 0.01 um).
The detector reads neighbouring frames and J2 cuts its walk there, so the two copies of a nucleus drift apart
after smoothing. Consensus: right after the line fit, every chain of nodes linked 1:1 across frozen
transitions gets the mean of its smoothed coordinates.

Linked edge t -> t+1: kind(t) == "frozen", the source has 1 child and <= 1 parent, the target 1 parent and
<= 1 child, and the pre-smoothing endpoints are <= 1.625 um apart. A chain whose mean is > 1.625 um from any of
its nodes is left alone. Node ids, times, edges and divisions never change.

88-movie CPU replay: +0.00103 official score (6bba +0.00115, 44b6 unchanged: it has no frozen frames).
"""

from __future__ import annotations

import numpy as np

MAX_UM = 1.625
VOXEL_SCALE_UM = (1.625, 0.40625, 0.40625)


def written(v) -> int:
    """The submission writer's rounding."""
    return max(0, int(round(float(v))))


def frozen_consensus(nodes_by_id: dict, edges: list[dict], before: dict, kinds: dict[int, str],
                     scale_um=VOXEL_SCALE_UM, max_um: float = MAX_UM) -> tuple[dict, dict]:
    """(new coordinates {node_id: (z, y, x)}, counts). Reads its arguments only.

    ``before``: {node_id: (z, y, x)} pre-smoothing coordinates (the distance gate); ``nodes_by_id`` holds the
    smoothed ones (averaged). Chains are walked from their first node, first nodes in id order; the mean is
    taken in time order, so the result is deterministic.
    """
    scale = np.asarray(scale_um, dtype=np.float64)
    children: dict[int, int] = {}
    parents: dict[int, int] = {}
    for edge in edges:
        source_id, target_id = int(edge["source_id"]), int(edge["target_id"])
        children[source_id] = children.get(source_id, 0) + 1
        parents[target_id] = parents.get(target_id, 0) + 1
    link: dict[int, int] = {}
    for edge in edges:
        source_id, target_id = int(edge["source_id"]), int(edge["target_id"])
        source, target = nodes_by_id.get(source_id), nodes_by_id.get(target_id)
        if source is None or target is None or int(target["t"]) != int(source["t"]) + 1:
            continue
        if kinds.get(int(source["t"]), "normal") != "frozen":
            continue
        if (children[source_id] != 1 or parents[target_id] != 1 or parents.get(source_id, 0) > 1
                or children.get(target_id, 0) > 1):
            continue
        gap = np.asarray(before[source_id], dtype=np.float64) - np.asarray(before[target_id], dtype=np.float64)
        if not float(np.sqrt(np.sum((gap * scale) ** 2))) <= max_um:
            continue
        link[source_id] = target_id
    counts = {"frozen_consensus_edges": len(link), "frozen_consensus_groups": 0, "frozen_consensus_nodes": 0,
              "frozen_consensus_skipped_groups": 0, "frozen_consensus_moved_rounded": 0}
    moves: dict[int, tuple[float, float, float]] = {}
    linked_targets = set(link.values())
    for start in sorted(s for s in link if s not in linked_targets):
        chain = [start]
        while chain[-1] in link:
            chain.append(link[chain[-1]])
        pts = np.array([[float(nodes_by_id[i]["z"]), float(nodes_by_id[i]["y"]), float(nodes_by_id[i]["x"])]
                        for i in chain], dtype=np.float64)
        mean = pts.mean(axis=0)
        if not float(np.max(np.sqrt((((pts - mean) * scale) ** 2).sum(axis=1)))) <= max_um:
            counts["frozen_consensus_skipped_groups"] += 1
            continue
        new = tuple(float(v) for v in mean)
        counts["frozen_consensus_groups"] += 1
        counts["frozen_consensus_nodes"] += len(chain)
        for node_id, old in zip(chain, pts, strict=True):
            moves[node_id] = new
            counts["frozen_consensus_moved_rounded"] += int(any(written(a) != written(b)
                                                                for a, b in zip(new, old, strict=True)))
    return moves, counts


def apply_frozen_consensus(nodes_by_id: dict, edges: list[dict], before: dict, kinds: dict[int, str],
                           stats: dict | None = None, **kwargs) -> dict:
    """Write the consensus coordinates; on any exception restore what was written and keep the input."""
    stats = {} if stats is None else stats
    done: dict[int, tuple] = {}
    try:
        moves, counts = frozen_consensus(nodes_by_id, edges, before, kinds, **kwargs)
        for node_id, pos in moves.items():
            node = nodes_by_id[node_id]
            done[node_id] = (node["z"], node["y"], node["x"])
            node["z"], node["y"], node["x"] = pos
    except Exception:
        for node_id, pos in done.items():
            nodes_by_id[node_id]["z"], nodes_by_id[node_id]["y"], nodes_by_id[node_id]["x"] = pos
        stats["frozen_consensus_failed"] = 1
        return nodes_by_id
    stats.update(counts)
    stats["frozen_consensus_failed"] = 0
    return nodes_by_id
