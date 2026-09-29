"""Shared in-memory track-graph representation.

A `TrackGraph` is two DataFrames: nodes `(node_id, t, z, y, x)` and edges `(source_id, target_id)`,
referencing node IDs (not row positions), matching the geff / submission-CSV convention.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

NODE_COLUMNS: tuple[str, ...] = ("node_id", "t", "z", "y", "x")
EDGE_COLUMNS: tuple[str, ...] = ("source_id", "target_id")


def empty_frame(columns: tuple[str, ...], dtype: str = "int64") -> pd.DataFrame:
    """A zero-row DataFrame with `columns` typed as `dtype` (an untyped frame would come back as object)."""
    return pd.DataFrame({c: pd.Series(dtype=dtype) for c in columns})


@dataclass
class TrackGraph:
    """One dataset's detections + links.

    `nodes.z/y/x` are floats (sub-voxel precision is kept internally; only `io.submission` rounds to int,
    per the competition's CSV schema). `estimated_number_of_nodes` mirrors the geff metadata field used by
    the adjusted-Jaccard node-count penalty; it is `None` for predictions.
    """

    nodes: pd.DataFrame
    edges: pd.DataFrame
    estimated_number_of_nodes: float | None = None

    def __post_init__(self) -> None:
        missing_n = set(NODE_COLUMNS) - set(self.nodes.columns)
        if missing_n:
            raise ValueError(f"TrackGraph.nodes missing columns: {sorted(missing_n)}")
        missing_e = set(EDGE_COLUMNS) - set(self.edges.columns)
        if missing_e:
            raise ValueError(f"TrackGraph.edges missing columns: {sorted(missing_e)}")

    @staticmethod
    def empty() -> TrackGraph:
        return TrackGraph(nodes=empty_frame(NODE_COLUMNS), edges=empty_frame(EDGE_COLUMNS))

    @property
    def num_nodes(self) -> int:
        return len(self.nodes)

    @property
    def num_edges(self) -> int:
        return len(self.edges)


@dataclass
class ForestViolations:
    """Result of :func:`validate_forest`.

    The official scorer silently drops dt>1 edges and never rewards merges or >2-way splits, so a
    submission graph should have none of these.
    """

    merge_node_ids: list[int]
    multi_branch_node_ids: list[int]
    dt_violations: list[tuple[int, int]]
    self_loop_node_ids: list[int]
    dangling_edge_ids: list[tuple[int, int]]

    @property
    def is_valid(self) -> bool:
        return not (
            self.merge_node_ids
            or self.multi_branch_node_ids
            or self.dt_violations
            or self.self_loop_node_ids
            or self.dangling_edge_ids
        )


def validate_forest(graph: TrackGraph) -> ForestViolations:
    """Check the edges form a valid directed forest: no merges, no >2 children, every edge advances exactly
    one timepoint, no self-loops or edges to node IDs absent from `graph.nodes`."""
    t_by_id = dict(zip(graph.nodes["node_id"], graph.nodes["t"], strict=True))
    src = graph.edges["source_id"]
    tgt = graph.edges["target_id"]

    dangling = [(int(s), int(t)) for s, t in zip(src, tgt, strict=True) if s not in t_by_id or t not in t_by_id]

    valid_edges = graph.edges[src.isin(t_by_id) & tgt.isin(t_by_id)]
    out_deg = valid_edges["source_id"].value_counts()
    in_deg = valid_edges["target_id"].value_counts()

    self_loops = []
    dt_violations = []
    for s, t in zip(valid_edges["source_id"], valid_edges["target_id"], strict=True):
        if s == t:
            self_loops.append(int(s))
            continue
        if (t_by_id[t] - t_by_id[s]) != 1:
            dt_violations.append((int(s), int(t)))

    return ForestViolations(
        merge_node_ids=[int(x) for x in in_deg[in_deg > 1].index],
        multi_branch_node_ids=[int(x) for x in out_deg[out_deg > 2].index],
        dt_violations=dt_violations,
        self_loop_node_ids=self_loops,
        dangling_edge_ids=dangling,
    )


def next_free_node_id(graphs: list[TrackGraph] | TrackGraph, start: int = 1) -> int:
    """Smallest integer >= start not used as a node_id in any of `graphs`."""
    if isinstance(graphs, TrackGraph):
        graphs = [graphs]
    used = np.concatenate([g.nodes["node_id"].to_numpy() for g in graphs]) if graphs else np.array([])
    if used.size == 0:
        return start
    return max(start, int(used.max()) + 1)
