"""Local scoring with the official metric (skipped when the ``metric`` extra is not installed)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from biohub_tracking import VOXEL_SCALE_UM
from biohub_tracking.io.graph import TrackGraph
from biohub_tracking.io.submission import roundtrip_integer_graph
from biohub_tracking.metrics.official import node_ratio


def _gt(n_tracks=6, n_frames=5, divide=True):
    rows, edges, nid = [], [], 0
    last = {}
    for k in range(n_tracks):
        for t in range(n_frames):
            rows.append((nid, t, 10.0, 40.0 * k + 20, 40.0 + 2 * t))
            if t:
                edges.append((last[k], nid))
            last[k] = nid
            nid += 1
    if divide:                                           # track 0 divides at t=2 -> a new daughter at t=3
        rows.append((nid, 3, 10.0, 40.0 * 0 + 32, 44.0))
        edges.append((2, nid))
    nodes = pd.DataFrame(rows, columns=["node_id", "t", "z", "y", "x"])
    return TrackGraph(nodes, pd.DataFrame(edges, columns=["source_id", "target_id"]),
                      estimated_number_of_nodes=float(len(nodes)))


def test_integer_roundtrip_rounds_coordinates():
    g = TrackGraph(pd.DataFrame([(1, 0, 1.4, 2.6, 3.5)], columns=["node_id", "t", "z", "y", "x"]),
                   pd.DataFrame(columns=["source_id", "target_id"], dtype="int64"))
    assert roundtrip_integer_graph(g).nodes[["z", "y", "x"]].values.tolist() == [[1, 3, 4]]


def test_node_ratio():
    gt = _gt()
    assert node_ratio(gt, gt) == 0.0
    assert np.isnan(node_ratio(gt, TrackGraph(gt.nodes, gt.edges)))


def test_perfect_and_degraded_predictions_score_as_expected():
    pytest.importorskip("tracking_cellmot")
    from biohub_tracking.metrics.official import score_graphs

    gt = _gt()
    perfect = score_graphs({"44b6_a": TrackGraph(gt.nodes, gt.edges)}, {"44b6_a": gt}, VOXEL_SCALE_UM)
    assert perfect.official_score == pytest.approx(1.1)          # adjusted edge J 1.0 + 0.1 x division J 1.0
    no_div = TrackGraph(gt.nodes, gt.edges.iloc[:-1])
    worse = score_graphs({"44b6_a": no_div}, {"44b6_a": gt}, VOXEL_SCALE_UM)
    assert worse.official_division_jaccard == 0.0 and worse.official_score < perfect.official_score
    report = score_graphs({"44b6_a": no_div, "6bba_b": TrackGraph(gt.nodes, gt.edges)},
                          {"44b6_a": gt, "6bba_b": gt, "6bba_c": gt}, VOXEL_SCALE_UM)
    assert report.skipped == ["6bba_c"] and report.worst_embryo == "44b6"
    assert set(report.per_embryo) == {"44b6", "6bba"}


def test_read_geff_round_trip(tmp_path):
    zarr = pytest.importorskip("zarr")
    from biohub_tracking.io.geff import GeffReadError, read_geff

    path = tmp_path / "44b6_x.geff"
    g = zarr.open_group(str(path), mode="w")
    g.create_array("nodes/ids", data=np.array([5, 6, 7], dtype=np.uint64))
    for axis, values in {"t": [0, 1, 1], "z": [3, 3, 4], "y": [10, 11, 30], "x": [20, 21, 40]}.items():
        g.create_array(f"nodes/props/{axis}/values", data=np.array(values, dtype=np.int64))
    g.create_array("edges/ids", data=np.array([[5, 6], [5, 7]], dtype=np.uint64))
    g.attrs["geff"] = {"extra": {"estimated_number_of_nodes": 4.0}}
    graph = read_geff(path)
    assert graph.nodes.node_id.tolist() == [5, 6, 7] and graph.nodes.x.tolist() == [20.0, 21.0, 40.0]
    assert graph.edges.values.tolist() == [[5, 6], [5, 7]] and graph.estimated_number_of_nodes == 4.0
    with pytest.raises(GeffReadError):
        read_geff(tmp_path / "missing.geff")
