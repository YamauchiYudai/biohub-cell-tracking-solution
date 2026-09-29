"""J2 line fit and frozen-frame consensus: behaviour, and parity with the notebook blocks of every candidate."""

from __future__ import annotations

import ast
import copy
from collections import defaultdict

import numpy as np
import pytest

from biohub_tracking.kaggle import blocks
from biohub_tracking.kaggle import notebook as nbk
from biohub_tracking.postprocessing import consensus as FC
from biohub_tracking.postprocessing import linefit as LF

SCALE = np.array([1.625, 0.40625, 0.40625])


class _FakeOS:
    def __init__(self, env):
        self.environ = env


def _namespace(cells: list[str], j2_block: str | None = None, fc: bool = False) -> dict:
    """Upstream line fit (cell 5), its constants (cell 2), the relink helpers, then the J2 (and FC) block."""
    env = dict(nbk.env_overrides(cells))
    env["BIOHUB_OUTPUT_FROZEN_CONSENSUS"] = "1" if fc else "0"
    ns: dict = {"np": np, "os": _FakeOS(env)}
    for node in ast.parse(cells[2]).body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name) \
                and node.targets[0].id.startswith("OUTPUT_LINEFIT"):
            exec(ast.get_source_segment(cells[2], node), ns)
    c5 = cells[nbk.POSTPROC_CELL]
    exec("VOXEL_SCALE_UM = (1.625, 0.40625, 0.40625)", ns)
    exec(nbk.extract_block(c5, nbk.RELINK_BEGIN, nbk.RELINK_END), ns)
    exec(nbk.extract_function(c5, "linefit_smooth_output_graph"), ns)
    ns["_upstream_linefit"] = ns["linefit_smooth_output_graph"]
    exec(j2_block or nbk.extract_block(c5, nbk.J2_BEGIN, nbk.J2_END), ns)
    if fc:
        exec(blocks.FC_BLOCK, ns)
    return ns


def _forest(seed: int, n_tracks: int = 25, n_frames: int = 10):
    """Random tracks with a few divisions and merges-free topology; voxel coordinates."""
    rng = np.random.default_rng(seed)
    nodes, edges, nid = {}, [], 0
    heads = []
    for _ in range(n_tracks):
        nodes[nid] = {"t": 0, "z": float(rng.uniform(2, 20)), "y": float(rng.uniform(20, 200)),
                      "x": float(rng.uniform(20, 200))}
        heads.append(nid)
        nid += 1
    for t in range(1, n_frames):
        new_heads = []
        for h in heads:
            if rng.random() < 0.05:
                continue                                         # track ends
            for _ in range(2 if rng.random() < 0.06 else 1):     # division
                p = nodes[h]
                nodes[nid] = {"t": t, "z": p["z"] + rng.normal(0, 0.3), "y": p["y"] + rng.normal(1.5, 1.0),
                              "x": p["x"] + rng.normal(-1.0, 1.0)}
                edges.append({"source_id": h, "target_id": nid})
                new_heads.append(nid)
                nid += 1
        heads = new_heads
    return nodes, edges


def _kinds(seed: int, n_frames: int = 10):
    rng = np.random.default_rng(seed + 100)
    kinds = {t: str(rng.choice(["normal", "normal", "normal", "frozen", "jump"])) for t in range(n_frames - 1)}
    shifts = {t: [float(v) for v in rng.normal(0, 4, 3)]
              for t, k in kinds.items() if k == "jump" and rng.random() < 0.7}
    return kinds, shifts


def _set_ctx(ns, kinds, shifts):
    ns["X138_RELINK_CTX"]["dataset"] = "6bba_test"
    table = {"_movie_median": 1.0, **{t: {"kind": k} for t, k in kinds.items()}}
    ns["X138_RELINK_CTX"]["transitions"]["6bba_test"] = table
    ns["X138_RELINK_CTX"].setdefault("applied_shift", {})["6bba_test"] = shifts


# ------------------------------------------------------------------ J2
@pytest.mark.notebook
@pytest.mark.parametrize("which", ["insured", "v1"])
@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_j2_matches_the_notebook_block(core_cells, which, seed):
    ns = _namespace(core_cells, None if which == "insured" else blocks.J2_V1_BLOCK)
    nodes, edges = _forest(seed)
    kinds, shifts = _kinds(seed)
    _set_ctx(ns, kinds, shifts)
    nb_stats, lib_stats = defaultdict(int), {}
    theirs = ns["linefit_smooth_output_graph"](copy.deepcopy(nodes), edges, nb_stats)
    ours = LF.jump_aware_linefit(copy.deepcopy(nodes), edges, kinds, shifts, stats=lib_stats)
    assert ours == theirs
    assert {k: v for k, v in lib_stats.items()} == {k: v for k, v in nb_stats.items()}


@pytest.mark.notebook
def test_j2_equals_the_upstream_smoothing_without_special_transitions(core_cells):
    ns = _namespace(core_cells)
    nodes, edges = _forest(7)
    _set_ctx(ns, {t: "normal" for t in range(9)}, {})
    upstream = ns["_upstream_linefit"](copy.deepcopy(nodes), edges, defaultdict(int))
    assert LF.linefit(copy.deepcopy(nodes), edges) == upstream
    assert ns["linefit_smooth_output_graph"](copy.deepcopy(nodes), edges, defaultdict(int)) == upstream


def test_j2_cuts_frozen_and_compensates_jumps():
    """One cell moving +1 voxel in x per frame; the embryo steps +6 um in y between t=3 and t=4."""
    jump_vox = 6.0 / 0.40625
    nodes = {i: {"t": i, "z": 5.0, "y": 50.0 + (jump_vox if i >= 4 else 0.0) + 0.3 * (-1) ** i, "x": 10.0 + i}
             for i in range(7)}
    edges = [{"source_id": i, "target_id": i + 1} for i in range(6)]
    upstream = LF.linefit(copy.deepcopy(nodes), edges)
    for kinds, shifts in (({3: "frozen"}, {}), ({3: "jump"}, {})):
        cut = LF.jump_aware_linefit(copy.deepcopy(nodes), edges, kinds, shifts)
        assert abs(cut[3]["y"] - nodes[3]["y"]) < abs(upstream[3]["y"] - nodes[3]["y"]) - 1.0
    stats = {}
    comp = LF.jump_aware_linefit(copy.deepcopy(nodes), edges, {3: "jump"}, {3: [0.0, 6.0, 0.0]}, stats=stats)
    assert abs(comp[4]["y"] - (50.0 + jump_vox)) < 0.3 and abs(comp[3]["y"] - 50.0) < 0.3
    assert stats["linefit_jump_aware_shifts"] == 1


# ------------------------------------------------------------------ frozen-frame consensus
def _fc_track(dy_vox: float = 0.0):
    """One nucleus over 7 frames; t=3 -> 4 is the frozen transition (node 4 is node 3 again, 0.5 voxel off in x and
    dy_vox off in y before smoothing)."""
    nodes = {i: {"t": i, "z": 5.0, "y": 50.0 + 0.2 * (-1) ** i, "x": 10.0 + min(i, 3) + (0.5 + i - 4 if i >= 4 else 0)}
             for i in range(7)}
    nodes[4]["y"] += dy_vox
    return nodes, [{"source_id": i, "target_id": i + 1} for i in range(6)]


def _before(nodes):
    return {i: (n["z"], n["y"], n["x"]) for i, n in nodes.items()}


def _xyz(n):
    return np.array([n["z"], n["y"], n["x"]], dtype=float)


def test_consensus_averages_the_two_copies_across_a_frozen_transition():
    nodes, edges = _fc_track()
    before = _before(nodes)
    kinds = {t: ("frozen" if t == 3 else "normal") for t in range(6)}
    smoothed = LF.jump_aware_linefit(copy.deepcopy(nodes), edges, kinds, {})
    stats = {}
    out = FC.apply_frozen_consensus(copy.deepcopy(smoothed), edges, before, kinds, stats=stats)
    mean = (_xyz(smoothed[3]) + _xyz(smoothed[4])) / 2
    assert np.array_equal(_xyz(out[3]), mean) and np.array_equal(_xyz(out[4]), mean)
    assert all(out[i] == smoothed[i] for i in (0, 1, 2, 5, 6))
    counts = [stats[f"frozen_consensus_{k}"] for k in ("groups", "nodes", "failed")]
    assert counts == [1, 2, 0]
    kinds[4] = "frozen"                                         # two consecutive frozen transitions: one group of 3
    moves, counts = FC.frozen_consensus(smoothed, edges, before, kinds)
    assert set(moves) == {3, 4, 5} and counts["frozen_consensus_groups"] == 1


@pytest.mark.parametrize("dy_vox, linked", [(3.5, True), (4.5, False)])
def test_consensus_distance_gate_is_1625_um_before_smoothing(dy_vox, linked):
    nodes, edges = _fc_track(dy_vox)
    assert (float(np.linalg.norm((_xyz(nodes[3]) - _xyz(nodes[4])) * SCALE)) <= 1.625) == linked
    _, counts = FC.frozen_consensus(nodes, edges, _before(nodes), {3: "frozen"})
    assert counts["frozen_consensus_edges"] == int(linked)


@pytest.mark.parametrize("extra_node, extra_edge", [
    ({"t": 4, "z": 5.0, "y": 60.0, "x": 13.0}, (3, 99)),       # the source divides
    ({"t": 3, "z": 5.0, "y": 60.0, "x": 13.0}, (99, 4)),       # the target has two parents
])
def test_consensus_does_not_cross_forks_or_merges(extra_node, extra_edge):
    nodes, edges = _fc_track()
    nodes[99] = extra_node
    edges = edges + [{"source_id": extra_edge[0], "target_id": extra_edge[1]}]
    moves, counts = FC.frozen_consensus(nodes, edges, _before(nodes), {3: "frozen"})
    assert counts["frozen_consensus_edges"] == 0 and not moves


def test_consensus_failure_restores_coordinates():
    nodes, edges = _fc_track()
    original = copy.deepcopy(nodes)
    stats = {}
    out = FC.apply_frozen_consensus(nodes, edges, {}, {3: "frozen"}, stats=stats)     # missing `before`: KeyError
    assert out == original and stats["frozen_consensus_failed"] == 1


@pytest.mark.notebook
@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_consensus_after_j2_matches_the_notebook_block(core_cells, seed):
    ns = _namespace(core_cells, fc=True)
    nodes, edges = _forest(seed)
    kinds, shifts = _kinds(seed)
    kinds = {t: ("frozen" if k == "jump" else k) for t, k in kinds.items()}   # more frozen transitions
    _set_ctx(ns, kinds, {})
    nb_stats = defaultdict(int)
    theirs = ns["linefit_smooth_output_graph"](copy.deepcopy(nodes), edges, nb_stats)
    before = _before(nodes)
    lib_stats: dict = {}
    ours = LF.jump_aware_linefit(copy.deepcopy(nodes), edges, kinds, {}, stats=lib_stats)
    ours = FC.apply_frozen_consensus(ours, edges, before, kinds, stats=lib_stats)
    assert ours == theirs and lib_stats["frozen_consensus_groups"] > 0
    nb_stats.pop("frozen_consensus_seconds")
    assert lib_stats == dict(nb_stats)
    moves, counts = ns["_fc_consensus"](theirs, edges, before, kinds, (1.625, 0.40625, 0.40625), 1.625)
    assert (moves, counts) == FC.frozen_consensus(theirs, edges, before, kinds)
