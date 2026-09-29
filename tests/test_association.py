"""Frame-to-frame association repair: the b1c jump-aware relink.

The upstream `motion_relink_edges` and the embedded b1c helpers are executed straight from the production
notebook (cell 0 env values, cell 2 relink constants, cell 5 functions) on synthetic movies with a duplicated
frame; the library restatement in `biohub_tracking.postprocessing.relink` must match them exactly.
"""

from __future__ import annotations

import ast
import math

import numpy as np
import pytest
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree

from biohub_tracking.kaggle import notebook as nbk
from biohub_tracking.postprocessing import relink as R

VOX = (1.625, 0.40625, 0.40625)


class _FakeOS:
    def __init__(self, env):
        self.environ = env


def _namespace(cells: list[str], variant: str) -> dict:
    env = nbk.env_overrides(cells)
    env["BIOHUB_X138_RELINK_VARIANT"] = variant
    ns: dict = {"np": np, "math": math, "cKDTree": cKDTree, "linear_sum_assignment": linear_sum_assignment,
                "os": _FakeOS(env)}
    for node in ast.parse(cells[2]).body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if name.startswith("MOTION_RELINK_") or name == "OUTPUT_MOTION_RELINK":
                exec(ast.get_source_segment(cells[2], node), ns)
    c5 = cells[nbk.POSTPROC_CELL]
    for node in ast.parse(c5).body:
        if isinstance(node, ast.Assign) and ast.unparse(node.targets[0]) == "VOXEL_SCALE_UM":
            exec(ast.get_source_segment(c5, node), ns)
    exec(nbk.extract_block(c5, nbk.RELINK_BEGIN, nbk.RELINK_END), ns)
    ns["X138_RELINK"] = ns["x138_relink_config"](variant)      # the block reads the variant from os.environ
    for name in ("_position_um", "motion_relink_edges"):
        exec(nbk.extract_function(c5, name), ns)
    return ns


def _run(ns, nodes, probs, frozen=()):
    ns["X138_RELINK_CTX"].update({"dataset": "syn", "frozen": frozenset(frozen), "transitions": {}, "seconds": {}})
    stats = {"motion_relink_tight_edges": 0, "motion_relink_relaxed_edges": 0, "motion_relink_frames": 0}
    edges = ns["motion_relink_edges"]({k: dict(v) for k, v in nodes.items()}, stats, dict(probs))
    return sorted((e["source_id"], e["target_id"]) for e in edges), stats


def _movie(seed=0, n=64, T=12, step_um=4.5, freeze_at=None, spacing_um=6.5):
    """Cells on a jittered grid drifting together. freeze_at=f: frame f+1 repeats frame f, so transition f is
    frozen and transition f+1 carries two steps of motion (a jump)."""
    rng = np.random.default_rng(seed)
    side = int(np.ceil(n ** (1 / 3))) + 1
    grid = np.stack(np.meshgrid(*[np.arange(side)] * 3, indexing="ij"), -1).reshape(-1, 3)[:n] * spacing_um
    pos = grid + rng.normal(0, 1.0, grid.shape) + 20.0
    vel = np.array([0.6, 0.8, 0.0]) * step_um + rng.normal(0, 0.4, (n, 3))
    frames, t_real = [], 0
    for t in range(T):
        if freeze_at is not None and t == freeze_at + 1:
            frames.append(frames[-1].copy())
            t_real += 1
            continue
        frames.append(pos + vel * t_real)
        t_real += 1
    nodes, probs, truth, ids, nid = {}, {}, set(), [], 1
    for t, P in enumerate(frames):
        row = []
        for p in P:
            nodes[nid] = {"node_id": nid, "t": t, "z": p[0] / VOX[0], "y": p[1] / VOX[1], "x": p[2] / VOX[2]}
            row.append(nid)
            nid += 1
        ids.append(row)
    for t in range(T - 1):
        for a, b in zip(ids[t], ids[t + 1], strict=True):
            truth.add((a, b))
            if rng.random() < 0.7:
                probs[(a, b)] = float(rng.uniform(0.6, 1.0))
    return nodes, probs, truth


# ------------------------------------------------------------------ library behaviour
def test_translation_recovers_a_global_shift_with_missing_and_extra_cells():
    rng = np.random.default_rng(0)
    src = rng.uniform([0, 0, 0], [30, 120, 120], (300, 3))
    shift = np.array([1.2, 8.5, -6.0])
    tgt = np.concatenate([src[:280] + shift + rng.normal(0, 0.3, (280, 3)), rng.uniform(0, 120, (15, 3))])
    est = R.estimate_translation(src, tgt)
    assert np.linalg.norm(est - shift) < 0.2
    assert R.estimate_translation(src[:3], tgt[:3]) is None


def test_classification_and_frozen_detection():
    frames = {0: np.zeros((2, 2)), 1: np.ones((2, 2)), 2: np.ones((2, 2)), 3: np.full((2, 2), 2.0)}
    frozen = R.frozen_transitions([0, 1, 2, 3], lambda t: frames[t])
    assert frozen == [1]
    info = R.classify_transitions([0, 1, 2, 3], frozen, {0: 2.0, 1: 0.0, 2: 4.5, 3: 2.2})
    assert [info[t]["kind"] for t in range(4)] == ["normal", "frozen", "jump", "normal"]
    assert info[2]["reason"] == "after_frozen"
    info = R.classify_transitions([0, 1, 2], [], {0: 2.0, 1: 4.1, 2: 2.0})
    assert info[1] == {"kind": "jump", "k": 0, "seed_median": 4.1, "reason": "seed_median"}


# ------------------------------------------------------------------ parity with the production notebook
@pytest.mark.notebook
def test_helpers_match_the_notebook(core_cells):
    ns = _namespace(core_cells, "b1c")
    rng = np.random.default_rng(1)
    for _ in range(5):
        src = rng.uniform(0, 80, (150, 3))
        tgt = src[rng.permutation(150)[:140]] + rng.normal(0, 0.4, (140, 3)) + rng.uniform(-9, 9, 3)
        assert np.array_equal(R.estimate_translation(src, tgt), ns["x138_estimate_translation"](src, tgt))
        assert np.array_equal(R.vote_translation(src, tgt), ns["x138_vote_translation"](src, tgt))
    times = list(range(20))
    frozen = [4, 5, 11]
    medians = {t: float(rng.uniform(1, 6)) for t in times if t not in (7,)}
    ours = R.classify_transitions(times, frozen, medians)
    theirs = ns["x138_classify_transitions"](times, frozen, medians)
    assert ours.keys() == theirs.keys()
    def same(a, b):          # NaN-aware (a transition without enough seeds has seed_median = nan)
        return a == b or (isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b))

    for k in ours:
        entries = [(ours[k], theirs[k])] if k == "_movie_median" else [(ours[k][f], theirs[k][f]) for f in ours[k]]
        assert all(same(a, b) for a, b in entries), k
    assert math.isnan(ours[7]["seed_median"])


@pytest.mark.notebook
def test_jump_prior_matches_the_notebook(core_cells):
    ns = _namespace(core_cells, "b1c")
    rng = np.random.default_rng(2)
    position = {i: rng.uniform(0, 60, 3) for i in range(80)}
    info = {3: {"kind": "jump", "k": 1}, 4: {"kind": "normal", "k": 0}, 2: {"kind": "frozen", "k": 0}}
    shift = np.array([0.5, 7.0, -2.0])
    for i in range(40, 80):
        position[i] = position[i - 40] + shift + rng.normal(0, 0.2, 3)

    def flow(pos, exclude_um):
        return np.array([0.1, 0.2, -0.1])

    for prev in (flow, None):
        s1, s2 = {}, {}
        ours = R.JumpAwarePrior(info, position, s1).seed_prior(3, list(range(40)), list(range(40, 80)), prev)
        state = ns["X138RelinkState"](ns["x138_relink_config"]("b1c"), info, position, s2)
        theirs = state.seed_prior(3, list(range(40)), list(range(40, 80)), prev)
        for pos in (np.zeros(3), np.array([3.0, 4.0, 5.0])):
            assert np.array_equal(ours(pos, 1.5), theirs(pos, 1.5))
        assert s1 == s2
        assert R.JumpAwarePrior(info, position).seed_prior(4, [], [], prev) is prev


@pytest.mark.notebook
def test_b1c_recovers_links_across_a_freeze_and_the_library_prior_is_a_drop_in(core_cells):
    """Dense movie, 4.5 um/frame: the step after a duplicated frame moves 9 um and the upstream relink loses
    most links there; b1c recovers them. Swapping in the library's prior gives the identical edge set."""
    freeze_at = 5
    nodes, probs, truth = _movie(seed=5, freeze_at=freeze_at)
    t_of = {nid: n["t"] for nid, n in nodes.items()}
    jump_truth = {e for e in truth if t_of[e[0]] == freeze_at + 1}
    base, _ = _run(_namespace(core_cells, "base"), nodes, probs, frozen=(freeze_at,))
    ns = _namespace(core_cells, "b1c")
    b1c, stats = _run(ns, nodes, probs, frozen=(freeze_at,))
    assert stats["x138_transitions_frozen"] == 1 and stats["x138_transitions_jump"] >= 1
    assert len(jump_truth & set(base)) < 0.5 * len(jump_truth)
    assert len(jump_truth & set(b1c)) >= 0.9 * len(jump_truth)
    assert len(truth & set(b1c)) >= len(truth & set(base))

    ns2 = _namespace(core_cells, "b1c")
    def library_prior(self, t, s, g, prev):
        return R.JumpAwarePrior(self.info, self.position_um, self.stats).seed_prior(t, s, g, prev)

    ns2["X138RelinkState"].seed_prior = library_prior
    swapped, stats2 = _run(ns2, nodes, probs, frozen=(freeze_at,))
    assert swapped == b1c and stats2 == stats


@pytest.mark.notebook
def test_b1c_is_the_upstream_relink_on_movies_without_duplicated_frames(core_cells):
    nodes, probs, _ = _movie(seed=3, step_um=3.2, spacing_um=9.0, n=45)
    base, _ = _run(_namespace(core_cells, "base"), nodes, probs)
    b1c, stats = _run(_namespace(core_cells, "b1c"), nodes, probs)
    assert stats["x138_transitions_jump"] == 0 and b1c == base
