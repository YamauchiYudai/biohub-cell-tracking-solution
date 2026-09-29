"""Shared fixtures: the committed production notebook and small synthetic movies."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "final_submission.ipynb"
UM_Y = 1 / 0.40625          # voxels per um in y / x


@pytest.fixture(scope="session")
def notebook_cells() -> list[str]:
    from biohub_tracking.kaggle import notebook as nbk

    return nbk.code_cells(nbk.load_notebook(NOTEBOOK))


@pytest.fixture(scope="session")
def core_cells(notebook_cells) -> list[str]:
    """The core notebook (submission 56643954) recovered from whichever candidate is committed."""
    import yaml

    from biohub_tracking.kaggle import notebook as nbk
    from biohub_tracking.kaggle import variants

    core_sha = yaml.safe_load((ROOT / "configs" / "final.yaml").read_text())["candidates"]["core"]["code_sha256"]
    try:
        core = variants.to_core(notebook_cells)
    except ValueError:
        core = None
    if core is None or nbk.code_sha256(core) != core_sha:
        pytest.skip("the committed notebook is not an edit of the core (e.g. v_add): core-based tests do not apply")
    return core


@pytest.fixture()
def embedded(notebook_cells, tmp_path):
    """The division runtime modules exactly as embedded in the notebook, importable by their own names."""
    from biohub_tracking.kaggle import notebook as nbk

    modules = nbk.embedded_modules(notebook_cells)
    for name, src in modules.items():
        (tmp_path / name).write_text(src)
    names = [n[:-3] for n in modules]
    saved = {n: sys.modules.pop(n) for n in names if n in sys.modules}
    sys.path.insert(0, str(tmp_path))
    try:
        yield {n: importlib.import_module(n) for n in names}
    finally:
        sys.path.remove(str(tmp_path))
        for n in names:
            sys.modules.pop(n, None)
        sys.modules.update(saved)


def small_graph():
    """Two cells over 4 frames. Cell 1 divides between t=1 and t=2: P=11 at t=1 -> D1=21, and an orphan D2=22
    opposite D1. Cell 2 (ids 4x) continues and is far away."""
    rows = [
        (10, 0, 20, 100, 100), (11, 1, 20, 100, 100), (21, 2, 20, 100 - 5 * UM_Y, 100),
        (22, 2, 20, 100 + 5 * UM_Y, 100), (31, 3, 20, 100 - 7 * UM_Y, 100), (32, 3, 20, 100 + 7 * UM_Y, 100),
        (40, 0, 20, 200, 200), (41, 1, 20, 200, 200), (42, 2, 20, 200 - 2 * UM_Y, 200), (43, 3, 20, 200, 200),
    ]
    nodes = pd.DataFrame(rows, columns=["node_id", "t", "z", "y", "x"])
    edges = np.array([[10, 11], [11, 21], [21, 31], [22, 32], [40, 41], [41, 42], [42, 43]])
    return nodes, edges


def synthetic_movie(seed: int, n_cells: int = 40, n_frames: int = 8):
    """A random movie with dividing cells, an aligned edge cache and deterministic raw frames.

    Returns (nodes_by_id, pre_edges, safe_edges, cache, read_frame). Node ids are the cache row numbers, as in
    the notebook (ILP node ids index the cached detections).
    """
    rng = np.random.default_rng(seed)
    rows, edges = [], []
    tracks = [(rng.uniform([4, 20, 20], [12, 236, 236]), None) for _ in range(n_cells)]
    next_id = 0
    for t in range(n_frames):
        new_tracks = []
        for pos, parent in tracks:
            pos = pos + rng.normal(0, [0.3, 2.0, 2.0])
            nid = next_id
            next_id += 1
            rows.append((nid, t, *pos))
            if parent is not None:
                edges.append((parent, nid))
            if t < n_frames - 2 and rng.random() < 0.08:        # division: an orphan sister appears at t + 1
                offset = rng.normal(0, [0.5, 8.0, 8.0])
                new_tracks.append((pos + offset, nid))
                new_tracks.append((pos - offset, None))          # the sister starts a new track (no parent)
            else:
                new_tracks.append((pos, nid))
        tracks = new_tracks
    nodes = pd.DataFrame(rows, columns=["node_id", "t", "z", "y", "x"])
    nodes_by_id = {int(r.node_id): {"node_id": int(r.node_id), "t": int(r.t), "z": float(r.z), "y": float(r.y),
                                    "x": float(r.x)} for r in nodes.itertuples()}
    pre = [{"source_id": int(s), "target_id": int(d)} for s, d in edges]
    # safe-div made a few forks: parents with one child and a nearby orphan at t + 1
    parented = {d for _, d in edges}
    children: dict[int, int] = {}
    for s, _ in edges:
        children[s] = children.get(s, 0) + 1
    child_of = {s: d for s, d in edges}
    scale = np.array([1.625, 0.40625, 0.40625])

    def um(a, b):
        return float(np.linalg.norm((np.array([a["z"], a["y"], a["x"]]) - np.array([b["z"], b["y"], b["x"]])) * scale))

    safe = list(pre)
    for p, n in nodes_by_id.items():
        if children.get(p) != 1 or rng.random() > 0.5:
            continue
        d1 = nodes_by_id[child_of[p]]
        if um(n, d1) > 10.0:
            continue
        orphans = [o for o, m in nodes_by_id.items() if m["t"] == n["t"] + 1 and o not in parented
                   and um(n, m) <= 9.0 and um(d1, m) <= 14.0]          # safe-div's own geometry limits
        if orphans:
            safe.append({"source_id": p, "target_id": orphans[0], "safe_division": 1})
            parented.add(orphans[0])
    coords = nodes[["t", "z", "y", "x"]].to_numpy(float)
    src, tgt, prob = [], [], []
    for p, n in nodes_by_id.items():
        for o, m in nodes_by_id.items():
            if m["t"] == n["t"] + 1 and abs(m["y"] - n["y"]) < 30 and abs(m["x"] - n["x"]) < 30:
                src.append(p)
                tgt.append(o)
                prob.append(float(rng.uniform(0.3, 1.0)))
    keep = np.asarray(prob) > 0.48
    cache = {"coords": coords, "edge_src": np.asarray(src)[keep], "edge_tgt": np.asarray(tgt)[keep],
             "edge_prob": np.asarray(prob)[keep],
             "low_coords": np.empty((0, 4)), "low_score": np.empty(0)}
    frames = {t: rng.integers(0, 1000, (16, 256, 256), dtype=np.uint16) for t in range(n_frames)}

    def read_frame(dataset, t, cache_dict=None):
        return frames[int(t)]

    return nodes_by_id, pre, safe, cache, read_frame
