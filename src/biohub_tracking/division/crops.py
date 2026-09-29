"""Export T3 training crops from annotated train movies (CPU).

Each row is a 3-frame (t-1, t, t+1) native-image crop. The model input is z16 x y48 x x48 (26 x 19.5 x 19.5 um);
the export keeps a margin (z18 x y64 x x64) so training can shift the input by up to 1 z-slice and 8 xy-pixels
(3.3 um): at inference the crop is centred on a *predicted* parent, which sits a voxel or two off the GT
centre. Positives are centred at the frame immediately before the daughters appear, plus the adjacent event
frames. Negatives are annotated continuing cells away from any annotated division lineage (at most 30 per
movie). The 10x positive repetition belongs to training, not to this export.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np

CROP_SHAPE = np.array([18, 64, 64], dtype=int)
MAX_NEG_PER_MOVIE = 30


def read_gt_arrays(geff_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(node ids, (N, 4) t/z/y/x, (E, 2) edges) of a GT .geff."""
    import zarr

    g = zarr.open_group(str(geff_dir), mode="r")
    ids = np.asarray(g["nodes/ids"], dtype=np.int64)
    pos = np.column_stack([np.asarray(g[f"nodes/props/{axis}/values"]) for axis in "tzyx"]).astype(np.float64)
    edges = np.asarray(g["edges/ids"], dtype=np.int64).reshape(-1, 2)
    return ids, pos, edges


def division_events(ids: np.ndarray, pos: np.ndarray, edges: np.ndarray, n_frames: int, max_neg: int,
                    seed: int) -> list[dict]:
    """Positive (division) and negative (continuing cell) crop centres of one movie."""
    node = {int(n): row for n, row in zip(ids, pos, strict=True)}
    children: dict[int, list[int]] = {}
    parent: dict[int, int] = {}
    for a, b in edges:
        children.setdefault(int(a), []).append(int(b))
        parent[int(b)] = int(a)
    dividing = {n: ch for n, ch in children.items() if len(ch) == 2}
    around_division = set()
    positive = []
    for divider, daughters in dividing.items():
        p = node[divider]
        t = int(p[0])
        local = {divider, *daughters}
        predecessor = parent.get(divider)
        if predecessor is not None:
            local.add(predecessor)
            if parent.get(predecessor) is not None:
                local.add(parent[predecessor])
        for daughter in daughters:
            local.update(children.get(daughter, []))
        around_division.update(local)
        centers = [(t, p[1:], 0)]
        if predecessor is not None:
            centers.append((t - 1, node[predecessor][1:], -1))
        centers.append((t + 1, (node[daughters[0]][1:] + node[daughters[1]][1:]) / 2.0, 1))
        for center_t, xyz, offset in centers:
            if 1 <= center_t < n_frames - 1:
                positive.append({"t": center_t, "center": xyz, "label": 1, "node_id": divider,
                                 "event_parent_id": divider, "event_offset": offset})
    negative = []
    for n, row in node.items():
        t = int(row[0])
        if (n in around_division or n not in parent or len(children.get(n, [])) != 1
                or not 1 <= t < n_frames - 1):
            continue
        negative.append({"t": t, "center": row[1:], "label": 0, "node_id": n, "event_parent_id": -1,
                         "event_offset": 0})
    rng = np.random.default_rng(seed)
    if len(negative) > max_neg:
        chosen = rng.choice(len(negative), max_neg, replace=False)
        negative = [negative[int(i)] for i in sorted(chosen)]
    return sorted([*positive, *negative], key=lambda r: (r["t"], -r["label"]))


def crop(image, center_t: int, center: np.ndarray) -> np.ndarray:
    spatial = np.asarray(image.shape[1:], dtype=int)
    start = np.rint(center).astype(int) - CROP_SHAPE // 2
    lo, hi = np.maximum(start, 0), np.minimum(start + CROP_SHAPE, spatial)
    result = np.zeros((3, *CROP_SHAPE), dtype=np.uint16)
    if np.any(lo >= hi):
        return result
    dest_lo = lo - start
    dest_hi = dest_lo + (hi - lo)
    for i, frame in enumerate((center_t - 1, center_t, center_t + 1)):
        result[i, dest_lo[0]:dest_hi[0], dest_lo[1]:dest_hi[1], dest_lo[2]:dest_hi[2]] = (
            image[frame, lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]])
    return result


def export(train_dir: Path, out: Path, stems: list[str] | None = None, max_neg: int = MAX_NEG_PER_MOVIE) -> dict:
    """Write ``<out>/<movie>.npz`` crops for every annotated train movie and a ``manifest.json``."""
    import zarr

    geffs = sorted(Path(train_dir).glob("*.geff"))
    if stems is not None:
        chosen = set(stems)
        geffs = [p for p in geffs if p.stem in chosen]
        missing = chosen - {p.stem for p in geffs}
        if missing:
            raise FileNotFoundError(f"no .geff for {sorted(missing)}")
    out.mkdir(parents=True, exist_ok=True)
    counts: dict[str, dict] = {}
    for rank, gt_path in enumerate(geffs):
        stem = gt_path.stem
        image = zarr.open(str(Path(train_dir) / f"{stem}.zarr" / "0"), mode="r")
        if len(image.shape) != 4:
            raise ValueError(f"{stem}: expected a (t, z, y, x) image, got {image.shape}")
        ids, pos, edges = read_gt_arrays(gt_path)
        events = division_events(ids, pos, edges, int(image.shape[0]), max_neg, seed=20260926 + rank)
        if not events:
            continue
        crops = np.stack([crop(image, e["t"], np.asarray(e["center"])) for e in events])
        np.savez_compressed(
            out / f"{stem}.npz", crops=crops,
            label=np.asarray([e["label"] for e in events], dtype=np.uint8),
            t=np.asarray([e["t"] for e in events], dtype=np.int16),
            center_zyx=np.stack([e["center"] for e in events]).astype(np.float32),
            node_id=np.asarray([e["node_id"] for e in events], dtype=np.int64),
            event_parent_id=np.asarray([e["event_parent_id"] for e in events], dtype=np.int64),
            event_offset=np.asarray([e["event_offset"] for e in events], dtype=np.int8))
        labels = Counter(e["label"] for e in events)
        counts[stem] = {"positive": labels[1], "negative": labels[0],
                        "gt_divisions": int(sum(1 for _, f in Counter(edges[:, 0]).items() if f == 2))}
        print(stem, counts[stem], flush=True)
    manifest = {
        "shape": f"N x time=3 x z={CROP_SHAPE[0]} x y={CROP_SHAPE[1]} x x={CROP_SHAPE[2]}", "dtype": "uint16",
        "model_input": "z=16 x y=48 x x=48 (centre crop at evaluation, shifted crop in training)",
        "voxel_um": [1.625, 0.40625, 0.40625], "time_order": "t-1,t,t+1",
        "positive": "GT division at t, plus centres at t-1 and t+1",
        "negative": "annotated continuing cell, excluding division and adjacent lineage nodes",
        "negative_cap_per_movie": max_neg, "counts": counts,
        "totals": {"movies": len(counts), "positive": sum(v["positive"] for v in counts.values()),
                   "negative": sum(v["negative"] for v in counts.values()),
                   "gt_divisions": sum(v["gt_divisions"] for v in counts.values())},
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest
