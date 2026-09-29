"""Read a ground-truth `.geff` track graph (competition train data) into a :class:`TrackGraph`.

Layout used by the competition (geff v1.x on zarr v3):

    {name}.geff/
      zarr.json                         group; attributes.geff.extra.estimated_number_of_nodes
      nodes/ids                         (N,) node ids
      nodes/props/{t,z,y,x}/values      (N,) voxel coordinates
      edges/ids                         (E, 2) (source_id, target_id) as node-id values

Needs the optional ``zarr`` dependency (``pip install -e ".[metric]"``).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .graph import EDGE_COLUMNS, TrackGraph, empty_frame


class GeffReadError(RuntimeError):
    """A missing or incomplete .geff (for example a partially downloaded dataset)."""


def read_geff(path: Path | str) -> TrackGraph:
    import zarr

    geff_dir = Path(path)
    if not (geff_dir / "zarr.json").exists():
        raise GeffReadError(f"{geff_dir} is not a geff group")
    group = zarr.open_group(str(geff_dir), mode="r")
    try:
        ids = np.asarray(group["nodes/ids"])
        props = {axis: np.asarray(group[f"nodes/props/{axis}/values"]) for axis in "tzyx"}
    except KeyError as exc:
        raise GeffReadError(f"{geff_dir}: missing node array {exc}") from exc
    if any(len(v) != len(ids) for v in props.values()):
        raise GeffReadError(f"{geff_dir}: node id / property length mismatch")
    if len(ids) and np.all(props["t"] == 0) and np.all(props["z"] == 0):
        # zarr returns the fill value for absent chunks: a truncated download reads as every node at the origin
        raise GeffReadError(f"{geff_dir}: node coordinates are all zero (truncated download?)")

    nodes = pd.DataFrame({
        "node_id": ids.astype("int64"), "t": props["t"].astype("int64"),
        "z": props["z"].astype("float64"), "y": props["y"].astype("float64"), "x": props["x"].astype("float64"),
    })
    edges = empty_frame(EDGE_COLUMNS)
    if "edges" in group and "ids" in group["edges"]:
        pairs = np.asarray(group["edges/ids"]).reshape(-1, 2)
        if len(pairs):
            edges = pd.DataFrame({"source_id": pairs[:, 0].astype("int64"),
                                  "target_id": pairs[:, 1].astype("int64")})
    extra = (dict(group.attrs).get("geff") or {}).get("extra") or {}
    return TrackGraph(nodes=nodes, edges=edges, estimated_number_of_nodes=extra.get("estimated_number_of_nodes"))
