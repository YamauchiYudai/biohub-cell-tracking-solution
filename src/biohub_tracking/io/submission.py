"""Read/write the competition's `submission.csv` format.

Schema (checked against the competition overview's worked example and the organisers'
`scripts/geffs_to_csv.py` in royerlab/kaggle-cell-tracking-competition):

    id,dataset,row_type,node_id,t,z,y,x,source_id,target_id
    0,44b6_0113de3b,node,1,0,32,128,128,-1,-1
    1,44b6_0113de3b,edge,-1,-1,-1,-1,-1,1,2

- node rows: `node_id/t/z/y/x` are integer voxel coordinates; `source_id = target_id = -1`.
- edge rows: `source_id/target_id` are node_id values (not row indices); `node_id/t/z/y/x = -1`.
- `id` is a 0..N-1 row index.
- `dataset` equals the test `.zarr` folder stem (`{embryo_id}_{fov_id}`), and every test dataset must appear.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .graph import TrackGraph

SUBMISSION_COLUMNS: tuple[str, ...] = (
    "id", "dataset", "row_type", "node_id", "t", "z", "y", "x", "source_id", "target_id",
)


def graphs_to_submission_df(graphs: dict[str, TrackGraph]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for name, g in graphs.items():
        n = g.nodes
        frames.append(pd.DataFrame({
            "dataset": name, "row_type": "node",
            "node_id": n["node_id"].astype("int64"), "t": n["t"].astype("int64"),
            "z": n["z"].round().astype("int64"), "y": n["y"].round().astype("int64"),
            "x": n["x"].round().astype("int64"),
            "source_id": -1, "target_id": -1,
        }))
        e = g.edges
        frames.append(pd.DataFrame({
            "dataset": name, "row_type": "edge",
            "node_id": -1, "t": -1, "z": -1, "y": -1, "x": -1,
            "source_id": e["source_id"].astype("int64"), "target_id": e["target_id"].astype("int64"),
        }))

    if frames:
        df = pd.concat(frames, ignore_index=True)
    else:
        df = pd.DataFrame({c: pd.Series(dtype="int64") for c in SUBMISSION_COLUMNS[1:]})
        df["dataset"] = df["dataset"].astype("object")
        df["row_type"] = df["row_type"].astype("object")

    df.insert(0, "id", range(len(df)))
    return df[list(SUBMISSION_COLUMNS)]


def write_submission(graphs: dict[str, TrackGraph], path: Path | str) -> Path:
    path = Path(path)
    graphs_to_submission_df(graphs).to_csv(path, index=False)
    return path


def graphs_from_submission_df(df: pd.DataFrame) -> dict[str, TrackGraph]:
    """Rebuild per-dataset graphs from a submission-schema DataFrame."""
    missing = set(SUBMISSION_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"submission csv missing columns: {sorted(missing)}")

    graphs: dict[str, TrackGraph] = {}
    for name, group in df.groupby("dataset", sort=False):
        nodes = (group.loc[group["row_type"] == "node", ["node_id", "t", "z", "y", "x"]]
                 .astype("int64").reset_index(drop=True))
        edges = (group.loc[group["row_type"] == "edge", ["source_id", "target_id"]]
                 .astype("int64").reset_index(drop=True))
        graphs[str(name)] = TrackGraph(nodes=nodes, edges=edges)
    return graphs


def roundtrip_integer_graph(graph: TrackGraph) -> TrackGraph:
    """Integerize z/y/x the way `write_submission` then `read_submission` would.

    Local scoring goes through this so that an edge whose endpoints sit near the 7 um matching gate is
    scored on the same integer grid a real submission writes.
    """
    restored = graphs_from_submission_df(graphs_to_submission_df({"_": graph}))
    return restored.get("_", TrackGraph.empty())


def read_submission(path: Path | str) -> dict[str, TrackGraph]:
    return graphs_from_submission_df(pd.read_csv(Path(path)))


def check_submission_covers(graphs: dict[str, TrackGraph], required_dataset_names: list[str]) -> list[str]:
    """Return the subset of `required_dataset_names` missing from `graphs` (empty list == all present)."""
    return [name for name in required_dataset_names if name not in graphs]
