"""Submission CSV schema, graph validity, and parity of the committed notebook with configs/ and the kernels."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pandas as pd
import pytest
import yaml

from biohub_tracking.io.graph import TrackGraph, next_free_node_id, validate_forest
from biohub_tracking.io.submission import (
    SUBMISSION_COLUMNS,
    check_submission_covers,
    graphs_from_submission_df,
    graphs_to_submission_df,
    read_submission,
    roundtrip_integer_graph,
    write_submission,
)
from biohub_tracking.kaggle import notebook as nbk
from biohub_tracking.kaggle import variants

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import prepare_notebook  # noqa: E402

CONFIG = yaml.safe_load((ROOT / "configs" / "final.yaml").read_text())


def _graph(node_rows, edge_rows):
    return TrackGraph(pd.DataFrame(node_rows, columns=["node_id", "t", "z", "y", "x"]),
                      pd.DataFrame(edge_rows, columns=["source_id", "target_id"], dtype="int64"))


# ------------------------------------------------------------------ CSV schema
def test_round_trip_preserves_nodes_and_edges(tmp_path):
    g = _graph([(1, 0, 3.4, 10.6, 20.0), (2, 1, 3.6, 11.0, 21.2)], [(1, 2)])
    path = write_submission({"44b6_a": g, "6bba_b": TrackGraph.empty()}, tmp_path / "s.csv")
    back = read_submission(path)
    assert list(back) == ["44b6_a"]                       # an empty graph writes no rows
    assert back["44b6_a"].nodes[["z", "y", "x"]].values.tolist() == [[3, 11, 20], [4, 11, 21]]
    assert back["44b6_a"].edges.values.tolist() == [[1, 2]]


def test_schema_matches_the_competition_example():
    g = _graph([(1, 0, 32, 128, 128), (2, 1, 32, 128, 129)], [(1, 2)])
    df = graphs_to_submission_df({"44b6_0113de3b": g})
    assert tuple(df.columns) == SUBMISSION_COLUMNS
    assert df.iloc[0].tolist() == [0, "44b6_0113de3b", "node", 1, 0, 32, 128, 128, -1, -1]
    assert df.iloc[2].tolist() == [2, "44b6_0113de3b", "edge", -1, -1, -1, -1, -1, 1, 2]


def test_roundtrip_integer_graph_matches_write_read(tmp_path):
    g = _graph([(5, 0, 1.49, 2.51, 3.5), (6, 1, 1.51, 2.49, 4.5)], [(5, 6)])
    via_disk = read_submission(write_submission({"x": g}, tmp_path / "s.csv"))["x"]
    in_memory = roundtrip_integer_graph(g)
    assert via_disk.nodes.equals(in_memory.nodes) and via_disk.edges.equals(in_memory.edges)


def test_missing_columns_and_coverage():
    with pytest.raises(ValueError):
        graphs_from_submission_df(pd.DataFrame({"id": [0]}))
    assert check_submission_covers({"a": TrackGraph.empty()}, ["a", "b"]) == ["b"]


@pytest.mark.parametrize("edges, attr", [
    ([(1, 3), (2, 3)], "merge_node_ids"),
    ([(1, 2), (1, 3), (1, 4)], "multi_branch_node_ids"),
    ([(1, 5)], "dt_violations"),
    ([(1, 1)], "self_loop_node_ids"),
    ([(1, 99)], "dangling_edge_ids"),
])
def test_validate_forest_detects_each_violation(edges, attr):
    nodes = [(1, 0, 0, 0, 0), (2, 1, 0, 0, 0), (3, 1, 0, 0, 0), (4, 1, 0, 0, 0), (5, 2, 0, 0, 0)]
    v = validate_forest(_graph(nodes, edges))
    assert not v.is_valid and getattr(v, attr)


def test_validate_forest_accepts_a_division_and_next_free_id():
    g = _graph([(1, 0, 0, 0, 0), (2, 1, 0, 0, 0), (3, 1, 0, 0, 0)], [(1, 2), (1, 3)])
    assert validate_forest(g).is_valid
    assert next_free_node_id(g) == 4


# ------------------------------------------------------------------ notebook / config parity
def test_committed_notebook_matches_its_recorded_candidate(notebook_cells):
    name = CONFIG["notebook"]["candidate"]
    assert nbk.code_sha256(notebook_cells) == CONFIG["candidates"][name]["code_sha256"]
    assert prepare_notebook.check_cells(name, notebook_cells, CONFIG) == []


def test_notebook_has_no_outputs_and_only_markdown_additions():
    nb = nbk.load_notebook(ROOT / CONFIG["notebook"]["path"])
    assert {c["cell_type"] for c in nb["cells"]} <= {"code", "markdown"}
    assert all(not c.get("outputs") and c.get("execution_count") is None
               for c in nb["cells"] if c["cell_type"] == "code")
    assert set(nb["metadata"]) <= {"kernelspec", "language_info", "kaggle"}


@pytest.mark.parametrize("name", [n for n, c in CONFIG["candidates"].items() if c.get("edits") is not None])
def test_every_candidate_is_reproducible_from_the_committed_notebook(core_cells, name):
    """Whichever candidate becomes final, its exact submitted code can be regenerated and checked."""
    cand = CONFIG["candidates"][name]
    assert nbk.code_sha256(core_cells) == CONFIG["candidates"]["core"]["code_sha256"]
    cells = variants.build(core_cells, cand["edits"])
    assert nbk.code_sha256(cells) == cand["code_sha256"]
    assert variants.features(cells) == set(cand["edits"])
    assert variants.to_core(cells) == core_cells
    assert prepare_notebook.check_cells(name, cells, CONFIG) == []


def test_team_overrides_are_the_only_changes_to_upstream_settings(core_cells):
    upstream = yaml.safe_load((ROOT / "configs" / "default.yaml").read_text())["upstream_env"]
    env = nbk.env_overrides(core_cells)
    team = CONFIG["production_env"]
    assert not set(upstream) & set(team)
    assert {k: env[k] for k in upstream} == upstream
    assert {k: env[k] for k in team} == team
    assert set(env) == set(upstream) | set(team)


def test_production_constants_match_the_library(core_cells):
    """Numbers restated in src/ agree with the settings the notebook actually runs with."""
    from biohub_tracking.division import processor

    env = nbk.env_overrides(core_cells)
    assert float(env["BIOHUB_SAFE_DIV_FRAME_FRAC_CAP"]) == processor.FRAME_FRACTION
    assert float(env["BIOHUB_SAFE_DIV_GLOBAL_FRAC_CAP"]) == processor.EDGE_FRACTION
    assert CONFIG["production_env"]["BIOHUB_SUBMIT_DIVISION_ADD_CAP"] == "exempt"


def test_candidate_edits_reject_foreign_notebooks(core_cells):
    broken = list(core_cells)
    broken[nbk.POSTPROC_CELL] = broken[nbk.POSTPROC_CELL].replace("stats[\"linefit_jump_aware_failed\"] = 1", "")
    with pytest.raises(ValueError):
        variants.features(broken)
    with pytest.raises(ValueError):
        variants.add_frozen_consensus(variants.add_frozen_consensus(core_cells))


SENSITIVE = [r"ghp_[A-Za-z0-9]{20,}", r"github_pat_", r"\bsk-[A-Za-z0-9]{20,}", r"KAGGLE_KEY", r"/Users/",
             r"/home/[a-z]", r"@gmail\.com", r"CloudStorage", r"(?i)\bclaude\b", r"(?i)\bchatgpt\b",
             r"(?i)\bcodex\b", r"CLEARML_API_(ACCESS|SECRET)_KEY"]


def test_notebook_contains_no_credentials_or_personal_paths():
    text = (ROOT / CONFIG["notebook"]["path"]).read_text()
    found = {p: len(re.findall(p, text)) for p in SENSITIVE if re.search(p, text)}
    assert not found, found


def test_prepare_notebook_check_passes():
    old = sys.argv
    try:
        sys.argv = ["prepare_notebook.py", "--check"]
        assert prepare_notebook.main() == 0
    finally:
        sys.argv = old
    json.loads((ROOT / CONFIG["notebook"]["path"]).read_text())


def test_selected_submissions_are_reproducible_and_differ_only_in_the_head():
    selected = CONFIG["result"]["selected_submissions"]
    cands = [CONFIG["candidates"][name] for name in selected]
    assert len(selected) == 2 and CONFIG["notebook"]["candidate"] in selected
    assert all(c["edits"] is not None and c["code_sha256"] and isinstance(c["public_lb"], float) for c in cands)
    assert set(cands[0]["edits"]) ^ set(cands[1]["edits"]) == {"head_f03"}
    best = max(CONFIG["candidates"].values(), key=lambda c: c["public_lb"] if isinstance(c["public_lb"], float) else 0)
    assert best is CONFIG["candidates"][CONFIG["notebook"]["candidate"]]
