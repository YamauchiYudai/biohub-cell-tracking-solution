"""Division recovery: unit behaviour, and parity with the code embedded in the production notebook."""

from __future__ import annotations

import copy
import json
import types

import numpy as np
import pandas as pd
import pytest
from conftest import UM_Y, small_graph, synthetic_movie

from biohub_tracking.division import candidates as C
from biohub_tracking.division import features as F
from biohub_tracking.division import processor as P
from biohub_tracking.division import select as S
from biohub_tracking.division import t3 as T3
from biohub_tracking.division import thresholds as TH
from biohub_tracking.division.scoring import T3Scorer
from biohub_tracking.kaggle import notebook as nbk

NO_GEOMETRY = {"max_mirror_um": None, "min_p_d2_um": None, "max_p_d2_um": None, "min_angle_deg": None}


def _nodes_by_id(nodes: pd.DataFrame) -> dict:
    return {int(r.node_id): {"node_id": int(r.node_id), "t": int(r.t), "z": r.z, "y": r.y, "x": r.x}
            for r in nodes.itertuples()}


def _stub_scores(crops: np.ndarray) -> np.ndarray:
    """Deterministic pseudo-scores spread over [0, 1) (exercises both thresholds)."""
    return (crops.reshape(len(crops), -1).astype(np.int64).sum(1) % 997) / 997.0


# ------------------------------------------------------------------ unit behaviour
def test_enumerate_finds_the_mirror_orphan_and_free_low_peaks():
    nodes, edges = small_graph()
    low = (np.array([0, 1]), np.array([[2, 20, 100 + 5 * UM_Y, 100], [2, 20, 100, 100 + 5 * UM_Y]], float),
           np.array([0.5, 0.4]))
    c = C.enumerate_candidates(nodes, edges, np.empty((0, 2), int), low)
    a = c[c["shape"] == "a"]
    assert list(zip(a.p_id, a.d1_id, a.d2_id, strict=True)) == [(11, 21, 22)]
    r = a.iloc[0]
    assert r.angle_deg == pytest.approx(180.0) and r.p_d2_um == pytest.approx(5.0) and r.mirror_um < 1e-6
    assert r.t2_gain_um == pytest.approx(14.0 - 10.0)
    d = c[c["shape"] == "d"]
    assert list(d.d2_id) == [-2] and d.iloc[0].angle_deg == pytest.approx(90.0)   # the first peak sits on node 22


def test_choose_adds_best_mirror_dedups_and_removes_low_safe_forks():
    nodes, edges = small_graph()
    c = C.enumerate_candidates(nodes, edges, np.empty((0, 2), int), None)
    c["s"] = 0.95
    adds, removes = S.choose(c, "s", 0.9, 0.5)
    assert list(adds.d2_id) == [22] and removes.empty
    assert S.choose(c, "s", 0.99, None)[0].empty
    c = C.enumerate_candidates(nodes, edges, np.array([[11, 22]]), None)       # safe-div already made the fork
    c["s"] = 0.2
    adds, removes = S.choose(c, "s", 0.1, 0.5)
    assert adds.empty and list(zip(removes.p_id, removes.d2_id, strict=True)) == [(11, 22)]
    c["s"] = 0.9
    adds, removes = S.choose(c, "s", 0.1, 0.5)
    assert adds.empty and removes.empty


def test_apply_forks_edits_only_the_division_edges():
    nodes, edges = small_graph()
    nbi = _nodes_by_id(nodes)
    elist = [{"source_id": int(s), "target_id": int(t)} for s, t in edges]
    elist.append({"source_id": 11, "target_id": 22, "safe_division": 1})
    low = (np.array([7]), np.array([[2, 20, 200 + 4 * UM_Y, 200]], float), np.array([0.6]))
    c = C.enumerate_candidates(nodes, edges, np.array([[11, 22]]), low)
    c["s"] = np.where(c.p_id == 11, 0.1, 0.95)
    adds, removes = S.choose(c, "s", 0.9, 0.5)
    stats = {}
    out = S.apply_forks(nbi, elist, adds, removes, stats)
    assert (11, 22) not in {(e["source_id"], e["target_id"]) for e in out}
    new = [e for e in out if e.get("learned_division")]
    assert len(new) == 1 and (new[0]["source_id"], new[0]["target_id"]) == (41, 44)
    assert nbi[44]["t"] == 2 and nbi[44]["learned_division_node"] == 1
    assert stats == {"learned_div_added": 1, "learned_div_added_nodes": 1, "learned_div_removed": 1}


def test_geometry_filter_keeps_safe_forks():
    nodes, edges = small_graph()
    c = C.enumerate_candidates(nodes, edges, np.array([[11, 22]]), None)
    gated = P.geometry_filter(c, max_mirror_um=0.1, min_angle_deg=150)
    assert set(gated[gated.in_safe_div == 1].d2_id) == {22} and len(gated) <= len(c)


def test_learned_adds_share_the_safe_div_budget_unless_exempt():
    """V5a: learned adds lose to safe-div's division budget unless exempt (the production setting)."""
    nodes, edges = small_graph()
    nodes = pd.concat([nodes, pd.DataFrame([(44, 2, 20, 200 + 4 * UM_Y, 200)], columns=nodes.columns)],
                      ignore_index=True)
    pre = [{"source_id": int(s), "target_id": int(t)} for s, t in edges]
    safe = [*pre, {"source_id": 41, "target_id": 44, "safe_division": 1}]

    def score(c, *_):
        return c.assign(s=np.where(c.p_id == 11, 0.9, 0.0))

    kept = {}
    for scale in (1.0, 1.5, None):
        proc = P.DivisionProcessor(score_fn=score, score_column="s", tau_add=0.5, add_cap_scale=scale)
        out = proc.process(_nodes_by_id(nodes), list(safe), {}, "44b6_test", pre)
        kept[scale] = {(e["source_id"], e["target_id"]) for e in out if e.get("learned_division")}
        assert proc.log["44b6_test"]["adds_before_cap"] == 1
        assert (41, 44) in {(e["source_id"], e["target_id"]) for e in out}
    assert kept[1.0] == set() and kept[1.5] == set() and kept[None] == {(11, 22)}
    with pytest.raises(ValueError):
        P.DivisionProcessor(score_fn=score, add_cap_scale=0)


def test_t3_crop_pads_and_repeats_edge_frames():
    frames = {t: np.full((20, 60, 60), t + 1, dtype=np.uint16) for t in range(3)}
    crop = T3.crop_from_frames(lambda t: frames[t], 0, 3, (2, 5, 30))
    assert crop.shape == (3, *T3.CROP)
    assert crop[0].max() == 1 and crop[1].max() == 1 and crop[2].max() == 2
    assert crop[1, 0, 0, 0] == 0 and crop[1, 8, 24, 24] == 1
    assert T3.center_crop(np.zeros((2, 3, *T3.EXPORT), np.uint16)).shape == (2, 3, *T3.CROP)
    x = T3.normalize(np.random.default_rng(0).integers(0, 1000, (2, 3, *T3.CROP)).astype(np.uint16))
    assert x.dtype == np.float32 and x.min() >= -1 and x.max() <= 4


def test_t3_model_and_views():
    torch = pytest.importorskip("torch")
    model = T3.build_model(width=4)
    assert model(torch.zeros(2, 3, *T3.CROP)).shape == (2, 1)
    crops = np.random.default_rng(1).integers(0, 500, (3, 3, *T3.CROP)).astype(np.uint16)
    for views in (1, 2, 8):
        s = T3.predict([model], crops, views=views)
        assert s.shape == (3,) and np.all((s > 0) & (s < 1))


def test_threshold_selection_rules():
    rows = []
    for i, (score, label) in enumerate([(0.9, "tp"), (0.8, "tp"), (0.7, "fp"), (0.6, "tp"), (0.4, "fp"),
                                        (0.35, "fp"), (0.2, "fp")]):
        rows.append({"movie": "44b6_m", "embryo": "44b6", "shape": "a", "p_id": i, "p_t": 3 * i, "p_z": 0.0,
                     "p_y": 50.0 * i, "p_x": 0.0, "d2_id": 100 + i, "mirror_um": 0.0, "in_safe_div": 0,
                     "label": label, "s": score})
    cands = pd.DataFrame(rows)
    table = TH.sweep(cands, "s", {"tp": 0, "fp": 0, "gt": 5}, "add")
    tau = TH.pick_add(table)
    row = table[table.tau == tau].iloc[0]
    assert row.precision_add >= TH.MIN_PRECISION and row.added_tp + row.added_fp >= TH.MIN_ADDS
    lower = table[table.tau < tau]
    assert (lower.precision_add < TH.MIN_PRECISION).all() or (lower.added_tp + lower.added_fp < TH.MIN_ADDS).all()


# ------------------------------------------------------------------ parity with the embedded modules
@pytest.mark.notebook
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_candidates_select_and_processor_match_the_notebook(embedded, seed):
    nbi, pre, safe, cache, _ = synthetic_movie(seed)
    nodes = P.node_table(nbi)
    pre_pairs, selected = P.edge_pairs(pre), P.selected_pairs(pre, safe)
    low = C.low_peaks(cache)
    ours = C.enumerate_candidates(nodes, pre_pairs, selected, low)
    theirs = embedded["division_candidates"].enumerate_candidates(nodes, pre_pairs, selected, low)
    pd.testing.assert_frame_equal(ours, theirs)
    assert len(ours) > 0 and ours.in_safe_div.sum() > 0

    ours["s"] = theirs["s"] = np.random.default_rng(seed).uniform(size=len(ours))
    for tau_add, tau_del in ((0.3, 0.01), (0.5, 0.2), (None, 0.5)):
        a1, r1 = S.choose(ours, "s", tau_add, tau_del, shapes=("a",))
        a2, r2 = embedded["division_apply"].choose(theirs, "s", tau_add, tau_del, shapes=("a",))
        pd.testing.assert_frame_equal(a1, a2)
        pd.testing.assert_frame_equal(r1, r2)

    def score(c, *_):
        return c.assign(s=np.random.default_rng(seed + 10).uniform(size=len(c)))

    for scale in (None, 1.0):
        n1, n2 = copy.deepcopy(nbi), copy.deepcopy(nbi)
        ours_p = P.DivisionProcessor(score_fn=score, score_column="s", tau_add=0.3, tau_del=0.01,
                                     geometry=NO_GEOMETRY, cache_fn=lambda d: cache, add_cap_scale=scale)
        theirs_p = embedded["division_postprocess"].DivisionProcessor(
            score_fn=score, score_column="s", tau_add=0.3, tau_del=0.01, geometry=NO_GEOMETRY,
            cache_fn=lambda d: cache, add_cap_scale=scale)
        out1 = ours_p.process(n1, list(safe), {}, "6bba_test", pre)
        out2 = theirs_p.process(n2, list(safe), {}, "6bba_test", pre)
        assert out1 == out2 and n1 == n2
        assert ours_p.log == theirs_p.log


@pytest.mark.notebook
def test_edge_probability_features_match_the_notebook(embedded):
    nbi, pre, safe, cache, _ = synthetic_movie(3)
    nodes = P.node_table(nbi)
    cands = C.enumerate_candidates(nodes, P.edge_pairs(pre), P.selected_pairs(pre, safe), None)
    ours = F.add_edge_probabilities(cands.copy(), nodes, cache)
    theirs = embedded["division_planb"].enrich_candidates(
        cands.copy(), nodes, P.edge_pairs(pre), edge_cache=cache,
        required={"prob_p_d2", "prob_p_d2_observed"})
    cols = ["prob_p_d1", "prob_p_d2", "prob_p_d1_observed", "prob_p_d2_observed"]
    pd.testing.assert_frame_equal(ours[cols], theirs[cols])
    assert ours.prob_p_d2_observed.sum() > 0
    assert F.cache_alignment(nodes, cache) and embedded["division_planb"].cache_alignment(nodes, cache)


@pytest.mark.notebook
def test_t3_crop_and_normalisation_match_the_notebook(embedded):
    rng = np.random.default_rng(4)
    frames = {t: rng.integers(0, 4000, (12, 70, 70), dtype=np.uint16) for t in range(4)}
    for t, center in ((0, (2, 5, 30)), (2, (6.4, 35.6, 69.0)), (3, (11, 0, 0))):
        a = T3.crop_from_frames(lambda k: frames[k], t, 4, center)
        b = embedded["division_t3"].crop_from_frames(lambda k: frames[k], t, 4, center)
        assert np.array_equal(a, b)
    crops = rng.integers(0, 4000, (5, 3, *T3.CROP)).astype(np.uint16)
    assert np.array_equal(T3.normalize(crops), embedded["division_t3"].normalize(crops))
    assert (T3.CROP, T3.VIEWS) == (embedded["division_t3"].CROP, embedded["division_t3"].VIEWS)


@pytest.mark.notebook
@pytest.mark.parametrize("seed", [5, 6])
def test_production_scorer_and_hook_match_the_notebook(embedded, notebook_cells, tmp_path, seed):
    """The notebook's `_dv_score` (prob_angle_t2 gate + T3) and our T3Scorer give identical decisions."""
    nbi, pre, safe, cache, read_frame = synthetic_movie(seed)
    n_frames = 1 + max(n["t"] for n in nbi.values())
    dataset = "6bba_test"
    (tmp_path / f"{dataset}.zarr" / "0").mkdir(parents=True)
    (tmp_path / f"{dataset}.zarr" / "0" / "zarr.json").write_text(json.dumps({"shape": [n_frames, 16, 256, 256]}))
    t3_stub = types.SimpleNamespace(crop_from_frames=embedded["division_t3"].crop_from_frames,
                                    predict=lambda models, crops, device=None: _stub_scores(crops))
    ns = {"np": np, "pd": pd, "_dv_json": json, "_DvPath": type(tmp_path), "TEST_DIR": tmp_path,
          "_DV_ADD_GATE": "prob_angle_t2", "_DV_USE_BL": False, "_DV_USE_A": True, "_DV_USE_B": False,
          "_DV_METHOD": "A", "_DV_SCORE_LOG": {}, "_dv_b": embedded["division_planb"], "_dv_t3": t3_stub,
          "_DV_T3_MODELS": [], "_DV_DEVICE": "cpu", "read_test_frame": read_frame}
    exec(nbk.extract_function(notebook_cells[nbk.POSTPROC_CELL], "_dv_score"), ns)

    scorer = T3Scorer(predict=_stub_scores, read_frame=read_frame, n_frames=lambda d: n_frames)
    nodes = P.node_table(nbi)
    cands = C.enumerate_candidates(nodes, P.edge_pairs(pre), P.selected_pairs(pre, safe), C.low_peaks(cache))
    ours = scorer(cands.copy(), nodes, P.edge_pairs(pre), dataset, cache)
    theirs = ns["_dv_score"](cands.copy(), nodes, P.edge_pairs(pre), dataset, cache)
    cols = ["add_gate", "A", "division_score"]
    pd.testing.assert_frame_equal(ours[cols], theirs[cols])
    assert ours.add_gate.sum() > 0

    graphs = [{k: dict(v) for k, v in nbi.items()} for _ in range(2)]
    procs = [P.DivisionProcessor(score_fn=scorer, score_column="division_score", tau_add=0.3, tau_del=0.01,
                                 geometry=NO_GEOMETRY, cache_fn=lambda d: cache, add_cap_scale=None),
             embedded["division_postprocess"].DivisionProcessor(
                 score_fn=ns["_dv_score"], score_column="division_score", tau_add=0.3, tau_del=0.01,
                 geometry=NO_GEOMETRY, cache_fn=lambda d: cache, add_cap_scale=None)]
    outs = [proc.process(g, list(safe), {}, dataset, pre) for proc, g in zip(procs, graphs, strict=True)]
    assert outs[0] == outs[1] and graphs[0] == graphs[1]


def test_official_rule_labels_agree_with_the_scorer():
    pytest.importorskip("tracking_cellmot")
    from biohub_tracking.division import labels as L
    from biohub_tracking.io.graph import TrackGraph

    nodes, edges = small_graph()
    gt = TrackGraph(nodes.assign(node_id=nodes.node_id + 1000),
                    pd.DataFrame(np.vstack([edges, [[11, 22]]]) + 1000, columns=["source_id", "target_id"]))
    out, inc = L.adjacency(edges)
    lab = L.ForkLabeler(nodes, edges, gt, L.Adj(out, inc))
    assert lab.label(11, 21, 22, L.Adj(out, inc, [], [(11, 22)]))["label"] == "tp"
    nodes2 = pd.concat([nodes, pd.DataFrame([(50, 2, 20, 200 + 4 * UM_Y, 200)], columns=nodes.columns)])
    res = L.ForkLabeler(nodes2, edges, gt, L.Adj(out, inc)).label(41, 42, 50, L.Adj(out, inc, [], [(41, 50)]))
    assert res["label"] == "fp" and res["fp_reason"] == "annotated_continuing"
    empty = np.empty((0, 2), int)
    checks = (L.verify_labels(pd.DataFrame([{"p_id": 11, "d2_id": 22, "shape": "a", "label": "tp"}]),
                              nodes, edges, empty, gt)
              + L.verify_labels(pd.DataFrame([{"p_id": 41, "d2_id": 50, "shape": "a", "label": "fp"}]),
                                nodes2, edges, empty, gt))
    assert all(v["agree"] for v in checks), checks
    table, summary = L.label_movie("44b6_test", nodes, edges, empty, gt)
    assert summary["gt_divisions"] == 1 and set(table.label) <= {"tp", "fp", "ignored"}
