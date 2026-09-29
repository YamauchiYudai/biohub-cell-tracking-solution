"""V1284 coordinate head fine-tuning (F03). Needs torch; skipped otherwise."""

from __future__ import annotations

import numpy as np
import pytest

from biohub_tracking.detection import coordinate_head as CH


def test_mutual_pairs_are_mutual_nearest_within_7um():
    det = np.array([[0, 0, 0], [0, 0, 3.0], [0, 0, 20.0]])
    gt = np.array([[0, 0, 0.5], [0, 0, 2.5]])
    t_det, t_gt = np.zeros(3, int), np.zeros(2, int)
    di, gi = CH.mutual_pairs(det, t_det, gt, t_gt)
    assert sorted(zip(di.tolist(), gi.tolist(), strict=True)) == [(0, 0), (1, 1)]
    assert len(CH.mutual_pairs(det, t_det, gt, np.ones(2, int))[0]) == 0      # different frames never pair


def test_f03_finetune_improves_on_a_biased_public_head():
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(0)
    features = rng.normal(size=(400, CH.FEATURES)).astype(np.float32)
    w = rng.normal(scale=0.05, size=(CH.FEATURES, 3)).astype(np.float32)
    target = np.clip(features @ w, -1.5, 1.5).astype(np.float32)                  # the true offset, um
    public = CH.make_head()
    with torch.no_grad():
        public[0].weight.normal_(0, 0.05)
        public[2].weight.normal_(0, 0.05)
    public = (public.eval(), torch.zeros(CH.FEATURES), torch.ones(CH.FEATURES))
    tuned = CH.finetune(public, features, target, lam=0.3, epochs=150)
    before = np.linalg.norm(target - CH.predict_um(*public, features), axis=1).mean()
    after = np.linalg.norm(target - CH.predict_um(*tuned, features), axis=1).mean()
    assert after < 0.8 * before
    assert np.all(np.linalg.norm(CH.predict_um(*tuned, features * 50), axis=1) < 2.0)     # bounded below 2 um
    movies = np.array([f"{'44b6' if i % 2 else '6bba'}_m{i % 6}" for i in range(len(features))])
    report = CH.evaluate(public, features, target, movies, epochs=60)
    assert report["movie_folds_vs_public"] < 0 and "6bba->44b6_vs_public" in report
