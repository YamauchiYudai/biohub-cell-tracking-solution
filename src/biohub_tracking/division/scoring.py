"""Production division scorer: the ``prob_angle_t2`` add gate plus the T3 3D-CNN score.

Which forks may be *added* (gate, no model): the candidate is shape a, safe-div did not already make it,
the association Transformer's probability p(P->D2) was cached (> 0.48, i.e. the pair was a candidate edge),
the angle D1-P-D2 is >= 90 deg, and the two sisters separate further at t+2. Which forks are *kept*: every
safe-div fork is scored so that a clear non-division (T3 < tau_del) can be removed.

T3 then scores only the parents that matter (gated adds + safe-div forks), in blocks of 128 crops, and every
other candidate gets -1 so it can never pass tau_add. Production thresholds: tau_add = 0.3, tau_del = 0.01.
"""

from __future__ import annotations

import time
from collections.abc import Callable

import numpy as np
import pandas as pd

from . import t3
from .features import add_edge_probabilities, cache_alignment

T3_BLOCK = 128


def prob_angle_t2_gate(cands: pd.DataFrame, nodes: pd.DataFrame, edges: np.ndarray,
                       edge_cache: dict | None) -> pd.Series:
    """Boolean add gate per candidate row (index aligned with ``cands``)."""
    e = add_edge_probabilities(cands.copy(), nodes, edge_cache)
    gate = ((e.in_safe_div == 0) & (e["shape"] == "a") & (e.prob_p_d2_observed == 1)
            & (e.angle_deg >= 90) & (e.t2_gain_um.fillna(-1) > 0))
    return gate.reindex(cands.index, fill_value=False)


class T3Scorer:
    """``score_fn`` for :class:`~biohub_tracking.division.processor.DivisionProcessor`.

    ``read_frame(dataset, t, cache)`` returns the raw (z, y, x) frame; ``n_frames(dataset)`` the movie length;
    ``predict(crops_u16) -> scores`` wraps :func:`biohub_tracking.division.t3.predict` with the loaded models.
    """

    def __init__(self, predict: Callable[[np.ndarray], np.ndarray],
                 read_frame: Callable[[str, int, dict], np.ndarray], n_frames: Callable[[str], int],
                 score_column: str = "division_score"):
        self.predict = predict
        self.read_frame = read_frame
        self.n_frames = n_frames
        self.score_column = score_column
        self.log: dict[str, dict] = {}

    def __call__(self, cands: pd.DataFrame, nodes: pd.DataFrame, pre: np.ndarray, dataset: str,
                 cache: dict | None) -> pd.DataFrame:
        frame_cache: dict = {}

        def frame(tt):
            return self.read_frame(dataset, int(tt), frame_cache)

        t0 = time.time()
        cands["add_gate"] = prob_angle_t2_gate(cands, nodes, pre, cache).astype(int).to_numpy()
        todo = cands[(cands.in_safe_div == 1) | (cands.add_gate == 1)]
        self.log[dataset] = {"gated_adds": int(cands.add_gate.sum()), "safe_div": int(cands.in_safe_div.sum()),
                             "cache_aligned": bool(cache_alignment(nodes, cache)),
                             "gate_seconds": round(time.time() - t0, 2)}
        par = todo.drop_duplicates("p_id").sort_values("p_t")
        n_frames = int(self.n_frames(dataset))
        scores: list[float] = []
        for start in range(0, len(par), T3_BLOCK):
            block = par.iloc[start:start + T3_BLOCK]
            crops = np.stack([t3.crop_from_frames(frame, int(r.p_t), n_frames, (r.p_z, r.p_y, r.p_x))
                              for r in block.itertuples()])
            scores.extend(self.predict(crops))
            frame_cache.clear()
        cands["A"] = cands.p_id.map(dict(zip(par.p_id.astype(int), scores, strict=True)))
        keep = (cands.in_safe_div == 1) | (cands.add_gate == 1)
        cands["A"] = np.where(keep, cands["A"].fillna(-1.0), -1.0)
        self.log[dataset]["scored_parents"] = int(len(par))
        cands[self.score_column] = cands["A"]
        return cands
