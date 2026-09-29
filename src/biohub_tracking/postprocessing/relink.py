"""Jump-aware motion relink ("b1c"): a global-translation prior for transitions after duplicated frames.

The upstream post-processing re-links every transition t -> t+1 with a Hungarian assignment whose seed pass
predicts each cell's position with ``previous_flow`` (the median displacement of nearby confident links in
the previous transition). Some movies contain duplicated acquisition frames (frame t+1 is a bit-identical copy
of frame t). Across such a *frozen* transition every cell moves 0 um, so the flow prior collapses to zero,
and on the next transition the real motion (two frames' worth, often a whole-embryo step) falls outside the
5.5 um seed gate: cells get linked to their neighbours and the error propagates.

b1c classifies transitions and, on *jump* transitions only, adds a global translation T to the prior:

  frozen   frames t and t+1 are bit-identical
  jump     directly after a frozen transition, or seed-match median displacement >= 2 x the movie median
           (median over non-frozen transitions)
  normal   everything else: the upstream prior is returned unchanged

T is a point-set cross-correlation vote (the shift that makes the most source -> target pairs coincide within
1.5 um), refined by iterated mutual nearest neighbours (a translation-only ICP). Measured on the 4 visible
train copies: score 0.9180 -> 0.9290, all of it on the movie with duplicated frames; Public LB 0.953 -> 0.954.

This module restates the production code path (the ``b1c`` variant of the relink helpers embedded in the
submission notebook); ``tests/test_postprocessing.py`` checks it against the notebook's own code.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable

import numpy as np
from scipy.spatial import cKDTree

JUMP_RATIO = 2.0
MIN_SEEDS_FOR_MEDIAN = 4
TRANSLATION_ITERS = 6
TRANSLATION_MIN_PAIRS = 4
TRANSLATION_VOTE_RADIUS_UM = 16.0
TRANSLATION_VOTE_BANDWIDTH_UM = 1.5


# ------------------------------------------------------------------ transition classification
def frozen_transitions(t_values: Iterable[int], read_frame: Callable[[int], np.ndarray]) -> list[int]:
    """Transitions t (frame t -> t+1) whose raw frames are bit-identical. Holds two frames at a time."""
    ts = sorted({int(t) for t in t_values})
    if not ts:
        return []
    frozen: list[int] = []
    prev_t, prev = None, None
    for t in range(ts[0], ts[-1] + 1):
        cur = np.asarray(read_frame(t))
        if prev is not None and prev_t == t - 1 and prev.shape == cur.shape and np.array_equal(prev, cur):
            frozen.append(prev_t)
        prev_t, prev = t, cur
    return frozen


def preceding_frozen(t: int, frozen) -> int:
    """Number of frozen transitions immediately before transition t."""
    k = 0
    while (t - 1 - k) in frozen:
        k += 1
    return k


def seed_medians(times, ids_by_t: dict, seed_fn: Callable) -> dict[int, float]:
    """Median seed-match distance per transition (``seed_fn(src_ids, tgt_ids)`` returns (s, g, dist, ...)
    tuples: the upstream seed pass with no motion prior); transitions with < 4 seeds are left out."""
    out: dict[int, float] = {}
    for t in times:
        src, tgt = ids_by_t.get(t, []), ids_by_t.get(t + 1, [])
        if not src or not tgt:
            continue
        seeds = seed_fn(src, tgt)
        if len(seeds) >= MIN_SEEDS_FOR_MEDIAN:
            out[int(t)] = float(np.median([m[2] for m in seeds]))
    return out


def classify_transitions(times, frozen, seed_median: dict, ratio: float = JUMP_RATIO) -> dict:
    """{t: {"kind", "k", "seed_median", "reason"}} plus key "_movie_median" (float or nan)."""
    frozen = {int(t) for t in frozen}
    usable = [float(v) for t, v in seed_median.items() if int(t) not in frozen and v is not None and np.isfinite(v)]
    movie_median = float(np.median(usable)) if usable else float("nan")
    out: dict = {"_movie_median": movie_median}
    for t in sorted(int(t) for t in times):
        med = seed_median.get(t)
        med = float(med) if med is not None and np.isfinite(med) else float("nan")
        k = preceding_frozen(t, frozen)
        if t in frozen:
            kind, reason = "frozen", "frozen"
        elif k >= 1:
            kind, reason = "jump", "after_frozen"
        elif np.isfinite(med) and np.isfinite(movie_median) and movie_median > 0 and med >= ratio * movie_median:
            kind, reason = "jump", "seed_median"
        else:
            kind, reason = "normal", ""
        out[t] = {"kind": kind, "k": k, "seed_median": med, "reason": reason}
    return out


# ------------------------------------------------------------------ global translation
def vote_translation(src: np.ndarray, tgt: np.ndarray, radius_um: float = TRANSLATION_VOTE_RADIUS_UM,
                     bandwidth_um: float = TRANSLATION_VOTE_BANDWIDTH_UM, max_vectors: int = 200_000):
    """Point-set cross-correlation: the shift that makes the most source -> target pairs coincide.

    Every (source, target) pair closer than ``radius_um`` casts its displacement as a vote; the vote with the
    most other votes within ``bandwidth_um`` wins. Unlike nearest neighbours it does not lock onto a neighbour
    one cell spacing away (the true shift collects a vote from every cell, a wrong one from fewer).
    """
    near = cKDTree(tgt).query_ball_point(src, r=radius_um)
    vec = [tgt[j] - src[i] for i, js in enumerate(near) for j in js]
    if not vec:
        return None
    vec = np.asarray(vec, dtype=np.float64)
    if len(vec) > max_vectors:
        vec = vec[np.random.default_rng(0).choice(len(vec), max_vectors, replace=False)]
    tree = cKDTree(vec)
    counts = np.asarray(tree.query_ball_point(vec, r=bandwidth_um, return_length=True))
    best = vec[int(np.argmax(counts))]
    return np.median(vec[tree.query_ball_point(best, r=bandwidth_um)], axis=0)


def estimate_translation(src_um, tgt_um, iters: int = TRANSLATION_ITERS, min_pairs: int = TRANSLATION_MIN_PAIRS):
    """Global shift T from sources (already moved by the base prior) to targets, or None.

    Starts from :func:`vote_translation`, then refines with the median residual of mutual nearest neighbours,
    re-matched after each shift. None when fewer than ``min_pairs`` points or mutual pairs exist.
    """
    src = np.asarray(src_um, dtype=np.float64).reshape(-1, 3)
    tgt = np.asarray(tgt_um, dtype=np.float64).reshape(-1, 3)
    if len(src) < min_pairs or len(tgt) < min_pairs:
        return None
    tree_t = cKDTree(tgt)
    vote = vote_translation(src, tgt)
    shift = vote if vote is not None else np.zeros(3)
    found = False
    for _ in range(max(1, iters)):
        moved = src + shift
        _, j = tree_t.query(moved)
        _, i_back = cKDTree(moved).query(tgt)
        mutual = i_back[j] == np.arange(len(src))
        if int(mutual.sum()) < min_pairs:
            break
        new_shift = np.median(tgt[j[mutual]] - src[mutual], axis=0)
        found = True
        if np.allclose(new_shift, shift, atol=1e-6):
            shift = new_shift
            break
        shift = new_shift
    return shift if found else None


# ------------------------------------------------------------------ per-movie state used by the relink
class JumpAwarePrior:
    """Per relink call: returns the upstream prior on non-jump transitions, prior + T on jump transitions."""

    def __init__(self, info: dict, position_um: dict, stats: dict | None = None):
        self.info = info
        self.position_um = position_um
        self.stats = stats if stats is not None else {}

    def kind(self, t: int) -> str:
        entry = self.info.get(int(t))
        return entry["kind"] if entry else "normal"

    def seed_prior(self, t: int, source_ids, target_ids, previous_flow):
        """``previous_flow(pos_um, exclude_um) -> displacement or None`` is the upstream prior."""
        if self.kind(t) != "jump":
            return previous_flow
        self.stats["x138_jump_prior_transitions"] = self.stats.get("x138_jump_prior_transitions", 0) + 1

        def base_step(pos, exclude_um):
            return previous_flow(pos, exclude_um) if previous_flow is not None else None

        src = []
        for sid in source_ids:
            pos = self.position_um[sid]
            step = base_step(pos, 0.0)
            src.append(pos + (step if step is not None else 0.0))
        tgt = [self.position_um[g] for g in target_ids]
        shift = estimate_translation(src, tgt)
        if shift is None:
            return previous_flow
        self.stats["x138_jump_translation_um_sum"] = (self.stats.get("x138_jump_translation_um_sum", 0)
                                                      + float(np.linalg.norm(shift)))

        def predict(pos, exclude_um):
            step = base_step(pos, exclude_um)
            return (step if step is not None else np.zeros(3)) + shift

        return predict
