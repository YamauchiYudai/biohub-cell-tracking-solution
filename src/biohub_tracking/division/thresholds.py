"""Pick tau_add / tau_del for a per-parent division score from a labelled candidate table.

The table carries two score columns: a *fit* score (within-embryo movie-OOF, ``t3_inner``) used to choose
thresholds, and a *held-out* score (``t3_holdout``, models trained on the other embryo) used to report the
chosen thresholds on the embryo that did not choose them. Decisions use the submission's own
:func:`~biohub_tracking.division.select.choose` and, when pre-safe graphs are given, its budget cap.

  added tp / fp      forks added (ignored rows are not scored by the metric)
  precision_add      added tp / (added tp + added fp)     gate >= 0.40 (break-even of +1 TP vs +1 FP is ~0.29)
  kept_tp_frac       1 - removed tp / safe-div tp         gate >= 0.95
  division J         (TP0 + added tp - removed tp) / (GT + FP0 + added fp - removed fp)

tau_add is the lowest tau whose precision_add >= 0.40 with at least 3 scored adds; tau_del the highest tau with
kept_tp_frac >= 0.95.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .processor import cap_additions
from .select import choose

MIN_PRECISION = 0.40
MIN_KEEP = 0.95
MIN_ADDS = 3
GRID = np.round(np.concatenate([np.arange(0.05, 0.9, 0.05), np.arange(0.9, 1.0, 0.01), [0.995, 0.999]]), 4)


def decide(cands: pd.DataFrame, score: str, tau_add: float | None, tau_del: float | None,
           shapes=("a",), pre_safe: dict | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-movie adds / removes. ``pre_safe``: {movie: (nodes, pre edge pairs, safe-div forks)} to apply
    the division budget exactly as the submission does."""
    adds, removes = [], []
    for movie, g in cands.groupby("movie"):
        a, r = choose(g, score, tau_add, tau_del, shapes)
        if pre_safe is not None and len(a):
            nodes, pre, selected = pre_safe[movie]
            before = [{"source_id": int(s), "target_id": int(d)} for s, d in pre]
            safe = [*before, *({"source_id": int(s), "target_id": int(d), "safe_division": 1} for s, d in selected)]
            a = cap_additions(a, nodes, before, safe, r, score)
        adds.append(a)
        removes.append(r)
    return (pd.concat(adds) if adds else cands.iloc[:0], pd.concat(removes) if removes else cands.iloc[:0])


def count(cands: pd.DataFrame, adds: pd.DataFrame, removes: pd.DataFrame, base: dict) -> dict:
    """``base``: tp / fp / gt divisions of the unchanged post-safe graphs of the same movies."""
    at, af = int((adds.label == "tp").sum()), int((adds.label == "fp").sum())
    rt, rf = int((removes.label == "tp").sum()), int((removes.label == "fp").sum())
    safe = cands[cands.in_safe_div == 1].drop_duplicates(["movie", "p_id", "d2_id"])
    safe_tp = int((safe.label == "tp").sum())
    tp, fp = base["tp"] + at - rt, base["fp"] + af - rf
    j0 = base["tp"] / max(1, base["gt"] + base["fp"])
    j = tp / max(1, base["gt"] + fp)
    return {"movies": int(cands.movie.nunique()), "added": len(adds), "added_tp": at, "added_fp": af,
            "added_ignored": int((adds.label == "ignored").sum()),
            "precision_add": at / (at + af) if at + af else np.nan,
            "removed": len(removes), "removed_tp": rt, "removed_fp": rf,
            "kept_tp_frac": 1 - rt / safe_tp if safe_tp else 1.0,
            "division_tp": tp, "division_fp": fp, "division_fn": base["gt"] - tp,
            "division_j_before": j0, "division_j_after": j, "delta_score": 0.1 * (j - j0)}


def sweep(cands: pd.DataFrame, score: str, base: dict, kind: str, pre_safe: dict | None = None) -> pd.DataFrame:
    """The greedy takes parents in descending score, so the adds at tau are the adds of the lowest-tau run
    restricted to score >= tau (no removals); removals are a plain threshold on safe-div forks."""
    rows = []
    if kind == "add":
        adds_all, _ = decide(cands, score, float(GRID.min()), None, pre_safe=pre_safe)
        for tau in GRID:
            rows.append({"tau": float(tau), **count(cands, adds_all[adds_all[score] >= tau], cands.iloc[:0], base)})
    elif kind == "del":
        _, rem_all = decide(cands, score, None, float(GRID.max()) + 1.0)
        for tau in GRID:
            rows.append({"tau": float(tau), **count(cands, cands.iloc[:0], rem_all[rem_all[score] < tau], base)})
    else:
        raise ValueError(kind)
    return pd.DataFrame(rows)


def pick_add(table: pd.DataFrame) -> float | None:
    ok = table[(table.precision_add >= MIN_PRECISION) & (table.added_tp + table.added_fp >= MIN_ADDS)]
    return float(ok.tau.min()) if len(ok) else None


def pick_del(table: pd.DataFrame) -> float | None:
    ok = table[(table.kept_tp_frac >= MIN_KEEP) & (table.removed > 0)]
    return float(ok.tau.max()) if len(ok) else None


def select(cands: pd.DataFrame, fit_score: str, heldout_score: str, base_by_movie: dict[str, dict],
           pre_safe: dict | None = None) -> dict:
    """Thresholds on all movies (fit score) plus the embryo cross-check: fit on one embryo, count on the other
    with the held-out score. ``base_by_movie``: {movie: {"tp", "fp", "gt"}} of the unchanged run."""
    if fit_score == heldout_score:
        raise ValueError("the fit score and the held-out score must be different columns")

    def base(movies):
        return {k: sum(base_by_movie[m][k] for m in movies) for k in ("tp", "fp", "gt")}

    report: dict = {"fit_score": fit_score, "heldout_score": heldout_score,
                    "gates": {"precision_add": MIN_PRECISION, "kept_tp_frac": MIN_KEEP, "min_adds": MIN_ADDS}}
    for kind, picker in (("add", pick_add), ("del", pick_del)):
        report[f"tau_{kind}"] = picker(sweep(cands, fit_score, base(set(cands.movie)), kind, pre_safe))
        embryos = sorted(set(cands.embryo))
        for fit_e in embryos:
            for test_e in (e for e in embryos if e != fit_e):
                fit, test = cands[cands.embryo == fit_e], cands[cands.embryo == test_e]
                tau = picker(sweep(fit, fit_score, base(set(fit.movie)), kind, pre_safe))
                res: dict = {"tau": tau}
                if tau is not None:
                    adds, removes = decide(test, heldout_score, tau if kind == "add" else None,
                                           tau if kind == "del" else None, pre_safe=pre_safe)
                    res.update(count(test, adds, removes, base(set(test.movie))))
                report[f"{kind}_fit{fit_e}_test{test_e}"] = res
    return report
