"""Learned division recovery on top of the rule-based safe-division stage.

The upstream ILP never selects a fork (its division cost exceeds any second-daughter edge probability), so
every division in the output comes from post-processing. The production path:

  candidates.enumerate_candidates   possible forks P -> {D1, D2} on the graph entering safe-div
  scoring.T3Scorer                  prob_angle_t2 add gate + T3 3D-CNN score per parent
  select.choose / apply_forks       add forks with T3 >= tau_add, remove safe-div forks with T3 < tau_del
  processor.DivisionProcessor       the per-movie hook (budget shared with safe-div unless exempt, V5a)

Training / validation: crops (T3 training data), t3 (model and training), labels (official-rule candidate
labels) and thresholds (tau selection by within-embryo movie CV).
"""
