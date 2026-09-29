# Validation

The Public LB is computed on a hidden test set from embryos not in the training data, is shown with three
decimals, and its division term rests on roughly 30 annotated divisions (one division is worth about 0.003).
It cannot tell a +0.001 change from noise, so every decision was made offline and the LB was used only as a
sanity check. This page describes how the offline numbers were made trustworthy.

## 1. Official metric parity

All local scores come from the organisers' own implementation (`tracking_cellmot`, called through
`biohub_tracking.metrics.official`), never from a re-implementation:

- predictions are written and re-read through the submission CSV schema first, so nodes sit on the same
  integer voxel grid as in a real submission (an edge near the 7 um matching gate can flip otherwise);
- aggregation uses the official `summarise` (weighted mean for edges, pooled division Jaccard);
- scores are reported per embryo next to the overall number, because 6bba carries most of the edges;
- the division candidate labeller (`division/labels.py`) reproduces the official division rules and is
  checked against the real `score_divisions` on sampled forks (`verify_labels`).

## 2. Leakage: what is in-sample

| Data | Status for the final pipeline |
| --- | --- |
| Competition train, 199 movies (44b6: 71, 6bba: 128) | **In-sample for every public pretrained model.** The support pack's split manifest lists all 199 movies as training data. Any score on train movies is a diagnostic of post-processing changes, never an estimate of test performance, and was never called out-of-fold. |
| The 4 visible "test" movies | Copies of train movies: in-sample, used only as a smoke test of the real GPU notebook run. |
| Hidden test set | Other embryos. Public LB only. |
| `val_biohub` (external) | Zebrahub tail-bud movies with Ultrack-generated tracks (automatic, not human-verified). Used **only** for evaluation, never for gradients, pseudo-labels or fine-tuning. |

Consequences that shaped decisions:

- Changes to the detector, the Transformer or the association (anything upstream of post-processing) had to be
  compared with the frozen control on `val_biohub` before any submission. For example, a local-softmax
  fine-tune of the public weights degraded both the internal diagnostic (-0.046) and `val_biohub` (-0.046),
  so it was stopped.
- A post-processing change that improves train movies only because the detector is already accurate there
  (in-sample) is suspect. "Lane A" (weaker smoothing, +0.0035 on the 88-movie replay) was rejected because
  the gain on 44b6 came from a single movie, the visible GPU run got worse, and the detector coordinates it
  relies on are in-sample.

## 3. Embryo separation

The two embryos are the only independent units in the training data, so every learned or tuned component was
evaluated across them.

<a id="division-thresholds"></a>
**T3 division CNN and its thresholds.**

| Model set | Trained on | Used for |
| --- | --- | --- |
| holdout | one embryo, scored on the other | AUC / AP, and the held-out scores thresholds are reported on |
| inner | movie-grouped 3-fold CV inside each embryo | the scores thresholds are *selected* on |
| full | all 199 movies, 3 seeds | the submission |

`tau_add` is the lowest threshold whose added forks have precision >= 40 % (break-even for +1 TP vs +1 FP is
about 29 %) with at least 3 scored additions; `tau_del` the highest threshold that keeps >= 95 % of safe
division's true forks. Thresholds chosen on one embryo are applied to the other. This check caught a real
problem with the combined T3 x ranking-model score tried in S3: `tau_del` chosen on 6bba (0.92) would have
deleted true forks on 44b6, whereas 0.10 chosen on 44b6 was safe in both directions. The production S2 uses
T3 alone with `tau_add = 0.3` and a conservative `tau_del = 0.01` (just below the largest value that removes
no true fork in the within-embryo CV).

**Coordinate head (F03).** Five movie folds plus both cross-embryo directions, relative to the public head,
with the pass rule fixed before looking at results. F03 did not pass it; it was submitted only as a final-day
candidate after a positive visible-movie run. Combined with the frozen-frame consensus it then scored 0.966 on the
public board, the only final-day change that moved it. Because the offline gate and the public board disagree, the
final selection keeps both answers: `fc_f03` (with F03) and `fc` (the same notebook with the public head).

## 4. Exact CPU replay <a id="cpu-replay"></a>

GPU time was the scarce resource (about 30 h per week). The expensive stages run once:

1. **Capture (GPU, once).** The production notebook runs on 88 train movies (the 87 movies with annotated
   divisions plus one; 44b6: 22, 6bba: 66) and stores the ILP output graph, the edge cache, the
   low-score peaks, the DeepCenter heatmaps and the frozen-frame table per movie.
2. **Replay (CPU, many times).** The notebook's own post-processing code is executed from the notebook text
   (split at anchors, never re-typed) with environment overrides per configuration, and the final graphs are
   scored with the official metric.
3. **Parity.** Before a replay is trusted, the base configuration must reproduce the submitted notebook: same
   graphs entering safe division, same per-movie TP / FP / FN; the final replay agreed with the real
   pipeline to -0.00003.

Every post-processing decision after 2026-09-24 was made on this replay, with the same readout:

- official score difference, split into the edge-Jaccard part, the node-count part and the division part;
- per embryo, and "leave the most helpful movie out" (one movie must not carry the result);
- GT edges rescued vs broken.

Typical pre-registered gate: >= +0.001 on the 88 movies, both embryos non-negative, still positive with the
best movie removed, and no regression on the visible GPU run. Gap-filling (G1-G6), re-admission (R1-R3),
node relocalisation (M1-M4), smoothing weight and the low-probability daughter branch all failed such gates
and were not submitted ([experiments.md](experiments.md)).

## 5. Submission-time checks

Before any submission: the notebook is built by anchored patches from the previous submitted notebook, the
configuration guard and the division manifest record the resolved settings, the visible 4-movie GPU run is
checked for identical node sets where only coordinates should change, and the run fails if the
post-processing deadline degraded any movie. The runtime margin to the 12 h limit is checked from the scoring
times of earlier submissions.
