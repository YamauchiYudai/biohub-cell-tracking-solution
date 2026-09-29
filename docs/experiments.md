# Experiments

The competition ran for about two months with more than 100 logged experiments. This page keeps only the
experiments that changed a decision. LB = Public LB as shown on Kaggle (three decimals); "replay" = the exact
CPU replay on 88 train movies described in [validation.md](validation.md#cpu-replay); "visible 4" = the real
GPU notebook on the 4 visible test movies (copies of train movies, in-sample).

## Decision timeline

| # | Experiment | Change | Validation | LB | Decision |
| --- | --- | --- | --- | --- | --- |
| 1 | Classical baseline (2026-08-01) | Tissue mask + DoG detection + LAP linking + gap interpolation | 1-movie edge J | 0.616 | Baseline of the team's own pipeline |
| 2 | Own 3D U-Net detector (08-15) | Sparse-point heatmap U-Net with local softmax + centre BCE 0.1 | LB (trained on both embryos, so no held-out CV) | 0.669 | Best own-pipeline result |
| 3 | Calibration / distillation variants (08-17 .. 08-28) | Weaker calibration term; distillation from a frozen external nucleus detector only | CV, LB | 0.366 / 0.373 | Rejected; training was unstable (predicted-positive fraction exploded) |
| 4 | Public UNet + Transformer + ILP stack (09-02 .. 09-04) | Adopt a public notebook as a working control, inject our code at fixed points | built-in validator, bit-identical detection check | 0.936 | **Pivot**: 0.936 vs our 0.669; own detector line stopped |
| 5 | Our detection post-processing on the public stack (09-02) | 6 arms: sub-voxel refinement, Gaussian smoothing, NMS 6 um, combinations | built-in validator (2 held-out movies) | - | All 6 below the control; `wrong_association = 0` in every arm: losses were missing detections, not wrong links |
| 6 | Reproduce public 0.941 / 0.946 configurations (09-05 .. 09-17) | Public settings in our account (not our tuning) | LB | 0.941 / 0.946 | 0.946 frozen as the control for the association work |
| 7 | Association replacements (09-21 .. 09-24) | Hard-negative margin, relative-rank prior, soft mutual-best support, FOCUS-3D linker, SuperGlue-style dense matching, Trackastra-style 5-frame, MeMOTR-style memory | control vs candidate, external `val_biohub` | 0.945 / 0.945 / 0.944 / 0.941 / 0.870 / 0.946 / 0.946 | **None beat the control**; association is not the bottleneck |
| 8 | Error attribution of 0.946 (09-21) | Stage-by-stage attribution of every FN / FP edge on 72 movies | exact replay | - | Largest FN source: candidate pairs never generated (45 %); FP: confident wrong Transformer edges (68 %). Candidate-generation repair tried: -0.010, stopped |
| 9 | x138 public notebook + own V1284 head (09-23 .. 09-24) | Adopt the 0.953 public notebook; retrain its private coordinate head from our own capture (30 movies) | leave-movie-out centre error -17 %, LB | 0.952 (public head: 0.953) | Adopt x138 as the base, keep the public head |
| 10 | **Relink b1c** (09-25) | Global-translation prior on jump transitions after duplicated frames | visible 4: 0.9180 -> 0.9290; replay | **0.954** | **Adopted** (team best at the time) |
| 11 | Relink b1fg (09-26) | + undetected-jump trigger + no raw-distance admission inside jumps | visible 4 relink-stage TP 2046 -> 2049 | 0.954 | Same LB; kept b1c (simpler) |
| 12 | Safe division off (09-25) | Measure what divisions are worth on hidden data | LB difference | 0.914 | Divisions are worth about 0.04: **division becomes the main lever** |
| 13 | S1: geometric fork removal (09-26) | Remove safe-div forks whose continuing child barely moved (< 3 um) | candidate table | 0.950 | Rejected: removing forks costs on hidden data |
| 14 | **S2: T3 CNN add / remove** (09-26) | Learned division score; gated adds at T3 >= 0.3, removals at T3 < 0.01 | embryo holdout (AUC 0.86 / 0.94), within-embryo CV thresholds, replay | **0.964** | **Adopted** (+0.010, largest single gain) |
| 15 | S3: T3 x ranking model (09-26) | sqrt(T3 x conditional-logit ranker over competing parents), add + remove | holdout, replay (40 movies) | 0.958 | Rejected: removals hurt again |
| 16 | S2 variants (09-27) | No removals + taken-daughter reassignment (b); budget exemption (V5a) or x1.5 (V5b); LR + T3 reassignment | replay (88) | 0.964 / 0.964 / 0.964 / 0.961 | V5a kept (replay +1 division); S2 neighbourhood saturated |
| 17 | Low-probability second daughter (09-28) | Cache p(P->D2) down to 0.2-0.3 and re-score with T3 | 19-movie capture, 6bba cross-validation | - | NO-GO: gain not reproduced across folds, rests on 3 events |
| 18 | S2 production audit (09-27) | Independent code / failure audit of the submitted notebook | replay + GT attribution | - | No P0 bugs; 46 % of remaining FN edges are node localisation, not linking |
| 19 | Rule-based relocalisation, smoothing weight, DeepCenter threshold (09-28) | M1-M4 heatmap peak / sub-voxel / Gaussian fit; line-fit weight 0.6; DeepCenter veto threshold | replay per embryo + visible 4 | - | NO-GO (see failure analysis) |
| 20 | **J2 jump-aware line fit** (09-28) | Smoothing that respects frozen / jump transitions | replay +0.00301 (J2 + V5a +0.00353), both embryos positive | **0.964** | **Adopted** (insured version: falls back to the upstream smoothing on error) |
| 21 | Gap filling G1-G6, re-admission R1-R3 (09-29) | One post-processing knob each | replay, gate +0.001 | - | NO-GO: best +0.00010 (G6), re-admission -0.00004 .. +0.00003 |
| 22 | V1284 head retrain (09-29) | Pre-registered N / Ns / F1 / F03 comparison on 155 movies | 5 movie folds + cross-embryo; visible 4 +0.00135 | no score (J2 v1 + F03); **0.966** with the consensus | Failed its offline gate but was the only final-day change that moved the public score; kept in one of the two selected submissions (`fc_f03`) |
| 23 | Frozen-frame coordinate consensus (09-29) | Average the two copies of each nucleus across a frozen transition after J2 | replay +0.00103 (6bba +0.00115), all pre-registered checks pass; visible 4 +0.00259 | 0.964 (`fc`) | **Adopted** in both selected submissions |
| 24 | Teammate variants (09-29) | Extra nuclei from a high-resolution detection head (V-add); a fine-tuned detector ("fullstack") | visible 4 | 0.964 / 0.955 | Not selected: no public gain, and V-add adds up to 5 % nodes |
| 25 | **Final selection** (09-30) | `fc_f03` + `fc`: identical except for the coordinate head | Public 0.966 / 0.964 | private TBD | The higher private score of the two counts; the pair covers both outcomes of F03 |

## Approaches that were dropped

The code for these is not part of this repository; the outcome is what matters.

### Own detector trained from sparse annotations
- **Hypothesis:** a 3D U-Net trained on the sparse centre annotations, pre-trained on external data or
  distilled from a frozen nucleus detector, can match the public detector.
- **Result:** best LB 0.669; calibration and distillation variants collapsed (0.366 / 0.373). The public
  pretrained stack scored 0.936 on day one of the comparison.
- **Decision:** stop the own-detector line and build on the public stack, spending effort where the public
  stack was weak.

### Replacing or re-scoring the association Transformer
- **Hypothesis:** better association (hard negatives, rank priors, multi-frame or memory models, dense
  matching) raises edge Jaccard.
- **Result:** seven variants scored 0.870-0.946 against the 0.946 control; the error attribution showed that
  correct pairs are usually either missing from the candidate set or lost at detection.
- **Decision:** stop association work; move to detection-side and division-side errors.

### Fine-tuning the public detector (local softmax)
- **Hypothesis:** a local-softmax loss sharpens detections.
- **Result:** internal diagnostic -0.046 and external `val_biohub` -0.046; the run stopped at epoch 1 on the
  detection-explosion guard.
- **Decision:** NO-GO for this configuration; the loss family itself was not ruled out.

### Removing rule-based division forks
- **Hypothesis:** forks the classifier finds unlikely are false positives.
- **Result:** every submission that removed forks more aggressively scored lower (S1 0.950, S3 0.958) even when
  offline tables showed no true fork removed; sparse annotation hides many real divisions.
- **Decision:** add conservatively, remove only at T3 < 0.01.

### More post-processing knobs on the final base
- **Hypothesis:** gap filling, re-admission, relocalisation or smoothing weight have headroom.
- **Result:** every single-knob change was below +0.0002 on the replay, or positive only because of one movie
  or only in-sample.
- **Decision:** stop tuning; remaining errors need a better detector / coordinate model.

### Embryo-holdout retraining of the public models
- **Hypothesis:** retraining detector, DeepCenter and head per embryo would give truly held-out validation.
- **Result:** training code, splits and a leakage guard were wired and smoke-tested on CPU; not enough time
  or GPU budget remained for full training.
- **Decision:** documented as the main next step.
