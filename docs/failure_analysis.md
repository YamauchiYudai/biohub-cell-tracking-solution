# Failure analysis

Aggregate error analyses that drove the decisions in [experiments.md](experiments.md). All numbers come from
the exact CPU replay on train movies scored with the official metric; train movies are in-sample for the
public models, so absolute values are optimistic and only the *relative* sizes are used. No images or ground
truth are included in this repository.

## 1. Where the 0.946 pipeline lost edges (72 movies)

Every false-negative GT edge was assigned to the first stage that lost it.

| Stage of the first loss | FN edges | Share |
| --- | ---: | ---: |
| Candidate pair never generated for the Transformer | 469 | 45.2 % |
| Broken by the motion relink | 188 | 18.1 % |
| Rejected by the ILP | 143 | 13.8 % |
| Removed by output filters / topology repair | 128 | 12.3 % |
| Detection missing | 97 | 9.4 % |
| Other post-processing | 12 | 1.2 % |

False positives were dominated by confident but wrong Transformer edges (68 %) and edges introduced by the
relink (25 %). Two conclusions: re-scoring the Transformer could not fix most FNs (the pair was never a
candidate), and the relink was a real error source, which later led to b1c. FN edges were spread evenly
over dense, medium and sparse frames (324 / 346 / 367), so "crowded frames" was not the explanation.

## 2. Remaining errors of the S2 production pipeline (88 movies)

| Scope | Movies | Edge TP / FP / FN | Adjusted edge J | Division TP / FP / FN | Official score |
| --- | ---: | --- | ---: | --- | ---: |
| 44b6 | 22 | 6,463 / 406 / 318 | 0.9145 | 6 / 10 / 20 | 0.9312 |
| 6bba | 66 | 57,064 / 2,459 / 2,492 | 0.9199 | 38 / 25 / 87 | 0.9452 |
| All | 88 | 63,527 / 2,865 / 2,810 | 0.9193 | 44 / 35 / 107 | 0.9430 |

Cause of each of the 2,810 missed GT edges (exclusive, first match wins):

| Cause | Definition | Share | Fixable by linking? |
| --- | --- | ---: | --- |
| Node localisation | both ends matched, but a predicted node is > 3.5 um from its GT node or the correct pair is >= 7 um apart | 46.2 % | No: detector / coordinate model |
| Detection present but not kept by the ILP | a detection within 7 um reached the ILP but is not in the graph | 18.3 % | Partly; changes the node set |
| Removed by the short-track filter | in the graph before the filter, 4-5 node fragments | 12.7 % | Only with a better fragment selector |
| No detection | nothing within 7 um | 11.4 % | No |
| Wrong partner, correct pair not a candidate | positions correct, p(correct pair) <= 0.48 | 4.3 % | Needs a better association model |
| Only below-threshold detector peaks | peaks 0.5-0.965 within 7 um | 3.5 % | No |
| Nearby prediction taken by another GT node | matching conflict | 2.2 % | No |
| Wrong partner, correct pair was a candidate | the only purely post-processing error | 1.5 % | Yes, but worth at most +0.0006 |

**Reading:** almost two thirds of the remaining loss is detection / localisation. Linking changes are capped
at about +0.0006 to +0.0018 on this set, which is why the last days went into smoothing, coordinates and
divisions rather than linking.

## 3. Divisions

- **The ILP never creates a fork.** With costs edge -p, disappearance 2 and division 1.2, adding a second
  daughter always costs more than it gains; on the public x138 output no node had two children. Every output
  division comes from post-processing.
- **Divisions are worth about 0.04 on the hidden test set.** Turning safe division off: Public 0.954 -> 0.914.
- **Annotation is sparse.** On 40 train movies, 565 of 598 predicted forks sat on unannotated cells and were
  not scored at all; only forks on annotated cells count as TP or FP. Offline division Jaccard (0.11 on those
  movies) therefore does not translate into the Public number, and decisions were made on differences
  between configurations on the same movies.
- **Removing forks hurts on hidden data.** Every version that removed safe-division forks more aggressively
  lost on the Public LB (S1 0.950, S3 0.958) although the offline tables showed no true fork removed. Adding
  forks with a learned score (S2) gained +0.010.
- **Division recall is the weak point.** S2 on 88 movies: 44 TP / 35 FP / 107 FN. Of 120 FN edges caused by
  divisions, 80 are localisation errors of the daughters.

## 4. Duplicated frames and jumps

- 466 frozen transitions (bit-identical consecutive frames) in 88 movies, all in 6bba (55 of 66 movies).
- The upstream motion prior collapses to 0 um across a frozen transition, and the next transition carries
  two frames of motion. On the 4 visible movies, jump transitions held 31 of 82 FN and 49 of 105 FP edges
  before b1c (23 / 31 after).
- The output smoothing then mixed the two alignments around each jump, which J2 removes (+0.003 on the
  replay), and the two copies of a nucleus on either side of a frozen transition drift apart after smoothing,
  which the frozen-frame consensus addresses (+0.001, 6bba only).

## 5. Embryo and domain differences

- 44b6 scores lower (0.931 vs 0.945 on the replay) and has no duplicated frames; 6bba has most edges and
  every frozen-frame fix acts only there.
- Scores are not calibrated across embryos: a removal threshold chosen on 6bba (0.92) removed true forks on
  44b6, while the one chosen on 44b6 (0.10) was safe in both directions.
- The DeepCenter prior was trained on 44b6, so its 0.25 veto threshold is in-sample there; 25 % of the forks
  decided by that rule lie within +-0.05 of the threshold, a known sensitivity for hidden embryos.

## 6. Node-count effects <a id="node-count-effects"></a>

- The node factor `1 - 0.1 x (n_pred - n_est) / n_est` has no upper bound: predicting fewer nodes than the
  estimate *raises* the adjusted edge Jaccard even at equal edge Jaccard.
- Shortening the minimum track length from 6 to 5 or 4 recovers +43 / +81 TP edges but adds 20,000 / 35,000
  nodes and lowers the score (-0.00011 / -0.00022); 7 or 8 moves the embryos in opposite directions. The
  short-track filter was left unchanged.
- Re-admission and gap filling both add nodes; their small edge gains were cancelled by the node factor.

## 7. Validation mismatch

- **In-sample optimism.** Weaker smoothing (weight 0.6) gained +0.0035 on the replay but lost edges on the
  real visible GPU run; the detector's coordinates are more accurate on its own training movies, which favours
  less smoothing.
- **One-movie effects.** On 44b6 the smoothing gain came almost entirely from one movie (+0.0004 without
  it); the "leave the best movie out" check is now part of every gate.
- **LB resolution.** Several real offline improvements (J2 +0.003, V5a +1 division) left the three-decimal
  Public score unchanged at 0.964.
