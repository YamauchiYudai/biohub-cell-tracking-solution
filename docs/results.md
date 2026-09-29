# Results

Final rank and private LB: see the **Competition Result** table in [README.md](../README.md) (the only place
they are recorded). Everything below is a confirmed number from the Kaggle submissions list or from the
offline replay; nothing is estimated.

## Public LB progression (confirmed submissions)

| Date (UTC) | Submission | What changed | Public LB |
| --- | --- | --- | ---: |
| 2026-08-01 | classical baseline | own pipeline: DoG detection + LAP | 0.616 |
| 2026-08-15 | own 3D U-Net | own detector | 0.669 |
| 2026-09-04 | 56003694 | public UNet + Transformer + ILP notebook as the team control | 0.936 |
| 2026-09-05 | 56027658 | public 0.941 configuration reproduced in our account | 0.941 |
| 2026-09-17 | 56298082 | public 0.947 pipeline run in our account (frozen control for association work) | 0.946 |
| 2026-09-24 | 56508017 | x138 + own V1284 head (30 movies) | 0.952 |
| 2026-09-25 | 56537913 | x138 + **relink b1c** | 0.954 |
| 2026-09-25 | 56551441 | same, safe division off (measurement) | 0.914 |
| 2026-09-26 | 56578893 | **S2**: + T3 division add / remove | **0.964** |
| 2026-09-27 | 56603553 | S2 + V5a (learned adds exempt from the division budget) | 0.964 |
| 2026-09-28 | 56632306 | S2 + J2 + V5a | 0.964 |
| 2026-09-28 | 56643954 | S2 + J2 + V5a, J2 insured (the committed core) | 0.964 |
| 2026-09-29 | 56662358 / 56663004 / 56663011 / 56665741 | final-day candidates (F03 head / frozen consensus / both / teammate's V-add) | pending |

Best confirmed Public LB: **0.964**.

## Offline numbers of the production components

| Component | Measurement | Result |
| --- | --- | --- |
| Relink b1c | 4 visible movies, official score | 0.9180 -> 0.9290 |
| T3 division CNN | embryo holdout AUC / AP | 44b6 -> 6bba 0.864 / 0.554; 6bba -> 44b6 0.942 / 0.694 |
| S2 (b1c + T3) | 88-movie replay, official score | 0.9430 (44b6 0.9312, 6bba 0.9452); divisions 44 TP / 35 FP / 107 FN |
| J2 + V5a over S2 | 88-movie replay | +0.00353 (44b6 +0.00316, 6bba +0.00374); edges +54 TP / -116 FP / -54 FN |
| Frozen-frame consensus over J2 + V5a | 88-movie replay | +0.00103 (6bba +0.00115, 44b6 0); +0.00087 without the best movie |
| F03 head over J2 + V5a | 4 visible movies (GPU run) | +0.00135 |

## Final result

| | Value |
| --- | --- |
| Final submission | TBD (`configs/final.yaml` `result.final_submission`) |
| Final rank / Private LB | see README |
