# Third-party notices

The final solution is built on public work from other competitors and the competition organisers. This
file lists what is redistributed in this repository, what is only used at run time, and under which terms.
Licenses were checked on the Kaggle pages and GitHub repositories on 2026-09-29.

## Redistributed in this repository

| Project | Source | License | What is in this repository |
| --- | --- | --- | --- |
| **biohub x138** notebook, Anvith Pothula (now titled "Biohub 0.953 LB \| ORIGINAL") | https://www.kaggle.com/code/anvithpothula/biohub-0-953-lb-original (originally `anvithpothula/biohub-x138`) | Apache License 2.0 (as shown on the Kaggle page) | `notebooks/final_submission.ipynb` is a modified copy: cells 1-4 and 6-11 unchanged, cell 0 with appended environment overrides, cell 5 with inserted blocks, cells 12-13 added. Short excerpts of cells 4 and 5 are used as patch anchors in `src/biohub_tracking/kaggle/variants.py`. The V1284 head architecture (Linear(224, 32) - SiLU - Linear(32, 3), bounded output) is mirrored in `src/biohub_tracking/detection/coordinate_head.py` so trained weights load into the notebook. |
| **Biohub frontier947 readmit v1** notebook, Teddy Tennant (flow-prior relink, re-admission of discarded detections, low-score gap filling, repair deadline; the x138 notebook adds the V1284 head to it) | https://www.kaggle.com/code/thtennant/biohub-frontier947-readmit-v1 | Apache License 2.0 (as shown on the Kaggle page) | Contained in the notebook above through the x138 notebook. |
| **Biohub Cell Tracking: 0.947 LB** notebook, Reyhan Ksatria (the post-processing the x138 notebook extends: motion relink, gap closing, safe division, DeepCenter veto, short-track filter) | https://www.kaggle.com/code/reyhanksatria/biohub-cell-tracking-0-947-lb | Apache License 2.0 (as shown on the Kaggle page) | Contained in the notebook above through the x138 notebook. |

Statement of changes (Apache License 2.0, section 4(b)): the files above were modified by the authors of this
repository between 2026-09-23 and 2026-09-29. Every team change to the notebook is listed in the notebook's
markdown cells and in `docs/solution.md`. The full license text is in `LICENSES/Apache-2.0.txt`.

## Used at run time, not redistributed

| Project | Source | License | Use |
| --- | --- | --- | --- |
| Biohub Tracking Support Pack (pipeline code + TemporalUNet3D detector + association Transformer weights), pilkwang | https://www.kaggle.com/datasets/pilkwang/biohub-tracking-support-pack-50ep-v1 | CC0-1.0 | Attached to the Kaggle notebook |
| Biohub TemporalUNet3D Seed 314159 V1, pilkwang | https://www.kaggle.com/datasets/pilkwang/biohub-temporal-unet3d-seed314159-v1 | CC0-1.0 | Second detector seed |
| Biohub DeepCenterUNet3D Center Prior V1, pilkwang | https://www.kaggle.com/datasets/pilkwang/biohub-deepcenter-unet3d-center-prior-v1 | CC0-1.0 | Centre prior used by the gap-closing and safe-division vetoes |
| biohub v1284 head s075, Anvith Pothula | https://www.kaggle.com/datasets/anvithpothula/biohub-v1284-head-s075 | CC0-1.0 | Coordinate-refinement head (public-head candidates) and the initialisation of the F03 fine-tuning |
| kaggle-cell-tracking-competition (`tracking_cellmot`: official metric, UNet + Transformer baseline), royerlab | https://github.com/royerlab/kaggle-cell-tracking-competition | BSD-3-Clause | Official metric for local scoring (`metric` extra); the support pack's pipeline derives from its baseline |
| tracksdata, royerlab | https://github.com/royerlab/tracksdata | BSD-3-Clause | Graph container for the official metric and the notebook |
| geff | https://github.com/live-image-tracking-tools/geff | MIT | Ground-truth graph format |
| numpy, pandas, scipy, PyYAML, scikit-learn, PyTorch, zarr, polars | PyPI | BSD / MIT / Apache-2.0 (per package) | Python dependencies (`pyproject.toml`) |

Competition data (images and ground truth) is not included and must be downloaded from Kaggle under the
competition rules.

## Ideas credited, no code copied

- Oversampling positive division examples during training (T3 uses 10x) follows the practice described for
  OrganoidTracker 2.0; the implementation is the team's.
- The frozen-frame, jump-aware relink and the learned division scorer were designed by the team after the
  error analyses in `docs/failure_analysis.md`.
