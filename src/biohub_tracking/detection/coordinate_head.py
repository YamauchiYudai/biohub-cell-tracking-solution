"""V1284 coordinate-refinement head: fine-tuning recipe "F03" (final-day candidate component).

The upstream x138 notebook refines every fused detection with a small head on frozen U-Net features: the
feature at the detection voxel plus the differences to its 6 axis neighbours (224 values) go through
Linear(224, 32) - SiLU - Linear(32, 3); the output d is squashed to |shift| < 2 um by 2d / (1 + |d|), and
features are standardised with a stored (mean, scale). The architecture below mirrors that module so the
weights load into the notebook unchanged (file format {"state_dict", "mean", "scale"}).

F03 fine-tunes the public head instead of training from scratch:
  - start from the public head's weights and keep its standardisation;
  - loss = smooth-L1(pred, target; beta 0.5) + lam * mean ||pred - public(x)||^2, lam = 0.3 (distillation
    keeps the head near the public one, which was trained on data we cannot see);
  - AdamW lr 1e-3, weight decay 0, 300 full-batch epochs.
Targets are (GT - detection) in um for detection / GT pairs that are mutual nearest neighbours <= 7 um in the
same frame, on the model's isotropic 1.625 um grid.

Evaluation reports the centre error relative to the public head by 5 movie folds and across embryos.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

GRID_UM = 1.625
FEATURES = 224
EPOCHS = 300
PAIR_MAX_UM = 7.0


def make_head():
    import torch

    head = torch.nn.Sequential(torch.nn.Linear(FEATURES, 32), torch.nn.SiLU(), torch.nn.Linear(32, 3))
    torch.nn.init.zeros_(head[-1].weight)
    torch.nn.init.zeros_(head[-1].bias)
    return head


def bounded(head, x):
    import torch

    d = head(x)
    return 2.0 * d / (1.0 + torch.linalg.vector_norm(d, dim=-1, keepdim=True))


def load_head(path):
    import torch

    saved = torch.load(path, map_location="cpu", weights_only=True)
    head = make_head()
    head.load_state_dict(saved["state_dict"])
    head.eval()
    return head, saved["mean"].float(), saved["scale"].float()


def save_head(path, head, mean, scale) -> None:
    import torch

    torch.save({"state_dict": head.state_dict(), "mean": mean.clone(), "scale": scale.clone()}, path)


def predict_um(head, mean, scale, features: np.ndarray) -> np.ndarray:
    """Displacement in um for (N, 224) features, exactly the notebook's bounded(head, (x - mean) / scale)."""
    import torch

    with torch.no_grad():
        return bounded(head, (torch.from_numpy(features.astype(np.float32)) - mean) / scale).numpy()


def finetune(public, features: np.ndarray, target_um: np.ndarray, lam: float = 0.3, lr: float = 1e-3,
             epochs: int = EPOCHS, seed: int = 0):
    """F03: public-initialised, distillation-regularised fine-tuning. ``public`` = (head, mean, scale)."""
    import torch

    torch.manual_seed(seed)
    h0, mean, scale = public
    teacher = torch.from_numpy(predict_um(h0, mean, scale, features).astype(np.float32))
    head = make_head()
    head.load_state_dict(h0.state_dict())
    xt = (torch.from_numpy(features.astype(np.float32)) - mean) / scale
    yt = torch.from_numpy(target_um.astype(np.float32))
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=0.0)
    for _ in range(epochs):
        opt.zero_grad()
        pred = bounded(head, xt)
        loss = torch.nn.functional.smooth_l1_loss(pred, yt, beta=0.5) + lam * ((pred - teacher) ** 2).sum(-1).mean()
        loss.backward()
        opt.step()
    head.eval()
    return head, mean, scale


def mutual_pairs(det_grid: np.ndarray, t_det: np.ndarray, gt_grid: np.ndarray, t_gt: np.ndarray,
                 max_um: float = PAIR_MAX_UM) -> tuple[np.ndarray, np.ndarray]:
    """Per frame, detection / GT pairs that are each other's nearest and <= ``max_um`` apart (grid coords)."""
    di, gi = [], []
    for tt in np.unique(t_det):
        a = np.where(t_det == tt)[0]
        b = np.where(t_gt == tt)[0]
        if not len(a) or not len(b):
            continue
        det, gt = det_grid[a] * GRID_UM, gt_grid[b] * GRID_UM
        d1, j1 = cKDTree(det).query(gt)
        _, j2 = cKDTree(gt).query(det)
        for k, (dist, j) in enumerate(zip(d1, j1, strict=True)):
            if dist <= max_um and j2[j] == k:
                di.append(a[j])
                gi.append(b[k])
    return np.asarray(di, int), np.asarray(gi, int)


def relative_error(pred_um: np.ndarray, target_um: np.ndarray, public_um: np.ndarray) -> float:
    """Mean centre error of ``pred`` relative to the public head's (negative = better)."""
    e = np.linalg.norm(target_um - pred_um, axis=1).mean()
    e_public = np.linalg.norm(target_um - public_um, axis=1).mean()
    return float(e / e_public - 1.0)


def evaluate(public, features: np.ndarray, target_um: np.ndarray, movies: np.ndarray, lam: float = 0.3,
             folds: int = 5, epochs: int = EPOCHS) -> dict:
    """Leave-movies-out (5 folds, seed 0) and cross-embryo error of F03 relative to the public head."""
    public_um = predict_um(*public, features)
    embryo = np.array([m[:4] for m in movies])
    report: dict = {"n_pairs": int(len(features)), "n_movies": int(len(np.unique(movies)))}
    pred = np.zeros_like(target_um)
    for fold in np.array_split(np.random.default_rng(0).permutation(np.unique(movies)), folds):
        te = np.isin(movies, fold)
        head = finetune(public, features[~te], target_um[~te], lam, epochs=epochs)
        pred[te] = predict_um(*head, features[te])
    report["movie_folds_vs_public"] = relative_error(pred, target_um, public_um)
    for train_e, test_e in (("6bba", "44b6"), ("44b6", "6bba")):
        tr, te = embryo == train_e, embryo == test_e
        if tr.any() and te.any():
            head = finetune(public, features[tr], target_um[tr], lam, epochs=epochs)
            report[f"{train_e}->{test_e}_vs_public"] = relative_error(
                predict_um(*head, features[te]), target_um[te], public_um[te])
    report["mean_diff_from_public_um"] = float(np.linalg.norm(pred - public_um, axis=1).mean())
    return report
