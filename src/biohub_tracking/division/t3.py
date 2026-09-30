"""T3: a small 3D CNN that scores "a division happens here" from a 3-frame crop.

Input: the raw image around a (candidate) parent P at frame t, frames t-1, t, t+1 stacked as 3 channels,
z16 x y48 x x48 native voxels (26 x 19.5 x 19.5 um at 1.625 / 0.40625 / 0.40625 um). A dividing parent shows a
condensed, bright nucleus at t and two nuclei at t+1, so the t+1 frame carries most of the signal.

The same crop / normalisation code is used for training, offline candidate scoring and the submission
notebook. Training data (:mod:`biohub_tracking.division.crops`): positives are the GT divider at t (the frame
before the daughters appear) and the same event at t-1 and t+1 (the metric accepts a fork one frame early or
late); negatives are annotated cells with exactly one child, away from any division. Positives are repeated
10x per epoch with random xy flips, 90-degree xy rotations, a shift of up to 1 z-slice / 8 xy-pixels and an
intensity scale.

Three training modes, all with the same recipe:
  holdout  train on one embryo, score the other (44b6 -> 6bba, 6bba -> 44b6): the leak-free estimate
  inner    movie-grouped K-fold inside each embryo: scores used to *select* tau_add / tau_del
  full     all 199 train movies, 3 seeds: the weights used in the submission
"""

from __future__ import annotations

import contextlib
import json
import time
from pathlib import Path

import numpy as np

from biohub_tracking.security import load_weights, model_path

CROP = (16, 48, 48)              # model input z, y, x (native voxels)
EXPORT = (18, 64, 64)            # training export keeps a margin for the random shift
MAX_SHIFT = ((EXPORT[0] - CROP[0]) // 2, (EXPORT[1] - CROP[1]) // 2, (EXPORT[2] - CROP[2]) // 2)
VOXEL_UM = (1.625, 0.40625, 0.40625)
POS_REPEAT = 10
# xy test-time views (8 = 4 rotations x flip; 2 = identity + flip). Scoring every hidden movie inside the
# 12 h Kaggle limit bounds this to 2.
VIEWS = 2
EMBRYOS = ("44b6", "6bba")


# ------------------------------------------------------------------ crop and normalisation (shared)
def crop_from_frames(get_frame, t: int, n_frames: int, center_zyx, shape=CROP) -> np.ndarray:
    """(3, *shape) uint16 crop of frames t-1, t, t+1 centred on ``center_zyx`` (native voxels).
    Outside the image is 0; a missing t-1 / t+1 (first / last frame) repeats frame t."""
    shape = np.asarray(shape, dtype=int)
    out = np.zeros((3, *shape), dtype=np.uint16)
    for i, tt in enumerate((t - 1, t, t + 1)):
        frame = get_frame(int(min(max(tt, 0), n_frames - 1)))
        spatial = np.asarray(frame.shape, dtype=int)
        start = np.rint(np.asarray(center_zyx, dtype=float)).astype(int) - shape // 2
        lo, hi = np.maximum(start, 0), np.minimum(start + shape, spatial)
        if np.any(lo >= hi):
            continue
        d0 = lo - start
        d1 = d0 + (hi - lo)
        out[i, d0[0]:d1[0], d0[1]:d1[1], d0[2]:d1[2]] = frame[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
    return out


def normalize(crops: np.ndarray) -> np.ndarray:
    """Per-crop robust scaling over all 3 frames: (x - p1) / (p99.9 - p1), clipped to [-1, 4]."""
    x = crops.astype(np.float32)
    flat = x.reshape(len(x), -1)
    lo = np.percentile(flat, 1.0, axis=1)
    hi = np.percentile(flat, 99.9, axis=1)
    scale = np.maximum(hi - lo, 1.0)
    x = (x - lo[:, None, None, None, None]) / scale[:, None, None, None, None]
    return np.clip(x, -1.0, 4.0)


def center_crop(export_crops: np.ndarray) -> np.ndarray:
    """Model-size crop from the middle of an export crop (no shift)."""
    s = MAX_SHIFT
    return export_crops[:, :, s[0]:s[0] + CROP[0], s[1]:s[1] + CROP[1], s[2]:s[2] + CROP[2]]


# ------------------------------------------------------------------ model
def build_model(width: int = 16, p_drop: float = 0.3):
    import torch.nn as nn

    def block(cin, cout, k):
        pad = tuple(v // 2 for v in k)
        return nn.Sequential(nn.Conv3d(cin, cout, k, padding=pad, bias=False), nn.BatchNorm3d(cout), nn.SiLU())

    w = width
    return nn.Sequential(
        # z voxels are 4x larger than xy voxels: pool xy first
        block(3, w, (1, 3, 3)), block(w, w, (3, 3, 3)), nn.MaxPool3d((1, 2, 2)),           # 16 x 24 x 24
        block(w, 2 * w, (3, 3, 3)), block(2 * w, 2 * w, (3, 3, 3)), nn.MaxPool3d(2),        # 8 x 12 x 12
        block(2 * w, 4 * w, (3, 3, 3)), block(4 * w, 4 * w, (3, 3, 3)), nn.MaxPool3d(2),    # 4 x 6 x 6
        block(4 * w, 8 * w, (3, 3, 3)), nn.AdaptiveAvgPool3d(1), nn.Flatten(),
        nn.Dropout(p_drop), nn.Linear(8 * w, 1))


def predict(models: list, crops_u16: np.ndarray, device: str = "cpu", batch: int = 256,
            views: int | None = None) -> np.ndarray:
    """Mean sigmoid over ``models`` and ``views`` xy views (8: 4 rotations x flip, 4: rotations, 2: identity +
    flip, 1: identity; default VIEWS); crops are model-size uint16. fp16 autocast on CUDA."""
    import torch

    out = np.zeros(len(crops_u16), dtype=np.float64)
    if not len(crops_u16):
        return out
    views = VIEWS if views is None else int(views)
    rots, flips = {8: (4, (False, True)), 4: (4, (False,)), 2: (1, (False, True)), 1: (1, (False,))}[views]
    for model in models:
        model.eval().to(device)
    n_views = views * len(models)
    amp = (torch.autocast("cuda", dtype=torch.float16) if str(device).startswith("cuda")
           else contextlib.nullcontext())
    with torch.no_grad(), amp:
        for i in range(0, len(crops_u16), batch):
            # the per-crop scaling does not depend on the view: normalise once, rotate / flip on the device
            xt = torch.from_numpy(normalize(crops_u16[i:i + batch])).to(device)
            acc = torch.zeros(len(xt), dtype=torch.float64, device=device)
            for model in models:
                for k in range(rots):
                    view = torch.rot90(xt, k, dims=(3, 4))
                    for flip in flips:
                        acc += torch.sigmoid(model(view.flip(-1) if flip else view)[:, 0].float()).double()
            out[i:i + len(xt)] = acc.cpu().numpy()
    return out / n_views


def load_models(weight_dir: Path, names: list[str] | None = None, device: str = "cpu") -> list:
    """Models listed in ``<weight_dir>/t3_models.json`` (optionally only ``names``)."""
    meta = json.loads((Path(weight_dir) / "t3_models.json").read_text())
    models = []
    for entry in meta["models"]:
        if names is not None and entry["name"] not in names:
            continue
        model = build_model(**meta["arch"])
        model.load_state_dict(load_weights(model_path(weight_dir, entry["file"]), map_location=device))
        models.append(model.eval().to(device))
    return models


# ------------------------------------------------------------------ training
def load_crops(crop_dir: Path) -> dict:
    crops, label, movie, offset, event, t, node = [], [], [], [], [], [], []
    for f in sorted(Path(crop_dir).glob("*.npz")):
        with np.load(f) as d:
            crops.append(d["crops"])
            label.append(d["label"].astype(np.int64))
            offset.append(d["event_offset"].astype(np.int64))
            event.append(d["event_parent_id"].astype(np.int64))
            t.append(d["t"].astype(np.int64))
            node.append(d["node_id"].astype(np.int64) if "node_id" in d.files else np.full(len(d["label"]), -1))
            movie += [f.stem] * len(d["label"])
    data = {"crops": np.concatenate(crops), "label": np.concatenate(label), "offset": np.concatenate(offset),
            "event": np.concatenate(event), "t": np.concatenate(t), "node": np.concatenate(node),
            "movie": np.asarray(movie)}
    data["embryo"] = np.asarray([m.split("_")[0] for m in data["movie"]])
    if data["crops"].shape[1:] != (3, *EXPORT):
        raise ValueError(f"unexpected crop shape {data['crops'].shape}")
    return data


def augment(batch_u16: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    out = np.empty((len(batch_u16), 3, *CROP), dtype=np.uint16)
    for i, x in enumerate(batch_u16):
        o = [int(rng.integers(0, 2 * s + 1)) for s in MAX_SHIFT]
        c = x[:, o[0]:o[0] + CROP[0], o[1]:o[1] + CROP[1], o[2]:o[2] + CROP[2]]
        c = np.rot90(c, int(rng.integers(0, 4)), axes=(2, 3))
        if rng.random() < 0.5:
            c = c[..., ::-1]
        if rng.random() < 0.5:
            c = c[..., ::-1, :]
        out[i] = c
    x = normalize(out)
    scale = rng.uniform(0.85, 1.15, size=(len(x), 1, 1, 1, 1)).astype(np.float32)
    return x * scale


def train_one(data: dict, train_idx: np.ndarray, seed: int, epochs: int, device: str, width: int,
              lr: float = 1e-3, batch: int = 64, log=print):
    import torch

    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    model = build_model(width=width).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    pos = train_idx[data["label"][train_idx] == 1]
    neg = train_idx[data["label"][train_idx] == 0]
    per_epoch = len(neg) + POS_REPEAT * len(pos)
    steps = epochs * int(np.ceil(per_epoch / batch))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.15)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    for epoch in range(epochs):
        model.train()
        order = rng.permutation(np.concatenate([neg, np.repeat(pos, POS_REPEAT)]))
        total, n = 0.0, 0
        for i in range(0, len(order), batch):
            idx = order[i:i + batch]
            x = torch.from_numpy(np.ascontiguousarray(augment(data["crops"][idx], rng))).to(device)
            y = torch.from_numpy(data["label"][idx].astype(np.float32)).to(device)
            loss = loss_fn(model(x)[:, 0], y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            total += float(loss) * len(idx)
            n += len(idx)
        if epoch == 0 or (epoch + 1) % 5 == 0 or epoch + 1 == epochs:
            log(f"    epoch {epoch + 1}/{epochs} loss {total / max(n, 1):.4f}")
    return model.eval()


def metrics(label: np.ndarray, score: np.ndarray) -> dict:
    from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

    label = np.asarray(label).astype(int)
    out = {"n_pos": int(label.sum()), "n_neg": int((1 - label).sum())}
    if out["n_pos"] == 0 or out["n_neg"] == 0:
        return out
    out["auc"] = float(roc_auc_score(label, score))
    out["ap"] = float(average_precision_score(label, score))
    prec, rec, _ = precision_recall_curve(label, score)
    for r in (0.3, 0.5, 0.7, 0.9):
        out[f"p_at_r{int(r * 100)}"] = float(prec[rec >= r].max())
    return out


def run_holdout(crop_dir: Path, out: Path, epochs: int, seeds: list[int], width: int, device: str) -> dict:
    """Embryo holdout: train on one embryo, score the other, both directions."""
    import torch

    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    data = load_crops(crop_dir)
    report = {"arch": {"width": width}, "epochs": epochs, "seeds": seeds, "pos_repeat": POS_REPEAT,
              "crop": CROP, "folds": {}}
    models_meta = []
    oof = np.full(len(data["label"]), np.nan)
    for train_e, test_e in ((EMBRYOS[0], EMBRYOS[1]), (EMBRYOS[1], EMBRYOS[0])):
        tr = np.where(data["embryo"] == train_e)[0]
        te = np.where(data["embryo"] == test_e)[0]
        test_x = center_crop(data["crops"][te])
        scores = []
        for seed in seeds:
            print(f"fold train={train_e} test={test_e} seed={seed}", flush=True)
            model = train_one(data, tr, seed, epochs, device, width, log=lambda s: print(s, flush=True))
            name = f"t3_train{train_e}_seed{seed}"
            torch.save(model.state_dict(), out / f"{name}.pt")
            models_meta.append({"name": name, "file": f"{name}.pt", "train_embryo": train_e})
            scores.append(predict([model], test_x, device=device))
        mean = np.mean(scores, axis=0)
        oof[te] = mean
        lab, off = data["label"][te], data["offset"][te]
        exact = (lab == 0) | (off == 0)
        report["folds"][f"{train_e}->{test_e}"] = {
            "all_positives": metrics(lab, mean),
            "offset0_positives_only": metrics(lab[exact], mean[exact]),
            "per_seed_auc": [metrics(lab, s).get("auc") for s in scores],
        }
    (out / "t3_models.json").write_text(json.dumps({"arch": {"width": width}, "crop": CROP,
                                                    "models": models_meta}, indent=1))
    np.savez_compressed(out / "t3_oof.npz", score=oof, label=data["label"], movie=data["movie"],
                        offset=data["offset"], event=data["event"], t=data["t"], node=data["node"])
    report["seconds"] = round(time.time() - t0, 1)
    (out / "t3_report.json").write_text(json.dumps(report, indent=1))
    return report


def run_inner(crop_dir: Path, out: Path, epochs: int, width: int, device: str, folds: int = 3,
              seed: int = 26) -> dict:
    """Movie-grouped K-fold inside each embryo. Threshold selection must not use a model trained on the
    embryo it is evaluated on, and the holdout models cannot score their own training embryo."""
    import torch
    from sklearn.model_selection import GroupKFold

    out.mkdir(parents=True, exist_ok=True)
    data = load_crops(crop_dir)
    inner = np.full(len(data["label"]), np.nan)
    models = []
    for embryo in sorted(set(data["embryo"])):
        indices = np.where(data["embryo"] == embryo)[0]
        groups = data["movie"][indices]
        n = min(folds, len(set(groups)))
        if n < 2:
            raise ValueError(f"at least two movies required for {embryo} inner CV")
        for fold, (tr_local, te_local) in enumerate(GroupKFold(n_splits=n).split(indices, groups=groups)):
            tr, te = indices[tr_local], indices[te_local]
            name = f"t3_inner_{embryo}_fold{fold}"
            model = train_one(data, tr, seed + fold, epochs, device, width, log=lambda s: print(s, flush=True))
            torch.save(model.state_dict(), out / f"{name}.pt")
            inner[te] = predict([model], center_crop(data["crops"][te]), device=device)
            models.append({"name": name, "file": f"{name}.pt", "embryo": embryo,
                           "heldout_movies": sorted(set(data["movie"][te]))})
    if not np.isfinite(inner).all():
        raise RuntimeError("some crops were not scored by an inner fold")
    meta = {"arch": {"width": width}, "crop": CROP, "epochs": epochs, "seed": seed, "folds": folds,
            "models": models, "metrics": metrics(data["label"], inner)}
    (out / "t3_inner_models.json").write_text(json.dumps(meta, indent=1))
    np.savez_compressed(out / "t3_inner_oof.npz", score=inner, label=data["label"],
                        movie=data["movie"], node=data["node"])
    return meta


def run_full(crop_dir: Path, out: Path, epochs: int, seeds: list[int], width: int, device: str) -> dict:
    """All train movies of both embryos, one model per seed (the submission weights). No held-out score:
    thresholds come from the holdout / inner models."""
    import torch

    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    data = load_crops(crop_dir)
    idx = np.arange(len(data["label"]))
    models_meta = []
    for seed in seeds:
        model = train_one(data, idx, seed, epochs, device, width, log=lambda s: print(s, flush=True))
        name = f"t3_full_seed{seed}"
        torch.save(model.state_dict(), out / f"{name}.pt")
        models_meta.append({"name": name, "file": f"{name}.pt", "train_embryo": "all"})
    (out / "t3_models.json").write_text(json.dumps({"arch": {"width": width}, "crop": CROP, "views": VIEWS,
                                                    "models": models_meta}, indent=1))
    report = {"epochs": epochs, "seeds": seeds, "n": int(len(idx)), "positives": int(data["label"].sum()),
              "seconds": round(time.time() - t0, 1)}
    (out / "t3_full_report.json").write_text(json.dumps(report, indent=1))
    return report


# ------------------------------------------------------------------ offline scoring of a candidate table
def score_candidate_table(df, train_dir: Path, holdout_dir: Path, device: str = "cpu",
                          inner_dir: Path | None = None):
    """Score every candidate parent of a labelled candidate table from the raw train images.

    Adds ``t3_holdout`` (models NOT trained on the row's embryo), ``t3_all`` (every holdout model) and, with
    ``inner_dir``, ``t3_inner`` (the within-embryo movie-OOF model that held this movie out). Threshold
    selection fits on ``t3_inner`` and reports on ``t3_holdout`` of the other embryo.
    """
    import zarr

    meta = json.loads((Path(holdout_dir) / "t3_models.json").read_text())
    all_models = load_models(holdout_dir, device=device)
    by_embryo = {e: load_models(holdout_dir, [m["name"] for m in meta["models"] if m["train_embryo"] != e], device)
                 for e in EMBRYOS}
    inner_by_movie = {}
    if inner_dir is not None:
        imeta = json.loads((Path(inner_dir) / "t3_inner_models.json").read_text())
        for entry in imeta["models"]:
            model = build_model(**imeta["arch"])
            model.load_state_dict(load_weights(model_path(inner_dir, entry["file"]), map_location=device))
            for movie in entry["heldout_movies"]:
                if movie in inner_by_movie:
                    raise ValueError(f"duplicate inner model for {movie}")
                inner_by_movie[movie] = model.eval().to(device)
    out = {k: np.full(len(df), np.nan) for k in ("t3_holdout", "t3_all", "t3_inner")}
    for movie, rows in df.groupby("movie"):
        image = zarr.open(str(Path(train_dir) / f"{movie}.zarr" / "0"), mode="r")
        cache: dict[int, np.ndarray] = {}

        def get_frame(tt, image=image, cache=cache):
            if tt not in cache:
                cache[tt] = np.asarray(image[tt])
            return cache[tt]

        par = rows.drop_duplicates("p_id").sort_values("p_t")          # one crop per parent, in frame order
        crops = []
        for r in par.itertuples():
            crops.append(crop_from_frames(get_frame, int(r.p_t), int(image.shape[0]), (r.p_z, r.p_y, r.p_x)))
            for old in [k for k in cache if k < int(r.p_t) - 1]:
                del cache[old]
        crops = np.stack(crops)
        index = rows.index.to_numpy()
        columns = {"t3_holdout": by_embryo[movie.split("_")[0]], "t3_all": all_models}
        if inner_dir is not None:
            if movie not in inner_by_movie:
                raise ValueError(f"no within-embryo movie-OOF T3 model for {movie}")
            columns["t3_inner"] = [inner_by_movie[movie]]
        for column, models in columns.items():
            scores = dict(zip(par.p_id, predict(models, crops, device=device), strict=True))
            out[column][df.index.get_indexer(index)] = rows.p_id.map(scores).to_numpy()
    df = df.copy()
    for column, values in out.items():
        if column != "t3_inner" or inner_dir is not None:
            df[column] = values
    return df
