"""Output line-fit smoothing, and its jump-aware version J2.

The upstream post-processing smooths every node of the final graph: it walks up to ``window`` (2) frames back
and forward along single-parent / single-child links, fits a straight line per axis through those positions
and moves the node a fraction ``weight`` (0.8) toward the fitted value at its own time. Nodes with fewer than
3 points on their walk are left alone.

J2 (production since 2026-09-28): across a *frozen* transition (duplicated frame) or a *jump* (whole-embryo
step) that walk mixes positions from two alignments. J2 stops the walk at a frozen transition; across a jump
it subtracts the translation the b1c relink applied there (converted um -> voxels) and continues; it stops at
a jump for which the relink applied no translation. On movies without special transitions J2 equals the
upstream smoothing exactly.
"""

from __future__ import annotations

import numpy as np

VOXEL_SCALE_UM = np.array([1.625, 0.40625, 0.40625])


def _links(nodes_by_id: dict, edges: list[dict]) -> tuple[dict, dict]:
    predecessor: dict[int, list[int]] = {}
    successor: dict[int, list[int]] = {}
    for edge in edges:
        source_id, target_id = int(edge["source_id"]), int(edge["target_id"])
        source, target = nodes_by_id.get(source_id), nodes_by_id.get(target_id)
        if source is None or target is None or int(target["t"]) != int(source["t"]) + 1:
            continue
        successor.setdefault(source_id, []).append(target_id)
        predecessor.setdefault(target_id, []).append(source_id)
    return predecessor, successor


def jump_aware_linefit(nodes_by_id: dict, edges: list[dict], kinds: dict[int, str],
                       shifts_um: dict[int, np.ndarray], *, weight: float = 0.8, window: int = 2,
                       voxel_scale_um=VOXEL_SCALE_UM, stats: dict | None = None) -> dict:
    """J2 on a graph ``{node_id: {"t", "z", "y", "x"}}`` (voxel coordinates, edited in place and returned).

    ``kinds``: {transition t: "frozen" | "jump" | "normal"} from the relink classification;
    ``shifts_um``: {transition t: (dz, dy, dx) um} the relink added on jump transitions.
    All calculations finish before any coordinate is written. With no frozen / jump transitions this is the
    upstream smoothing.
    """
    stats = {} if stats is None else stats
    if weight <= 0 or window <= 0 or not edges:
        return nodes_by_id
    scale = np.asarray(voxel_scale_um, dtype=np.float64)
    shifts = {int(t): np.asarray(s, dtype=np.float64) / scale for t, s in shifts_um.items()}
    zero = np.zeros(3, dtype=np.float64)

    def cut(tt):
        kind = kinds.get(int(tt), "normal")
        return kind == "frozen" or (kind == "jump" and int(tt) not in shifts)

    def shift_at(tt):
        return shifts.get(int(tt), zero) if kinds.get(int(tt), "normal") == "jump" else zero

    predecessor, successor = _links(nodes_by_id, edges)
    original_pos = {node_id: np.array([float(n["z"]), float(n["y"]), float(n["x"])], dtype=np.float64)
                    for node_id, n in nodes_by_id.items()}
    updated_pos = {}
    w = float(np.clip(weight, 0.0, 1.0))
    n_cut = n_shifted = n_skipped = 0
    for node_id in sorted(nodes_by_id):
        neighbourhood = [(0, node_id, zero)]
        current, offset = node_id, zero
        for step in range(1, window + 1):                          # backward walk
            prev_ids = predecessor.get(current, [])
            if len(prev_ids) != 1:
                break
            tt = int(nodes_by_id[prev_ids[0]]["t"])                 # transition tt -> tt + 1
            if cut(tt):
                n_cut += 1
                break
            offset = offset - shift_at(tt)
            current = prev_ids[0]
            if current not in original_pos:
                break
            neighbourhood.append((-step, current, offset))
        current, offset = node_id, zero
        for step in range(1, window + 1):                          # forward walk
            next_ids = successor.get(current, [])
            if len(next_ids) != 1:
                break
            tt = int(nodes_by_id[current]["t"])
            if cut(tt):
                n_cut += 1
                break
            offset = offset + shift_at(tt)
            current = next_ids[0]
            if current not in original_pos:
                break
            neighbourhood.append((step, current, offset))
        if len(neighbourhood) < 3:
            n_skipped += 1
            continue
        n_shifted += int(any(np.any(o != 0) for _, _, o in neighbourhood))
        dts = np.array([delta for delta, _, _ in neighbourhood], dtype=np.float64)
        coords = np.stack([original_pos[nid] - o for _, nid, o in neighbourhood])
        fitted = np.array([np.polyval(np.polyfit(dts, coords[:, axis], 1), 0.0) for axis in range(3)],
                          dtype=np.float64)
        if not np.isfinite(fitted).all():
            n_skipped += 1
            continue
        updated_pos[node_id] = (1.0 - w) * original_pos[node_id] + w * fitted
    new_positions = {node_id: tuple(float(v) for v in pos) for node_id, pos in updated_pos.items()}
    stats["linefit_skipped_nodes"] = stats.get("linefit_skipped_nodes", 0) + n_skipped
    stats["linefit_smoothed_nodes"] = len(updated_pos)
    stats["linefit_jump_aware_cut_walks"] = n_cut
    stats["linefit_jump_aware_shifted_nodes"] = n_shifted
    stats["linefit_jump_aware_special_transitions"] = sum(k != "normal" for k in kinds.values())
    stats["linefit_jump_aware_shifts"] = len(shifts)
    for node_id, pos in new_positions.items():
        nodes_by_id[node_id]["z"], nodes_by_id[node_id]["y"], nodes_by_id[node_id]["x"] = pos
    return nodes_by_id


def linefit(nodes_by_id: dict, edges: list[dict], *, weight: float = 0.8, window: int = 2) -> dict:
    """The upstream smoothing (no special transitions)."""
    return jump_aware_linefit(nodes_by_id, edges, {}, {}, weight=weight, window=window)
