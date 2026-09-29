"""Team-authored notebook blocks that distinguish the final-day submission candidates.

These are the exact texts of the submitted Kaggle kernels (verified by the code-cell SHA-256 recorded in
configs/final.yaml), kept as data so every candidate notebook can be regenerated from
notebooks/final_submission.ipynb (the insured J2 + V5a core):

  J2_INSURED_BLOCK  J2 with the fallback-to-S2 insurance (the committed core, 56643954)
  J2_V1_BLOCK   J2 as first submitted (56632306 / 56662358), before the fallback-to-S2 insurance was added
  FC_BLOCK      frozen-frame coordinate consensus (56663004 / 56663011)
  FC_ENV_LINE   the cell-0 line that switches FC on

The equivalent library code is biohub_tracking.postprocessing.linefit / .consensus; the tests execute these
blocks and compare them with the library.
"""

# ruff: noqa: E501

J2_INSURED_BLOCK = r'''# ---- J2 BEGIN: jump-aware output line fit (BIOHUB_OUTPUT_LINEFIT_JUMP_AWARE)
# S2's line fit (weight 0.8, window 2) walks a track 2 frames back and forward and pulls every node toward
# the fitted line. Across a frozen transition (bit-identical frames) or a jump (a whole-embryo step) that walk
# mixes positions from two alignments. J2: stop the walk at a frozen transition; across a jump, subtract the
# translation b1c's relink applied there (x138_estimate_translation, um -> voxels) and go on; stop at a jump
# without one. Fewer than 3 points: no smoothing, as in S2. Off (0): S2's function unchanged.
OUTPUT_LINEFIT_JUMP_AWARE = os.environ.get("BIOHUB_OUTPUT_LINEFIT_JUMP_AWARE", "0").strip() == "1"
if OUTPUT_LINEFIT_JUMP_AWARE:
    X138_RELINK_CTX.setdefault("applied_shift", {})
    _J2_LAST_SHIFT = [None]
    _j2_estimate_translation = x138_estimate_translation
    _j2_relink_setup = x138_relink_setup
    _j2_seed_prior = X138RelinkState.seed_prior
    _j2_linefit_s2 = linefit_smooth_output_graph

    def x138_estimate_translation(src_um, tgt_um, *args, **kwargs):
        shift = _j2_estimate_translation(src_um, tgt_um, *args, **kwargs)
        try:
            _J2_LAST_SHIFT[0] = shift
        except Exception:
            pass  # Recording must not change the relink result.
        return shift

    def x138_relink_setup(times, ids_by_t, position_um, seed_fn, stats):
        # one relink call per setup: keep the shifts of the last call, like X138_RELINK_CTX["transitions"]
        try:
            X138_RELINK_CTX["applied_shift"][X138_RELINK_CTX.get("dataset")] = {}
        except Exception:
            pass  # The shift log is optional; always run the original relink.
        return _j2_relink_setup(times, ids_by_t, position_um, seed_fn, stats)

    def _j2_recording_seed_prior(self, t, source_ids, target_ids, previous_flow):
        try:
            _J2_LAST_SHIFT[0] = None
        except Exception:
            pass
        out = _j2_seed_prior(self, t, source_ids, target_ids, previous_flow)
        try:
            shift = _J2_LAST_SHIFT[0]
            if shift is not None and out is not previous_flow:  # the relink used this translation at t
                X138_RELINK_CTX["applied_shift"].setdefault(X138_RELINK_CTX.get("dataset"), {})[int(t)] = [
                    float(v) for v in shift]
        except Exception:
            pass  # Recording must not change the seed prior returned by relink.
        return out

    X138RelinkState.seed_prior = _j2_recording_seed_prior

    def _j2_linefit_jump_aware(nodes_by_id, edges, stats):
        if not OUTPUT_LINEFIT_SMOOTH or OUTPUT_LINEFIT_WEIGHT <= 0 or OUTPUT_LINEFIT_WINDOW <= 0 or not edges:
            return nodes_by_id
        dataset = X138_RELINK_CTX.get("dataset")
        info = (X138_RELINK_CTX.get("transitions") or {}).get(dataset) or {}
        kinds = {int(t): v["kind"] for t, v in info.items() if t != "_movie_median"}
        scale = np.asarray(VOXEL_SCALE_UM, dtype=np.float64)
        shifts = {int(t): np.asarray(s, dtype=np.float64) / scale
                  for t, s in (X138_RELINK_CTX.get("applied_shift") or {}).get(dataset, {}).items()}
        zero = np.zeros(3, dtype=np.float64)

        def cut(tt):
            kind = kinds.get(int(tt), "normal")
            return kind == "frozen" or (kind == "jump" and int(tt) not in shifts)

        def shift_at(tt):
            return shifts.get(int(tt), zero) if kinds.get(int(tt), "normal") == "jump" else zero

        predecessor, successor = {}, {}
        for edge in edges:
            source_id, target_id = int(edge["source_id"]), int(edge["target_id"])
            source, target = nodes_by_id.get(source_id), nodes_by_id.get(target_id)
            if source is None or target is None or int(target["t"]) != int(source["t"]) + 1:
                continue
            successor.setdefault(source_id, []).append(target_id)
            predecessor.setdefault(target_id, []).append(source_id)
        original_pos = {node_id: np.array([float(n["z"]), float(n["y"]), float(n["x"])], dtype=np.float64)
                        for node_id, n in nodes_by_id.items()}
        updated_pos = {}
        weight = float(np.clip(OUTPUT_LINEFIT_WEIGHT, 0.0, 1.0))
        n_cut = n_shifted = 0
        for node_id in sorted(nodes_by_id):
            neighbourhood = [(0, node_id, zero)]
            current, offset = node_id, zero
            for step in range(1, OUTPUT_LINEFIT_WINDOW + 1):
                prev_ids = predecessor.get(current, [])
                if len(prev_ids) != 1:
                    break
                tt = int(nodes_by_id[prev_ids[0]]["t"])              # transition tt -> tt + 1
                if cut(tt):
                    n_cut += 1
                    break
                offset = offset - shift_at(tt)
                current = prev_ids[0]
                if current not in original_pos:
                    break
                neighbourhood.append((-step, current, offset))
            current, offset = node_id, zero
            for step in range(1, OUTPUT_LINEFIT_WINDOW + 1):
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
                stats["linefit_skipped_nodes"] += 1
                continue
            n_shifted += int(any(np.any(o != 0) for _, _, o in neighbourhood))
            dts = np.array([delta for delta, _, _ in neighbourhood], dtype=np.float64)
            coords = np.stack([original_pos[nid] - o for _, nid, o in neighbourhood])
            fitted = np.array([np.polyval(np.polyfit(dts, coords[:, axis], 1), 0.0) for axis in range(3)],
                              dtype=np.float64)
            if not np.isfinite(fitted).all():
                stats["linefit_skipped_nodes"] += 1
                continue
            updated_pos[node_id] = (1.0 - weight) * original_pos[node_id] + weight * fitted
        # Finish all fallible calculations before changing any node coordinates.
        new_positions = {node_id: tuple(float(v) for v in pos) for node_id, pos in updated_pos.items()}
        stats["linefit_smoothed_nodes"] = len(updated_pos)
        stats["linefit_jump_aware_cut_walks"] = n_cut
        stats["linefit_jump_aware_shifted_nodes"] = n_shifted
        stats["linefit_jump_aware_special_transitions"] = sum(k != "normal" for k in kinds.values())
        stats["linefit_jump_aware_shifts"] = len(shifts)
        for node_id, pos in new_positions.items():
            nodes_by_id[node_id]["z"], nodes_by_id[node_id]["y"], nodes_by_id[node_id]["x"] = pos
        return nodes_by_id

    def linefit_smooth_output_graph(nodes_by_id, edges, stats):
        before = stats.copy()
        try:
            return _j2_linefit_jump_aware(nodes_by_id, edges, stats)
        except Exception:
            stats.clear()
            stats.update(before)
            stats["linefit_jump_aware_failed"] = 1
            return _j2_linefit_s2(nodes_by_id, edges, stats)
# ---- J2 END
'''

J2_V1_BLOCK = r'''# ---- J2 BEGIN: jump-aware output line fit (BIOHUB_OUTPUT_LINEFIT_JUMP_AWARE)
# S2's line fit (weight 0.8, window 2) walks a track 2 frames back and forward and pulls every node toward
# the fitted line. Across a frozen transition (bit-identical frames) or a jump (a whole-embryo step) that walk
# mixes positions from two alignments. J2: stop the walk at a frozen transition; across a jump, subtract the
# translation b1c's relink applied there (x138_estimate_translation, um -> voxels) and go on; stop at a jump
# without one. Fewer than 3 points: no smoothing, as in S2. Off (0): S2's function unchanged.
OUTPUT_LINEFIT_JUMP_AWARE = os.environ.get("BIOHUB_OUTPUT_LINEFIT_JUMP_AWARE", "0").strip() == "1"
if OUTPUT_LINEFIT_JUMP_AWARE:
    X138_RELINK_CTX.setdefault("applied_shift", {})
    _J2_LAST_SHIFT = [None]
    _j2_estimate_translation = x138_estimate_translation
    _j2_relink_setup = x138_relink_setup
    _j2_seed_prior = X138RelinkState.seed_prior
    _j2_linefit_s2 = linefit_smooth_output_graph

    def x138_estimate_translation(src_um, tgt_um, *args, **kwargs):
        shift = _j2_estimate_translation(src_um, tgt_um, *args, **kwargs)
        _J2_LAST_SHIFT[0] = shift
        return shift

    def x138_relink_setup(times, ids_by_t, position_um, seed_fn, stats):
        # one relink call per setup: keep the shifts of the last call, like X138_RELINK_CTX["transitions"]
        X138_RELINK_CTX["applied_shift"][X138_RELINK_CTX.get("dataset")] = {}
        return _j2_relink_setup(times, ids_by_t, position_um, seed_fn, stats)

    def _j2_recording_seed_prior(self, t, source_ids, target_ids, previous_flow):
        _J2_LAST_SHIFT[0] = None
        out = _j2_seed_prior(self, t, source_ids, target_ids, previous_flow)
        shift = _J2_LAST_SHIFT[0]
        if shift is not None and out is not previous_flow:        # the relink used this translation at t
            X138_RELINK_CTX["applied_shift"].setdefault(X138_RELINK_CTX.get("dataset"), {})[int(t)] = [
                float(v) for v in shift]
        return out

    X138RelinkState.seed_prior = _j2_recording_seed_prior

    def linefit_smooth_output_graph(nodes_by_id, edges, stats):
        if not OUTPUT_LINEFIT_SMOOTH or OUTPUT_LINEFIT_WEIGHT <= 0 or OUTPUT_LINEFIT_WINDOW <= 0 or not edges:
            return nodes_by_id
        dataset = X138_RELINK_CTX.get("dataset")
        info = (X138_RELINK_CTX.get("transitions") or {}).get(dataset) or {}
        kinds = {int(t): v["kind"] for t, v in info.items() if t != "_movie_median"}
        scale = np.asarray(VOXEL_SCALE_UM, dtype=np.float64)
        shifts = {int(t): np.asarray(s, dtype=np.float64) / scale
                  for t, s in (X138_RELINK_CTX.get("applied_shift") or {}).get(dataset, {}).items()}
        zero = np.zeros(3, dtype=np.float64)

        def cut(tt):
            kind = kinds.get(int(tt), "normal")
            return kind == "frozen" or (kind == "jump" and int(tt) not in shifts)

        def shift_at(tt):
            return shifts.get(int(tt), zero) if kinds.get(int(tt), "normal") == "jump" else zero

        predecessor, successor = {}, {}
        for edge in edges:
            source_id, target_id = int(edge["source_id"]), int(edge["target_id"])
            source, target = nodes_by_id.get(source_id), nodes_by_id.get(target_id)
            if source is None or target is None or int(target["t"]) != int(source["t"]) + 1:
                continue
            successor.setdefault(source_id, []).append(target_id)
            predecessor.setdefault(target_id, []).append(source_id)
        original_pos = {node_id: np.array([float(n["z"]), float(n["y"]), float(n["x"])], dtype=np.float64)
                        for node_id, n in nodes_by_id.items()}
        updated_pos = {}
        weight = float(np.clip(OUTPUT_LINEFIT_WEIGHT, 0.0, 1.0))
        n_cut = n_shifted = 0
        for node_id in sorted(nodes_by_id):
            neighbourhood = [(0, node_id, zero)]
            current, offset = node_id, zero
            for step in range(1, OUTPUT_LINEFIT_WINDOW + 1):
                prev_ids = predecessor.get(current, [])
                if len(prev_ids) != 1:
                    break
                tt = int(nodes_by_id[prev_ids[0]]["t"])              # transition tt -> tt + 1
                if cut(tt):
                    n_cut += 1
                    break
                offset = offset - shift_at(tt)
                current = prev_ids[0]
                if current not in original_pos:
                    break
                neighbourhood.append((-step, current, offset))
            current, offset = node_id, zero
            for step in range(1, OUTPUT_LINEFIT_WINDOW + 1):
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
                stats["linefit_skipped_nodes"] += 1
                continue
            n_shifted += int(any(np.any(o != 0) for _, _, o in neighbourhood))
            dts = np.array([delta for delta, _, _ in neighbourhood], dtype=np.float64)
            coords = np.stack([original_pos[nid] - o for _, nid, o in neighbourhood])
            fitted = np.array([np.polyval(np.polyfit(dts, coords[:, axis], 1), 0.0) for axis in range(3)],
                              dtype=np.float64)
            if not np.isfinite(fitted).all():
                stats["linefit_skipped_nodes"] += 1
                continue
            updated_pos[node_id] = (1.0 - weight) * original_pos[node_id] + weight * fitted
        for node_id, pos in updated_pos.items():
            nodes_by_id[node_id]["z"] = float(pos[0])
            nodes_by_id[node_id]["y"] = float(pos[1])
            nodes_by_id[node_id]["x"] = float(pos[2])
        stats["linefit_smoothed_nodes"] = len(updated_pos)
        stats["linefit_jump_aware_cut_walks"] = n_cut
        stats["linefit_jump_aware_shifted_nodes"] = n_shifted
        stats["linefit_jump_aware_special_transitions"] = sum(k != "normal" for k in kinds.values())
        stats["linefit_jump_aware_shifts"] = len(shifts)
        return nodes_by_id
# ---- J2 END
'''

FC_BLOCK = r'''# ---- FC BEGIN: frozen-frame coordinate consensus (BIOHUB_OUTPUT_FROZEN_CONSENSUS)
# A frozen transition t (frames t and t+1 bit-identical; X138_RELINK_CTX kind "frozen") shows the same nuclei
# twice, and GT does not move across it (6bba, 88 train movies: 99.5 % of 4,420 GT edges < 0.01 um). The detector
# reads neighbouring frames and J2 cuts its walk there, so the two copies of a nucleus drift apart. Consensus: right
# after the line fit, every chain of nodes linked across frozen transitions gets the mean of its smoothed
# coordinates. Linked edge t -> t+1: kind(t) == "frozen", the source has 1 child and <= 1 parent, the target 1 parent
# and <= 1 child, and the pre-smoothing endpoints are <= 1.625 um apart. A chain whose mean is > 1.625 um from any
# of its nodes is left alone. Node ids, times, edges and divisions never change; the writer rounds as before.
# Any exception: this movie keeps the line fit's output (frozen_consensus_failed = 1). Off (0): not installed.
OUTPUT_FROZEN_CONSENSUS = os.environ.get("BIOHUB_OUTPUT_FROZEN_CONSENSUS", "0").strip() == "1"
FROZEN_CONSENSUS_MAX_UM = 1.625
import time as _fc_time


def _fc_written(v):
    return max(0, int(round(float(v))))                  # write_test_submission's rounding


def _fc_consensus(nodes_by_id, edges, before, kinds, scale_um, max_um):
    """(new coordinates {node_id: (z, y, x)}, counts). Reads its arguments only.
    Chains are walked from their first node, first nodes in id order; the mean is taken in time order."""
    scale = np.asarray(scale_um, dtype=np.float64)
    children, parents = {}, {}
    for edge in edges:
        source_id, target_id = int(edge["source_id"]), int(edge["target_id"])
        children[source_id] = children.get(source_id, 0) + 1
        parents[target_id] = parents.get(target_id, 0) + 1
    link = {}                                            # source -> target of each linked edge
    for edge in edges:
        source_id, target_id = int(edge["source_id"]), int(edge["target_id"])
        source, target = nodes_by_id.get(source_id), nodes_by_id.get(target_id)
        if source is None or target is None or int(target["t"]) != int(source["t"]) + 1:
            continue
        if kinds.get(int(source["t"]), "normal") != "frozen":
            continue
        if (children[source_id] != 1 or parents[target_id] != 1 or parents.get(source_id, 0) > 1
                or children.get(target_id, 0) > 1):
            continue
        gap = (np.asarray(before[source_id], dtype=np.float64) - np.asarray(before[target_id], dtype=np.float64))
        if not float(np.sqrt(np.sum((gap * scale) ** 2))) <= max_um:
            continue
        link[source_id] = target_id
    counts = {"frozen_consensus_edges": len(link), "frozen_consensus_groups": 0, "frozen_consensus_nodes": 0,
              "frozen_consensus_skipped_groups": 0, "frozen_consensus_moved_rounded": 0}
    moves = {}
    linked_targets = set(link.values())
    for start in sorted(s for s in link if s not in linked_targets):
        chain = [start]
        while chain[-1] in link:
            chain.append(link[chain[-1]])
        pts = np.array([[float(nodes_by_id[i]["z"]), float(nodes_by_id[i]["y"]), float(nodes_by_id[i]["x"])]
                        for i in chain], dtype=np.float64)
        mean = pts.mean(axis=0)
        if not float(np.max(np.sqrt((((pts - mean) * scale) ** 2).sum(axis=1)))) <= max_um:
            counts["frozen_consensus_skipped_groups"] += 1
            continue
        new = tuple(float(v) for v in mean)
        counts["frozen_consensus_groups"] += 1
        counts["frozen_consensus_nodes"] += len(chain)
        for node_id, old in zip(chain, pts):
            moves[node_id] = new
            counts["frozen_consensus_moved_rounded"] += int(any(_fc_written(a) != _fc_written(b)
                                                                for a, b in zip(new, old)))
    return moves, counts


if OUTPUT_FROZEN_CONSENSUS:
    _fc_linefit = linefit_smooth_output_graph

    def linefit_smooth_output_graph(nodes_by_id, edges, stats):
        try:
            before = {int(i): (float(n["z"]), float(n["y"]), float(n["x"])) for i, n in nodes_by_id.items()}
        except Exception:
            before = None                                # the line fit still runs; the consensus is skipped
        nodes_by_id = _fc_linefit(nodes_by_id, edges, stats)
        t0 = _fc_time.time()
        written = {}
        try:
            if before is None:
                raise ValueError("no pre-smoothing coordinates")
            dataset = X138_RELINK_CTX.get("dataset")
            info = (X138_RELINK_CTX.get("transitions") or {}).get(dataset) or {}
            kinds = {int(t): v["kind"] for t, v in info.items() if t != "_movie_median"}
            moves, counts = _fc_consensus(nodes_by_id, edges, before, kinds, VOXEL_SCALE_UM,
                                          FROZEN_CONSENSUS_MAX_UM)
            for node_id, pos in moves.items():
                node = nodes_by_id[node_id]
                written[node_id] = (node["z"], node["y"], node["x"])
                node["z"], node["y"], node["x"] = pos
        except Exception:
            for node_id, pos in written.items():         # undo any coordinate already written
                nodes_by_id[node_id]["z"], nodes_by_id[node_id]["y"], nodes_by_id[node_id]["x"] = pos
            stats["frozen_consensus_failed"] = 1
            return nodes_by_id
        stats.update(counts)
        stats["frozen_consensus_failed"] = 0
        stats["frozen_consensus_seconds"] = round(_fc_time.time() - t0, 3)
        return nodes_by_id
# ---- FC END
'''

FC_ENV_LINE = (
    "\n# 2026-09-29 frozen-frame coordinate consensus after J2 (same smoothed mean across frozen transitions)\n"
    'os.environ["BIOHUB_OUTPUT_FROZEN_CONSENSUS"] = "1"\n'
)
