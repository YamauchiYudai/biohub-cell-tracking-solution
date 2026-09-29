"""Convert between the final-day submission candidates, all built on the same core notebook.

Every candidate is the core (S2 + J2 + V5a with the J2 insurance, submission 56643954) plus a subset of three
independent, anchor-checked edits. Each edit is invertible, so the committed notebook can be converted to any
candidate and verified against the code-cell SHA-256 recorded in configs/final.yaml:

  j2_v1             J2 without the fallback-to-S2 insurance (as first submitted)
  frozen_consensus  the FC block after J2 plus its cell-0 switch
  head_f03          the V1284 coordinate head loaded from the team's F03 dataset (sha256-checked)
"""

from __future__ import annotations

from .blocks import FC_BLOCK, FC_ENV_LINE, J2_INSURED_BLOCK, J2_V1_BLOCK
from .notebook import FC_BEGIN, HEAD_CELL, HOOK_MARKER, J2_BEGIN, J2_END, POSTPROC_CELL, extract_block

PUBLIC_HEAD = "anvithpothula/biohub-v1284-head-s075"
F03_HEAD = "yudaiyamauchi/biomed-x138-v1284-head-f03"
F03_SHA256 = "924a24f8316c095a1a0ef986087611d68559bb4fcb5b11322d3854ae683b5dd3"

_MOUNT_CHECK = ("if len(_myhead) != 1:\n"
                "    raise RuntimeError(('my V1284 head mount mismatch', [str(p) for p in _myhead]))\n")


def _lookup(dataset: str) -> str:
    return f"_myhead = sorted(Path('/kaggle/input').rglob('{dataset.split('/')[1]}/v1284_head.pt'))\n"


def _sha_check(sha256: str) -> str:
    return ("import hashlib as _v1284_hashlib\n"
            "_v1284_sha = _v1284_hashlib.sha256(_myhead[0].read_bytes()).hexdigest()\n"
            f"if _v1284_sha != {sha256!r}:\n"
            "    raise RuntimeError(('V1284 head sha256 mismatch', _v1284_sha))\n"
            "print('V1284 head', _myhead[0], 'sha256', _v1284_sha)\n")


def _label(dataset: str) -> str:
    return f"X138_HEAD_DATASET = '{dataset}'\n"


def _swap(src: str, old: str, new: str) -> str:
    if src.count(old) != 1:
        raise ValueError(f"anchor occurs {src.count(old)} times: {old[:80]!r}")
    return src.replace(old, new)


def _replace(cells: list[str], index: int, new: str) -> list[str]:
    out = list(cells)
    out[index] = new
    for i, cell in enumerate(out):
        compile(cell, f"cell{i}", "exec")
    return out


# ------------------------------------------------------------------ edits and their inverses
def features(cells: list[str]) -> set[str]:
    """Which edits a notebook carries."""
    c5 = cells[POSTPROC_CELL]
    found = set()
    if extract_block(c5, J2_BEGIN, J2_END) == J2_V1_BLOCK:
        found.add("j2_v1")
    elif extract_block(c5, J2_BEGIN, J2_END) != J2_INSURED_BLOCK:
        raise ValueError("unknown J2 block: not a notebook of this solution")
    if FC_BEGIN in c5:
        found.add("frozen_consensus")
    if _lookup(F03_HEAD) in cells[HEAD_CELL]:
        found.add("head_f03")
    return found


def add_j2_v1(cells: list[str]) -> list[str]:
    return _replace(cells, POSTPROC_CELL, _swap(cells[POSTPROC_CELL], J2_INSURED_BLOCK, J2_V1_BLOCK))


def remove_j2_v1(cells: list[str]) -> list[str]:
    return _replace(cells, POSTPROC_CELL, _swap(cells[POSTPROC_CELL], J2_V1_BLOCK, J2_INSURED_BLOCK))


def add_frozen_consensus(cells: list[str]) -> list[str]:
    j2_on = 'os.environ["BIOHUB_OUTPUT_LINEFIT_JUMP_AWARE"] = "1"' in cells[0]
    if "BIOHUB_OUTPUT_FROZEN_CONSENSUS" in cells[0] or not j2_on:
        raise ValueError("cell 0 must turn on J2 and not set the consensus yet")
    out = list(cells)
    out[0] = cells[0].rstrip("\n") + FC_ENV_LINE
    out = _replace(out, POSTPROC_CELL, _swap(cells[POSTPROC_CELL], J2_END + HOOK_MARKER,
                                             J2_END + "\n" + FC_BLOCK + HOOK_MARKER))
    return out


def remove_frozen_consensus(cells: list[str]) -> list[str]:
    if not cells[0].endswith(FC_ENV_LINE):
        raise ValueError("cell 0 does not end with the FC switch")
    out = list(cells)
    out[0] = cells[0][:-len(FC_ENV_LINE)] + "\n"
    return _replace(out, POSTPROC_CELL, _swap(cells[POSTPROC_CELL], "\n" + FC_BLOCK, ""))


def add_head_f03(cells: list[str]) -> list[str]:
    c4 = _swap(cells[HEAD_CELL], _lookup(PUBLIC_HEAD), _lookup(F03_HEAD))
    c4 = _swap(c4, _MOUNT_CHECK, _MOUNT_CHECK + _sha_check(F03_SHA256))
    out = _replace(cells, HEAD_CELL, c4)
    return _replace(out, POSTPROC_CELL, _swap(out[POSTPROC_CELL], _label(PUBLIC_HEAD), _label(F03_HEAD)))


def remove_head_f03(cells: list[str]) -> list[str]:
    c4 = _swap(cells[HEAD_CELL], _lookup(F03_HEAD), _lookup(PUBLIC_HEAD))
    c4 = _swap(c4, _MOUNT_CHECK + _sha_check(F03_SHA256), _MOUNT_CHECK)
    out = _replace(cells, HEAD_CELL, c4)
    return _replace(out, POSTPROC_CELL, _swap(out[POSTPROC_CELL], _label(F03_HEAD), _label(PUBLIC_HEAD)))


EDITS = {  # name: (add, remove); applied in this order when building
    "j2_v1": (add_j2_v1, remove_j2_v1),
    "frozen_consensus": (add_frozen_consensus, remove_frozen_consensus),
    "head_f03": (add_head_f03, remove_head_f03),
}


def to_core(cells: list[str]) -> list[str]:
    for name in reversed(list(EDITS)):
        if name in features(cells):
            cells = EDITS[name][1](cells)
    return cells


def build(core: list[str], edits: list[str]) -> list[str]:
    unknown = set(edits) - set(EDITS)
    if unknown:
        raise ValueError(f"unknown edits {sorted(unknown)}")
    if features(core):
        raise ValueError("build starts from the core notebook")
    cells = core
    for name in EDITS:
        if name in edits:
            cells = EDITS[name][0](cells)
    return cells
