"""Read, fingerprint and take apart the production Kaggle notebook.

The submission is a Kaggle notebook built on the public "biohub x138" notebook (Apache-2.0, see
THIRD_PARTY_NOTICES.md). The upstream cells are untouched except for three kinds of team additions:

  cell 0   environment overrides appended at the end (relink variant, division method / thresholds, J2, ...)
  cell 5   inserted blocks: the relink helpers (b1c), the embedded division runtime modules, the division
           scorer + post-safe-div hook, and J2 (and FC in some candidates)
  cells 12-13  manifest cells that record the resolved configuration and fail the run on a degraded deadline

Parity is defined on the code cells only: ``code_sha256`` hashes the list of code-cell sources, so markdown
added for readers does not change it.
"""

from __future__ import annotations

import ast
import hashlib
import json
import re
from pathlib import Path

POSTPROC_CELL = 5
HEAD_CELL = 4
RELINK_BEGIN = "# ===== X138_RELINK_HELPERS BEGIN (scripts/x138/relink_helpers.py) =====\n"
RELINK_END = "# ===== X138_RELINK_HELPERS END =====\n"
J2_BEGIN = "# ---- J2 BEGIN: jump-aware output line fit (BIOHUB_OUTPUT_LINEFIT_JUMP_AWARE)\n"
J2_END = "# ---- J2 END\n"
FC_BEGIN = "# ---- FC BEGIN: frozen-frame coordinate consensus (BIOHUB_OUTPUT_FROZEN_CONSENSUS)\n"
FC_END = "# ---- FC END\n"
HOOK_MARKER = "\n# One post-safe-div hook in replay and submission.\n"
SCORE_SETUP_BEGIN = "import json as _dv_json\n_DV_METHOD = "
_ENV_ASSIGN = re.compile(r'^os\.environ\["([A-Z0-9_]+)"\]\s*=\s*(.+?)\s*(?:#.*)?$')


def load_notebook(path: Path | str) -> dict:
    return json.loads(Path(path).read_text())


def cell_source(cell: dict) -> str:
    src = cell["source"]
    return "".join(src) if isinstance(src, list) else src


def code_cells(nb: dict) -> list[str]:
    return [cell_source(c) for c in nb["cells"] if c["cell_type"] == "code"]


def code_sha256(cells: list[str]) -> str:
    """Fingerprint of a notebook's code (markdown and outputs excluded)."""
    return hashlib.sha256(json.dumps(cells).encode()).hexdigest()


def extract_block(src: str, begin: str, end: str) -> str:
    """The text from ``begin`` through ``end`` (inclusive); both must occur exactly once."""
    if src.count(begin) != 1 or src.count(end) != 1:
        raise ValueError(f"block markers must occur once: {begin.strip()!r} x{src.count(begin)}, "
                         f"{end.strip()!r} x{src.count(end)}")
    i = src.index(begin)
    return src[i:src.index(end, i) + len(end)]


def has_block(src: str, begin: str) -> bool:
    return begin in src


def embedded_modules(cells: list[str]) -> dict[str, str]:
    """The division runtime modules the notebook writes to disk and imports (``_DV_RT = {name: source}``)."""
    for node in ast.parse(cells[POSTPROC_CELL]).body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "_DV_RT" for t in node.targets):
            return ast.literal_eval(node.value)
    raise KeyError("_DV_RT not found in the post-processing cell")


def extract_function(src: str, name: str) -> str:
    """Source of the top-level ``def name`` in ``src`` (exact slice of the original text)."""
    for node in ast.parse(src).body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            segment = ast.get_source_segment(src, node)
            if segment is None:
                break
            return segment + "\n"
    raise KeyError(name)


def env_overrides(cells: list[str]) -> dict[str, str]:
    """``os.environ["KEY"] = "value"`` assignments of cell 0, in order (a later one wins)."""
    out: dict[str, str] = {}
    for line in cells[0].splitlines():
        m = _ENV_ASSIGN.match(line.strip())
        if m:
            try:
                out[m.group(1)] = str(ast.literal_eval(m.group(2)))
            except (ValueError, SyntaxError):
                continue
    return out


def publishable(nb: dict, code: list[str], markdown: dict[int, str]) -> dict:
    """A clean notebook: ``markdown[i]`` is inserted before code cell i, outputs and run metadata dropped."""
    cells = []
    for i, src in enumerate(code):
        if i in markdown:
            cells.append({"cell_type": "markdown", "metadata": {}, "source": markdown[i].splitlines(keepends=True)})
        cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
                      "source": src.splitlines(keepends=True)})
    metadata = {k: v for k, v in nb.get("metadata", {}).items() if k in ("kernelspec", "language_info", "kaggle")}
    return {"cells": cells, "metadata": metadata, "nbformat": 4, "nbformat_minor": 4}
