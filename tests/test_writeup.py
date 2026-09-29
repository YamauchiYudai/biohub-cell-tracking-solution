"""The Kaggle solution write-up runs without competition data and contains no private information."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from biohub_tracking.kaggle import notebook as nbk

ROOT = Path(__file__).resolve().parents[1]
WRITEUP = ROOT / "notebooks" / "solution_writeup.ipynb"


def test_writeup_code_cells_run(monkeypatch):
    pytest.importorskip("matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    monkeypatch.setattr(plt, "show", lambda *args, **kwargs: None)       # headless: figures are only built
    namespace: dict = {}
    for src in nbk.code_cells(nbk.load_notebook(WRITEUP)):
        exec(compile(src, "writeup", "exec"), namespace)
        plt.close("all")
    assert "estimate_translation" in namespace and "smooth" in namespace


def test_writeup_has_no_outputs_or_private_information():
    nb = nbk.load_notebook(WRITEUP)
    assert all(not c.get("outputs") for c in nb["cells"] if c["cell_type"] == "code")
    text = WRITEUP.read_text()
    for pattern in (r"/Users/", r"@gmail\.com", r"CloudStorage", r"KAGGLE_KEY", r"ghp_[A-Za-z0-9]{20}",
                    r"docs/yamauchi", r"prj_Biohub"):
        assert not re.search(pattern, text), pattern
    assert "Final rank:** TBD" in text and "Private LB:** TBD" in text
