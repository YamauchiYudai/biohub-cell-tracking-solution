"""Security boundaries: reject pickle execution and files outside model bundles."""

import json
import os
import pickle
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from biohub_tracking.security import load_weights, model_path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import prepare_notebook  # noqa: E402
import prepare_safe_notebook  # noqa: E402

from biohub_tracking.kaggle import notebook as nbk  # noqa: E402


def test_old_torch_rejected_before_loading(monkeypatch):
    fake = SimpleNamespace(__version__="2.5.1", load=lambda *a, **kw: pytest.fail("unsafe load called"))
    monkeypatch.setitem(sys.modules, "torch", fake)
    with pytest.raises(RuntimeError, match="upgrade"):
        load_weights("unused.pt")


def test_weights_only_cannot_be_disabled(monkeypatch):
    calls = []
    fake = SimpleNamespace(__version__="2.14.0", load=lambda *a, **kw: calls.append(kw))
    monkeypatch.setitem(sys.modules, "torch", fake)
    load_weights("weights.pt")
    assert calls == [{"map_location": "cpu", "weights_only": True}]


@pytest.mark.parametrize("filename", ["../outside.pt", "/tmp/outside.pt"])
def test_manifest_cannot_escape(tmp_path, filename):
    with pytest.raises(ValueError, match="escapes"):
        model_path(tmp_path, filename)


def test_manifest_symlink_cannot_escape(tmp_path):
    root = tmp_path / "models"
    root.mkdir()
    (root / "link.pt").symlink_to(tmp_path / "outside.pt")
    with pytest.raises(ValueError, match="escapes"):
        model_path(root, "link.pt")
    assert model_path(root, "nested/ok.pt") == root / "nested/ok.pt"


def test_malicious_checkpoint_never_executes(tmp_path):
    torch = pytest.importorskip("torch")
    marker = tmp_path / "executed"

    class Payload:
        def __reduce__(self):
            return os.system, (f"touch {marker}",)

    path = tmp_path / "bad.pt"
    torch.save(Payload(), path)
    with pytest.raises(pickle.UnpicklingError):
        load_weights(path)
    assert not marker.exists()
    good = tmp_path / "good.pt"
    torch.save({"weight": torch.ones(2)}, good)
    assert torch.equal(load_weights(good)["weight"], torch.ones(2))


def test_safe_notebook_preserves_archive_and_disables_legacy_pickle():
    cfg = prepare_notebook.load_config()
    path = ROOT / cfg["notebook"]["path"]
    before = path.read_bytes()
    safe = prepare_safe_notebook.hardened_notebook(json.loads(before), cfg)
    cells = nbk.code_cells(safe)
    assert cells[0].startswith(prepare_safe_notebook.GUARD)
    assert all("weights_only=False" not in c for c in cells)
    modules = nbk.embedded_modules(cells)
    assert any("Legacy pickle models are disabled" in c for c in modules.values())
    assert all("pickle.loads(" not in c for c in modules.values())
    for name, code in modules.items():
        compile(code, name, "exec")
    assert path.read_bytes() == before


def test_preflight_overrides_unsafe_environment(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(__version__="2.14.0"))
    monkeypatch.setenv("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
    monkeypatch.setenv("TORCH_FORCE_WEIGHTS_ONLY_LOAD", "0")
    exec(prepare_safe_notebook.GUARD, {})
    assert os.environ["TORCH_FORCE_WEIGHTS_ONLY_LOAD"] == "1"
    assert "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD" not in os.environ


def test_release_scanner_rejects_secret_without_printing_value(tmp_path):
    pytest.importorskip("detect_secrets")
    import hashlib

    import check_release_security as scanner

    value = hashlib.sha256(b"synthetic scanner regression fixture").hexdigest()
    path = tmp_path / "settings.py"
    path.write_text('API_KEY = "' + value + '"\n')
    findings = scanner.inspect_file(path, "settings.py", set(), "test")
    assert findings
    assert all(value not in finding for finding in findings)


def test_release_scanner_rejects_private_files_and_notebook_output(tmp_path):
    pytest.importorskip("detect_secrets")
    import check_release_security as scanner

    model = tmp_path / "weights.pth"
    model.write_bytes(b"")
    assert scanner.inspect_file(model, "weights.pth", set(), "test")
    nb = tmp_path / "result.ipynb"
    nb.write_text(json.dumps({"cells": [{"cell_type": "code", "execution_count": 1,
                                        "outputs": [{"text": "private output"}]}]}))
    assert scanner.inspect_file(nb, "result.ipynb", set(), "test")
