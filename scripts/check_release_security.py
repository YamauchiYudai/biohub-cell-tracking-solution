#!/usr/bin/env python3
"""Fail on new secrets, private artifacts, or executable-notebook output.

Scans tracked and non-ignored untracked files. --history also scans every blob reachable from local refs.
Findings never print secret values. .secrets.baseline contains reviewed hash fingerprints only.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

from detect_secrets import SecretsCollection
from detect_secrets.settings import default_settings

ROOT = Path(__file__).resolve().parents[1]
PRIVATE_SUFFIXES = {".pt", ".pth", ".ckpt", ".onnx", ".safetensors", ".npz", ".npy", ".tif", ".tiff",
                    ".h5", ".pem", ".key", ".pkl", ".pickle"}
PRIVATE_DIRS = {"data", "outputs", "artifacts", "checkpoints", "weights", ".kaggle", ".clearml"}


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT)  # noqa: S603


def private_path(name):
    p = Path(name)
    return (p.suffix.lower() in PRIVATE_SUFFIXES or any(
        part in PRIVATE_DIRS or part.endswith((".zarr", ".geff")) for part in p.parts)
        or p.name in {"kaggle.json", "clearml.conf", ".netrc", ".env"}
        or (p.name.startswith(".env.") and p.name != ".env.example"))


def inspect_file(path, name, allowed, label):
    problems = []
    if private_path(name):
        problems.append(f"{label}:{name}: private artifact")
    if name == ".secrets.baseline":
        return problems  # Scanner hashes, not original secret values; reviewed separately.
    with default_settings():
        collection = SecretsCollection()
        collection.scan_file(str(path))
    for findings in collection.json().values():
        for f in findings:
            key = (name, f["type"], f["hashed_secret"])
            if key not in allowed:
                problems.append(f"{label}:{name}:{f['line_number']}: {f['type']} (redacted)")
    if path.suffix == ".ipynb":
        nb = json.loads(path.read_text())
        for i, cell in enumerate(nb["cells"]):
            if cell.get("cell_type") == "code" and (cell.get("outputs") or cell.get("execution_count") is not None):
                problems.append(f"{label}:{name}: code cell {i} contains execution output/metadata")
    return problems


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--history", action="store_true")
    args = ap.parse_args()
    baseline = json.loads((ROOT / ".secrets.baseline").read_text())
    allowed = {(name, f["type"], f["hashed_secret"]) for name, fs in baseline["results"].items() for f in fs}
    files = set(git("ls-files", "-z", "--cached", "--others", "--exclude-standard").decode().strip("\0").split("\0"))
    problems = []
    for name in sorted(files):
        path = ROOT / name
        if path.is_file():
            problems.extend(inspect_file(path, name, allowed, "working-tree"))
    blobs = set()
    if args.history:
        with tempfile.TemporaryDirectory(prefix="biohub-security-") as tmp:
            for ref in git("rev-list", "--all").decode().splitlines():
                for entry in git("ls-tree", "-rz", ref).split(b"\0"):
                    if not entry:
                        continue
                    meta, name_bytes = entry.split(b"\t", 1)
                    _, kind, oid = meta.split()
                    name = name_bytes.decode()
                    if kind != b"blob" or (oid, name) in blobs:
                        continue
                    blobs.add((oid, name))
                    path = Path(tmp) / ("scan" + Path(name).suffix)
                    path.write_bytes(git("cat-file", "blob", oid.decode()))
                    problems.extend(inspect_file(path, name, allowed, ref[:8]))
    for p in problems:
        print(p)
    print(f"Scanned {len(files)} working files and {len(blobs)} historical blobs; {len(problems)} findings")
    return int(bool(problems))


if __name__ == "__main__":
    raise SystemExit(main())
