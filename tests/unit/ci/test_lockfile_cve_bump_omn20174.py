# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-20174: verify the fixed packages in the repository root's real uv.lock.

Dev's uv.lock locked virtualenv 21.14.12 (GHSA-cx6f-vjhg-g3cc, HIGH, fixed in
21.14.13), brotli 1.2.0 (GHSA-9v8h-qqgm-4m35, PYSEC-2026-4322, fixed in
1.2.1), and filelock 3.32.7 (GHSA-v4x8-3wr5-3wq8, fixed in 4.2.1). The
Lockfile CVE Scan blocks every PR while these vulnerable versions are locked.
The tests read the real uv.lock at the repository root.
"""

from __future__ import annotations

import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[3]
FIXED_FLOORS = {
    "brotli": (1, 2, 1),
    "filelock": (4, 2, 1),
    "virtualenv": (21, 14, 13),
}


def _version_tuple(version: str) -> tuple[int, ...]:
    """Parse dotted integer versions, refusing components we cannot compare."""
    try:
        return tuple(int(part) for part in version.split("."))
    except ValueError:
        pytest.fail(f"version has a non-integer component: {version!r}")


def _locked_package(name: str) -> dict:
    packages = tomllib.loads((REPO_ROOT / "uv.lock").read_text(encoding="utf-8"))[
        "package"
    ]
    matches = [package for package in packages if package["name"] == name]
    assert len(matches) == 1, f"expected exactly one {name} entry, got {len(matches)}"
    return matches[0]


@pytest.mark.parametrize(("name", "floor"), FIXED_FLOORS.items())
def test_fixed_versions_locked(name: str, floor: tuple[int, ...]) -> None:
    package = _locked_package(name)
    assert _version_tuple(package["version"]) >= floor, (
        f"{name} {package['version']} is below fixed floor {floor}"
    )
    assert package["source"] == {"registry": "https://pypi.org/simple"}
    wheels = package["wheels"]
    assert isinstance(wheels, list) and wheels, f"{name} has no locked wheels"
    artifacts = [package["sdist"], *wheels] if "sdist" in package else wheels
    for artifact in artifacts:
        assert artifact["hash"].startswith("sha256:"), artifact
        assert artifact["url"].startswith("https://files.pythonhosted.org/"), artifact


def test_runner_image_identity_binds_fixed_lock() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/ci/runner_image_identity.py", "--mode", "verify"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, (
        f"runner image identity verification failed ({result.returncode})\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
