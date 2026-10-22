# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The runner image gets Python 3.12 from uv, not from a Launchpad PPA (OMN-17427).

On 2026-10-21 the deadsnakes PPA on ppa.launchpadcontent.net answered 503 to
every runner-image-build-smoke and runner-image-multiarch-smoke from 19:46Z to
20:27Z. Any PR that touches ``uv.lock`` re-binds the runner image and builds it,
so each one went red in CI Summary for a third party's outage. One of them was
the virtualenv CVE fix that every other PR's Lockfile CVE Scan was waiting on.

The image now installs a uv-managed CPython (python-build-standalone), fetched
from GitHub releases like the uv binary itself. These tests pin that source, and
the order it needs: uv must be on PATH before the Python step runs.

Like test_runner_image_multiarch.py, this is a TEXT-level test. The unit lane
has no builder; build-smoke proves the built image.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]
DOCKERFILE = REPO_ROOT / "docker" / "runners" / "Dockerfile"


def _instructions() -> list[str]:
    """Dockerfile instructions with comments dropped and continuations joined."""
    joined: list[str] = []
    current = ""
    for raw in DOCKERFILE.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.endswith("\\"):
            current += line[:-1] + " "
            continue
        joined.append(current + line)
        current = ""
    if current:
        joined.append(current)
    return joined


def _index(predicate_text: str) -> int:
    for i, instruction in enumerate(_instructions()):
        if predicate_text in instruction:
            return i
    raise AssertionError(f"no Dockerfile instruction contains {predicate_text!r}")


def test_no_launchpad_ppa() -> None:
    offenders = [
        i
        for i in _instructions()
        if "add-apt-repository" in i or "deadsnakes" in i or "launchpad" in i
    ]
    assert not offenders, (
        "docker/runners/Dockerfile installs from a Launchpad PPA again; its 503s "
        f"red every PR that builds the image (OMN-17427): {offenders!r}"
    )


def test_python_comes_from_uv_after_uv_is_installed() -> None:
    uv_install = _index("/releases/download/${UV_VERSION}/uv-")
    python_install = _index("uv python install 3.12")
    assert uv_install < python_install, (
        "the Python step runs `uv python install`, so uv must be installed first"
    )


def test_python_entry_points_resolve_to_the_managed_interpreter() -> None:
    step = _instructions()[_index("uv python install 3.12")]
    assert "UV_PYTHON_INSTALL_DIR=/opt/uv-python" in step
    assert "uv python find --managed-python 3.12" in step
    for link in (
        "/usr/local/bin/python3.12",
        "/usr/local/bin/python3",
        "/usr/local/bin/python",
        "/usr/local/bin/pip3",
        "/usr/local/bin/pip",
    ):
        assert link in step, f"the Python step no longer provides {link}"
    assert "sys.version_info[:2] == (3, 12)" in step, (
        "the step must fail the build when the interpreter is not 3.12"
    )
