# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The runner image must ship the IANA timezone database (OMN-19998).

The ubuntu:22.04 base carries no ``/usr/share/zoneinfo``. On an image without
the ``tzdata`` package, ``zoneinfo.ZoneInfo("America/New_York")`` raises
``ZoneInfoNotFoundError``, and a merge group for a private repository failed on
exactly that on two runners. The fleet has no passwordless package manager, so a
job cannot install the package itself; it has to be in the image.

Image v12 (runner 2.338.2) was built without tzdata, so the image version moves
to 13 with the package. The lock must not fall back below that, or the live
fleet could be rolled to an image that lacks the database.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]
RUNNER_DOCKERFILE = REPO_ROOT / "docker" / "runners" / "Dockerfile"
RUNNER_IMAGE_LOCK = REPO_ROOT / "docker" / "runners" / "runner-image.lock.json"

TZDATA_BAKED_IMAGE_VERSION = 13


def _apt_installed_packages(source: str) -> set[str]:
    """Every package name an ``apt-get install`` command in the Dockerfile names.

    Comment lines are dropped so prose naming a package cannot stand in for an
    install, and line continuations are joined so a split list is read whole.
    """
    code = "\n".join(
        line for line in source.splitlines() if not line.lstrip().startswith("#")
    )
    joined = code.replace("\\\n", " ")
    packages: set[str] = set()
    for command in re.findall(r"apt-get install([^\n&;|]*)", joined):
        for token in command.split():
            if re.fullmatch(r"[a-z0-9][a-z0-9.+-]+", token):
                packages.add(token)
    return packages


def test_dockerfile_installs_tzdata() -> None:
    installed = _apt_installed_packages(RUNNER_DOCKERFILE.read_text(encoding="utf-8"))
    assert "tzdata" in installed, (
        "runner Dockerfile must apt-install tzdata: without it "
        'zoneinfo.ZoneInfo("America/New_York") raises ZoneInfoNotFoundError on '
        "the fleet and a job there cannot install it itself (OMN-19998)"
    )


def test_tzdata_install_is_in_the_toolchain_stage_the_runner_inherits() -> None:
    source = RUNNER_DOCKERFILE.read_text(encoding="utf-8")
    runner_stage = source.index("FROM toolchain AS runner")
    toolchain_only = source[:runner_stage]
    assert "tzdata" in _apt_installed_packages(toolchain_only), (
        "tzdata must be installed before the runner stage, which inherits the "
        "toolchain stage (OMN-19998)"
    )


def test_lock_image_version_carries_tzdata() -> None:
    lock = json.loads(RUNNER_IMAGE_LOCK.read_text(encoding="utf-8"))
    assert lock["image_version"] >= TZDATA_BAKED_IMAGE_VERSION, (
        "tzdata is baked from image v13; a lock below it describes an image "
        "without the timezone database (OMN-19998)"
    )
