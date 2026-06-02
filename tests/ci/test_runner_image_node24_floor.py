# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Rollout gate: the runner image RUNNER_VERSION must be node24-capable and must
not downgrade the live fleet (OMN-12585).

Background. The OMN-12567 versioned runner image pinned
``ARG RUNNER_VERSION=2.323.0`` while the live fleet ran ``2.334.0``. Rolling that
image to the fleet *downgraded* the GitHub Actions runner, and ``2.323.0`` cannot
execute ``node24`` actions (``actions/checkout@v6``, used 73x across workflows).
It fails at "Set up job" with ``'using: node24' is not supported``. node24
execution support landed in actions/runner ``2.327.0`` (actions/runner#3940).

The OMN-12584 build-smoke proves the image *builds*, but it runs on whatever
runner picks up the PR (the OLD fleet), so it never exercises the NEW image's
runner version against a node24 action. No gate asserted the baked
``RUNNER_VERSION`` was node24-capable or not a downgrade — this regression merged
green. This test is that gate, enforced as a unit check on every PR (and the
build-smoke workflow / pre-commit run it before the runner image can ship).

Two invariants, both load-bearing:

* **node24 capability** — ``RUNNER_VERSION >= 2.327.0`` so the baked runner can
  execute ``node24`` actions.
* **non-downgrade vs the live fleet** — ``RUNNER_VERSION >= 2.334.0`` so rolling
  the image to the fleet never rolls the runner backward.

The Dockerfile ``ARG``, the SHA256 comment, and the lock ``runner_version`` must
all agree, so a partial bump (version updated but checksum stale, or lock not
regenerated) also fails here.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]
RUNNER_DOCKERFILE = REPO_ROOT / "docker" / "runners" / "Dockerfile"
LOCK_FILE = REPO_ROOT / "docker" / "runners" / "runner-image.lock.json"

# node24 *execution* support landed in actions/runner 2.327.0 (actions/runner
# #3940). Below this, `actions/checkout@v6` (using: node24) fails at "Set up job".
NODE24_FLOOR = (2, 327, 0)

# The live self-hosted fleet runs 2.334.0 (verified 2026-06-02, OMN-12585). The
# baked image must never downgrade the fleet's runner. Raise this floor whenever
# the fleet is rolled forward to a newer runner.
LIVE_FLEET_FLOOR = (2, 334, 0)

_VERSION_RE = re.compile(r"^(\d+)\.(\d+)\.(\d+)$")


def _parse(version: str) -> tuple[int, int, int]:
    """Parse a dotted ``major.minor.patch`` runner version into a comparable tuple."""
    match = _VERSION_RE.match(version.strip())
    assert match is not None, f"unparseable runner version: {version!r}"
    return (int(match.group(1)), int(match.group(2)), int(match.group(3)))


def _fmt(version: tuple[int, int, int]) -> str:
    return ".".join(str(part) for part in version)


def _dockerfile_runner_version() -> str:
    source = RUNNER_DOCKERFILE.read_text(encoding="utf-8")
    match = re.search(r"^ARG RUNNER_VERSION=(\S+)\s*$", source, re.MULTILINE)
    assert match is not None, "ARG RUNNER_VERSION not found in runner Dockerfile"
    return match.group(1)


def _lock_runner_version() -> str:
    data = json.loads(LOCK_FILE.read_text(encoding="utf-8"))
    version = data.get("runner_version")
    assert isinstance(version, str) and version, "lock missing runner_version"
    return version


def test_dockerfile_runner_version_is_node24_capable() -> None:
    """The baked runner must be able to execute node24 actions (>= 2.327.0)."""
    version = _parse(_dockerfile_runner_version())
    assert version >= NODE24_FLOOR, (
        f"RUNNER_VERSION {_fmt(version)} predates node24 execution support "
        f"({_fmt(NODE24_FLOOR)}); actions/checkout@v6 (using: node24) would fail "
        "at 'Set up job' with \"'using: node24' is not supported\""
    )


def test_dockerfile_runner_version_does_not_downgrade_live_fleet() -> None:
    """The baked runner must not roll the live fleet backward (>= 2.334.0)."""
    version = _parse(_dockerfile_runner_version())
    assert version >= LIVE_FLEET_FLOOR, (
        f"RUNNER_VERSION {_fmt(version)} is a downgrade from the live fleet "
        f"({_fmt(LIVE_FLEET_FLOOR)}); rolling this image to the fleet would "
        "downgrade the runner — the exact OMN-12585 regression"
    )


def test_lock_runner_version_matches_dockerfile() -> None:
    """The lock's runner_version must track the Dockerfile ARG.

    The runner version is part of the bound image identity, so a mismatch means
    the lock was not regenerated after the bump — the baked binding would not
    describe the runner actually installed.
    """
    docker_version = _dockerfile_runner_version()
    lock_version = _lock_runner_version()
    assert docker_version == lock_version, (
        f"Dockerfile RUNNER_VERSION {docker_version!r} != lock runner_version "
        f"{lock_version!r}; run scripts/ci/runner_image_identity.py --mode generate"
    )


def test_lock_runner_version_is_node24_capable_and_not_a_downgrade() -> None:
    """The same floors apply to the lock-recorded runner version."""
    version = _parse(_lock_runner_version())
    assert version >= NODE24_FLOOR, (
        f"lock runner_version {_fmt(version)} predates node24 support "
        f"({_fmt(NODE24_FLOOR)})"
    )
    assert version >= LIVE_FLEET_FLOOR, (
        f"lock runner_version {_fmt(version)} downgrades the live fleet "
        f"({_fmt(LIVE_FLEET_FLOOR)})"
    )


def test_runner_sha256_comment_tracks_runner_version() -> None:
    """The SHA256 comment must name the same version as the ARG.

    The Dockerfile verifies the downloaded runner tarball against a pinned
    RUNNER_SHA256. The comment naming the version that checksum belongs to must
    track the ARG, or a version bump with a stale checksum (wrong file) slips
    through: the build would fail the sha256 check, but only at image-build time.
    This static check catches the inconsistency earlier and unambiguously.
    """
    source = RUNNER_DOCKERFILE.read_text(encoding="utf-8")
    version = _dockerfile_runner_version()
    assert f"actions-runner-linux-x64-{version}.tar.gz" in source, (
        f"runner Dockerfile SHA256 comment does not reference "
        f"actions-runner-linux-x64-{version}.tar.gz; update the checksum + "
        "comment in lockstep with RUNNER_VERSION"
    )
    # A non-empty pinned checksum must be present.
    assert re.search(r"^ENV RUNNER_SHA256=[0-9a-f]{64}\s*$", source, re.MULTILINE), (
        "runner Dockerfile must pin a 64-hex-char RUNNER_SHA256"
    )
