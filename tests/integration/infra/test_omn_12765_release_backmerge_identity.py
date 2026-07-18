# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Release backmerge identity checks for OMN-12765."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def test_release_backmerge_preserves_proven_runtime_core_pin() -> None:
    """The main-lane backmerge must carry the proven PyPI core/spi releases.

    OMN-13762 R3: infra is cut off the unreleased git-rev pins (core dev HEAD
    48cf8b0, spi 3c99ed4) onto the published PyPI releases so main can build a
    clean, reproducible runtime image from immutable artifacts. The proven
    runtime inputs are now the released versions, not git revs.

    OMN-14600 refresh: the proven runtime advances to core 0.46.8 while keeping
    the PyPI-sourced reproducible lock.
    """

    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    uv_lock = (ROOT / "uv.lock").read_text(encoding="utf-8")

    # The proven runtime now pins the published PyPI releases (exact versions).
    assert "omnibase-core==0.46.8" in pyproject
    assert "omnibase-spi==0.23.1" in pyproject

    # The retired git-rev overrides must be gone from both manifest and lock:
    # the OMN-13762 core rev and the OMN-12549 seam core/spi revs.
    for retired_rev in (
        "48cf8b0be1c1f6d04d1e92c7f18ceb58c812471d",
        "2a07385dec0ff06903f62572c546ec201f964aaf",
        "cdfe1a470e96cbe8414ba6b08bbc99a452f09018",
    ):
        assert retired_rev not in pyproject
        assert retired_rev not in uv_lock


def test_release_backmerge_preserves_runner_identity_lock() -> None:
    """The runner image identity matches the dev runtime deploy proof."""

    lock = json.loads(
        (ROOT / "docker/runners/runner-image.lock.json").read_text(encoding="utf-8")
    )

    # OMN-12765 / #2306 (dependabot uvicorn <0.51.0 -> <0.52.0 bound bump): the
    # runner identity is a binding, not a label (OMN-12567). It folds in the full
    # dependency-manifest bytes (pyproject.toml + uv.lock), so a dependency-range
    # update legitimately rebinds the runner lock. Regenerated with
    # scripts/ci/runner_image_identity.py --mode generate and build-proven by the
    # runner-image-build-smoke gate (baked image label == lock identity_digest).
    # Precedent: #2227 (prior uvicorn bump) and #2228 (fastapi bump) rebound the
    # same lock and updated this anchor in lockstep.
    assert lock["identity_digest"] == "65f4c724695e15a5b6dd10c5eb296e89"
    assert lock["shared_env_digest"] == "0a37bc0a00d924c761a88251"
