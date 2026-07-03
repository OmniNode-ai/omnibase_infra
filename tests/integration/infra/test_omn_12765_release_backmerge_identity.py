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
    """

    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    uv_lock = (ROOT / "uv.lock").read_text(encoding="utf-8")

    # The proven runtime now pins the published PyPI releases.
    assert "omnibase-core>=0.46.1,<0.47.0" in pyproject
    assert "omnibase-spi>=0.23.0,<0.24.0" in pyproject

    # The retired git-rev override must be gone from both manifest and lock.
    retired_core_rev = "48cf8b0be1c1f6d04d1e92c7f18ceb58c812471d"
    assert retired_core_rev not in pyproject
    assert retired_core_rev not in uv_lock


def test_release_backmerge_preserves_runner_identity_lock() -> None:
    """The runner image identity matches the dev runtime deploy proof."""

    lock = json.loads(
        (ROOT / "docker/runners/runner-image.lock.json").read_text(encoding="utf-8")
    )

    # OMN-13762 R3: identity regenerated after relocking onto the published
    # PyPI core 0.46.1 / spi 0.23.0 releases, which changed the runtime
    # dependency-manifest and shared-env inputs.
    assert lock["identity_digest"] == "585efe7a10449e68db1823034cad3814"
    assert lock["shared_env_digest"] == "34829d3bf76bf07612be9295"
