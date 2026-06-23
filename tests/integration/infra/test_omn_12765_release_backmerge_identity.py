# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Release backmerge identity checks for OMN-12765."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def test_release_backmerge_preserves_proven_runtime_core_pin() -> None:
    """The main-lane backmerge must not move off the dev runtime proof inputs."""

    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    uv_lock = (ROOT / "uv.lock").read_text(encoding="utf-8")
    # OMN-13445: pin advanced to the Phase-1b core SHA (OMN-13444 / core #1296)
    # that relocated RuntimeLocal + adapter + the 5 local-runtime protocols into
    # omnibase_core.runtime / omnibase_core.protocols.runtime; infra re-points
    # onto the core-resident runtime here.
    expected_core = "db7f341353e598df84629b1c32514c78d554ef57"

    assert expected_core in pyproject
    assert expected_core in uv_lock


def test_release_backmerge_preserves_runner_identity_lock() -> None:
    """The runner image identity matches the dev runtime deploy proof."""

    lock = json.loads(
        (ROOT / "docker/runners/runner-image.lock.json").read_text(encoding="utf-8")
    )

    # OMN-13486: identity regenerated for the current dev/runtime proof inputs.
    assert lock["identity_digest"] == "da0dc5140fae53e84399503a0e5f8eb3"
    assert lock["shared_env_digest"] == "8238ed24bfdfc3816dde8016"
