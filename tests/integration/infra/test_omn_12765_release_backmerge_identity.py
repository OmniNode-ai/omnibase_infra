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
    expected_core = "2defabef4d1a7558f59f9d2a3f7f42831b00fb51"

    assert expected_core in pyproject
    assert expected_core in uv_lock


def test_release_backmerge_preserves_runner_identity_lock() -> None:
    """The runner image identity matches the dev runtime deploy proof."""

    lock = json.loads(
        (ROOT / "docker/runners/runner-image.lock.json").read_text(encoding="utf-8")
    )

    assert lock["identity_digest"] == "3937d0b4aff241981c88f7f01e13bc22"
    assert lock["shared_env_digest"] == "99eb863ed6e8573cdffac599"
