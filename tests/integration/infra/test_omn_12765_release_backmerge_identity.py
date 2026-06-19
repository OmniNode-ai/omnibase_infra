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
    # OMN-13290: pin advanced to the backend secret-discipline core validator
    # consumed by the infra gate.
    expected_core = "d6e9c3d845a9560f9ba203c92963970c0e325c75"

    assert expected_core in pyproject
    assert expected_core in uv_lock


def test_release_backmerge_preserves_runner_identity_lock() -> None:
    """The runner image identity matches the dev runtime deploy proof."""

    lock = json.loads(
        (ROOT / "docker/runners/runner-image.lock.json").read_text(encoding="utf-8")
    )

    # OMN-13290: identity regenerated after merging the backend-secret gate with
    # the coding-agent nodes and validator-requirements baseline changes.
    assert lock["identity_digest"] == "88303b6571205b4c2602f93a4912324d"
    assert lock["shared_env_digest"] == "c028f4f1d619fb0101204054"
