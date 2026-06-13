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
    # OMN-13094: pin advanced to the merged ArtifactStore slice (OMN-13093) —
    # receipt mode consumes omnibase_core.artifacts + ModelSkillResult.
    expected_core = "ae8793bdad96c12a0b47e2f4d1d0f179618553b8"

    assert expected_core in pyproject
    assert expected_core in uv_lock


def test_release_backmerge_preserves_runner_identity_lock() -> None:
    """The runner image identity matches the dev runtime deploy proof."""

    lock = json.loads(
        (ROOT / "docker/runners/runner-image.lock.json").read_text(encoding="utf-8")
    )

    # OMN-13094: identity regenerated for the advanced core pin (both digests
    # bind pyproject.toml + uv.lock).
    assert lock["identity_digest"] == "9bce87358ac31006180ffdc8eaa1d370"
    assert lock["shared_env_digest"] == "0f0276f1dab9c86d39fef288"
