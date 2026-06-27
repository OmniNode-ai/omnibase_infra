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
    # OMN-12546 S-1b: pin advanced to core dev HEAD 48cf8b0 (successor of
    # 287511f20 from OMN-13507) so infra imports the promoted rich dispatch
    # model types from the proven core commit.
    expected_core = "48cf8b0be1c1f6d04d1e92c7f18ceb58c812471d"

    assert expected_core in pyproject
    assert expected_core in uv_lock


def test_release_backmerge_preserves_runner_identity_lock() -> None:
    """The runner image identity matches the dev runtime deploy proof."""

    lock = json.loads(
        (ROOT / "docker/runners/runner-image.lock.json").read_text(encoding="utf-8")
    )

    # OMN-13664: identity regenerated after the uv dependency refresh updated
    # the runtime shared-env inputs.
    assert lock["identity_digest"] == "917d844856cec327a8e5c4220348526a"
    assert lock["shared_env_digest"] == "366971b13c707215ad917d59"
