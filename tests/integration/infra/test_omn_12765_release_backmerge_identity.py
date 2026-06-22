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
    # OMN-13507: pin advanced to core dev HEAD 1b22fd18b (a strict descendant of
    # the Phase-1b SHA db7f341) so the runtime image bundles a
    # ModelTaskDelegatedEvent carrying tokens_input/tokens_output (#1300/OMN-13408)
    # + context_pack_hash (#1299/OMN-13407); terminal-emit no longer raises
    # ValidationError under frozen+extra=forbid.
    expected_core = "8ecb7efc17721dda2ce468b2e5051816ff8e89bc"

    assert expected_core in pyproject
    assert expected_core in uv_lock


def test_release_backmerge_preserves_runner_identity_lock() -> None:
    """The runner image identity matches the dev runtime deploy proof."""

    lock = json.loads(
        (ROOT / "docker/runners/runner-image.lock.json").read_text(encoding="utf-8")
    )

    # OMN-13507: identity regenerated after advancing the core pin to dev HEAD
    # 1b22fd18b (scripts/ci/runner_image_identity.py --mode generate).
    assert lock["identity_digest"] == "bb8ae038ac2051ed6dba61d5f0302540"
    assert lock["shared_env_digest"] == "2c4bc0ccf99e2f9dab9ab79a"
