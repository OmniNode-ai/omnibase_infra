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
    # OMN-13507: pin advanced to core dev HEAD bd98c188 (a strict descendant of
    # the Phase-1b SHA db7f341) so the runtime image bundles a
    # ModelTaskDelegatedEvent carrying tokens_input/tokens_output (#1300/OMN-13408)
    # + context_pack_hash (#1299/OMN-13407); terminal-emit no longer raises
    # ValidationError under frozen+extra=forbid.
    expected_core = "bd98c188f4bf5d173b9054527d8459efe9e7bddc"

    assert expected_core in pyproject
    assert expected_core in uv_lock


def test_release_backmerge_preserves_runner_identity_lock() -> None:
    """The runner image identity matches the dev runtime deploy proof."""

    lock = json.loads(
        (ROOT / "docker/runners/runner-image.lock.json").read_text(encoding="utf-8")
    )

    # OMN-13507: identity regenerated after merging the current dev/runtime
    # proof inputs with the advanced omnibase-core pin.
    assert lock["identity_digest"] == "8a8d47cf703f5c590517ce203b2c3e83"
    assert lock["shared_env_digest"] == "0846c76fac6fae4673127b05"
