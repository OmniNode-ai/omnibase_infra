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
    # OMN-13131: pin advanced to the renderer-capability runtime core release
    # consumed by the W5 runtime image.
    expected_core = "dc69b726248fd0212a3abc0a5a587bd974278739"

    assert expected_core in pyproject
    assert expected_core in uv_lock


def test_release_backmerge_preserves_runner_identity_lock() -> None:
    """The runner image identity matches the dev runtime deploy proof."""

    lock = json.loads(
        (ROOT / "docker/runners/runner-image.lock.json").read_text(encoding="utf-8")
    )

    # OMN-13247: identity regenerated because adding the 4 coding-agent onex.nodes
    # entry-points to pyproject.toml changed the runner image binding inputs
    # (pyproject.toml feeds both the manifest_digest and the shared_env_digest).
    assert lock["identity_digest"] == "e9825ff53fdc27ad57ad61cc133ea830"
    assert lock["shared_env_digest"] == "8cf7e1e1a85a1d2433a9a355"
