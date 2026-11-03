# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The pre-merge pool driver's contract overlay for tests (OMN-22153).

``prepr_runtime_pool.pool_setting`` reads its numbers and command prefixes from
``config/onex/overlays/prepr_runtime_pool/overlay.yaml`` under the workspace
config root. A test module that drives the pool imports ``pool_overlay_env`` so
the root names the test double beside the pool fixture.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

POOL_CONFIG_ROOT = (
    Path(__file__).resolve().parents[1]
    / "unit"
    / "scripts"
    / "fixtures"
    / "pool_config_root"
)


@pytest.fixture(autouse=True)
def pool_overlay_env(
    monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest
) -> Iterator[Path]:
    # a lab_overlay test reads the lab's own files from the root its job fetched
    if request.node.get_closest_marker("lab_overlay") is None:
        monkeypatch.setenv("ONEX_WORKSPACE_CONFIG_ROOT", str(POOL_CONFIG_ROOT))
    return POOL_CONFIG_ROOT
