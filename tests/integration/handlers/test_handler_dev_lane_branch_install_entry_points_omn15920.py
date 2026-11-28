# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""A declared entry point survives a refresh and the runtime's reader lists it (OMN-15920).

One store writes the record, a container recreate resets the installed dist-info, and
a second store over the same state directory writes the entry point back. The check
reads the dist-info with ``importlib.metadata``, the call ``read_package_inventory``
makes to find nodes.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from omnibase_infra.handlers.handler_dev_lane_branch_install import (
    DevLaneBranchInstallStore,
    HandlerDevLaneBranchInstall,
)
from tests.unit.handlers.test_handler_dev_lane_branch_install_entry_points_omn15920 import (
    EP,
    EXISTING,
    _declare,
    _Lane,
    _listed,
    _make_dist,
)

pytestmark = pytest.mark.integration


def test_a_recreated_container_gets_the_entry_point_back(tmp_path: Path) -> None:
    site = tmp_path / "site"
    ep_file = _make_dist(site, f"[onex.nodes]\n{EXISTING}\n")
    _declare(tmp_path, DevLaneBranchInstallStore(tmp_path / "state"), (EP,))
    now = datetime(2026, 11, 28, 5, 0, tzinfo=UTC)
    lane = _Lane(site)
    fresh = HandlerDevLaneBranchInstall(
        DevLaneBranchInstallStore(tmp_path / "state"),
        "omnibase-infra",
        run=lane,
        sleep=lambda _s: None,
        clock=lambda: now,
    )
    assert "node_mcp_server_effect" not in _listed(ep_file)
    assert fresh.apply().ok
    assert _listed(ep_file)["node_mcp_server_effect"] == EP.value
