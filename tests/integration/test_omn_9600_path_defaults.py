# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Integration proof for OMN-9600 workspace-local .onex_state defaults.

OMN-9600 fixes path defaults in AgentRegistry and LocalStubProjectTracker:
both previously anchored to ``Path.home() / ".onex_state"`` which breaks
cross-machine portability and conflicts with the documented workspace-local
.onex_state convention.

After the fix:
- ``AgentRegistry()`` defaults to ``Path(".onex_state")`` (workspace-local)
- ``LocalStubProjectTracker()`` defaults to ``Path(".onex_state") / "local-tracker"``

This test asserts both defaults do NOT reference the user home directory.

Ticket: OMN-9600
Integration Test Coverage gate: OMN-7005 (hard gate since 2026-04-13).
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from omnibase_infra.adapters.project_tracker.local_stub_project_tracker import (
    LocalStubProjectTracker,
)
from omnibase_infra.services.agent_registry.registry import AgentRegistry


@pytest.mark.integration
def test_local_stub_project_tracker_default_is_workspace_local() -> None:
    """Default state_root must not reference Path.home()."""
    tracker = LocalStubProjectTracker()

    assert Path.home() not in tracker._state_root.parents
    assert tracker._state_root == Path(".onex_state") / "local-tracker"
    assert ".onex_state" in tracker._state_root.parts, (
        "LocalStubProjectTracker.__init__ must not reference Path.home() — "
        "defaults must be workspace-local per OMN-9600"
    )


@pytest.mark.integration
def test_agent_registry_default_is_workspace_local() -> None:
    """Default state_dir must not reference Path.home()."""
    with patch.object(Path, "mkdir") as mkdir:
        registry = AgentRegistry()

    mkdir.assert_called_once_with(parents=True, exist_ok=True)
    assert Path.home() not in registry._state_dir.parents
    assert registry._state_dir == Path(".onex_state")
    assert ".onex_state" in registry._state_dir.parts, (
        "AgentRegistry.__init__ must not reference Path.home() — "
        "defaults must be workspace-local per OMN-9600"
    )


@pytest.mark.integration
def test_local_stub_project_tracker_state_root_under_workspace(
    tmp_path: Path,
) -> None:
    """When instantiated with a provided path, state_root resolves correctly."""
    tracker = LocalStubProjectTracker(state_root=tmp_path)
    assert tracker._state_root == Path(tmp_path)


@pytest.mark.integration
def test_agent_registry_state_dir_under_workspace(
    tmp_path: Path,
) -> None:
    """When instantiated with a provided path, state_dir resolves correctly."""
    registry = AgentRegistry(state_dir=tmp_path)
    assert registry._state_dir == Path(tmp_path)
