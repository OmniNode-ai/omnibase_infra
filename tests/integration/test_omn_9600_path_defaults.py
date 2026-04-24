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

import pytest

from omnibase_infra.adapters.project_tracker.local_stub_project_tracker import (
    LocalStubProjectTracker,
)
from omnibase_infra.services.agent_registry.registry import AgentRegistry


@pytest.mark.integration
def test_local_stub_project_tracker_default_is_workspace_local(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Default state_root must resolve to workspace-local .onex_state, not Path.home()."""
    monkeypatch.chdir(tmp_path)
    tracker = LocalStubProjectTracker()
    home = Path.home()
    assert not str(tracker._state_root).startswith(str(home)), (
        f"LocalStubProjectTracker._state_root ({tracker._state_root}) must not be "
        f"under Path.home() ({home}) — defaults must be workspace-local per OMN-9600"
    )
    assert ".onex_state" in str(tracker._state_root), (
        "LocalStubProjectTracker._state_root must contain .onex_state"
    )


@pytest.mark.integration
def test_agent_registry_default_is_workspace_local(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Default state_dir must resolve to workspace-local .onex_state, not Path.home()."""
    monkeypatch.chdir(tmp_path)
    registry = AgentRegistry()
    home = Path.home()
    assert not str(registry._state_dir).startswith(str(home)), (
        f"AgentRegistry._state_dir ({registry._state_dir}) must not be "
        f"under Path.home() ({home}) — defaults must be workspace-local per OMN-9600"
    )
    assert ".onex_state" in str(registry._state_dir), (
        "AgentRegistry._state_dir must contain .onex_state"
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
