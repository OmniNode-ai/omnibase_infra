# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Integration test for agent identity lifecycle.

Tests: create agent -> register -> bind -> project session -> query snapshot -> unbind.
"""

from datetime import UTC, datetime
from pathlib import Path

import pytest

from omnibase_infra.models.agent_identity import (
    ModelAgentBinding,
    ModelAgentEntity,
)
from omnibase_infra.services.agent_registry.registry import AgentRegistry
from omnibase_infra.services.session_projector.projector import (
    project_session_ended,
    project_session_started,
    project_tool_executed,
)


@pytest.mark.integration
class TestAgentIdentityRoundTrip:
    def test_full_lifecycle(self, tmp_path: Path) -> None:
        # 1. Create and register
        registry = AgentRegistry(state_dir=tmp_path)
        caia = ModelAgentEntity(
            agent_id="CAIA",
            display_name="CAIA — Primary Development Agent",
            created_at=datetime.now(tz=UTC),
        )
        registry.register(caia)

        # 2. Bind to terminal
        binding = ModelAgentBinding(
            terminal_id="terminal-mac-3",
            session_id="sess-test-001",
            machine="test-machine",
            bound_at=datetime.now(tz=UTC),
        )
        registry.bind("CAIA", binding)
        bound_agent = registry.get("CAIA")
        assert bound_agent is not None
        assert bound_agent.current_binding is not None

        # 3. Project session events
        snapshot = project_session_started(
            agent_id="CAIA",
            session_id="sess-test-001",
            terminal_id="terminal-mac-3",
            machine="test-machine",
            working_directory="/Volumes/PRO-G40/Code/omni_worktrees/OMN-7241/omnibase_infra",
            git_branch="jonah/omn-7241-learning-models",
            started_at=datetime.now(tz=UTC),
        )
        assert snapshot["agent_id"] == "CAIA"
        assert snapshot["current_ticket"] == "OMN-7241"

        # 4. Project tool events
        snapshot = project_tool_executed(
            snapshot=snapshot,
            tool_name="Edit",
            success=True,
            summary="Edited src/models/agent_learning.py:42",
        )
        assert len(snapshot["files_touched"]) == 1

        snapshot = project_tool_executed(
            snapshot=snapshot,
            tool_name="Bash",
            success=False,
            summary="ImportError: cannot import 'foo'",
        )
        assert len(snapshot["errors_hit"]) == 1

        # 5. End session
        snapshot = project_session_ended(
            snapshot=snapshot,
            outcome="success",
            ended_at=datetime.now(tz=UTC),
        )
        assert snapshot["session_outcome"] == "success"

        # 6. Unbind
        registry.unbind("CAIA")
        unbound_agent = registry.get("CAIA")
        assert unbound_agent is not None
        assert unbound_agent.current_binding is None
        assert unbound_agent.status.value == "idle"

    def test_multi_agent_concurrent_sessions(self, tmp_path: Path) -> None:
        """Test that multiple agents can have independent sessions."""
        registry = AgentRegistry(state_dir=tmp_path)
        now = datetime.now(tz=UTC)

        for name in ["CAIA", "SENTINEL"]:
            registry.register(
                ModelAgentEntity(
                    agent_id=name,
                    display_name=f"{name} agent",
                    created_at=now,
                )
            )
            registry.bind(
                name,
                ModelAgentBinding(
                    terminal_id=f"terminal-{name.lower()}",
                    session_id=f"sess-{name.lower()}",
                    machine="test-machine",
                    bound_at=now,
                ),
            )

        caia = registry.get("CAIA")
        sentinel = registry.get("SENTINEL")
        assert caia is not None and sentinel is not None
        assert caia.current_binding is not None
        assert sentinel.current_binding is not None
        assert caia.current_binding.terminal_id != sentinel.current_binding.terminal_id

        # Unbinding one doesn't affect the other
        registry.unbind("CAIA")
        assert registry.get("CAIA").current_binding is None
        assert registry.get("SENTINEL").current_binding is not None
