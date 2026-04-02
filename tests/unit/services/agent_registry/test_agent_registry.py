# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Tests for YAML-based agent registry."""

from datetime import UTC, datetime, timezone
from pathlib import Path

import pytest

from omnibase_infra.models.agent_identity import (
    ModelAgentBinding,
    ModelAgentEntity,
)
from omnibase_infra.services.agent_registry.registry import AgentRegistry


@pytest.mark.unit
class TestAgentRegistry:
    def test_register_and_get(self, tmp_path: Path) -> None:
        registry = AgentRegistry(state_dir=tmp_path)
        agent = ModelAgentEntity(
            agent_id="CAIA",
            display_name="CAIA — Primary Development Agent",
            created_at=datetime.now(tz=UTC),
        )
        registry.register(agent)
        retrieved = registry.get("CAIA")
        assert retrieved is not None
        assert retrieved.agent_id == "CAIA"

    def test_get_nonexistent(self, tmp_path: Path) -> None:
        registry = AgentRegistry(state_dir=tmp_path)
        assert registry.get("NONEXISTENT") is None

    def test_list_agents(self, tmp_path: Path) -> None:
        registry = AgentRegistry(state_dir=tmp_path)
        for name in ["CAIA", "SENTINEL", "ARCHIVIST"]:
            registry.register(
                ModelAgentEntity(
                    agent_id=name,
                    display_name=f"{name} agent",
                    created_at=datetime.now(tz=UTC),
                )
            )
        agents = registry.list_agents()
        assert len(agents) == 3
        assert {a.agent_id for a in agents} == {"CAIA", "SENTINEL", "ARCHIVIST"}

    def test_update_binding(self, tmp_path: Path) -> None:
        registry = AgentRegistry(state_dir=tmp_path)
        registry.register(
            ModelAgentEntity(
                agent_id="CAIA",
                display_name="CAIA",
                created_at=datetime.now(tz=UTC),
            )
        )
        binding = ModelAgentBinding(
            terminal_id="terminal-mac-3",
            session_id="sess-abc",
            machine="jonahs-macbook",
            bound_at=datetime.now(tz=UTC),
        )
        registry.bind("CAIA", binding)
        agent = registry.get("CAIA")
        assert agent is not None
        assert agent.current_binding is not None
        assert agent.current_binding.terminal_id == "terminal-mac-3"

    def test_unbind(self, tmp_path: Path) -> None:
        registry = AgentRegistry(state_dir=tmp_path)
        registry.register(
            ModelAgentEntity(
                agent_id="CAIA",
                display_name="CAIA",
                created_at=datetime.now(tz=UTC),
            )
        )
        binding = ModelAgentBinding(
            terminal_id="terminal-mac-3",
            session_id="sess-abc",
            machine="jonahs-macbook",
            bound_at=datetime.now(tz=UTC),
        )
        registry.bind("CAIA", binding)
        registry.unbind("CAIA")
        agent = registry.get("CAIA")
        assert agent is not None
        assert agent.current_binding is None

    def test_persistence_across_instances(self, tmp_path: Path) -> None:
        registry1 = AgentRegistry(state_dir=tmp_path)
        registry1.register(
            ModelAgentEntity(
                agent_id="CAIA",
                display_name="CAIA",
                created_at=datetime.now(tz=UTC),
            )
        )
        registry2 = AgentRegistry(state_dir=tmp_path)
        agent = registry2.get("CAIA")
        assert agent is not None
        assert agent.agent_id == "CAIA"

    def test_yaml_file_created(self, tmp_path: Path) -> None:
        registry = AgentRegistry(state_dir=tmp_path)
        registry.register(
            ModelAgentEntity(
                agent_id="CAIA",
                display_name="CAIA",
                created_at=datetime.now(tz=UTC),
            )
        )
        yaml_path = tmp_path / "agents" / "CAIA.yaml"
        assert yaml_path.exists()
