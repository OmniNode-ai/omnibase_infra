# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""YAML-based agent registry.

Local-first agent identity store. Each agent is a YAML file in the state
directory. Survives without Postgres. Async sync to Postgres is a future
concern (cross-machine portability).

File layout:
    {state_dir}/agents/CAIA.yaml
    {state_dir}/agents/SENTINEL.yaml
    {state_dir}/agents/ARCHIVIST.yaml
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import TypeAdapter

from omnibase_infra.models.agent_identity.enum_agent_status import EnumAgentStatus
from omnibase_infra.models.agent_identity.model_agent_binding import ModelAgentBinding
from omnibase_infra.models.agent_identity.model_agent_entity import ModelAgentEntity

_AGENT_ADAPTER = TypeAdapter(ModelAgentEntity)


class AgentRegistry:
    """Local-first YAML agent registry."""

    def __init__(self, state_dir: Path | None = None) -> None:
        self._state_dir = state_dir or Path.home() / ".onex_state"
        self._agents_dir = self._state_dir / "agents"
        self._agents_dir.mkdir(parents=True, exist_ok=True)

    def _agent_path(self, agent_id: str) -> Path:
        return self._agents_dir / f"{agent_id}.yaml"

    def register(self, agent: ModelAgentEntity) -> None:
        """Register a new agent or overwrite existing.

        Uses atomic write-tmp-rename to prevent corruption from partial writes
        or concurrent mutations. Not fully concurrent-safe (no file locking),
        but prevents the worst case of truncated YAML.
        """
        data = agent.model_dump(mode="json")
        path = self._agent_path(agent.agent_id)
        tmp_path = path.with_suffix(".yaml.tmp")
        tmp_path.write_text(yaml.dump(data, default_flow_style=False, sort_keys=False))
        tmp_path.replace(path)

    def get(self, agent_id: str) -> ModelAgentEntity | None:
        """Get agent by ID, or None if not registered."""
        path = self._agent_path(agent_id)
        if not path.exists():
            return None
        data = yaml.safe_load(path.read_text())
        return _AGENT_ADAPTER.validate_python(data)

    def list_agents(self) -> list[ModelAgentEntity]:
        """List all registered agents."""
        agents: list[ModelAgentEntity] = []
        for path in sorted(self._agents_dir.glob("*.yaml")):
            data = yaml.safe_load(path.read_text())
            if data:
                agents.append(_AGENT_ADAPTER.validate_python(data))
        return agents

    def bind(self, agent_id: str, binding: ModelAgentBinding) -> None:
        """Bind an agent to a terminal session."""
        agent = self.get(agent_id)
        if agent is None:
            msg = f"Agent '{agent_id}' not registered"
            raise ValueError(msg)
        updated = agent.model_copy(
            update={
                "current_binding": binding,
                "status": EnumAgentStatus.ACTIVE,
            }
        )
        self.register(updated)

    def unbind(self, agent_id: str) -> None:
        """Unbind an agent from its current terminal session."""
        agent = self.get(agent_id)
        if agent is None:
            msg = f"Agent '{agent_id}' not registered"
            raise ValueError(msg)
        updated = agent.model_copy(
            update={
                "current_binding": None,
                "status": EnumAgentStatus.IDLE,
            }
        )
        self.register(updated)
