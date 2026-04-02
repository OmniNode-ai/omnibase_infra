# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Tests for agent identity models."""

from datetime import UTC, datetime, timedelta, timezone

import pytest

from omnibase_infra.models.agent_identity.model_agent_binding import (
    ModelAgentBinding,
)
from omnibase_infra.models.agent_identity.model_agent_entity import (
    EnumAgentStatus,
    ModelAgentEntity,
)


@pytest.mark.unit
class TestModelAgentEntity:
    def test_create_caia(self) -> None:
        agent = ModelAgentEntity(
            agent_id="CAIA",
            display_name="CAIA — Primary Development Agent",
            created_at=datetime.now(tz=UTC),
        )
        assert agent.agent_id == "CAIA"
        assert agent.status == EnumAgentStatus.IDLE
        assert agent.current_binding is None
        assert agent.active_tickets == ()

    def test_create_with_binding(self) -> None:
        binding = ModelAgentBinding(
            terminal_id="terminal-mac-3",
            session_id="sess-abc123",
            machine="jonahs-macbook",
            bound_at=datetime.now(tz=UTC),
        )
        agent = ModelAgentEntity(
            agent_id="SENTINEL",
            display_name="SENTINEL — CI/Review Agent",
            created_at=datetime.now(tz=UTC),
            current_binding=binding,
            status=EnumAgentStatus.ACTIVE,
            active_tickets=("OMN-7241", "OMN-7249"),
        )
        assert agent.current_binding is not None
        assert agent.current_binding.terminal_id == "terminal-mac-3"
        assert len(agent.active_tickets) == 2

    def test_frozen(self) -> None:
        agent = ModelAgentEntity(
            agent_id="TEST",
            display_name="Test Agent",
            created_at=datetime.now(tz=UTC),
        )
        with pytest.raises(Exception):
            agent.agent_id = "OTHER"  # type: ignore[misc]

    def test_agent_id_constraints(self) -> None:
        with pytest.raises(Exception):
            ModelAgentEntity(
                agent_id="",
                display_name="Bad",
                created_at=datetime.now(tz=UTC),
            )

    def test_updated_at_and_revision(self) -> None:
        agent = ModelAgentEntity(
            agent_id="CAIA",
            display_name="CAIA",
            created_at=datetime.now(tz=UTC),
            updated_at=datetime.now(tz=UTC),
            revision=3,
        )
        assert agent.revision == 3
        assert agent.updated_at is not None


@pytest.mark.unit
class TestModelAgentBinding:
    def test_create(self) -> None:
        binding = ModelAgentBinding(
            terminal_id="terminal-mac-1",
            session_id="sess-xyz",
            machine="jonahs-macbook",
            bound_at=datetime.now(tz=UTC),
        )
        assert binding.terminal_id == "terminal-mac-1"

    def test_stale_check(self) -> None:
        old = ModelAgentBinding(
            terminal_id="terminal-mac-1",
            session_id="sess-old",
            machine="jonahs-macbook",
            bound_at=datetime.now(tz=UTC) - timedelta(hours=2),
        )
        assert old.bound_at < datetime.now(tz=UTC) - timedelta(hours=1)
