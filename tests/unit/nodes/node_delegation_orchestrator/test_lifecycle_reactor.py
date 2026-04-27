# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import pytest

from omnibase_core.enums.enum_agent_task_lifecycle_type import (
    EnumAgentTaskLifecycleType,
)
from omnibase_infra.nodes.node_delegation_orchestrator.enums import (
    EnumDelegationState,
)
from omnibase_infra.nodes.node_delegation_orchestrator.lifecycle_reactor import (
    next_state_from_lifecycle,
)


@pytest.mark.unit
def test_submitted_advances_to_executing() -> None:
    assert (
        next_state_from_lifecycle(EnumAgentTaskLifecycleType.SUBMITTED)
        is EnumDelegationState.EXECUTING
    )


@pytest.mark.unit
def test_completed_terminal() -> None:
    assert (
        next_state_from_lifecycle(EnumAgentTaskLifecycleType.COMPLETED)
        is EnumDelegationState.COMPLETED
    )


@pytest.mark.unit
def test_failed_terminal() -> None:
    for lifecycle in (
        EnumAgentTaskLifecycleType.FAILED,
        EnumAgentTaskLifecycleType.TIMED_OUT,
        EnumAgentTaskLifecycleType.CANCELED,
    ):
        assert next_state_from_lifecycle(lifecycle) is EnumDelegationState.FAILED
