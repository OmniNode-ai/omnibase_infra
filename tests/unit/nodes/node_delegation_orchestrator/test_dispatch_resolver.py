# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import pytest

from omnibase_core.enums.enum_invocation_kind import EnumInvocationKind
from omnibase_core.topics import TopicBase
from omnibase_infra.nodes.node_delegation_orchestrator.dispatch_resolver import (
    resolve_effect_topic,
)


@pytest.mark.unit
def test_agent_routes_to_remote_agent_invoke() -> None:
    assert (
        resolve_effect_topic(EnumInvocationKind.AGENT) is TopicBase.REMOTE_AGENT_INVOKE
    )


@pytest.mark.unit
def test_model_raises_not_implemented() -> None:
    with pytest.raises(NotImplementedError, match="MODEL deferred to Part 2"):
        resolve_effect_topic(EnumInvocationKind.MODEL)
