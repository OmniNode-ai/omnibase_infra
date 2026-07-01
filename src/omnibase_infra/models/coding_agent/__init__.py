# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Shared models + enums for the canonical coding-agent workflow (OMN-13247).

These are the typed I/O models for the four coding-agent nodes
(orchestrator / FSM reducer / invoke effect / workspace compute). They live in
``omnibase_infra`` because all four nodes co-located here import them and the
inference handler being relocated (Phase D) already lives in this repo.

PROMOTION NOTE (memory ``feedback_models_in_core_not_market``): if/when Phase E
(omnimarket delegation orchestrator) emits ``ModelCodingAgentInvokeCommand`` at
the code_generation ceiling, the command crosses a second repo boundary
(>=2 importers) and graduates to ``omnibase_core``. Until that second consumer
exists, it stays repo-private here. Do NOT bump the ``omnibase_compat`` pin to
host it — compat is staging for wire DTOs, not a home for business commands.
"""

from __future__ import annotations

from omnibase_infra.models.coding_agent.enum_agent_sandbox import EnumAgentSandbox
from omnibase_infra.models.coding_agent.enum_agent_status import EnumAgentStatus
from omnibase_infra.models.coding_agent.enum_cli_backend_status import (
    EnumCliBackendStatus,
)
from omnibase_infra.models.coding_agent.enum_coding_agent import EnumCodingAgent
from omnibase_infra.models.coding_agent.enum_coding_agent_event_kind import (
    EnumCodingAgentEventKind,
)
from omnibase_infra.models.coding_agent.enum_coding_agent_fsm_state import (
    TERMINAL_STATES,
    EnumCodingAgentFsmState,
)
from omnibase_infra.models.coding_agent.enum_coding_agent_intent_kind import (
    EnumCodingAgentIntentKind,
)
from omnibase_infra.models.coding_agent.model_agent_invocation import (
    ModelAgentInvocation,
)
from omnibase_infra.models.coding_agent.model_coding_agent_event import (
    ModelCodingAgentEvent,
)
from omnibase_infra.models.coding_agent.model_coding_agent_fsm_state import (
    ModelCodingAgentFsmState,
)
from omnibase_infra.models.coding_agent.model_coding_agent_intent import (
    ModelCodingAgentIntent,
)
from omnibase_infra.models.coding_agent.model_coding_agent_invoke_command import (
    ModelCodingAgentInvokeCommand,
)
from omnibase_infra.models.coding_agent.model_coding_agent_result import (
    ModelCodingAgentResult,
)
from omnibase_infra.models.coding_agent.model_coding_agent_trace_projection import (
    ModelCodingAgentTraceProjection,
)
from omnibase_infra.models.coding_agent.model_subprocess_invocation import (
    ModelSubprocessInvocation,
)
from omnibase_infra.models.coding_agent.model_subprocess_outcome import (
    ModelSubprocessOutcome,
)
from omnibase_infra.models.coding_agent.model_workspace_validate_command import (
    ModelWorkspaceValidateCommand,
)
from omnibase_infra.models.coding_agent.model_workspace_validate_result import (
    ModelWorkspaceValidateResult,
)

__all__: list[str] = [
    "TERMINAL_STATES",
    "EnumAgentSandbox",
    "EnumAgentStatus",
    "EnumCliBackendStatus",
    "EnumCodingAgent",
    "EnumCodingAgentEventKind",
    "EnumCodingAgentFsmState",
    "EnumCodingAgentIntentKind",
    "ModelCodingAgentEvent",
    "ModelCodingAgentFsmState",
    "ModelCodingAgentIntent",
    "ModelCodingAgentInvokeCommand",
    "ModelCodingAgentResult",
    "ModelCodingAgentTraceProjection",
    "ModelAgentInvocation",
    "ModelSubprocessInvocation",
    "ModelSubprocessOutcome",
    "ModelWorkspaceValidateCommand",
    "ModelWorkspaceValidateResult",
]
