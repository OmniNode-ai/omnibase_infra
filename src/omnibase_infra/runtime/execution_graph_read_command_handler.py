# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Signed-only composition seam for execution-graph read commands.

The event-bus subcontract owns routing and verifies the gateway signature before
binding the authority context.  This boundary makes the remaining trust chain
explicit: owner-first current evidence, pure admission, a later pure fold, and
a terminal publisher supplied by the declared workflow composition.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable

from omnibase_core.models.execution_graph_replay.model_execution_graph_request import (
    ModelExecutionGraphRequest,
)
from omnibase_core.models.execution_graph_replay.model_execution_graph_stored_chain_annotation import (
    ModelExecutionGraphStoredChainAnnotation,
)
from omnibase_core.models.execution_graph_replay.model_execution_graph_terminal_refusal import (
    ModelExecutionGraphTerminalRefusal,
)
from omnibase_core.models.execution_graph_replay.model_execution_graph_terminal_result import (
    ModelExecutionGraphTerminalResult,
)
from omnibase_infra.runtime.db.execution_graph_read_adapters import (
    ExecutionGraphCurrentEvidence,
    ExecutionGraphCurrentEvidenceReader,
    ExecutionGraphOwnerNotFoundError,
)
from omnibase_infra.runtime.db.protocol_execution_graph_stored_chain_reader import (
    ProtocolExecutionGraphStoredChainReader,
)
from omnibase_infra.runtime.dispatch_envelope_context import (
    current_execution_graph_read_authority,
)
from omnibase_infra.runtime.execution_graph_ownership import (
    ExecutionGraphOwnershipAdmission,
    ExecutionGraphOwnershipRefusalError,
    admit_current_ownership,
)
from omnibase_infra.runtime.execution_graph_read_authority import (
    VerifiedExecutionGraphReadAuthority,
)
from omnibase_infra.runtime.execution_graph_topology_registry import (
    PinnedExecutionGraphTopology,
)

type ExecutionGraphReadFold = Callable[
    [
        ModelExecutionGraphRequest,
        VerifiedExecutionGraphReadAuthority,
        PinnedExecutionGraphTopology,
        ExecutionGraphOwnershipAdmission,
        tuple[ModelExecutionGraphStoredChainAnnotation, ...],
    ],
    Awaitable[ModelExecutionGraphTerminalResult],
]
type ExecutionGraphTerminalPublisher = Callable[
    [VerifiedExecutionGraphReadAuthority, ModelExecutionGraphTerminalResult],
    Awaitable[None],
]

_LOGGER = logging.getLogger(__name__)


class ExecutionGraphReadCommandError(PermissionError):
    """The command was not admitted to read or publish graph evidence."""


class ExecutionGraphReadCommandExecutor:
    """Run one contract-routed graph read only under verified ingress authority."""

    def __init__(
        self,
        *,
        evidence_reader: ExecutionGraphCurrentEvidenceReader,
        stored_chain_reader: ProtocolExecutionGraphStoredChainReader,
        topology: PinnedExecutionGraphTopology,
        workflow_type: str,
        fold: ExecutionGraphReadFold,
        publish_terminal: ExecutionGraphTerminalPublisher,
    ) -> None:
        if type(topology) is not PinnedExecutionGraphTopology:
            raise TypeError("Graph read command requires a sealed pinned topology")
        if not workflow_type or workflow_type != workflow_type.strip():
            raise ValueError("Graph read workflow type must be non-empty and canonical")
        self._evidence_reader = evidence_reader
        self._stored_chain_reader = stored_chain_reader
        self._topology = topology
        self._workflow_type = workflow_type
        self._fold = fold
        self._publish_terminal = publish_terminal

    async def handle(
        self, request: ModelExecutionGraphRequest
    ) -> ModelExecutionGraphTerminalResult:
        """Read, admit, fold, and publish one graph terminal under sealed authority."""
        authority = current_execution_graph_read_authority()
        if type(authority) is not VerifiedExecutionGraphReadAuthority:
            raise ExecutionGraphReadCommandError(
                "Execution graph command requires verified signed ingress authority"
            )
        if request != authority.request:
            raise ExecutionGraphReadCommandError(
                "Execution graph command conflicts with signed request"
            )

        try:
            evidence = await self._evidence_reader.read_authorized_current(
                authority, self._topology.read_set
            )
            admission = admit_current_ownership(evidence, self._topology.read_set)
        except (
            ExecutionGraphOwnerNotFoundError,
            ExecutionGraphOwnershipRefusalError,
        ) as exc:
            reason = (
                exc.reason.value
                if isinstance(exc, ExecutionGraphOwnershipRefusalError)
                else "owner_not_found"
            )
            _LOGGER.info("execution graph ownership refused: %s", reason)
            terminal = ModelExecutionGraphTerminalResult(
                workflow_id=authority.workflow_id,
                tenant_id=authority.tenant_id,
                correlation_id=authority.correlation_id,
                workflow_type=self._workflow_type,
                status="failed",
                refusal=ModelExecutionGraphTerminalRefusal(
                    code="not_found",
                    message="Execution graph correlation was not found.",
                ),
            )
            self._validate_terminal(terminal, authority)
            await self._publish_terminal(authority, terminal)
            return terminal
        stored_chain = await self._stored_chain_reader.read_current(
            authority, admission.owner, admission.owned_envelope_ids
        )
        owned_ids = set(admission.owned_envelope_ids)
        stored_ids = tuple(annotation.node_id for annotation in stored_chain)
        if len(stored_ids) != len(set(stored_ids)) or any(
            node_id not in owned_ids for node_id in stored_ids
        ):
            raise ExecutionGraphReadCommandError(
                "stored chain annotation conflicts with admitted ownership"
            )
        terminal = await self._fold(
            request, authority, self._topology, admission, stored_chain
        )
        self._validate_terminal(terminal, authority)
        await self._publish_terminal(authority, terminal)
        return terminal

    def _validate_terminal(
        self,
        terminal: ModelExecutionGraphTerminalResult,
        authority: VerifiedExecutionGraphReadAuthority,
    ) -> None:
        if type(terminal) is not ModelExecutionGraphTerminalResult:
            raise TypeError("Execution graph fold must return a typed terminal result")
        if (
            terminal.workflow_id != authority.workflow_id
            or terminal.tenant_id != authority.tenant_id
            or terminal.correlation_id != authority.correlation_id
            or terminal.workflow_type != self._workflow_type
        ):
            raise ExecutionGraphReadCommandError(
                "Execution graph terminal conflicts with signed workflow identity"
            )


__all__ = [
    "ExecutionGraphReadCommandError",
    "ExecutionGraphReadCommandExecutor",
    "ExecutionGraphReadFold",
    "ExecutionGraphTerminalPublisher",
]
