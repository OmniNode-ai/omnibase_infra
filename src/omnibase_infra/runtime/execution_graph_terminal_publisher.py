# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Signed-only terminal publisher for the execution-graph workflow."""

from __future__ import annotations

from collections.abc import Awaitable, Callable

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from omnibase_core.models.envelope.model_message_envelope import ModelMessageEnvelope
from omnibase_core.models.execution_graph_replay.model_execution_graph_terminal_result import (
    ModelExecutionGraphTerminalResult,
)
from omnibase_infra.runtime.execution_graph_read_authority import (
    VerifiedExecutionGraphReadAuthority,
)
from omnibase_infra.runtime.models.model_execution_graph_terminal_publisher_config import (
    ModelExecutionGraphTerminalPublisherConfig,
)

type ExecutionGraphTerminalTransport = Callable[
    [str, ModelMessageEnvelope[dict[str, object]]], Awaitable[None]
]


class ExecutionGraphTerminalPublisherError(PermissionError):
    """A graph terminal could not be published with its signed identity intact."""


class ExecutionGraphTerminalPublisher:
    """Emit one contract-topic signed terminal after strict authority binding."""

    def __init__(
        self,
        *,
        config: ModelExecutionGraphTerminalPublisherConfig,
        private_key: Ed25519PrivateKey,
        publish: ExecutionGraphTerminalTransport,
    ) -> None:
        if not isinstance(private_key, Ed25519PrivateKey):
            raise TypeError("Graph terminal publisher requires an Ed25519 private key")
        self._config = config
        self._private_key = private_key
        self._publish = publish

    async def publish(
        self,
        authority: VerifiedExecutionGraphReadAuthority,
        terminal: ModelExecutionGraphTerminalResult,
    ) -> None:
        """Sign and publish only a terminal bound to the verified command."""
        if type(authority) is not VerifiedExecutionGraphReadAuthority:
            raise ExecutionGraphTerminalPublisherError(
                "Graph terminal requires sealed signed ingress authority"
            )
        if type(terminal) is not ModelExecutionGraphTerminalResult:
            raise TypeError("Graph terminal publisher requires a typed terminal result")
        if (
            terminal.tenant_id != authority.tenant_id
            or terminal.correlation_id != authority.correlation_id
            or terminal.workflow_type != self._config.workflow_type
        ):
            raise ExecutionGraphTerminalPublisherError(
                "Graph terminal conflicts with signed workflow identity"
            )
        payload = terminal.model_dump(mode="json")
        envelope = ModelMessageEnvelope[dict[str, object]].create_signed(
            realm=self._config.realm,
            runtime_id=self._config.runtime_id,
            bus_id=self._config.bus_id,
            trace_id=terminal.correlation_id,
            tenant_id=str(terminal.tenant_id),
            payload=payload,
            private_key=self._private_key.private_bytes_raw(),
        )
        if (
            envelope.runtime_id != self._config.runtime_id
            or envelope.realm != self._config.realm
            or envelope.bus_id != self._config.bus_id
            or envelope.trace_id != terminal.correlation_id
            or envelope.tenant_id != str(terminal.tenant_id)
            or envelope.payload != payload
        ):
            raise ExecutionGraphTerminalPublisherError(
                "Graph terminal signer did not preserve configured identity"
            )
        await self._publish(self._config.terminal_topic, envelope)


__all__ = [
    "ExecutionGraphTerminalPublisher",
    "ExecutionGraphTerminalPublisherError",
    "ExecutionGraphTerminalTransport",
]
