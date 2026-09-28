# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Execution graph workflow terminals are always signed and identity-bound."""

from __future__ import annotations

from uuid import uuid4

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from omnibase_core.crypto.crypto_ed25519_signer import generate_keypair
from omnibase_core.models.envelope.model_message_envelope import ModelMessageEnvelope
from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope
from omnibase_core.models.execution_graph_replay.model_execution_graph_terminal_refusal import (
    ModelExecutionGraphTerminalRefusal,
)
from omnibase_core.models.execution_graph_replay.model_execution_graph_terminal_result import (
    ModelExecutionGraphTerminalResult,
)
from omnibase_infra.runtime.execution_graph_read_authority import (
    TrustedExecutionGraphGatewayPolicy,
    TrustedGatewaySignerScope,
    VerifiedExecutionGraphReadAuthority,
    verify_signed_execution_graph_read_authority,
)
from omnibase_infra.runtime.execution_graph_terminal_publisher import (
    ExecutionGraphTerminalPublisher,
    ExecutionGraphTerminalPublisherError,
)
from omnibase_infra.runtime.models.model_execution_graph_terminal_publisher_config import (
    ModelExecutionGraphTerminalPublisherConfig,
)
from tests.helpers.projection_tenant_authority import InMemoryKeyProvider

_GATEWAY_SCOPE = TrustedGatewaySignerScope(
    runtime_id="trusted-gateway", realm="test", bus_id="gateway-bus"
)
_CONFIG = ModelExecutionGraphTerminalPublisherConfig(
    terminal_topic="onex.evt.omnibase-infra.delegation-execution-graph-read-terminal.v1",
    runtime_id="infra-runtime",
    realm="test",
    bus_id="infra-bus",
    workflow_type="delegation_execution_graph_read",
)


def _authority() -> VerifiedExecutionGraphReadAuthority:
    tenant_id = uuid4()
    correlation_id = uuid4()
    keys = generate_keypair()
    payload = ModelEventEnvelope[dict[str, object]](
        tenant_id=str(tenant_id),
        correlation_id=correlation_id,
        event_type="delegation-execution-graph-requested",
        metadata={"tags": {"workflow_id": str(uuid4())}},
        payload={"correlation_id": str(correlation_id), "cursor_mode": "latest"},
    ).model_dump(mode="json")
    signed = ModelMessageEnvelope[dict[str, object]].create_signed(
        realm=_GATEWAY_SCOPE.realm,
        runtime_id=_GATEWAY_SCOPE.runtime_id,
        bus_id=_GATEWAY_SCOPE.bus_id,
        trace_id=correlation_id,
        tenant_id=str(tenant_id),
        payload=payload,
        private_key=keys.private_key_bytes,
    )
    return verify_signed_execution_graph_read_authority(
        signed,
        InMemoryKeyProvider({_GATEWAY_SCOPE.runtime_id: keys.public_key_bytes}),
        TrustedExecutionGraphGatewayPolicy(scopes=frozenset({_GATEWAY_SCOPE})),
    )


def _terminal(
    authority: VerifiedExecutionGraphReadAuthority,
) -> ModelExecutionGraphTerminalResult:
    return ModelExecutionGraphTerminalResult(
        workflow_id=authority.workflow_id,
        tenant_id=authority.tenant_id,
        correlation_id=authority.correlation_id,
        workflow_type=_CONFIG.workflow_type,
        status="failed",
        refusal=ModelExecutionGraphTerminalRefusal(code="refused", message="test"),
    )


@pytest.mark.unit
@pytest.mark.asyncio
async def test_terminal_is_signed_on_configured_topic_with_exact_identity() -> None:
    authority = _authority()
    keys = generate_keypair()
    published: list[tuple[str, ModelMessageEnvelope[dict[str, object]]]] = []

    async def transport(
        topic: str, envelope: ModelMessageEnvelope[dict[str, object]]
    ) -> None:
        published.append((topic, envelope))

    publisher = ExecutionGraphTerminalPublisher(
        config=_CONFIG,
        private_key=Ed25519PrivateKey.from_private_bytes(keys.private_key_bytes),
        publish=transport,
    )

    await publisher.publish(authority, _terminal(authority))

    topic, envelope = published[0]
    assert topic == _CONFIG.terminal_topic
    assert envelope.trace_id == authority.correlation_id
    assert envelope.tenant_id == str(authority.tenant_id)
    assert envelope.payload["workflow_id"] == str(authority.workflow_id)
    assert envelope.payload == _terminal(authority).model_dump(mode="json")
    assert envelope.verify_signature(
        InMemoryKeyProvider({_CONFIG.runtime_id: keys.public_key_bytes})
    )


@pytest.mark.unit
@pytest.mark.asyncio
async def test_foreign_terminal_is_refused_before_any_publish() -> None:
    authority = _authority()
    calls: list[str] = []
    keys = generate_keypair()

    async def transport(*_args: object) -> None:
        calls.append("publish")

    publisher = ExecutionGraphTerminalPublisher(
        config=_CONFIG,
        private_key=Ed25519PrivateKey.from_private_bytes(keys.private_key_bytes),
        publish=transport,
    )
    foreign = _terminal(authority).model_copy(update={"tenant_id": uuid4()})

    with pytest.raises(ExecutionGraphTerminalPublisherError, match="conflicts"):
        await publisher.publish(authority, foreign)
    assert calls == []


@pytest.mark.unit
@pytest.mark.asyncio
async def test_foreign_workflow_terminal_is_refused_before_publish() -> None:
    authority = _authority()
    calls: list[str] = []
    keys = generate_keypair()

    async def transport(*_args: object) -> None:
        calls.append("publish")

    publisher = ExecutionGraphTerminalPublisher(
        config=_CONFIG,
        private_key=Ed25519PrivateKey.from_private_bytes(keys.private_key_bytes),
        publish=transport,
    )
    foreign = _terminal(authority).model_copy(update={"workflow_id": uuid4()})

    with pytest.raises(ExecutionGraphTerminalPublisherError, match="conflicts"):
        await publisher.publish(authority, foreign)
    assert calls == []


@pytest.mark.unit
def test_publisher_requires_typed_private_key() -> None:
    with pytest.raises(TypeError, match="Ed25519"):
        ExecutionGraphTerminalPublisher(
            config=_CONFIG,
            private_key=None,  # type: ignore[arg-type]
            publish=lambda *_args: None,  # type: ignore[arg-type]
        )
