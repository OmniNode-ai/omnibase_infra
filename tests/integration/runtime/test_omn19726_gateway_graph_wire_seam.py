# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Offline wire seam: real Gateway command signing and Infra graph terminal."""

from __future__ import annotations

import importlib
import os
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType
from uuid import UUID

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from omnibase_core.models.envelope.model_message_envelope import ModelMessageEnvelope
from omnibase_core.models.execution_graph_replay.model_execution_graph_terminal_refusal import (
    ModelExecutionGraphTerminalRefusal,
)
from omnibase_core.models.execution_graph_replay.model_execution_graph_terminal_result import (
    ModelExecutionGraphTerminalResult,
)
from omnibase_infra.nodes.node_delegation_chain_ledger_effect.execution_graph_fold import (
    DelegationExecutionGraphFold,
)
from omnibase_infra.runtime.execution_graph_read_authority import (
    ExecutionGraphReadAuthorityError,
    TrustedExecutionGraphGatewayPolicy,
    TrustedGatewaySignerScope,
    verify_signed_execution_graph_read_authority,
)
from omnibase_infra.runtime.execution_graph_terminal_publisher import (
    ExecutionGraphTerminalPublisher,
)
from omnibase_infra.runtime.models.model_execution_graph_terminal_publisher_config import (
    ModelExecutionGraphTerminalPublisherConfig,
)
from tests.helpers.projection_tenant_authority import InMemoryKeyProvider
from tests.unit.nodes.node_delegation_chain_ledger_effect.test_execution_graph_fold import (
    CORRELATION,
    TENANT,
    _request,
)

pytestmark = [pytest.mark.integration, pytest.mark.asyncio]

_WORKFLOW = UUID("d2f30b4e-d83d-4ad0-a905-6833d5d68d53")
_WORKFLOW_TYPE = "delegation-execution-graph-read"
_COMMAND_TOPIC = "onex.cmd.omnibase-infra.delegation-execution-graph-requested.v1"
_TERMINAL_TOPIC = "onex.evt.omnibase-infra.delegation-execution-graph-read-terminal.v1"


def _gateway_modules(monkeypatch: pytest.MonkeyPatch) -> tuple[ModuleType, ...]:
    raw = os.environ.get("OMN19726_GATEWAY_SOURCE_DIR")
    if not raw:
        pytest.skip("set OMN19726_GATEWAY_SOURCE_DIR for cross-repo wire proof")
    source = Path(raw)
    if not (source / "workflow_publisher.py").is_file():
        pytest.fail("OMN19726_GATEWAY_SOURCE_DIR lacks Gateway source modules")
    monkeypatch.syspath_prepend(str(source))
    return (
        importlib.import_module("models.model_workflow_envelope"),
        importlib.import_module("models.model_signed_workflow_terminal_policy"),
        importlib.import_module("workflow_publisher"),
        importlib.import_module("workflow_contracts"),
        importlib.import_module("signed_workflow_terminal"),
    )


@pytest.mark.integration
async def test_gateway_signed_request_and_runtime_signed_terminal_roundtrip(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    (
        envelope_module,
        policy_module,
        publisher_module,
        contracts_module,
        terminal_module,
    ) = _gateway_modules(monkeypatch)
    contracts_module.load_workflow_contracts()
    contract = contracts_module.fenced_workflow_contract(_WORKFLOW_TYPE)
    assert contract is not None
    assert contract.contract_id == "node_execution_graph_read_effect:1.0.0"
    assert contract.command_topic == _COMMAND_TOPIC
    assert contract.terminal_topic == _TERMINAL_TOPIC
    request_payload = {
        "cursor_mode": "bounded",
        "source_cursors": [
            {"topic": topic, "partition": 0, "max_ingest_watermark": 3}
            for topic in ("head", "left", "right")
        ],
    }
    assert contracts_module.validate_workflow_payload(contract, request_payload) == []

    gateway_key = Ed25519PrivateKey.generate()
    gateway_key_path = tmp_path / "gateway-test-only.pem"
    gateway_key_path.write_bytes(
        gateway_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )
    )
    monkeypatch.setenv(
        "DELEGATION_EXECUTION_GRAPH_SIGNING_KEY_FILE", str(gateway_key_path)
    )
    monkeypatch.setenv("DELEGATION_EXECUTION_GRAPH_SIGNER_REALM", "offline-test")
    monkeypatch.setenv("DELEGATION_EXECUTION_GRAPH_SIGNER_RUNTIME_ID", "gateway-test")
    monkeypatch.setenv("DELEGATION_EXECUTION_GRAPH_SIGNER_BUS_ID", "gateway-test-bus")
    command = envelope_module.ModelWorkflowCommandEnvelope(
        workflow_id=_WORKFLOW,
        workflow_type=_WORKFLOW_TYPE,
        contract_id=contract.contract_id,
        command_topic=contract.command_topic,
        event_type=contract.event_type,
        payload=request_payload,
        envelope_id=UUID("7c2d8024-7673-42ca-a035-d69414716d9c"),
        correlation_id=CORRELATION,
        causation_id=UUID("fc7c9e43-4089-4ee9-b507-2e6ca8d35647"),
        source_tenant_id=TENANT,
        source_tenant_principal_id="offline-test-tenant",
        authenticated_user_id="offline-test-user",
        schema_version="1.1.0",
        emitted_at=datetime(2026, 9, 26, 12, tzinfo=UTC),
        source_gateway_instance="onex-api@offline-test",
    )
    signed_command = ModelMessageEnvelope[dict[str, object]].model_validate_json(
        publisher_module.serialize_envelope(command)
    )
    gateway_scope = TrustedGatewaySignerScope(
        runtime_id="gateway-test", realm="offline-test", bus_id="gateway-test-bus"
    )
    authority = verify_signed_execution_graph_read_authority(
        signed_command,
        InMemoryKeyProvider(
            {"gateway-test": gateway_key.public_key().public_bytes_raw()}
        ),
        TrustedExecutionGraphGatewayPolicy(scopes=frozenset({gateway_scope})),
    )
    assert authority.workflow_id == _WORKFLOW
    assert authority.request.source_cursors is not None
    assert authority.request.source_cursors[0].max_ingest_watermark == 3
    tampered_command = signed_command.model_copy(
        update={"tenant_id": str(UUID("aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"))}
    )
    with pytest.raises(ExecutionGraphReadAuthorityError, match="signature"):
        verify_signed_execution_graph_read_authority(
            tampered_command,
            InMemoryKeyProvider(
                {"gateway-test": gateway_key.public_key().public_bytes_raw()}
            ),
            TrustedExecutionGraphGatewayPolicy(scopes=frozenset({gateway_scope})),
        )
    with pytest.raises(ExecutionGraphReadAuthorityError, match="trusted gateway"):
        verify_signed_execution_graph_read_authority(
            signed_command,
            InMemoryKeyProvider(
                {"gateway-test": gateway_key.public_key().public_bytes_raw()}
            ),
            TrustedExecutionGraphGatewayPolicy(
                scopes=frozenset(
                    {
                        TrustedGatewaySignerScope(
                            runtime_id="other-gateway",
                            realm="offline-test",
                            bus_id="gateway-test-bus",
                        )
                    }
                )
            ),
        )

    runtime_key = Ed25519PrivateKey.generate()
    published: list[tuple[str, ModelMessageEnvelope[dict[str, object]]]] = []

    async def capture(
        topic: str, envelope: ModelMessageEnvelope[dict[str, object]]
    ) -> None:
        published.append((topic, envelope))

    runtime_publisher = ExecutionGraphTerminalPublisher(
        config=ModelExecutionGraphTerminalPublisherConfig(
            terminal_topic=_TERMINAL_TOPIC,
            runtime_id="infra-test",
            realm="offline-test",
            bus_id="infra-test-bus",
            workflow_type=_WORKFLOW_TYPE,
        ),
        private_key=runtime_key,
        publish=capture,
    )
    policy = policy_module.ModelSignedWorkflowTerminalPolicy(
        runtime_id="infra-test",
        realm="offline-test",
        bus_id="infra-test-bus",
        terminal_topic=_TERMINAL_TOPIC,
        workflow_type=_WORKFLOW_TYPE,
    )
    runtime_keys = InMemoryKeyProvider(
        {"infra-test": runtime_key.public_key().public_bytes_raw()}
    )

    graph = DelegationExecutionGraphFold().handle(_request())
    assert graph.replay.correlation_id == authority.correlation_id
    assert graph.replay.source_cursors[0].max_ingest_watermark == 3
    await runtime_publisher.publish(
        authority,
        ModelExecutionGraphTerminalResult(
            workflow_id=authority.workflow_id,
            tenant_id=authority.tenant_id,
            correlation_id=authority.correlation_id,
            workflow_type=_WORKFLOW_TYPE,
            status="completed",
            result=graph,
        ),
    )
    await runtime_publisher.publish(
        authority,
        ModelExecutionGraphTerminalResult(
            workflow_id=authority.workflow_id,
            tenant_id=authority.tenant_id,
            correlation_id=authority.correlation_id,
            workflow_type=_WORKFLOW_TYPE,
            status="failed",
            refusal=ModelExecutionGraphTerminalRefusal(
                code="offline-refusal", message="offline proof only"
            ),
        ),
    )

    assert len(published) == 2
    for topic, signed in published:
        parsed_wire = ModelMessageEnvelope[dict[str, object]].model_validate_json(
            signed.model_dump_json()
        )
        accepted = terminal_module.verify_signed_workflow_terminal(
            parsed_wire, topic=topic, key_provider=runtime_keys, policy=policy
        )
        assert accepted.workflow_id == _WORKFLOW
        assert accepted.tenant_id == TENANT
        assert accepted.correlation_id == CORRELATION
        assert accepted.workflow_type == _WORKFLOW_TYPE
        if accepted.status == "completed":
            assert accepted.refusal is None
            assert accepted.result is not None
            assert (
                accepted.result["replay"]["source_cursors"][0]["max_ingest_watermark"]
                == 3
            )
        else:
            assert accepted.status == "failed"
            assert accepted.result is None
            assert accepted.refusal == {
                "code": "offline-refusal",
                "message": "offline proof only",
            }
    completed_topic, completed_envelope = published[0]
    tampered_terminal = completed_envelope.model_copy(
        update={
            "payload": {
                **completed_envelope.payload,
                "workflow_id": str(UUID("aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee")),
            }
        }
    )
    with pytest.raises(terminal_module.SignedWorkflowTerminalError, match="signature"):
        terminal_module.verify_signed_workflow_terminal(
            tampered_terminal,
            topic=completed_topic,
            key_provider=runtime_keys,
            policy=policy,
        )
    wrong_policy = policy.model_copy(update={"runtime_id": "other-runtime"})
    with pytest.raises(terminal_module.SignedWorkflowTerminalError, match="signer"):
        terminal_module.verify_signed_workflow_terminal(
            completed_envelope,
            topic=completed_topic,
            key_provider=runtime_keys,
            policy=wrong_policy,
        )
