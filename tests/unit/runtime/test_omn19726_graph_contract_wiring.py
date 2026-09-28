# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Exercise graph contract discovery through the runtime wiring seams."""

from __future__ import annotations

import base64
import importlib
import json
import tomllib
from pathlib import Path
from typing import cast
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from omnibase_core.crypto.crypto_ed25519_signer import generate_keypair
from omnibase_core.models.container.model_onex_container import ModelONEXContainer
from omnibase_core.models.envelope.model_message_envelope import ModelMessageEnvelope
from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope
from omnibase_core.models.execution_graph_replay.model_execution_graph_topology_version import (
    ModelExecutionGraphTopologyVersion,
)
from omnibase_core.models.primitives.model_semver import ModelSemVer
from omnibase_infra.errors import ProtocolConfigurationError
from omnibase_infra.event_bus.event_bus_inmemory import EventBusInmemory
from omnibase_infra.runtime.auto_wiring.discovery import discover_contracts_from_paths
from omnibase_infra.runtime.auto_wiring.handler_wiring import (
    subscribe_wired_contract_topics,
    wire_from_manifest,
)
from omnibase_infra.runtime.auto_wiring.models.model_auto_wiring_manifest import (
    ModelAutoWiringManifest,
)
from omnibase_infra.runtime.db.execution_graph_read_adapters import (
    ExecutionGraphCurrentEvidenceReader,
)
from omnibase_infra.runtime.db.protocol_execution_graph_stored_chain_reader import (
    ProtocolExecutionGraphStoredChainReader,
)
from omnibase_infra.runtime.dispatch_envelope_context import (
    current_execution_graph_read_authority,
)
from omnibase_infra.runtime.execution_graph_read_authority import (
    TrustedExecutionGraphGatewayPolicy,
    TrustedGatewaySignerScope,
)
from omnibase_infra.runtime.execution_graph_read_command_handler import (
    ExecutionGraphReadCommandExecutor,
)
from omnibase_infra.runtime.execution_graph_runtime_composition import (
    GRAPH_READ_CONTRACT_NAME,
    GRAPH_READ_WORKFLOW_TYPE,
    select_execution_graph_contract,
)
from omnibase_infra.runtime.execution_graph_topology_registry import (
    PackagedExecutionGraphTopologyContract,
)
from omnibase_infra.runtime.message_dispatch_engine import MessageDispatchEngine
from omnibase_infra.runtime.models.model_execution_graph_read_ingress_config import (
    ModelExecutionGraphReadIngressConfig,
)
from omnibase_infra.runtime.runtime_host_process import RuntimeHostProcess
from omnibase_infra.runtime.service_kernel import _build_runtime_handler_dependencies
from omnibase_infra.utils.util_topic_event_type import derive_event_type_from_topic
from tests.helpers.projection_tenant_authority import InMemoryKeyProvider

_REPO_ROOT = Path(__file__).resolve().parents[3]
_CONTRACT = (
    _REPO_ROOT
    / "src"
    / "omnibase_infra"
    / "nodes"
    / GRAPH_READ_CONTRACT_NAME
    / "contract.yaml"
)
_COMMAND_TOPIC = "onex.cmd.omnibase-infra.delegation-execution-graph-requested.v1"


def _discovered_manifest() -> ModelAutoWiringManifest:
    project = tomllib.loads((_REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    module_name = project["project"]["entry-points"]["onex.nodes"][
        GRAPH_READ_CONTRACT_NAME
    ]
    module = importlib.import_module(module_name)
    assert module.__file__ is not None
    entrypoint_contract = Path(module.__file__).parent / "contract.yaml"
    assert entrypoint_contract == _CONTRACT

    manifest = discover_contracts_from_paths([entrypoint_contract])
    assert manifest.errors == ()
    assert len(manifest.contracts) == 1
    return manifest


def _executor() -> ExecutionGraphReadCommandExecutor:
    """Provide the real required executor type without opening external resources."""
    topology = PackagedExecutionGraphTopologyContract().resolve(
        ModelExecutionGraphTopologyVersion(
            contract_version=ModelSemVer(major=1, minor=3, patch=0),
            topology_sha256="0505ab0b163492380739a15646c0442a3ecfb54efb0acb53bc23d232640fbfd3",
        )
    )
    return ExecutionGraphReadCommandExecutor(
        evidence_reader=cast("ExecutionGraphCurrentEvidenceReader", object()),
        stored_chain_reader=cast("ProtocolExecutionGraphStoredChainReader", object()),
        topology=topology,
        workflow_type=GRAPH_READ_WORKFLOW_TYPE,
        fold=AsyncMock(),
        publish_terminal=AsyncMock(),
    )


def _write_gateway_key_file(path: Path) -> None:
    path.write_text(
        json.dumps(
            {"keys": {"trusted-gateway": base64.urlsafe_b64encode(b"g" * 32).decode()}}
        ),
        encoding="utf-8",
    )
    path.chmod(0o600)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_discovered_graph_contract_wires_handler_and_exact_command_subscription() -> (
    None
):
    discovered = _discovered_manifest()
    enabled, contract = select_execution_graph_contract(discovered, enabled=True)
    assert contract is discovered.contracts[0]
    assert contract.event_bus is not None
    assert contract.event_bus.subscribe_topics == (_COMMAND_TOPIC,)

    executor = _executor()
    observed_authorities = []

    async def observe_authority(*_args: object, **_kwargs: object) -> None:
        observed_authorities.append(current_execution_graph_read_authority())

    executor.handle = AsyncMock(side_effect=observe_authority)  # type: ignore[method-assign]
    dependencies = _build_runtime_handler_dependencies(
        None,
        execution_graph_read_executor=executor,
        execution_graph_read_container=ModelONEXContainer(),
    )
    assert dependencies is not None
    bus = EventBusInmemory(environment="test", group="graph-wiring")
    await bus.start()
    engine = MessageDispatchEngine()
    report = await wire_from_manifest(
        manifest=enabled,
        dispatch_engine=engine,
        event_bus=bus,
        environment="test",
        container=dependencies["HandlerExecutionGraphRead"]["container"],
        subscribe_immediately=False,
        materialized_explicit_dependencies=dependencies,
    )
    assert report.total_failed == 0
    assert [result.contract_name for result in report.results] == [
        GRAPH_READ_CONTRACT_NAME
    ]
    assert report.results[0].outcome.value == "wired"

    engine.freeze()
    keys = generate_keypair()
    scope = TrustedGatewaySignerScope(
        runtime_id="trusted-gateway", realm="test", bus_id="graph-read"
    )
    ingress = ModelExecutionGraphReadIngressConfig(
        command_topic=_COMMAND_TOPIC,
        gateway_policy=TrustedExecutionGraphGatewayPolicy(scopes=frozenset({scope})),
    )
    provider = InMemoryKeyProvider({scope.runtime_id: keys.public_key_bytes})
    with pytest.raises(ValueError, match="requires signed ingress"):
        await subscribe_wired_contract_topics(
            manifest=enabled,
            report=report,
            dispatch_engine=engine,
            event_bus=bus,
            environment="test",
        )
    assert bus._subscribers == {}
    subscribed = await subscribe_wired_contract_topics(
        manifest=enabled,
        report=report,
        dispatch_engine=engine,
        event_bus=bus,
        environment="test",
        execution_graph_read_ingress=ingress,
        execution_graph_read_key_provider=provider,
    )
    assert subscribed == {GRAPH_READ_CONTRACT_NAME: (_COMMAND_TOPIC,)}
    assert tuple(bus._subscribers) == (_COMMAND_TOPIC,)

    tenant_id = uuid4()
    correlation_id = uuid4()
    workflow_id = uuid4()
    inner = ModelEventEnvelope[dict[str, object]](
        tenant_id=str(tenant_id),
        correlation_id=correlation_id,
        event_type=derive_event_type_from_topic(_COMMAND_TOPIC),
        metadata={"tags": {"workflow_id": str(workflow_id)}},
        payload={
            "correlation_id": str(correlation_id),
            "cursor_mode": "latest",
            "source_cursors": None,
        },
    ).model_dump(mode="json")
    signed = ModelMessageEnvelope[dict[str, object]].create_signed(
        realm=scope.realm,
        runtime_id=scope.runtime_id,
        bus_id=scope.bus_id,
        trace_id=correlation_id,
        tenant_id=str(tenant_id),
        payload=inner,
        private_key=keys.private_key_bytes,
    )
    await bus.publish(_COMMAND_TOPIC, None, signed.model_dump_json().encode())
    assert len(observed_authorities) == 1
    authority = observed_authorities[0]
    assert authority is not None
    assert authority.tenant_id == tenant_id
    assert authority.correlation_id == correlation_id
    assert authority.workflow_id == workflow_id
    assert authority.payload_hash == signed.signature.payload_hash
    assert current_execution_graph_read_authority() is None

    tampered = signed.model_copy(update={"tenant_id": str(uuid4())})
    await bus.publish(_COMMAND_TOPIC, None, tampered.model_dump_json().encode())
    assert len(observed_authorities) == 1
    assert current_execution_graph_read_authority() is None

    async def fail_after_authority(*_args: object, **_kwargs: object) -> None:
        assert current_execution_graph_read_authority() is not None
        raise RuntimeError("simulated graph executor failure")

    executor.handle = AsyncMock(side_effect=fail_after_authority)  # type: ignore[method-assign]
    error_correlation = uuid4()
    error_payload = ModelEventEnvelope[dict[str, object]](
        tenant_id=str(tenant_id),
        correlation_id=error_correlation,
        event_type=derive_event_type_from_topic(_COMMAND_TOPIC),
        metadata={"tags": {"workflow_id": str(uuid4())}},
        payload={
            "correlation_id": str(error_correlation),
            "cursor_mode": "latest",
            "source_cursors": None,
        },
    ).model_dump(mode="json")
    failing_signed = ModelMessageEnvelope[dict[str, object]].create_signed(
        realm=scope.realm,
        runtime_id=scope.runtime_id,
        bus_id=scope.bus_id,
        trace_id=error_correlation,
        tenant_id=str(tenant_id),
        payload=error_payload,
        private_key=keys.private_key_bytes,
    )
    await bus.publish(_COMMAND_TOPIC, None, failing_signed.model_dump_json().encode())
    assert current_execution_graph_read_authority() is None
    assert (
        bus._topic_offsets.get(
            "onex.evt.omnibase-infra.delegation-execution-graph-read-terminal.v1", 0
        )
        == 0
    )
    await bus.shutdown()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_disabled_graph_selection_produces_no_dispatch_or_subscription() -> None:
    disabled, graph_contract = select_execution_graph_contract(
        _discovered_manifest(), enabled=False
    )
    assert graph_contract is None
    assert disabled.contracts == ()

    bus = EventBusInmemory(environment="test", group="graph-disabled")
    await bus.start()
    engine = MessageDispatchEngine()
    report = await wire_from_manifest(
        manifest=disabled,
        dispatch_engine=engine,
        event_bus=bus,
        environment="test",
        subscribe_immediately=False,
    )
    assert report.results == ()
    engine.freeze()
    subscribed = await subscribe_wired_contract_topics(
        manifest=disabled,
        report=report,
        dispatch_engine=engine,
        event_bus=bus,
        environment="test",
    )
    assert subscribed == {}
    assert bus._subscribers == {}
    await bus.shutdown()


@pytest.mark.unit
def test_graph_contract_ingress_configuration_requires_resolvable_gateway_key(
    tmp_path: Path,
) -> None:
    key_file = tmp_path / "gateway-keys.json"
    _write_gateway_key_file(key_file)
    host = RuntimeHostProcess(
        config={
            "service_name": "omnibase-infra",
            "node_name": "graph-read",
            "execution_graph_read_gateway": {
                "command_topic": _COMMAND_TOPIC,
                "runtime_id": "trusted-gateway",
                "realm": "test",
                "bus_id": "test-bus",
                "public_key_path": str(key_file),
            },
        }
    )
    ingress, provider = host._execution_graph_read_ingress_dependencies()
    assert ingress is not None and provider is not None
    assert ingress.command_topic == _COMMAND_TOPIC
    assert provider.get_public_key("trusted-gateway") is not None

    missing_key_host = RuntimeHostProcess(
        config={
            "service_name": "omnibase-infra",
            "node_name": "graph-read",
            "execution_graph_read_gateway": {
                "command_topic": _COMMAND_TOPIC,
                "runtime_id": "trusted-gateway",
                "realm": "test",
                "bus_id": "test-bus",
                "public_key_path": str(tmp_path / "missing.json"),
            },
        }
    )
    with pytest.raises(ProtocolConfigurationError, match="invalid execution graph"):
        missing_key_host._execution_graph_read_ingress_dependencies()
