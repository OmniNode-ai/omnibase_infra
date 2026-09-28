# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Graph owner reads use analytics before ledger evidence on separate databases."""

from __future__ import annotations

import base64
import getpass
import importlib
import json
import shutil
import socket
import subprocess
import tempfile
import tomllib
from collections.abc import AsyncGenerator
from pathlib import Path
from uuid import UUID, uuid4

import asyncpg
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from omnibase_core.crypto.crypto_ed25519_signer import generate_keypair
from omnibase_core.models.container.model_onex_container import ModelONEXContainer
from omnibase_core.models.envelope.model_message_envelope import ModelMessageEnvelope
from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope
from omnibase_core.models.execution_graph_replay.model_execution_graph_request import (
    ModelExecutionGraphRequest,
)
from omnibase_core.models.execution_graph_replay.model_execution_graph_terminal_refusal import (
    ModelExecutionGraphTerminalRefusal,
)
from omnibase_core.models.execution_graph_replay.model_execution_graph_terminal_result import (
    ModelExecutionGraphTerminalResult,
)
from omnibase_core.models.execution_graph_replay.model_execution_graph_topology_version import (
    ModelExecutionGraphTopologyVersion,
)
from omnibase_core.models.primitives.model_semver import ModelSemVer
from omnibase_infra.event_bus.event_bus_inmemory import EventBusInmemory
from omnibase_infra.runtime.auto_wiring.discovery import discover_contracts_from_paths
from omnibase_infra.runtime.auto_wiring.handler_wiring import (
    subscribe_wired_contract_topics,
    wire_from_manifest,
)
from omnibase_infra.runtime.auto_wiring.models.model_contract_version import (
    ModelContractVersion,
)
from omnibase_infra.runtime.auto_wiring.models.model_discovered_contract import (
    ModelDiscoveredContract,
)
from omnibase_infra.runtime.auto_wiring.models.model_event_bus_wiring import (
    ModelEventBusWiring,
)
from omnibase_infra.runtime.dispatch_envelope_context import (
    bind_execution_graph_read_authority,
)
from omnibase_infra.runtime.execution_graph_ownership import (
    ExecutionGraphOwnershipAdmission,
)
from omnibase_infra.runtime.execution_graph_read_authority import (
    TrustedExecutionGraphGatewayPolicy,
    TrustedGatewaySignerScope,
    VerifiedExecutionGraphReadAuthority,
    verify_signed_execution_graph_read_authority,
)
from omnibase_infra.runtime.execution_graph_read_command_handler import (
    ExecutionGraphReadCommandExecutor,
)
from omnibase_infra.runtime.execution_graph_read_composition import (
    compose_execution_graph_read_executor,
    open_execution_graph_read_pools,
)
from omnibase_infra.runtime.execution_graph_runtime_composition import (
    GRAPH_READ_CONTRACT_NAME,
    GRAPH_READ_WORKFLOW_TYPE,
    open_execution_graph_runtime_executor,
    select_execution_graph_contract,
)
from omnibase_infra.runtime.execution_graph_topology_registry import (
    PackagedExecutionGraphTopologyContract,
)
from omnibase_infra.runtime.message_dispatch_engine import MessageDispatchEngine
from omnibase_infra.runtime.models.model_execution_graph_read_databases import (
    ModelExecutionGraphReadDatabases,
)
from omnibase_infra.runtime.models.model_execution_graph_read_runtime_config import (
    ModelExecutionGraphReadRuntimeConfig,
)
from omnibase_infra.runtime.models.model_execution_graph_terminal_publisher_config import (
    ModelExecutionGraphTerminalPublisherConfig,
)
from omnibase_infra.runtime.models.model_execution_graph_trusted_gateway_config import (
    ModelExecutionGraphTrustedGatewayConfig,
)
from omnibase_infra.runtime.runtime_host_process import RuntimeHostProcess
from omnibase_infra.runtime.service_kernel import _build_runtime_handler_dependencies
from omnibase_infra.utils.util_topic_event_type import derive_event_type_from_topic
from tests.helpers.projection_tenant_authority import InMemoryKeyProvider

_REPO_ROOT = Path(__file__).resolve().parents[3]
_BASE_MIGRATION = _REPO_ROOT / "docker/migrations/forward/044_create_event_ledger.sql"
_WATERMARK_MIGRATION = (
    _REPO_ROOT / "docker/migrations/forward/109_add_event_ledger_ingest_watermark.sql"
)
_WORKFLOW_TYPE = "delegation_execution_graph_read"


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


@pytest.fixture
async def dual_pools(
    tmp_path: Path,
) -> AsyncGenerator[tuple[asyncpg.Pool, asyncpg.Pool], None]:
    """Start isolated local Postgres with distinct analytics and ledger databases."""
    initdb = shutil.which("initdb")
    pg_ctl = shutil.which("pg_ctl")
    if initdb is None or pg_ctl is None:
        pytest.skip("Local PostgreSQL binaries are unavailable")
    data_dir = tmp_path / "pgdata"
    socket_dir = Path(tempfile.mkdtemp(prefix="omn19728-dual-", dir="/tmp"))
    port = _free_port()
    subprocess.run(
        [initdb, "-D", str(data_dir), "--auth=trust", "--no-instructions"],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        [
            pg_ctl,
            "-D",
            str(data_dir),
            "-l",
            str(tmp_path / "postgres.log"),
            "-o",
            f"-h 127.0.0.1 -k {socket_dir} -p {port}",
            "-w",
            "start",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    dsn_prefix = f"postgresql://{getpass.getuser()}@127.0.0.1:{port}"
    config = ModelExecutionGraphReadDatabases(
        analytics_dsn=f"{dsn_prefix}/omn19728_analytics",
        ledger_dsn=f"{dsn_prefix}/omn19728_ledger",
    )
    admin_pool = await asyncpg.create_pool(
        database="postgres", user=getpass.getuser(), host="127.0.0.1", port=port
    )
    analytics_pool: asyncpg.Pool | None = None
    ledger_pool: asyncpg.Pool | None = None
    try:
        async with admin_pool.acquire() as admin:
            await admin.execute("CREATE DATABASE omn19728_analytics")
            await admin.execute("CREATE DATABASE omn19728_ledger")
        analytics_pool = await asyncpg.create_pool(
            dsn=config.analytics_dsn.get_secret_value()
        )
        ledger_pool = await asyncpg.create_pool(
            dsn=config.ledger_dsn.get_secret_value()
        )
        async with analytics_pool.acquire() as analytics:
            await analytics.execute(
                "CREATE SCHEMA omn19728_test_guard; "
                "CREATE TABLE omn19728_test_guard.disposable_instance (id INT)"
            )
            await analytics.execute(
                "CREATE TABLE public.delegation_events "
                "(correlation_id TEXT UNIQUE NOT NULL, tenant_id UUID NOT NULL)"
            )
        async with ledger_pool.acquire() as ledger:
            await ledger.execute(
                "CREATE SCHEMA omn19728_test_guard; "
                "CREATE TABLE omn19728_test_guard.disposable_instance (id INT)"
            )
            await ledger.execute(_BASE_MIGRATION.read_text(encoding="utf-8"))
            await ledger.execute(_WATERMARK_MIGRATION.read_text(encoding="utf-8"))
        await analytics_pool.close()
        analytics_pool = None
        await ledger_pool.close()
        ledger_pool = None
        async with open_execution_graph_read_pools(config) as pools:
            yield pools
    finally:
        if analytics_pool is not None:
            await analytics_pool.close()
        if ledger_pool is not None:
            await ledger_pool.close()
        await admin_pool.close()
        subprocess.run(
            [pg_ctl, "-D", str(data_dir), "-m", "immediate", "-w", "stop"],
            check=True,
            capture_output=True,
            text=True,
        )
        socket_dir.rmdir()


def _authority(
    tenant_id: UUID, correlation_id: UUID
) -> VerifiedExecutionGraphReadAuthority:
    scope = TrustedGatewaySignerScope("trusted-gateway", "test", "graph-read")
    keys = generate_keypair()
    inner = ModelEventEnvelope[dict[str, object]](
        tenant_id=str(tenant_id),
        correlation_id=correlation_id,
        metadata={"tags": {"workflow_id": str(uuid4())}},
        payload={"correlation_id": str(correlation_id), "cursor_mode": "latest"},
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
    return verify_signed_execution_graph_read_authority(
        signed,
        InMemoryKeyProvider({scope.runtime_id: keys.public_key_bytes}),
        TrustedExecutionGraphGatewayPolicy(frozenset({scope})),
    )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_owner_is_read_from_analytics_before_ledger_evidence(
    dual_pools: tuple[asyncpg.Pool, asyncpg.Pool],
) -> None:
    analytics_pool, ledger_pool = dual_pools
    async with analytics_pool.acquire() as analytics:
        async with ledger_pool.acquire() as ledger:
            analytics_marker = await analytics.fetchval(
                "SELECT to_regclass('omn19728_test_guard.disposable_instance') "
                "IS NOT NULL"
            )
            ledger_marker = await ledger.fetchval(
                "SELECT to_regclass('omn19728_test_guard.disposable_instance') "
                "IS NOT NULL"
            )
            if analytics_marker is not True or ledger_marker is not True:
                raise ValueError("both graph test databases need disposable markers")
            assert await analytics.fetchval("SELECT current_database()") != (
                await ledger.fetchval("SELECT current_database()")
            )

    tenant_id = uuid4()
    correlation_id = uuid4()
    envelope_id = uuid4()
    topic = "onex.cmd.omnimarket.delegate-skill.v1"
    async with analytics_pool.acquire() as analytics:
        await analytics.execute(
            "INSERT INTO public.delegation_events (correlation_id, tenant_id) "
            "VALUES ($1, $2)",
            str(correlation_id),
            tenant_id,
        )
    async with ledger_pool.acquire() as ledger:
        append_result = await ledger.fetchrow(
            "SELECT * FROM public.append_event_ledger_with_watermark("
            "$1, 0, 1, NULL, $2, '{}'::jsonb, $3, $4, NULL, NULL, NULL)",
            topic,
            json.dumps({"tenant_id": str(tenant_id)}).encode(),
            envelope_id,
            correlation_id,
        )
        assert append_result is not None
        assert append_result["ingest_watermark"] == 1
        assert append_result["duplicate"] is False

    topology = PackagedExecutionGraphTopologyContract().resolve(
        ModelExecutionGraphTopologyVersion(
            contract_version=ModelSemVer(major=1, minor=3, patch=0),
            topology_sha256=(
                "0505ab0b163492380739a15646c0442a3ecfb54efb0acb53bc23d232640fbfd3"
            ),
        )
    )
    calls: list[str] = []

    class StoredReader:
        async def read_current(self, *_args: object) -> tuple[()]:
            calls.append("stored")
            return ()

    def stored_factory(pool: object) -> StoredReader:
        assert pool is ledger_pool
        return StoredReader()

    async def fold(
        request: ModelExecutionGraphRequest,
        authority: VerifiedExecutionGraphReadAuthority,
        _topology: object,
        admission: ExecutionGraphOwnershipAdmission,
        _stored: object,
    ) -> ModelExecutionGraphTerminalResult:
        calls.append("fold")
        assert request == authority.request
        assert admission.owner.correlation_id == correlation_id
        assert len(admission.owned_rows) == 1
        return ModelExecutionGraphTerminalResult(
            workflow_id=authority.workflow_id,
            tenant_id=authority.tenant_id,
            correlation_id=authority.correlation_id,
            workflow_type=_WORKFLOW_TYPE,
            status="failed",
            refusal=ModelExecutionGraphTerminalRefusal(
                code="fixture", message="fixture terminal"
            ),
        )

    async def publish(
        _authority: VerifiedExecutionGraphReadAuthority,
        terminal: ModelExecutionGraphTerminalResult,
    ) -> None:
        calls.append("publish:" + (terminal.refusal.code if terminal.refusal else ""))

    executor = compose_execution_graph_read_executor(
        analytics_pool=analytics_pool,
        ledger_pool=ledger_pool,
        topology=topology,
        workflow_type=_WORKFLOW_TYPE,
        fold=fold,
        stored_chain_reader_factory=stored_factory,
        publish_terminal=publish,
    )
    authority = _authority(tenant_id, correlation_id)
    with bind_execution_graph_read_authority(authority):
        await executor.handle(authority.request)
    assert calls == ["stored", "fold", "publish:fixture"]

    # An unauthorized request must finish before any ledger query. Removing the
    # disposable ledger table makes an accidental read fail this test.
    async with ledger_pool.acquire() as ledger:
        await ledger.execute("DROP TABLE public.event_ledger")
    foreign = _authority(uuid4(), correlation_id)
    with bind_execution_graph_read_authority(foreign):
        refusal = await executor.handle(foreign.request)
    assert refusal.refusal is not None
    assert refusal.refusal.code == "not_found"
    assert calls == ["stored", "fold", "publish:fixture", "publish:not_found"]


@pytest.mark.integration
@pytest.mark.asyncio
async def test_boot_composition_reads_two_databases_and_signs_fold_terminal(
    dual_pools: tuple[asyncpg.Pool, asyncpg.Pool], tmp_path: Path
) -> None:
    """Exercise the real boot seam with disposable databases and a test signer."""
    analytics_pool, ledger_pool = dual_pools
    tenant_id, correlation_id, envelope_id = uuid4(), uuid4(), uuid4()
    topic = "onex.cmd.omnimarket.delegate-skill.v1"
    async with analytics_pool.acquire() as analytics:
        port = await analytics.fetchval("SELECT inet_server_port()")
        await analytics.execute(
            "INSERT INTO public.delegation_events (correlation_id, tenant_id) "
            "VALUES ($1, $2)",
            str(correlation_id),
            tenant_id,
        )
    async with ledger_pool.acquire() as ledger:
        await ledger.execute(
            "CREATE TABLE public.ledger_chain ("
            "correlation_id TEXT, envelope_id TEXT, hop_index INT, "
            "replay_green BOOLEAN, verifier_verdict TEXT)"
        )
        await ledger.fetchrow(
            "SELECT * FROM public.append_event_ledger_with_watermark("
            "$1, 0, 1, NULL, $2, '{}'::jsonb, $3, $4, NULL, NULL, NULL)",
            topic,
            json.dumps({"tenant_id": str(tenant_id)}).encode(),
            envelope_id,
            correlation_id,
        )

    signer = Ed25519PrivateKey.generate()
    private_path = tmp_path / "graph-terminal.pem"
    private_path.write_bytes(
        signer.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )
    )
    private_path.chmod(0o600)
    gateway_keys = generate_keypair()
    public_path = tmp_path / "gateway-keys.json"
    public_path.write_text(
        json.dumps(
            {
                "keys": {
                    "trusted-gateway": base64.urlsafe_b64encode(
                        gateway_keys.public_key_bytes
                    ).decode()
                }
            }
        ),
        encoding="utf-8",
    )
    public_path.chmod(0o600)
    databases = ModelExecutionGraphReadDatabases(
        analytics_dsn=f"postgresql://{getpass.getuser()}@127.0.0.1:{port}/omn19728_analytics",
        ledger_dsn=f"postgresql://{getpass.getuser()}@127.0.0.1:{port}/omn19728_ledger",
    )
    topology_version = ModelExecutionGraphTopologyVersion(
        contract_version=ModelSemVer(major=1, minor=3, patch=0),
        topology_sha256="0505ab0b163492380739a15646c0442a3ecfb54efb0acb53bc23d232640fbfd3",
    )
    terminal_topic = (
        "onex.evt.omnibase-infra.delegation-execution-graph-read-terminal.v1"
    )
    command_topic = "onex.cmd.omnibase-infra.delegation-execution-graph-requested.v1"
    config = ModelExecutionGraphReadRuntimeConfig(
        databases=databases,
        topology_version=topology_version,
        terminal_publisher=ModelExecutionGraphTerminalPublisherConfig(
            terminal_topic=terminal_topic,
            runtime_id="test-graph-runtime",
            realm="test",
            bus_id="graph-read",
            workflow_type=GRAPH_READ_WORKFLOW_TYPE,
        ),
        private_key_path=private_path,
    )
    gateway = ModelExecutionGraphTrustedGatewayConfig(
        command_topic=command_topic,
        runtime_id="trusted-gateway",
        realm="test",
        bus_id="graph-read",
        public_key_path=public_path,
    )
    project = tomllib.loads((_REPO_ROOT / "pyproject.toml").read_text())
    module_name = project["project"]["entry-points"]["onex.nodes"][
        GRAPH_READ_CONTRACT_NAME
    ]
    contract_module = importlib.import_module(module_name)
    assert contract_module.__file__ is not None
    contract_path = Path(contract_module.__file__).parent / "contract.yaml"
    discovered = discover_contracts_from_paths([contract_path])
    enabled, contract = select_execution_graph_contract(discovered, enabled=True)
    assert contract is not None
    assert contract.name == GRAPH_READ_CONTRACT_NAME

    bus = EventBusInmemory(environment="test", group="graph-dual-database")
    await bus.start()
    terminal_messages: list[ModelMessageEnvelope[dict[str, object]]] = []

    async def capture_terminal(message: object) -> None:
        raw = message.value
        terminal_messages.append(
            ModelMessageEnvelope[dict[str, object]].model_validate_json(raw)
        )

    await bus.subscribe(terminal_topic, on_message=capture_terminal, group_id="capture")
    runtime_host = RuntimeHostProcess(
        config={
            "service_name": "omnibase-infra",
            "node_name": "graph-read",
            "execution_graph_read_gateway": gateway.model_dump(mode="json"),
        },
        event_bus=bus,
    )
    ingress, gateway_provider = (
        runtime_host._execution_graph_read_ingress_dependencies()
    )
    assert ingress is not None and gateway_provider is not None
    container = ModelONEXContainer()
    engine = MessageDispatchEngine()
    async with open_execution_graph_runtime_executor(
        config=config,
        gateway=gateway,
        contract=contract,
        event_bus=bus,
    ) as executor:
        dependencies = _build_runtime_handler_dependencies(
            None,
            execution_graph_read_executor=executor,
            execution_graph_read_container=container,
        )
        assert dependencies is not None
        wiring_report = await wire_from_manifest(
            manifest=enabled,
            dispatch_engine=engine,
            event_bus=bus,
            environment="test",
            container=container,
            subscribe_immediately=False,
            materialized_explicit_dependencies=dependencies,
        )
        assert wiring_report.total_failed == 0
        engine.freeze()
        subscriptions = await subscribe_wired_contract_topics(
            manifest=enabled,
            report=wiring_report,
            dispatch_engine=engine,
            event_bus=bus,
            environment="test",
            execution_graph_read_ingress=ingress,
            execution_graph_read_key_provider=gateway_provider,
        )
        assert subscriptions == {GRAPH_READ_CONTRACT_NAME: (command_topic,)}

        workflow_id = uuid4()
        inner = ModelEventEnvelope[dict[str, object]](
            tenant_id=str(tenant_id),
            correlation_id=correlation_id,
            event_type=derive_event_type_from_topic(command_topic),
            metadata={"tags": {"workflow_id": str(workflow_id)}},
            payload=ModelExecutionGraphRequest(
                correlation_id=correlation_id, cursor_mode="latest"
            ).model_dump(mode="json"),
        ).model_dump(mode="json")
        command = ModelMessageEnvelope[dict[str, object]].create_signed(
            realm=gateway.realm,
            runtime_id=gateway.runtime_id,
            bus_id=gateway.bus_id,
            trace_id=correlation_id,
            tenant_id=str(tenant_id),
            payload=inner,
            private_key=gateway_keys.private_key_bytes,
        )
        await bus.publish(command_topic, None, command.model_dump_json().encode())

        # A later-arriving row with a lower Kafka offset must not alter the
        # replay selected by the cursor vector returned by the first read.
        initial = terminal_messages[-1].payload["result"]
        assert isinstance(initial, dict)
        initial_replay = initial["replay"]
        assert isinstance(initial_replay, dict)
        initial_cursors = initial_replay["source_cursors"]
        assert isinstance(initial_cursors, list)
        async with ledger_pool.acquire() as ledger:
            late_append = await ledger.fetchrow(
                "SELECT * FROM public.append_event_ledger_with_watermark("
                "$1, 0, 0, NULL, $2, '{}'::jsonb, $3, $4, NULL, NULL, NULL)",
                topic,
                json.dumps({"tenant_id": str(tenant_id)}).encode(),
                envelope_id,
                correlation_id,
            )
        assert late_append is not None
        assert late_append["ingest_watermark"] == 2
        bounded_workflow_id = uuid4()
        bounded_inner = ModelEventEnvelope[dict[str, object]](
            tenant_id=str(tenant_id),
            correlation_id=correlation_id,
            event_type=derive_event_type_from_topic(command_topic),
            metadata={"tags": {"workflow_id": str(bounded_workflow_id)}},
            payload={
                "correlation_id": str(correlation_id),
                "cursor_mode": "bounded",
                "source_cursors": initial_cursors,
            },
        ).model_dump(mode="json")
        bounded_command = ModelMessageEnvelope[dict[str, object]].create_signed(
            realm=gateway.realm,
            runtime_id=gateway.runtime_id,
            bus_id=gateway.bus_id,
            trace_id=correlation_id,
            tenant_id=str(tenant_id),
            payload=bounded_inner,
            private_key=gateway_keys.private_key_bytes,
        )
        await bus.publish(
            command_topic, None, bounded_command.model_dump_json().encode()
        )

        latest_workflow_id = uuid4()
        latest_inner = ModelEventEnvelope[dict[str, object]](
            tenant_id=str(tenant_id),
            correlation_id=correlation_id,
            event_type=derive_event_type_from_topic(command_topic),
            metadata={"tags": {"workflow_id": str(latest_workflow_id)}},
            payload=ModelExecutionGraphRequest(
                correlation_id=correlation_id, cursor_mode="latest"
            ).model_dump(mode="json"),
        ).model_dump(mode="json")
        latest_command = ModelMessageEnvelope[dict[str, object]].create_signed(
            realm=gateway.realm,
            runtime_id=gateway.runtime_id,
            bus_id=gateway.bus_id,
            trace_id=correlation_id,
            tenant_id=str(tenant_id),
            payload=latest_inner,
            private_key=gateway_keys.private_key_bytes,
        )
        await bus.publish(
            command_topic, None, latest_command.model_dump_json().encode()
        )

        # Reuse becomes visible only after the original cursor. Current-state
        # authorization must still refuse it rather than hiding the new head
        # behind the caller's older replay bounds.
        foreign_tenant = uuid4()
        async with ledger_pool.acquire() as ledger:
            foreign_append = await ledger.fetchrow(
                "SELECT * FROM public.append_event_ledger_with_watermark("
                "$1, 0, 2, NULL, $2, '{}'::jsonb, $3, $4, NULL, NULL, NULL)",
                topic,
                json.dumps({"tenant_id": str(foreign_tenant)}).encode(),
                uuid4(),
                correlation_id,
            )
        assert foreign_append is not None
        assert foreign_append["ingest_watermark"] == 3
        assert len(terminal_messages) == 3
        refusals: list[object] = []
        for caller, requested_correlation in (
            (tenant_id, uuid4()),  # Unknown correlation.
            (foreign_tenant, correlation_id),  # Another tenant's binding.
            (tenant_id, correlation_id),  # Own binding, reused correlation.
        ):
            refused_workflow = uuid4()
            refused_inner = ModelEventEnvelope[dict[str, object]](
                tenant_id=str(caller),
                correlation_id=requested_correlation,
                event_type=derive_event_type_from_topic(command_topic),
                metadata={"tags": {"workflow_id": str(refused_workflow)}},
                payload={
                    "correlation_id": str(requested_correlation),
                    "cursor_mode": "bounded",
                    "source_cursors": initial_cursors,
                },
            ).model_dump(mode="json")
            refused_command = ModelMessageEnvelope[dict[str, object]].create_signed(
                realm=gateway.realm,
                runtime_id=gateway.runtime_id,
                bus_id=gateway.bus_id,
                trace_id=requested_correlation,
                tenant_id=str(caller),
                payload=refused_inner,
                private_key=gateway_keys.private_key_bytes,
            )
            await bus.publish(
                command_topic, None, refused_command.model_dump_json().encode()
            )
            refused_terminal = terminal_messages[-1]
            assert refused_terminal.payload["status"] == "failed"
            assert refused_terminal.payload["workflow_id"] == str(refused_workflow)
            assert refused_terminal.payload["correlation_id"] == str(
                requested_correlation
            )
            assert refused_terminal.payload["result"] is None
            assert refused_terminal.verify_signature(
                InMemoryKeyProvider(
                    {"test-graph-runtime": signer.public_key().public_bytes_raw()}
                )
            )
            refusals.append(refused_terminal.payload["refusal"])
        assert refusals[0] == refusals[1] == refusals[2]
        assert isinstance(refusals[0], dict)
        assert refusals[0]["code"] == "not_found"

    assert len(terminal_messages) == 6
    signed = terminal_messages[0]
    assert signed.runtime_id == "test-graph-runtime"
    assert signed.payload["workflow_type"] == GRAPH_READ_WORKFLOW_TYPE
    assert signed.payload["status"] == "completed"
    assert signed.payload["workflow_id"] == str(workflow_id)
    assert signed.payload["tenant_id"] == str(tenant_id)
    assert signed.payload["correlation_id"] == str(correlation_id)
    result = signed.payload["result"]
    assert isinstance(result, dict)
    replay = result["replay"]
    assert isinstance(replay, dict)
    nodes = replay["nodes"]
    assert isinstance(nodes, list)
    assert tuple(node["id"] for node in nodes if isinstance(node, dict)) == (
        str(envelope_id),
    )
    cursors = replay["source_cursors"]
    assert isinstance(cursors, list) and cursors[0]["max_ingest_watermark"] == 1
    assert signed.verify_signature(
        InMemoryKeyProvider(
            {"test-graph-runtime": signer.public_key().public_bytes_raw()}
        )
    )
    bounded_payload = terminal_messages[1].payload["result"]
    assert isinstance(bounded_payload, dict)
    bounded_replay = bounded_payload["replay"]
    assert isinstance(bounded_replay, dict)
    assert bounded_replay == replay
    bounded_terminal = terminal_messages[1]
    assert bounded_terminal.payload["status"] == "completed"
    assert bounded_terminal.payload["workflow_id"] == str(bounded_workflow_id)
    assert bounded_terminal.payload["correlation_id"] == str(correlation_id)
    assert bounded_terminal.verify_signature(
        InMemoryKeyProvider(
            {"test-graph-runtime": signer.public_key().public_bytes_raw()}
        )
    )

    newest_terminal = terminal_messages[2]
    assert newest_terminal.payload["status"] == "completed"
    assert newest_terminal.payload["workflow_id"] == str(latest_workflow_id)
    assert newest_terminal.payload["correlation_id"] == str(correlation_id)
    assert newest_terminal.verify_signature(
        InMemoryKeyProvider(
            {"test-graph-runtime": signer.public_key().public_bytes_raw()}
        )
    )
    newest_result = newest_terminal.payload["result"]
    assert isinstance(newest_result, dict)
    newest_replay = newest_result["replay"]
    assert isinstance(newest_replay, dict)
    newest_cursors = newest_replay["source_cursors"]
    assert isinstance(newest_cursors, list)
    assert newest_cursors[0]["max_ingest_watermark"] == 2
    newest_nodes = newest_replay["nodes"]
    assert isinstance(newest_nodes, list)
    assert tuple(node["id"] for node in newest_nodes if isinstance(node, dict)) == (
        str(envelope_id),
    )
    await bus.shutdown()
