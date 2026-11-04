# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Hold the five I/O categories recorded on plugin.py before extraction."""

import ast
from pathlib import Path

import pytest

PLUGIN = "src/omnibase_infra/nodes/node_registration_orchestrator/plugin.py"
CONTRACT = "src/omnibase_infra/nodes/node_registration_orchestrator/contract.yaml"
ROOT = Path(__file__).resolve().parents[4]

# Hook output on dev 2bb3d116596c, before changing plugin.py. Each original
# offender remains a positive control, even after its source line moves.
RECORDED_FINDINGS = (
    (79, "file_read", '_CONTRACT_PATH.read_text(encoding="utf-8")'),
    (95, "import_database", "import asyncpg"),
    (280, "env_read", 'os.getenv("OMNIBASE_INFRA_DB_URL")'),
    (307, "import_database", "import asyncpg"),
    (309, "clock", "time.time()"),
    (315, "env_read", 'os.getenv("OMNIBASE_INFRA_DB_URL")'),
    (388, "clock", "time.time()"),
    (399, "clock", "time.time()"),
    (564, "env_read", 'os.getenv("ONEX_PROJECTOR_CONTRACTS_DIR", "default")'),
    (677, "file_read", "schema_file.read_text()"),
    (696, "import_database", "import asyncpg.exceptions"),
    (758, "env_read", 'os.getenv("OMNIBASE_INFRA_DB_URL")'),
    (840, "direct_bus", "AIOKafkaProducer(bootstrap_servers=kafka_bootstrap_servers)"),
    (950, "clock", "time.time()"),
    (968, "clock", "time.time()"),
    (990, "clock", "time.time()"),
    (1020, "clock", "time.time()"),
    (1035, "clock", "time.time()"),
    (1055, "clock", "time.time()"),
    (1188, "clock", "time.time()"),
    (1197, "clock", "time.time()"),
    (1480, "clock", "time.time()"),
    (1541, "clock", "time.time()"),
)


def _findings(source: str) -> list[tuple[int, str]]:
    # The runtime dependency and the hook pin can carry different checker
    # versions. Keep this regression tied to the recorded hook categories,
    # including clock/env/file reads absent from older runtime checkers.
    findings: list[tuple[int, str]] = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            modules = (
                [alias.name for alias in node.names]
                if isinstance(node, ast.Import)
                else [node.module or ""]
            )
            if any(module.split(".")[0] == "asyncpg" for module in modules):
                findings.append((node.lineno, "import_database"))
        elif isinstance(node, ast.Call):
            func = node.func
            name = ast.unparse(func)
            if name in {"time.time", "time.monotonic", "time.perf_counter"}:
                findings.append((node.lineno, "clock"))
            elif name in {"os.getenv", "os.environ.get"}:
                findings.append((node.lineno, "env_read"))
            elif isinstance(func, ast.Attribute) and func.attr == "read_text":
                findings.append((node.lineno, "file_read"))
            elif name == "AIOKafkaProducer":
                findings.append((node.lineno, "direct_bus"))
    return sorted(findings)


@pytest.mark.unit
@pytest.mark.parametrize(("line", "category", "source"), RECORDED_FINDINGS)
def test_recorded_registration_plugin_offender_is_detected(
    line: int,
    category: str,
    source: str,
) -> None:
    findings = _findings(source)
    assert len(findings) == 1, (line, category, findings)
    assert findings[0][1] == category


@pytest.mark.unit
def test_registration_orchestrator_plugin_has_no_io() -> None:
    findings = _findings((ROOT / PLUGIN).read_text())
    assert not findings, findings


@pytest.mark.unit
def test_recorded_registration_plugin_categories() -> None:
    from collections import Counter

    assert Counter(category for _, category, _ in RECORDED_FINDINGS) == {
        "clock": 13,
        "env_read": 4,
        "import_database": 3,
        "file_read": 2,
        "direct_bus": 1,
    }


def _config():
    from uuid import uuid4

    from omnibase_core.models.container.model_onex_container import ModelONEXContainer
    from omnibase_infra.event_bus.event_bus_inmemory import EventBusInmemory
    from omnibase_infra.runtime.protocol_domain_plugin import ModelDomainPluginConfig

    return ModelDomainPluginConfig(
        container=ModelONEXContainer(),
        event_bus=EventBusInmemory(),
        correlation_id=uuid4(),
        input_topic="registration-input",
        output_topic="registration-output",
        consumer_group="registration-test",
    )


@pytest.mark.unit
@pytest.mark.asyncio
async def test_registration_plugin_duration_uses_injected_clock() -> None:
    from unittest.mock import Mock

    from omnibase_infra.nodes.node_registration_orchestrator.plugin import (
        ServiceRegistration,
    )
    from omnibase_infra.protocols.protocol_clock import ProtocolClock

    clock = Mock(spec=ProtocolClock)
    clock.monotonic.side_effect = [100.0, 102.5]
    result = await ServiceRegistration(clock=clock).wire_dispatchers(_config())
    assert result.success
    assert result.duration_seconds == 2.5
    assert clock.monotonic.call_count == 2
    clock.now.assert_not_called()


@pytest.mark.unit
@pytest.mark.parametrize(
    "dsn", ["", "   ", "http://localhost/db", "postgresql://localhost/"]
)
@pytest.mark.asyncio
async def test_registration_bootstrap_rejects_bad_dsn_before_pool_creation(
    monkeypatch: pytest.MonkeyPatch,
    dsn: str,
) -> None:
    from unittest.mock import AsyncMock

    from omnibase_infra.errors import ContainerWiringError
    from omnibase_infra.handlers.handler_registration_bootstrap import (
        HandlerRegistrationBootstrap,
    )

    config = _config()
    create_pool = AsyncMock()
    monkeypatch.setenv("OMNIBASE_INFRA_DB_URL", dsn)
    monkeypatch.setattr("asyncpg.create_pool", create_pool)
    with pytest.raises(ContainerWiringError):
        await HandlerRegistrationBootstrap(config.container).handle(config)
    create_pool.assert_not_awaited()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_registration_bootstrap_preserves_pool_parameters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from unittest.mock import AsyncMock, Mock

    from asyncpg import Pool

    from omnibase_infra.handlers.handler_registration_bootstrap import (
        HandlerRegistrationBootstrap,
    )

    config = _config()
    pool = Mock(spec=Pool)
    create_pool = AsyncMock(return_value=pool)
    monkeypatch.setenv("OMNIBASE_INFRA_DB_URL", " postgresql://localhost/registration ")
    monkeypatch.setattr("asyncpg.create_pool", create_pool)
    result = await HandlerRegistrationBootstrap(config.container).handle(config)
    assert result.pool is pool
    create_pool.assert_awaited_once_with(
        "postgresql://localhost/registration",
        min_size=2,
        max_size=10,
    )


@pytest.mark.unit
def test_registration_bootstrap_settings_are_resolved_at_call_time(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from omnibase_infra.handlers.handler_registration_bootstrap import (
        HandlerRegistrationBootstrap,
    )
    from omnibase_infra.nodes.node_registration_orchestrator.plugin import (
        ServiceRegistration,
    )

    config = _config()
    plugin = ServiceRegistration()
    monkeypatch.delenv("OMNIBASE_INFRA_DB_URL", raising=False)
    monkeypatch.delenv("ONEX_PROJECTOR_CONTRACTS_DIR", raising=False)
    assert not plugin.should_activate(config)
    assert HandlerRegistrationBootstrap.projector_contracts_dir(tmp_path) == tmp_path

    monkeypatch.setenv("OMNIBASE_INFRA_DB_URL", " postgresql://localhost/registration ")
    monkeypatch.setenv("ONEX_PROJECTOR_CONTRACTS_DIR", str(tmp_path / "override"))
    assert plugin.should_activate(config)
    assert (
        HandlerRegistrationBootstrap.database_url()
        == "postgresql://localhost/registration"
    )
    assert (
        HandlerRegistrationBootstrap.projector_contracts_dir(tmp_path)
        == tmp_path / "override"
    )


@pytest.mark.unit
@pytest.mark.parametrize("content", ["[", "- a", "null"])
def test_registration_contract_read_remains_best_effort(
    tmp_path: Path,
    content: str,
) -> None:
    from omnibase_infra.handlers.handler_registration_bootstrap import (
        HandlerRegistrationBootstrap,
    )

    path = tmp_path / "contract.yaml"
    assert HandlerRegistrationBootstrap.load_contract(path) == {}
    path.write_text(content)
    assert HandlerRegistrationBootstrap.load_contract(path) == {}
    path.write_text("published_events: []")
    assert HandlerRegistrationBootstrap.load_contract(path) == {"published_events": []}


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize("start_fails", [False, True])
async def test_registration_snapshot_effect_preserves_auth_and_failed_start_cleanup(
    monkeypatch: pytest.MonkeyPatch,
    start_fails: bool,
) -> None:
    from unittest.mock import AsyncMock, Mock

    from omnibase_infra.nodes.node_registration_orchestrator.plugin import (
        ServiceRegistration,
    )
    from omnibase_infra.projectors.snapshot_publisher_registration import (
        SnapshotPublisherRegistration,
    )

    config = _config()
    config.kafka_bootstrap_servers = "localhost:19092"
    from aiokafka import AIOKafkaProducer

    producer = Mock(spec=AIOKafkaProducer)
    producer.stop = AsyncMock()
    producer_cls = Mock(return_value=producer)
    publisher = Mock(spec=SnapshotPublisherRegistration)
    publisher.start = AsyncMock(
        side_effect=RuntimeError("start failed") if start_fails else None
    )
    publisher_cls = Mock(return_value=publisher)
    monkeypatch.setattr("aiokafka.AIOKafkaProducer", producer_cls)
    monkeypatch.setattr(
        "omnibase_infra.event_bus.kafka_auth.build_aiokafka_auth_kwargs_from_env",
        lambda: {"security_protocol": "PLAINTEXT"},
    )
    monkeypatch.setattr(
        "omnibase_infra.projectors.snapshot_publisher_registration.SnapshotPublisherRegistration",
        publisher_cls,
    )
    plugin = ServiceRegistration()
    await plugin._initialize_snapshot_publisher(config)
    producer_cls.assert_called_once_with(
        bootstrap_servers="localhost:19092",
        security_protocol="PLAINTEXT",
    )
    publisher.start.assert_awaited_once()
    if start_fails:
        producer.stop.assert_awaited_once()
        assert plugin.snapshot_publisher is None
    else:
        producer.stop.assert_not_awaited()
        assert plugin.snapshot_publisher is publisher


@pytest.mark.unit
@pytest.mark.asyncio
async def test_registration_effect_contracts_resolve_real_handlers_and_boot_manifest() -> (
    None
):
    from importlib import import_module

    from omnibase_core.models.resolver.model_handler_resolver_context import (
        ModelHandlerResolverContext,
    )
    from omnibase_core.services.service_handler_resolver import ServiceHandlerResolver
    from omnibase_infra.runtime.auto_wiring.discovery import _parse_contract
    from omnibase_infra.runtime.auto_wiring.handler_wiring import wire_from_manifest
    from omnibase_infra.runtime.auto_wiring.models import ModelAutoWiringManifest
    from omnibase_infra.runtime.message_dispatch_engine import MessageDispatchEngine

    config = _config()
    contracts = tuple(
        _parse_contract(
            contract_path=ROOT / f"src/omnibase_infra/nodes/{name}/contract.yaml",
            entry_point_name=name,
            package_name="omnibase_infra",
            package_version="0.0.0",
        )
        for name in (
            "node_registration_bootstrap_effect",
            "node_event_bus_wiring_effect",
        )
    )
    expected = {"HandlerRegistrationBootstrap", "HandlerRegistrationSnapshotPublisher"}
    resolved = set()
    for contract in contracts:
        for entry in contract.handler_routing.handlers:
            handler_cls = getattr(
                import_module(entry.handler.module), entry.handler.name
            )
            resolution = ServiceHandlerResolver().resolve(
                ModelHandlerResolverContext(
                    handler_cls=handler_cls,
                    handler_module=entry.handler.module,
                    handler_name=entry.handler.name,
                    contract_name=contract.name,
                    node_name=contract.name,
                    container=config.container,
                )
            )
            assert isinstance(resolution.handler_instance, handler_cls)
            resolved.add(entry.handler.name)
    assert resolved == expected
    # These effects run before dispatch is wired and declare no subscriptions.
    # Use the kernel's wiring entrypoint and deferred-subscription mode anyway,
    # so their presence in discovery cannot break the runtime boot manifest.
    report = await wire_from_manifest(
        manifest=ModelAutoWiringManifest(contracts=contracts),
        dispatch_engine=MessageDispatchEngine(),
        event_bus=config.event_bus,
        environment="dev",
        container=config.container,
        subscribe_immediately=False,
        result_appliers_by_contract={},
        materialized_explicit_dependencies={},
        topology=None,
    )
    assert report.total_failed == 0


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize("handler_name", ["bootstrap", "snapshot_publisher"])
@pytest.mark.parametrize("injected", [False, True])
async def test_registration_effect_binding_respects_container_injection(
    handler_name: str,
    injected: bool,
) -> None:
    from importlib import import_module
    from unittest.mock import Mock

    config = _config()
    module = import_module(
        f"omnibase_infra.handlers.handler_registration_{handler_name}"
    )
    class_name = "HandlerRegistration" + "".join(
        part.title() for part in handler_name.split("_")
    )
    handler_cls = getattr(module, class_name)
    if injected:
        instance = Mock(spec=handler_cls)
        registry = config.container.service_registry
        assert registry is not None
        await registry.register_instance(
            interface=handler_cls,
            instance=instance,
        )
    first = await handler_cls.from_container(config.container)
    second = await handler_cls.from_container(config.container)
    assert first is second
    assert await config.container.get_service_async(handler_cls) is first
    if injected:
        assert first is instance
    else:
        assert isinstance(first, handler_cls)
