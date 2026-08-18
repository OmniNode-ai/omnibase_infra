# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Integration tests for runtime-owned handler dependency injection.

Verifies that the service_kernel bootstrap path correctly threads a postgres_pool
into HandlerPostgresRuntimeManifestInsert via materialized_explicit_dependencies
when ServiceRegistration.postgres_pool is non-None.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

from omnibase_infra.runtime.auto_wiring.discovery import discover_contracts_from_paths
from omnibase_infra.runtime.auto_wiring.handler_wiring import wire_from_manifest
from omnibase_infra.runtime.auto_wiring.models import (
    ModelAutoWiringManifest,
    ModelContractVersion,
    ModelDiscoveredContract,
    ModelEventBusWiring,
    ModelHandlerRef,
    ModelHandlerRouting,
    ModelHandlerRoutingEntry,
)
from omnibase_infra.runtime.service_kernel import _build_runtime_handler_dependencies

_GATEWAY_RESOLVER_CONFIG = """\
enable_convention_fallback: false
mappings:
  - logical_name: gateway.attach.keycloak.issuer
    source: {source_type: env, source_path: TEST_ISSUER}
  - logical_name: gateway.attach.keycloak.introspection
    source: {source_type: env, source_path: TEST_INTROSPECTION}
  - logical_name: gateway.attach.keycloak.admin_client_credentials.client_id
    source: {source_type: env, source_path: TEST_CLIENT_ID}
  - logical_name: gateway.attach.keycloak.admin_client_credentials.client_secret
    source: {source_type: env, source_path: TEST_CLIENT_SECRET}
  - logical_name: gateway.attach.keycloak.jwks
    source: {source_type: env, source_path: TEST_JWKS}
"""


def _write_gateway_resolver_config(tmp_path: Path) -> Path:
    config_path = tmp_path / "secret_resolver.yaml"
    config_path.write_text(_GATEWAY_RESOLVER_CONFIG, encoding="utf-8")
    return config_path


def _make_pool_backed_contract(
    *,
    name: str = "node_manifest_insert",
    handler_name: str = "HandlerPostgresRuntimeManifestInsert",
) -> ModelDiscoveredContract:
    return ModelDiscoveredContract(
        name=name,
        node_type="EFFECT_GENERIC",
        contract_version=ModelContractVersion(major=1, minor=0, patch=0),
        contract_path=Path("/fake/contract.yaml"),
        entry_point_name=name,
        package_name="test-pkg",
        event_bus=ModelEventBusWiring(
            subscribe_topics=(f"onex.evt.platform.{name}.v1",),
            publish_topics=(),
        ),
        handler_routing=ModelHandlerRouting(
            routing_strategy="payload_type_match",
            handlers=(
                ModelHandlerRoutingEntry(
                    handler=ModelHandlerRef(
                        name=handler_name,
                        module="fake.handler_module",
                    ),
                    event_model=None,
                    operation=None,
                ),
            ),
        ),
    )


def _make_manifest_insert_contract() -> ModelDiscoveredContract:
    return _make_pool_backed_contract()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_postgres_pool_threaded_via_materialized_deps() -> None:
    """postgres_pool is materialized and threaded into HandlerPostgresRuntimeManifestInsert."""
    from omnibase_infra.runtime.message_dispatch_engine import (
        MessageDispatchEngine,
    )

    class HandlerPostgresRuntimeManifestInsert:
        def __init__(self, pool: object) -> None:
            self.pool = pool

        async def handle(self, envelope: object) -> None:
            return None

    fake_pool = MagicMock()
    contract = _make_manifest_insert_contract()
    manifest = ModelAutoWiringManifest(contracts=(contract,))
    engine = MessageDispatchEngine()

    with patch(
        "omnibase_infra.runtime.auto_wiring.handler_wiring._import_handler_class",
        return_value=HandlerPostgresRuntimeManifestInsert,
    ):
        report = await wire_from_manifest(
            manifest,
            engine,
            materialized_explicit_dependencies={
                "HandlerPostgresRuntimeManifestInsert": {"pool": fake_pool}
            },
        )

    assert report.total_failed == 0
    assert report.total_wired == 1


@pytest.mark.integration
@pytest.mark.asyncio
async def test_kernel_runtime_pool_dependencies_cover_baselines_batch_compute() -> None:
    """Kernel dependency map covers every runtime-owned handler that requires pool."""
    from omnibase_infra.runtime.message_dispatch_engine import (
        MessageDispatchEngine,
    )

    class PoolBackedHandler:
        def __init__(self, pool: object) -> None:
            self.pool = pool

        async def handle(self, envelope: object) -> None:
            return None

    fake_pool = MagicMock()
    manifest = ModelAutoWiringManifest(
        contracts=(
            _make_pool_backed_contract(
                name="node_manifest_insert",
                handler_name="HandlerPostgresRuntimeManifestInsert",
            ),
            _make_pool_backed_contract(
                name="node_baselines_batch_compute",
                handler_name="HandlerBaselinesBatchCompute",
            ),
        )
    )
    engine = MessageDispatchEngine()

    with patch(
        "omnibase_infra.runtime.auto_wiring.handler_wiring._import_handler_class",
        return_value=PoolBackedHandler,
    ):
        report = await wire_from_manifest(
            manifest,
            engine,
            materialized_explicit_dependencies=_build_runtime_handler_dependencies(
                fake_pool
            ),
        )

    assert report.total_failed == 0
    assert report.total_wired == 2


@pytest.mark.integration
@pytest.mark.asyncio
async def test_no_materialized_deps_when_pool_is_none() -> None:
    """When postgres_pool is None, no materialized deps map is passed (empty dict -> None)."""
    from omnibase_infra.runtime.message_dispatch_engine import (
        MessageDispatchEngine,
    )

    class HandlerPostgresRuntimeManifestInsert:
        def __init__(self) -> None:
            pass

        async def handle(self, envelope: object) -> None:
            return None

    contract = _make_manifest_insert_contract()
    manifest = ModelAutoWiringManifest(contracts=(contract,))
    engine = MessageDispatchEngine()

    with patch(
        "omnibase_infra.runtime.auto_wiring.handler_wiring._import_handler_class",
        return_value=HandlerPostgresRuntimeManifestInsert,
    ):
        # No materialized_explicit_dependencies -> handler uses zero-arg construction
        report = await wire_from_manifest(manifest, engine)

    assert report.total_failed == 0
    assert report.total_wired == 1


@pytest.mark.integration
def test_empty_runtime_manifest_dependencies_is_falsy() -> None:
    """Invariant: empty dict evaluates falsy, so `or None` yields None.

    This mirrors the kernel pattern:
        materialized_explicit_dependencies=_build_runtime_handler_dependencies(pool)
    """
    result = _build_runtime_handler_dependencies(None)
    assert result is None


@pytest.mark.integration
def test_runtime_handler_dependencies_include_pool_backed_handlers() -> None:
    """Kernel exposes explicit pool deps for all service_kernel-owned handlers."""
    fake_pool = MagicMock()
    result = _build_runtime_handler_dependencies(fake_pool)
    assert result is not None
    assert "HandlerPostgresRuntimeManifestInsert" in result
    assert "HandlerBaselinesBatchCompute" in result
    assert result["HandlerPostgresRuntimeManifestInsert"]["pool"] is fake_pool
    assert result["HandlerBaselinesBatchCompute"]["pool"] is fake_pool


@pytest.mark.integration
def test_runtime_handler_dependencies_include_dlq_replay_when_kafka_configured() -> (
    None
):
    """Kernel exposes explicit Kafka deps for the DLQ replay handler."""
    from omnibase_infra.nodes.node_dlq_replay_effect.engine_dlq_replay import (
        DLQConsumer,
        DLQProducer,
        DLQQuarantineProducer,
    )

    result = _build_runtime_handler_dependencies(
        None, kafka_bootstrap_servers="redpanda:9092"
    )

    assert result is not None
    dlq_deps = result["HandlerDlqReplay"]
    assert isinstance(dlq_deps["consumer"], DLQConsumer)
    assert isinstance(dlq_deps["producer"], DLQProducer)
    assert isinstance(dlq_deps["quarantine_producer"], DLQQuarantineProducer)
    consumer = dlq_deps["consumer"]
    assert consumer.config.bootstrap_servers == "redpanda:9092"
    assert consumer.config.dlq_topic == "onex.dlq.omnibase-infra.events.v1"


@pytest.mark.integration
def test_runtime_handler_dependencies_share_gateway_state_and_resolver(
    tmp_path: Path,
) -> None:
    """All gateway lifecycle operations use one session authority and resolver."""
    config_path = _write_gateway_resolver_config(tmp_path)

    result = _build_runtime_handler_dependencies(
        None,
        gateway_secret_resolver_config_path=config_path,
    )

    assert result is not None
    attach = result["HandlerGatewayAttach"]
    heartbeat = result["HandlerGatewayHeartbeat"]
    detach = result["HandlerGatewayDetach"]
    assert attach["config"] is heartbeat["config"] is detach["config"]
    assert attach["session_store"] is heartbeat["session_store"]
    assert attach["session_store"] is detach["session_store"]
    assert attach["secret_resolver"] is heartbeat["secret_resolver"]
    assert attach["secret_resolver"] is detach["secret_resolver"]


@pytest.mark.integration
def test_gateway_runtime_dependencies_resolve_config_from_contract(
    tmp_path: Path,
) -> None:
    """The wired gateway config carries contract.yaml values, not field defaults.

    Drives the real seam: a config.gateway_attach edit in the node's
    contract.yaml must reach the renewal builder and session policy through
    the kernel wiring. A bare ModelGatewayAttachConfig() made that edit a
    silent runtime no-op.
    """
    import omnibase_infra.nodes.node_gateway_attach_effect as gateway_attach_pkg

    contract_path = Path(str(gateway_attach_pkg.__file__)).parent / "contract.yaml"
    declared = yaml.safe_load(contract_path.read_text(encoding="utf-8"))["config"][
        "gateway_attach"
    ]

    result = _build_runtime_handler_dependencies(
        None,
        gateway_secret_resolver_config_path=_write_gateway_resolver_config(tmp_path),
    )

    assert result is not None
    config = result["HandlerGatewayAttach"]["config"]
    for field_name, declared_value in declared.items():
        assert getattr(config, field_name) == declared_value, (
            f"contract.yaml config.gateway_attach.{field_name} did not reach the "
            "wired ModelGatewayAttachConfig"
        )


@pytest.mark.integration
def test_gateway_runtime_dependencies_fail_closed_on_invalid_config(
    tmp_path: Path,
) -> None:
    """A missing deploy-rendered resolver artifact cannot silently disable auth."""
    from omnibase_infra.errors import ProtocolConfigurationError

    with pytest.raises(ProtocolConfigurationError, match="valid rendered"):
        _build_runtime_handler_dependencies(
            None,
            gateway_secret_resolver_config_path=tmp_path / "missing.yaml",
        )


@pytest.mark.integration
def test_gateway_runtime_dependencies_fail_closed_on_missing_mapping(
    tmp_path: Path,
) -> None:
    """A partial resolver artifact cannot defer an auth failure to first traffic."""
    from omnibase_infra.errors import ProtocolConfigurationError

    config_path = tmp_path / "secret_resolver.yaml"
    config_path.write_text(
        "enable_convention_fallback: false\nmappings: []\n",
        encoding="utf-8",
    )

    with pytest.raises(ProtocolConfigurationError, match="missing explicit"):
        _build_runtime_handler_dependencies(
            None,
            gateway_secret_resolver_config_path=config_path,
        )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_real_gateway_contract_wires_in_strict_mode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The shipped contract and all real handlers satisfy strict boot wiring."""
    from omnibase_infra.runtime.message_dispatch_engine import MessageDispatchEngine

    contract_path = (
        Path(__file__).parents[3]
        / "src/omnibase_infra/nodes/node_gateway_attach_effect/contract.yaml"
    )
    config_path = _write_gateway_resolver_config(tmp_path)
    monkeypatch.setenv("ONEX_WIRING_STRICT_MODE", "1")

    report = await wire_from_manifest(
        discover_contracts_from_paths([contract_path]),
        MessageDispatchEngine(),
        materialized_explicit_dependencies=_build_runtime_handler_dependencies(
            None,
            gateway_secret_resolver_config_path=config_path,
        ),
    )

    assert report.total_failed == 0
    assert report.total_wired == 1
    assert len(report.results[0].wirings) == 3


@pytest.mark.integration
@pytest.mark.asyncio
async def test_kernel_runtime_dependencies_cover_dlq_replay_handler() -> None:
    """The runtime dependency map satisfies HandlerDlqReplay constructor deps."""
    from omnibase_infra.runtime.message_dispatch_engine import (
        MessageDispatchEngine,
    )

    class HandlerDlqReplay:
        def __init__(
            self,
            *,
            consumer: object,
            producer: object,
            quarantine_producer: object,
        ) -> None:
            self.consumer = consumer
            self.producer = producer
            self.quarantine_producer = quarantine_producer

        async def handle(self, envelope: object) -> None:
            return None

    manifest = ModelAutoWiringManifest(
        contracts=(
            _make_pool_backed_contract(
                name="node_dlq_replay_effect",
                handler_name="HandlerDlqReplay",
            ),
        )
    )
    engine = MessageDispatchEngine()

    with patch(
        "omnibase_infra.runtime.auto_wiring.handler_wiring._import_handler_class",
        return_value=HandlerDlqReplay,
    ):
        report = await wire_from_manifest(
            manifest,
            engine,
            materialized_explicit_dependencies=_build_runtime_handler_dependencies(
                None, kafka_bootstrap_servers="redpanda:9092"
            ),
        )

    assert report.total_failed == 0
    assert report.total_wired == 1
