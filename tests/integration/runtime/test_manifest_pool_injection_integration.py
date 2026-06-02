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
def test_runtime_handler_dependencies_include_dlq_replay_when_kafka_configured() -> None:
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
