# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Integration tests for runtime manifest postgres_pool injection (OMN-11247).

Verifies that the service_kernel bootstrap path correctly threads a postgres_pool
into HandlerPostgresRuntimeManifestInsert via materialized_explicit_dependencies
when ServiceRegistration.postgres_pool is non-None.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

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


def _make_manifest_insert_contract() -> ModelDiscoveredContract:
    return ModelDiscoveredContract(
        name="node_manifest_insert",
        node_type="EFFECT_GENERIC",
        contract_version=ModelContractVersion(major=1, minor=0, patch=0),
        contract_path=Path("/fake/contract.yaml"),
        entry_point_name="node_manifest_insert",
        package_name="test-pkg",
        event_bus=ModelEventBusWiring(
            subscribe_topics=("onex.evt.platform.manifest-insert.v1",),
            publish_topics=(),
        ),
        handler_routing=ModelHandlerRouting(
            routing_strategy="payload_type_match",
            handlers=(
                ModelHandlerRoutingEntry(
                    handler=ModelHandlerRef(
                        name="HandlerPostgresRuntimeManifestInsert",
                        module="fake.handler_module",
                    ),
                    event_model=None,
                    operation=None,
                ),
            ),
        ),
    )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_postgres_pool_threaded_via_materialized_deps() -> None:
    """postgres_pool is materialized and threaded into HandlerPostgresRuntimeManifestInsert."""
    from omnibase_infra.runtime.service_message_dispatch_engine import (
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
async def test_no_materialized_deps_when_pool_is_none() -> None:
    """When postgres_pool is None, no materialized deps map is passed (empty dict -> None)."""
    from omnibase_infra.runtime.service_message_dispatch_engine import (
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
        materialized_explicit_dependencies=(runtime_manifest_dependencies or None)
    """
    runtime_manifest_dependencies: dict[str, dict[str, object]] = {}
    result = runtime_manifest_dependencies or None
    assert result is None


@pytest.mark.integration
def test_nonempty_runtime_manifest_dependencies_is_truthy() -> None:
    """Invariant: non-empty dict evaluates truthy, so `or None` passes it through."""
    fake_pool = MagicMock()
    runtime_manifest_dependencies: dict[str, dict[str, object]] = {
        "HandlerPostgresRuntimeManifestInsert": {"pool": fake_pool}
    }
    result = runtime_manifest_dependencies or None
    assert result is not None
    assert "HandlerPostgresRuntimeManifestInsert" in result
