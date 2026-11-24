# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-26594: a lazy start wires a handler with the constructor dependencies the boot built.

The dev runtime of 2026-11-24T00:37:39Z logged ``did not wake for a message`` for
``node_dlq_replay_effect`` (``HandlerDlqReplay requires constructor parameters ['consumers',
'producer', 'quarantine_producer']``) and for ``node_runtime_manifest_reducer``
(``HandlerPostgresRuntimeManifestInsert requires constructor parameters ['pool']``), four attempts
per message. The kernel passes ``materialized_explicit_dependencies`` to ``wire_from_manifest`` for the
boot set; ``LazyStartRuntimeTransaction.build`` passed none, so a dormant contract's handler had no
provider for its pool or its producers.

The kernel records the map it built, the dormant store keeps it, the ingress hands it to the
transaction and the transaction hands it to ``wire_from_manifest``. The real contract is read from disk.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast
from unittest.mock import MagicMock

import pytest

from omnibase_core.container import ModelONEXContainer
from omnibase_core.protocols.event_bus.protocol_event_bus_subscriber import (
    ProtocolEventBusSubscriber,
)
from omnibase_infra.event_bus.model_topic_readiness_config import (
    ModelTopicReadinessConfig,
)
from omnibase_infra.handlers import handler_lazy_start_ingress as ingress_module
from omnibase_infra.handlers.handler_lazy_start_ingress import HandlerLazyStartIngress
from omnibase_infra.handlers.lazy_start_boot_split import StoreLazyStartDormant
from omnibase_infra.handlers.lazy_start_runtime_transaction import (
    LazyStartRuntimeTransaction,
)
from omnibase_infra.models.lazy_start import ModelLazyStartPolicy
from omnibase_infra.protocols.protocol_lazy_start_bus import ProtocolLazyStartBus
from omnibase_infra.runtime.auto_wiring.discovery import (
    discover_contracts_from_paths,
)
from omnibase_infra.runtime.service_kernel import _build_runtime_handler_dependencies

pytestmark = pytest.mark.unit

NAME = "node_runtime_manifest_reducer"
CONTRACT = (
    Path(__file__).resolve().parents[3]
    / "src/omnibase_infra/nodes"
    / NAME
    / "contract.yaml"
)


@pytest.fixture(autouse=True)
def _clean_store() -> Any:
    StoreLazyStartDormant.clear()
    yield
    StoreLazyStartDormant.clear()


def _transaction(
    dependencies: dict[str, dict[str, object]] | None,
) -> LazyStartRuntimeTransaction:
    manifest = discover_contracts_from_paths([CONTRACT])
    assert manifest.total_errors == 0
    contract = next(c for c in manifest.contracts if c.name == NAME)
    return LazyStartRuntimeTransaction(
        {NAME: contract},
        cast("ProtocolLazyStartBus", MagicMock()),
        "local",
        None,
        ModelONEXContainer(),
        ModelTopicReadinessConfig(),
        handler_dependencies=dependencies,
    )


@pytest.mark.asyncio
async def test_build_wires_the_real_contract_with_the_kernels_dependency_map() -> None:
    kernel_map = _build_runtime_handler_dependencies(MagicMock(name="postgres_pool"))
    assert kernel_map is not None

    transaction = _transaction(kernel_map)

    await transaction.build(
        NAME
    )  # raised "requires constructor parameters ['pool']" before OMN-26594
    assert transaction.node_engine(NAME) is not None


@pytest.mark.asyncio
async def test_build_without_the_dependency_map_still_refuses_the_contract() -> None:
    transaction = _transaction(None)

    with pytest.raises(RuntimeError, match="did not wire"):
        await transaction.build(NAME)


def test_the_kernels_dependency_builder_records_its_map_for_the_dormant_contracts() -> (
    None
):
    built = _build_runtime_handler_dependencies(MagicMock(name="postgres_pool"))

    assert built is not None
    assert StoreLazyStartDormant.handler_dependencies() is built
    assert "HandlerPostgresRuntimeManifestInsert" in built


def test_the_dormant_store_keeps_the_map_the_kernel_recorded_and_clears_it() -> None:
    dependencies: dict[str, dict[str, object]] = {"HandlerX": {"pool": object()}}

    StoreLazyStartDormant.record_handler_dependencies(dependencies)

    assert StoreLazyStartDormant.handler_dependencies() is dependencies
    StoreLazyStartDormant.clear()
    assert StoreLazyStartDormant.handler_dependencies() is None


@pytest.mark.asyncio
async def test_the_ingress_builds_its_transaction_with_the_recorded_dependencies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dependencies: dict[str, dict[str, object]] = {"HandlerX": {"pool": object()}}
    StoreLazyStartDormant.record(
        ModelLazyStartPolicy(ingress_contract="node_ingress_fixture"), (), None, ()
    )
    StoreLazyStartDormant.record_handler_dependencies(dependencies)
    built: dict[str, Any] = {}

    class _Capturing:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            built.update(kwargs)

    monkeypatch.setattr(ingress_module, "LazyStartRuntimeTransaction", _Capturing)
    monkeypatch.setattr(ingress_module, "install_publish_gate", lambda *_a, **_k: True)
    container = ModelONEXContainer()
    assert container.service_registry is not None
    key: type[object] = ProtocolEventBusSubscriber
    await container.service_registry.register_instance(key, MagicMock())

    HandlerLazyStartIngress(container)

    assert built.get("handler_dependencies") is dependencies
