# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Integration coverage for strict orchestrator dispatcher gating.

The CI integration-test gate requires feature-code PRs to prove behavior
through an integration surface, not only unit-level helpers. These tests keep
that proof local and side-effect free: no event bus, no Kafka, no live runtime.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from omnibase_core.models.errors import ModelOnexError
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
from omnibase_infra.runtime.service_message_dispatch_engine import MessageDispatchEngine

START_TOPIC = "onex.cmd.omnimarket.pr-lifecycle-orchestrator-start.v1"
START_ALIAS = "omnimarket.pr-lifecycle-orchestrator-start"


def _make_contract(
    handler_routing: ModelHandlerRouting | None,
) -> ModelDiscoveredContract:
    return ModelDiscoveredContract(
        name="pr_lifecycle_orchestrator",
        node_type="orchestrator",
        contract_version=ModelContractVersion(major=1, minor=0, patch=0),
        contract_path=Path("/fake/pr_lifecycle_orchestrator/contract.yaml"),
        entry_point_name="pr_lifecycle_orchestrator",
        package_name="omnimarket",
        event_bus=ModelEventBusWiring(
            subscribe_topics=(START_TOPIC,),
            publish_topics=(),
        ),
        handler_routing=handler_routing,
    )


def _make_handler_routing() -> ModelHandlerRouting:
    return ModelHandlerRouting(
        routing_strategy="payload_type_match",
        handlers=(
            ModelHandlerRoutingEntry(
                handler=ModelHandlerRef(
                    name="HandlerPrLifecycleOrchestrator",
                    module="fake.handlers",
                ),
                event_model=ModelHandlerRef(
                    name="ModelPrLifecycleStartCommand",
                    module="fake.models",
                ),
                event_type=START_ALIAS,
                message_category="command",
            ),
        ),
    )


class FakeHandler:
    async def handle(self, envelope: object) -> None:
        return None


@pytest.mark.asyncio
@pytest.mark.integration
async def test_strict_coverage_blocks_commit_before_dispatcher_registration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ONEX_STRICT_DISPATCHER_COVERAGE", "1")
    engine = MessageDispatchEngine()
    manifest = ModelAutoWiringManifest(contracts=(_make_contract(None),))

    with pytest.raises(ModelOnexError):
        await wire_from_manifest(manifest, engine)

    assert engine.dispatcher_count == 0


@pytest.mark.asyncio
@pytest.mark.integration
async def test_strict_coverage_allows_contract_alias_to_register_dispatcher(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ONEX_STRICT_DISPATCHER_COVERAGE", "1")
    engine = MessageDispatchEngine()
    manifest = ModelAutoWiringManifest(
        contracts=(_make_contract(_make_handler_routing()),),
    )

    with patch(
        "omnibase_infra.runtime.auto_wiring.handler_wiring._import_handler_class",
        return_value=FakeHandler,
    ):
        report = await wire_from_manifest(manifest, engine)

    assert report.total_wired == 1
    assert engine.dispatcher_count == 1
