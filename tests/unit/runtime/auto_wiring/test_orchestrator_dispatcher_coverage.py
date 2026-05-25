# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Strict dispatcher coverage tests for orchestrator start topics."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

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
from omnibase_infra.runtime.auto_wiring.report import EnumWiringOutcome

START_TOPIC = "onex.cmd.omnimarket.pr-lifecycle-orchestrator-start.v1"
START_ALIAS = "omnimarket.pr-lifecycle-orchestrator-start"


def _make_contract(
    *,
    node_type: str = "ORCHESTRATOR_GENERIC",
    handler_routing: ModelHandlerRouting | None = None,
) -> ModelDiscoveredContract:
    return ModelDiscoveredContract(
        name="pr_lifecycle_orchestrator",
        node_type=node_type,
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


def _make_handler_routing(
    *,
    event_type: str | None = START_ALIAS,
    event_model_name: str | None = "ModelPrLifecycleStartCommand",
) -> ModelHandlerRouting:
    event_model = (
        ModelHandlerRef(
            name=event_model_name,
            module="fake.models",
        )
        if event_model_name is not None
        else None
    )
    return ModelHandlerRouting(
        routing_strategy="payload_type_match",
        handlers=(
            ModelHandlerRoutingEntry(
                handler=ModelHandlerRef(
                    name="HandlerPrLifecycleOrchestrator",
                    module="fake.handlers",
                ),
                event_model=event_model,
                event_type=event_type,
                message_category="command",
            ),
        ),
    )


class FakeHandler:
    async def handle(self, envelope: object) -> None:
        return None


@pytest.mark.asyncio
async def test_orchestrator_start_coverage_is_default_off(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("ONEX_STRICT_DISPATCHER_COVERAGE", raising=False)
    manifest = ModelAutoWiringManifest(contracts=(_make_contract(),))

    report = await wire_from_manifest(manifest, MagicMock())

    assert report.total_skipped == 1
    assert report.results[0].outcome is EnumWiringOutcome.SKIPPED


@pytest.mark.asyncio
async def test_strict_coverage_rejects_orchestrator_start_without_handler(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ONEX_STRICT_DISPATCHER_COVERAGE", "1")
    manifest = ModelAutoWiringManifest(contracts=(_make_contract(),))

    with pytest.raises(ModelOnexError) as exc_info:
        await wire_from_manifest(manifest, MagicMock())

    message = str(exc_info.value)
    assert "Strict dispatcher coverage failed" in message
    assert "pr_lifecycle_orchestrator" in message
    assert START_ALIAS in message


@pytest.mark.asyncio
async def test_strict_coverage_accepts_contract_event_type_alias(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from omnibase_infra.runtime.message_dispatch_engine import (
        MessageDispatchEngine,
    )

    monkeypatch.setenv("ONEX_STRICT_DISPATCHER_COVERAGE", "1")
    manifest = ModelAutoWiringManifest(
        contracts=(
            _make_contract(
                handler_routing=_make_handler_routing(event_type=START_ALIAS)
            ),
        ),
    )

    with patch(
        "omnibase_infra.runtime.auto_wiring.handler_wiring._import_handler_class",
        return_value=FakeHandler,
    ):
        report = await wire_from_manifest(manifest, MessageDispatchEngine())

    assert report.total_wired == 1
    assert report.results[0].dispatchers_registered == (
        "dispatcher.auto.pr_lifecycle_orchestrator.HandlerPrLifecycleOrchestrator",
    )


@pytest.mark.asyncio
async def test_strict_coverage_accepts_topic_derived_event_type_alias(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from omnibase_infra.runtime.message_dispatch_engine import (
        MessageDispatchEngine,
    )

    monkeypatch.setenv("ONEX_STRICT_DISPATCHER_COVERAGE", "1")
    manifest = ModelAutoWiringManifest(
        contracts=(
            _make_contract(
                handler_routing=_make_handler_routing(
                    event_type=None,
                    event_model_name=None,
                )
            ),
        ),
    )

    with patch(
        "omnibase_infra.runtime.auto_wiring.handler_wiring._import_handler_class",
        return_value=FakeHandler,
    ):
        report = await wire_from_manifest(manifest, MessageDispatchEngine())

    assert report.total_wired == 1


@pytest.mark.asyncio
async def test_strict_coverage_rejects_orchestrator_start_when_prepare_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ONEX_STRICT_DISPATCHER_COVERAGE", "true")
    manifest = ModelAutoWiringManifest(
        contracts=(_make_contract(handler_routing=_make_handler_routing()),),
    )

    with patch(
        "omnibase_infra.runtime.auto_wiring.handler_wiring._import_handler_class",
        side_effect=ImportError("missing handler"),
    ):
        with pytest.raises(ModelOnexError) as exc_info:
            await wire_from_manifest(manifest, MagicMock())

    message = str(exc_info.value)
    assert "because handler preparation failed" in message
    assert START_ALIAS in message
