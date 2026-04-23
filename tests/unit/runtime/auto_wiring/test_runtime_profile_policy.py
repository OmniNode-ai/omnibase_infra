# SPDX-FileCopyrightText: 2026 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Runtime-profile eligibility and startup-state tests for auto-wiring."""

from __future__ import annotations

import sys
import types
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

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
    ModelRuntimeProfilePolicy,
)
from omnibase_infra.runtime.auto_wiring.report import EnumWiringOutcome

_MODULE_NAME = "tests.unit.runtime._runtime_profile_policy_fixture"


class _HandlerHealthy:
    async def handle(self, envelope: object) -> None:
        return None


class _HandlerRequiredCtor:
    def __init__(self, missing_dep: object) -> None:
        self._missing_dep = missing_dep

    async def handle(self, envelope: object) -> None:
        return None


class _HandlerMissingHandle:
    pass


@pytest.fixture
def installed_handler_module() -> types.ModuleType:
    module = types.ModuleType(_MODULE_NAME)
    module._HandlerHealthy = _HandlerHealthy  # type: ignore[attr-defined]
    module._HandlerRequiredCtor = _HandlerRequiredCtor  # type: ignore[attr-defined]
    module._HandlerMissingHandle = _HandlerMissingHandle  # type: ignore[attr-defined]
    sys.modules[_MODULE_NAME] = module
    try:
        yield module
    finally:
        sys.modules.pop(_MODULE_NAME, None)


def _make_contract(
    *,
    name: str,
    node_type: str,
    handler_name: str,
    runtime_profiles: dict[str, ModelRuntimeProfilePolicy] | None = None,
) -> ModelDiscoveredContract:
    return ModelDiscoveredContract(
        name=name,
        node_type=node_type,
        contract_version=ModelContractVersion(major=1, minor=0, patch=0),
        contract_path=Path("/fake/contract.yaml"),
        entry_point_name=name,
        package_name="test-package",
        event_bus=ModelEventBusWiring(
            subscribe_topics=("onex.evt.platform.test-input.v1",),
            publish_topics=(),
        ),
        handler_routing=ModelHandlerRouting(
            routing_strategy="payload_type_match",
            handlers=(
                ModelHandlerRoutingEntry(
                    handler=ModelHandlerRef(name=handler_name, module=_MODULE_NAME),
                    event_model=ModelHandlerRef(name="ModelEvt", module="fake.models"),
                    operation=None,
                ),
            ),
        ),
        runtime_profiles=runtime_profiles or {},
    )


def _make_engine() -> MagicMock:
    engine = MagicMock()
    engine._routes = {}
    engine._container = None
    engine.register_dispatcher = MagicMock()
    engine.register_route = MagicMock()
    return engine


def _make_bus() -> MagicMock:
    bus = MagicMock()
    bus.subscribe = AsyncMock()
    return bus


@pytest.mark.asyncio
async def test_effects_profile_skips_non_effect_contract_as_ineligible(
    installed_handler_module: types.ModuleType,
) -> None:
    contract = _make_contract(
        name="node_pr_review_bot",
        node_type="WORKFLOW",
        handler_name="_HandlerHealthy",
    )

    report = await wire_from_manifest(
        manifest=ModelAutoWiringManifest(contracts=(contract,), errors=()),
        dispatch_engine=_make_engine(),
        event_bus=_make_bus(),
        runtime_profile="effects",
    )

    assert report.total_failed == 0
    assert report.total_skipped == 1
    assert report.ineligible_skipped_contracts == 1
    assert report.startup_state == "healthy"
    result = report.results[0]
    assert result.outcome is EnumWiringOutcome.SKIPPED
    assert result.profile_skip_reason == "ineligible"


@pytest.mark.asyncio
async def test_optional_effects_contract_degrades_instead_of_failing(
    installed_handler_module: types.ModuleType,
) -> None:
    contract = _make_contract(
        name="node_optional_effect",
        node_type="EFFECT_GENERIC",
        handler_name="_HandlerRequiredCtor",
        runtime_profiles={"effects": ModelRuntimeProfilePolicy(optional=True)},
    )

    report = await wire_from_manifest(
        manifest=ModelAutoWiringManifest(contracts=(contract,), errors=()),
        dispatch_engine=_make_engine(),
        event_bus=_make_bus(),
        runtime_profile="effects",
    )

    assert report.total_failed == 0
    assert report.optional_skipped_contracts == 1
    assert report.startup_state == "degraded"
    result = report.results[0]
    assert result.outcome is EnumWiringOutcome.SKIPPED
    assert result.profile_optional is True
    assert result.profile_skip_reason == "optional_unresolved"


@pytest.mark.asyncio
async def test_missing_handle_is_rejected_before_dispatch(
    installed_handler_module: types.ModuleType,
) -> None:
    contract = _make_contract(
        name="node_structural_invalid",
        node_type="EFFECT_GENERIC",
        handler_name="_HandlerMissingHandle",
    )

    with pytest.raises(TypeError, match="does not expose a callable handle"):
        await wire_from_manifest(
            manifest=ModelAutoWiringManifest(contracts=(contract,), errors=()),
            dispatch_engine=_make_engine(),
            event_bus=_make_bus(),
            runtime_profile="default",
        )
