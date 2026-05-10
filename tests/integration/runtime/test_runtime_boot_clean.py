# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Runtime boot validation for the main auto-wiring profile.

OMN-10453 proves that the real installed contract manifest, filtered to the
production ``main`` runtime profile, can pass through ``wire_from_manifest``
without a runtime boot crash. The test uses an in-process dispatch engine and a
deterministic DI container so the gate validates manifest and routing shape
without requiring live external services.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from omnibase_infra.runtime.auto_wiring import (
    ModelAutoWiringManifest,
    discover_contracts,
    filter_manifest_for_runtime_profile,
    wire_from_manifest,
)
from omnibase_infra.runtime.service_message_dispatch_engine import (
    MessageDispatchEngine,
)

pytestmark = [pytest.mark.integration]

MAIN_RUNTIME_PROFILE = "main"
OMNIMARKET_PACKAGE = "omnimarket"
EXCLUDED_MAIN_PROFILE_NODES = frozenset(
    {
        "node_intelligence_orchestrator",
        "node_intent_event_consumer_effect",
    }
)


class _BootValidationHandler:
    """Minimal async handler instance returned by the boot-validation container."""

    async def handle(self, envelope: object) -> None:
        return None


class _BootValidationContainer:
    """Deterministic DI container for constructor-free boot validation."""

    async def get_service_async(self, handler_cls: type[object]) -> object:
        return _BootValidationHandler()

    def get_service(self, handler_cls: type[object]) -> object:
        return _BootValidationHandler()


def _main_profile_manifest() -> ModelAutoWiringManifest:
    discovered = discover_contracts()
    ownership = filter_manifest_for_runtime_profile(
        manifest=discovered,
        runtime_profile=MAIN_RUNTIME_PROFILE,
    )
    return ownership.manifest


@pytest.mark.asyncio
async def test_wire_from_manifest_main_profile_no_crash() -> None:
    """The real main-profile manifest wires cleanly without ModelOnexError."""
    manifest = _main_profile_manifest()
    assert manifest.total_discovered > 0

    engine = MessageDispatchEngine(logger=MagicMock())
    report = await wire_from_manifest(
        manifest=manifest,
        dispatch_engine=engine,
        event_bus=None,
        environment="test",
        container=_BootValidationContainer(),
        subscribe_immediately=False,
    )

    assert report.total_failed == 0
    assert len(report.results) == manifest.total_discovered


def test_manifest_includes_omnimarket_contracts() -> None:
    """The main-profile boot gate is not vacuous: omnimarket is present.

    Asserts a representative slice of well-known omnimarket workflow nodes.
    `merge_sweep` was split into `merge_sweep_compute` + state-reducer + arm-effect
    in OMN-10791 wave; this assertion was updated to match the new node layout.
    """
    manifest = _main_profile_manifest()

    omnimarket_contracts = {
        contract.name
        for contract in manifest.contracts
        if contract.package_name == OMNIMARKET_PACKAGE
    }

    assert omnimarket_contracts
    assert {
        "build_loop",
        "merge_sweep_compute",
        "runtime_sweep",
    }.issubset(omnimarket_contracts)


def test_main_profile_excludes_memory_and_intelligence() -> None:
    """Known non-main crashers must not be owned by the main runtime profile."""
    manifest = _main_profile_manifest()
    main_profile_contract_names = {contract.name for contract in manifest.contracts}

    assert EXCLUDED_MAIN_PROFILE_NODES.isdisjoint(main_profile_contract_names)
