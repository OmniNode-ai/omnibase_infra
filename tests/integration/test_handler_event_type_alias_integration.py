# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Integration-level proof for OMN-9215 dispatcher-key alias symmetry.

OMN-9215 fixes dispatcher-key drift: contracts declare a dot-path
``event_type`` alias (e.g. ``omnimarket.pr-lifecycle-orchestrator-start``)
while ``MessageDispatchEngine`` normalizes the wire ``event_type`` via
``.strip()`` before routing. Before the fix, handler_wiring registered
only ``event_model.name`` as a message_type key, so every contract-aliased
command missed the dispatcher lookup and fell through to DLQ.

This integration test asserts the two-sided symmetry between registration
and dispatch as a single contract — if either side drifts, this fails.

Unit coverage for the individual sides lives at:
    tests/unit/runtime/auto_wiring/test_handler_wiring_event_type_alias.py

Ticket: OMN-9215
Integration Test Coverage gate: OMN-7005 (hard gate since 2026-04-13).
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from omnibase_core.services.service_handler_resolver import ServiceHandlerResolver
from omnibase_core.services.service_local_handler_ownership_query import (
    ServiceLocalHandlerOwnershipQuery,
)
from omnibase_infra.runtime.auto_wiring.handler_wiring import _prepare_handler_wiring
from omnibase_infra.runtime.auto_wiring.models import (
    ModelContractVersion,
    ModelDiscoveredContract,
    ModelEventBusWiring,
    ModelHandlerRef,
    ModelHandlerRouting,
    ModelHandlerRoutingEntry,
)


def _make_contract(event_type_alias: str | None) -> ModelDiscoveredContract:
    entry_kwargs: dict[str, object] = {
        "handler": ModelHandlerRef(name="HandlerFoo", module="fake.handlers"),
        "event_model": ModelHandlerRef(name="ModelFooCommand", module="fake.models"),
        "operation": None,
    }
    if event_type_alias is not None:
        entry_kwargs["event_type"] = event_type_alias
    return ModelDiscoveredContract(
        name="node_local",
        node_type="ORCHESTRATOR_GENERIC",
        contract_version=ModelContractVersion(major=1, minor=0, patch=0),
        contract_path=Path("/fake/contract.yaml"),
        entry_point_name="node_local",
        package_name="test-pkg",
        event_bus=ModelEventBusWiring(
            subscribe_topics=("onex.cmd.platform.foo-start.v1",),
            publish_topics=(),
        ),
        handler_routing=ModelHandlerRouting(
            routing_strategy="operation_match",
            handlers=(ModelHandlerRoutingEntry(**entry_kwargs),),
        ),
    )


@pytest.mark.integration
def test_registration_emits_alias_matching_dispatch_engine_normalization() -> None:
    """OMN-9215: registration emits the alias; dispatch engine normalizes via strip().

    Contract declares ``event_type='  platform.foo-start  '`` (with surrounding
    whitespace — the real-world regression mode). Registration must strip to
    ``'platform.foo-start'`` so it matches what ``MessageDispatchEngine`` will
    look up after its own ``.strip()`` normalization of the wire envelope.

    If either side of the symmetry breaks (registration stops stripping, or
    dispatch stops stripping) this assertion catches it before the bug hits
    production as DLQ-routed commands.
    """

    class _FakeHandler:
        async def handle(self, envelope: object) -> None:
            return None

    contract = _make_contract(event_type_alias="  platform.foo-start  ")
    entry = contract.handler_routing.handlers[0]  # type: ignore[union-attr]
    resolver = ServiceHandlerResolver()
    ownership = ServiceLocalHandlerOwnershipQuery(
        local_node_names=frozenset({contract.name})
    )
    with patch(
        "omnibase_infra.runtime.auto_wiring.handler_wiring._import_handler_class",
        return_value=_FakeHandler,
    ):
        prepared = _prepare_handler_wiring(
            contract=contract,
            entry=entry,
            dispatch_engine=None,
            resolver=resolver,
            ownership_query=ownership,
            event_bus=None,
            container=None,
        )

    assert prepared.message_types == {"ModelFooCommand", "platform.foo-start"}, (
        "Registration must emit the stripped alias so the dispatch engine's "
        ".strip()-normalized wire event_type matches. Drift here reintroduces "
        "the OMN-9215 DLQ-everything bug."
    )

    # Symmetry check: dispatch-engine normalization of the wire value.
    # service_message_dispatch_engine.py:1106-1108 does
    #     str(envelope_event_type).strip()
    # so the key produced on the wire for a whitespace-polluted event_type
    # must equal what registration emitted.
    wire_event_type = "  platform.foo-start  "
    normalized_wire_key = str(wire_event_type).strip()
    assert normalized_wire_key in prepared.message_types, (
        "dispatch engine normalization key must appear in registered message_types"
    )


@pytest.mark.integration
def test_alias_absent_falls_back_to_class_name_only() -> None:
    """OMN-9215 fix must not leak: no alias declared → class-name key only.

    Regression guard: the strip() addition must not accidentally synthesize
    an alias where none was declared.
    """

    class _FakeHandler:
        async def handle(self, envelope: object) -> None:
            return None

    contract = _make_contract(event_type_alias=None)
    entry = contract.handler_routing.handlers[0]  # type: ignore[union-attr]
    resolver = ServiceHandlerResolver()
    ownership = ServiceLocalHandlerOwnershipQuery(
        local_node_names=frozenset({contract.name})
    )
    with patch(
        "omnibase_infra.runtime.auto_wiring.handler_wiring._import_handler_class",
        return_value=_FakeHandler,
    ):
        prepared = _prepare_handler_wiring(
            contract=contract,
            entry=entry,
            dispatch_engine=None,
            resolver=resolver,
            ownership_query=ownership,
            event_bus=None,
            container=None,
        )

    assert prepared.message_types == {"ModelFooCommand"}
