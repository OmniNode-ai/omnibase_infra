# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Tests for event_model-derived payload type scoping in auto-wiring (OMN-12416).

A multi-handler contract whose entries declare distinct ``event_model``s must
build a per-dispatcher ``payload_type_matcher`` so the dispatch engine routes
each consumed message only to the handler whose event_model matches the payload.
These tests exercise ``_prepare_handler_wiring`` to confirm:
  - a handler entry with an ``event_model`` produces a matcher that accepts its
    own model (instance and dict form) and rejects a sibling's model;
  - an operation-only handler entry (no ``event_model``) produces NO matcher,
    preserving legacy string-only matching for single-handler / untyped paths.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import BaseModel

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


class ModelTypeScopeAlpha(BaseModel):
    """Real importable event model for the alpha handler entry."""

    alpha_field: str


class ModelTypeScopeBeta(BaseModel):
    """Real importable event model for the beta handler entry."""

    beta_field: int


class _FakeHandler:
    async def handle(self, payload: object) -> None:
        return None


def _contract(entry: ModelHandlerRoutingEntry) -> ModelDiscoveredContract:
    return ModelDiscoveredContract(
        name="node_local",
        node_type="EFFECT_GENERIC",
        contract_version=ModelContractVersion(major=1, minor=0, patch=0),
        contract_path=Path("/fake/contract.yaml"),
        entry_point_name="node_local",
        package_name="test-pkg",
        event_bus=ModelEventBusWiring(
            subscribe_topics=("onex.cmd.test-service.shared-command.v1",),
            publish_topics=(),
        ),
        handler_routing=ModelHandlerRouting(
            routing_strategy="operation_match",
            handlers=(entry,),
        ),
    )


def _prepare(entry: ModelHandlerRoutingEntry) -> object:
    contract = _contract(entry)
    # The handler module here is a real local class, not an installed node —
    # patch import to return the fake. The contract name must be in the
    # ownership set so the resolver treats it as a live (non-skipped) handler.
    from unittest.mock import patch

    ownership = ServiceLocalHandlerOwnershipQuery(
        local_node_names=frozenset({contract.name})
    )
    resolver = ServiceHandlerResolver()
    with patch(
        "omnibase_infra.runtime.auto_wiring.handler_wiring._import_handler_class",
        return_value=_FakeHandler,
    ):
        return _prepare_handler_wiring(
            contract=contract,
            entry=entry,
            dispatch_engine=None,
            resolver=resolver,
            ownership_query=ownership,
            event_bus=None,
            container=None,
        )


_THIS_MODULE = "tests.unit.runtime.auto_wiring.test_handler_wiring_payload_type_matcher"


@pytest.mark.unit
class TestPayloadTypeMatcher:
    def test_typed_entry_builds_matcher_accepting_own_model(self) -> None:
        entry = ModelHandlerRoutingEntry(
            handler=ModelHandlerRef(name="HandlerAlpha", module=_THIS_MODULE),
            event_model=ModelHandlerRef(
                name="ModelTypeScopeAlpha", module=_THIS_MODULE
            ),
            operation="alpha_op",
        )
        prepared = _prepare(entry)
        matcher = prepared.payload_type_matcher  # type: ignore[attr-defined]
        assert matcher is not None
        # Accepts its own model instance.
        assert matcher(ModelTypeScopeAlpha(alpha_field="x")) is True
        # Accepts a dict that validates to its own model.
        assert matcher({"alpha_field": "x"}) is True

    def test_typed_entry_matcher_rejects_sibling_model(self) -> None:
        entry = ModelHandlerRoutingEntry(
            handler=ModelHandlerRef(name="HandlerAlpha", module=_THIS_MODULE),
            event_model=ModelHandlerRef(
                name="ModelTypeScopeAlpha", module=_THIS_MODULE
            ),
            operation="alpha_op",
        )
        prepared = _prepare(entry)
        matcher = prepared.payload_type_matcher  # type: ignore[attr-defined]
        assert matcher is not None
        # Rejects the sibling's model instance.
        assert matcher(ModelTypeScopeBeta(beta_field=1)) is False
        # Rejects a dict that does NOT satisfy the alpha model.
        assert matcher({"beta_field": 1}) is False

    def test_operation_only_entry_builds_no_matcher(self) -> None:
        """An entry without an event_model stays un-scoped (legacy matching)."""
        entry = ModelHandlerRoutingEntry(
            handler=ModelHandlerRef(name="HandlerUntyped", module=_THIS_MODULE),
            operation="untyped_op",
        )
        prepared = _prepare(entry)
        assert prepared.payload_type_matcher is None  # type: ignore[attr-defined]
