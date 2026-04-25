# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Tests for dispatcher-key event_type alias [OMN-9215].

Reproduces the dispatcher-key drift where a contract registers a handler under
a Pydantic class name but publishers emit `ModelEventEnvelope` with a contract
dot-path `event_type` (e.g., ``omnimarket.pr-lifecycle-orchestrator-start``).
Before this fix, ``_prepare_handler_wiring`` keyed the dispatcher only on
``event_model.name``, so the dispatcher's routing lookup never matched the
on-wire ``event_type`` and every command routed to DLQ.

This test proves:
  1. ``ModelHandlerRoutingEntry`` accepts an optional ``event_type`` alias.
  2. ``_prepare_handler_wiring`` includes both the class name AND the
     ``event_type`` alias in ``PreparedWiring.message_types``.
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


def _make_zero_arg_handler_cls() -> type:
    class FakeHandler:
        async def handle(self, envelope: object) -> None:
            return None

    return FakeHandler


def _make_contract_with_event_type_alias(
    *,
    event_model_name: str,
    event_type_alias: str | None,
    message_category: str | None = None,
    node_name: str = "node_local",
) -> ModelDiscoveredContract:
    event_model = ModelHandlerRef(name=event_model_name, module="fake.models")
    entry_kwargs: dict[str, object] = {
        "handler": ModelHandlerRef(name="HandlerFoo", module="fake.module"),
        "event_model": event_model,
        "operation": None,
    }
    if event_type_alias is not None:
        entry_kwargs["event_type"] = event_type_alias
    if message_category is not None:
        entry_kwargs["message_category"] = message_category

    return ModelDiscoveredContract(
        name=node_name,
        node_type="ORCHESTRATOR_GENERIC",
        contract_version=ModelContractVersion(major=1, minor=0, patch=0),
        contract_path=Path("/fake/contract.yaml"),
        entry_point_name=node_name,
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


class TestModelHandlerRoutingEntryAcceptsEventType:
    @pytest.mark.unit
    def test_event_type_alias_is_optional_and_defaults_to_none(self) -> None:
        entry = ModelHandlerRoutingEntry(
            handler=ModelHandlerRef(name="HandlerFoo", module="fake.module"),
            event_model=ModelHandlerRef(name="ModelFooCommand", module="fake.models"),
        )
        assert entry.event_type is None

    @pytest.mark.unit
    def test_event_type_alias_can_be_set(self) -> None:
        entry = ModelHandlerRoutingEntry(
            handler=ModelHandlerRef(name="HandlerFoo", module="fake.module"),
            event_model=ModelHandlerRef(name="ModelFooCommand", module="fake.models"),
            event_type="platform.foo-start",
        )
        assert entry.event_type == "platform.foo-start"

    @pytest.mark.unit
    def test_message_category_can_be_set(self) -> None:
        entry = ModelHandlerRoutingEntry(
            handler=ModelHandlerRef(name="HandlerFoo", module="fake.module"),
            event_model=ModelHandlerRef(name="ModelFooCommand", module="fake.models"),
            message_category="COMMAND",
        )
        assert entry.message_category == "COMMAND"


class TestPrepareHandlerWiringIncludesEventTypeAlias:
    @pytest.mark.unit
    def test_message_types_includes_both_class_name_and_event_type(self) -> None:
        """When event_type alias is declared, BOTH keys index the dispatcher.

        Before OMN-9215 dispatcher-key fix: message_types == {"ModelFooCommand"}.
        After: message_types == {"ModelFooCommand", "platform.foo-start"}.
        This lets a publisher's wire-level event_type resolve to the registered
        handler (primary lookup path) AND keeps the class-name fallback working.
        """
        contract = _make_contract_with_event_type_alias(
            event_model_name="ModelFooCommand",
            event_type_alias="platform.foo-start",
        )
        entry = contract.handler_routing.handlers[0]  # type: ignore[union-attr]
        handler_cls = _make_zero_arg_handler_cls()
        ownership = ServiceLocalHandlerOwnershipQuery(
            local_node_names=frozenset({contract.name})
        )
        resolver = ServiceHandlerResolver()
        with patch(
            "omnibase_infra.runtime.auto_wiring.handler_wiring._import_handler_class",
            return_value=handler_cls,
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
        assert prepared.message_types == {"ModelFooCommand", "platform.foo-start"}

    @pytest.mark.unit
    def test_message_types_strips_whitespace_from_event_type_alias(self) -> None:
        """Alias whitespace is stripped before indexing.

        The dispatch engine normalizes envelope ``event_type`` via ``.strip()``
        before lookup (service_message_dispatch_engine.py). If a contract accidentally
        carries surrounding whitespace in ``event_type``, registering the alias
        verbatim produces a key mismatch and routes every command to DLQ.
        Normalize on registration to keep both sides symmetric.
        """
        contract = _make_contract_with_event_type_alias(
            event_model_name="ModelFooCommand",
            event_type_alias="  platform.foo-start  ",
        )
        entry = contract.handler_routing.handlers[0]  # type: ignore[union-attr]
        handler_cls = _make_zero_arg_handler_cls()
        ownership = ServiceLocalHandlerOwnershipQuery(
            local_node_names=frozenset({contract.name})
        )
        resolver = ServiceHandlerResolver()
        with patch(
            "omnibase_infra.runtime.auto_wiring.handler_wiring._import_handler_class",
            return_value=handler_cls,
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
        assert prepared.message_types == {"ModelFooCommand", "platform.foo-start"}

    @pytest.mark.unit
    def test_message_types_omits_alias_when_only_whitespace(self) -> None:
        """A whitespace-only alias reduces to empty and is not registered."""
        contract = _make_contract_with_event_type_alias(
            event_model_name="ModelFooCommand",
            event_type_alias="   ",
        )
        entry = contract.handler_routing.handlers[0]  # type: ignore[union-attr]
        handler_cls = _make_zero_arg_handler_cls()
        ownership = ServiceLocalHandlerOwnershipQuery(
            local_node_names=frozenset({contract.name})
        )
        resolver = ServiceHandlerResolver()
        with patch(
            "omnibase_infra.runtime.auto_wiring.handler_wiring._import_handler_class",
            return_value=handler_cls,
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

    @pytest.mark.unit
    def test_message_category_overrides_contract_first_topic_category(self) -> None:
        """Mixed-topic contracts must route command handlers as commands.

        The registration contract subscribes to event topics first, then command
        topics. Before this regression fix, every handler inherited the first
        topic category, so request-introspection commands were registered as
        event routes and DLQ'd at runtime.
        """
        from omnibase_infra.enums import EnumMessageCategory

        contract = _make_contract_with_event_type_alias(
            event_model_name="ModelFooCommand",
            event_type_alias="platform.foo-start",
            message_category="COMMAND",
        )
        entry = contract.handler_routing.handlers[0]  # type: ignore[union-attr]
        handler_cls = _make_zero_arg_handler_cls()
        ownership = ServiceLocalHandlerOwnershipQuery(
            local_node_names=frozenset({contract.name})
        )
        resolver = ServiceHandlerResolver()
        with patch(
            "omnibase_infra.runtime.auto_wiring.handler_wiring._import_handler_class",
            return_value=handler_cls,
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

        assert prepared.category is EnumMessageCategory.COMMAND

    @pytest.mark.unit
    def test_registration_contract_skips_generic_direct_handler_dispatcher(
        self,
    ) -> None:
        """Registration auto-wiring owns subscriptions, not generic dispatchers."""
        from omnibase_core.enums.enum_handler_resolution_outcome import (
            EnumHandlerResolutionOutcome,
        )
        from omnibase_infra.enums import EnumMessageCategory

        contract = _make_contract_with_event_type_alias(
            event_model_name="ModelTopicCatalogRequest",
            event_type_alias="platform.request-introspection",
            message_category="COMMAND",
            node_name="node_registration_orchestrator",
        )
        entry = contract.handler_routing.handlers[0]  # type: ignore[union-attr]
        ownership = ServiceLocalHandlerOwnershipQuery(
            local_node_names=frozenset({contract.name})
        )
        resolver = ServiceHandlerResolver()
        with patch(
            "omnibase_infra.runtime.auto_wiring.handler_wiring._import_handler_class"
        ) as import_handler:
            prepared = _prepare_handler_wiring(
                contract=contract,
                entry=entry,
                dispatch_engine=None,
                resolver=resolver,
                ownership_query=ownership,
                event_bus=None,
                container=None,
            )

        import_handler.assert_not_called()
        assert prepared.is_skip is True
        assert prepared.category is EnumMessageCategory.COMMAND
        assert prepared.message_types == {
            "ModelTopicCatalogRequest",
            "platform.request-introspection",
        }
        assert (
            prepared.resolution_outcome
            is EnumHandlerResolutionOutcome.RESOLVED_VIA_LOCAL_OWNERSHIP_SKIP
        )
        assert "explicit dispatcher adapters" in prepared.skip_reason

    @pytest.mark.unit
    def test_message_types_only_class_name_when_no_alias(self) -> None:
        """Absent event_type alias: message_types == {event_model.name} — unchanged."""
        contract = _make_contract_with_event_type_alias(
            event_model_name="ModelFooCommand",
            event_type_alias=None,
        )
        entry = contract.handler_routing.handlers[0]  # type: ignore[union-attr]
        handler_cls = _make_zero_arg_handler_cls()
        ownership = ServiceLocalHandlerOwnershipQuery(
            local_node_names=frozenset({contract.name})
        )
        resolver = ServiceHandlerResolver()
        with patch(
            "omnibase_infra.runtime.auto_wiring.handler_wiring._import_handler_class",
            return_value=handler_cls,
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


class TestContractDiscoveryParsesEventType:
    """End-to-end check: YAML `event_type:` survives parsing onto the model."""

    @pytest.mark.unit
    def test_parse_handler_routing_threads_event_type(self) -> None:
        from omnibase_infra.runtime.auto_wiring.discovery import _parse_handler_routing

        hr_raw = {
            "routing_strategy": "operation_match",
            "handlers": [
                {
                    "handler": {
                        "name": "HandlerFoo",
                        "module": "fake.handlers",
                    },
                    "event_model": {
                        "name": "ModelFooCommand",
                        "module": "fake.models",
                    },
                    "event_type": "platform.foo-start",
                    "message_category": "COMMAND",
                },
            ],
        }

        parsed = _parse_handler_routing(hr_raw)

        assert len(parsed.handlers) == 1
        assert parsed.handlers[0].event_type == "platform.foo-start"
        assert parsed.handlers[0].message_category == "COMMAND"

    @pytest.mark.unit
    def test_parse_handler_routing_defaults_event_type_and_category_to_none(
        self,
    ) -> None:
        from omnibase_infra.runtime.auto_wiring.discovery import _parse_handler_routing

        hr_raw = {
            "routing_strategy": "operation_match",
            "handlers": [
                {
                    "handler": {"name": "HandlerFoo", "module": "fake.handlers"},
                    "event_model": {"name": "ModelFooCommand", "module": "fake.models"},
                },
            ],
        }

        parsed = _parse_handler_routing(hr_raw)

        assert parsed.handlers[0].event_type is None
        assert parsed.handlers[0].message_category is None
