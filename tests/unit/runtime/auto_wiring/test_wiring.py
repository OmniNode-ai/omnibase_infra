# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Tests for handler auto-wiring engine (OMN-7654)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from omnibase_infra.runtime.auto_wiring.handler_wiring import (
    _derive_dispatcher_id,
    _derive_message_category,
    _derive_route_id,
    _derive_topic_pattern_from_topic,
    _detect_duplicate_topics,
    _make_dispatch_callback,
    wire_from_manifest,
)
from omnibase_infra.runtime.auto_wiring.models import (
    ModelAutoWiringManifest,
    ModelContractVersion,
    ModelDiscoveredContract,
    ModelEventBusWiring,
    ModelHandlerRef,
    ModelHandlerRouting,
    ModelHandlerRoutingEntry,
)
from omnibase_infra.runtime.auto_wiring.report import (
    EnumWiringOutcome,
    ModelAutoWiringReport,
    ModelContractWiringResult,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_contract_version() -> ModelContractVersion:
    return ModelContractVersion(major=1, minor=0, patch=0)


def _make_handler_routing(
    handler_name: str = "HandlerTest",
    handler_module: str = "test.handlers.handler_test",
    event_model_name: str | None = None,
    event_model_module: str | None = None,
    operation: str | None = None,
) -> ModelHandlerRouting:
    event_model = None
    if event_model_name and event_model_module:
        event_model = ModelHandlerRef(name=event_model_name, module=event_model_module)
    return ModelHandlerRouting(
        routing_strategy="payload_type_match",
        handlers=(
            ModelHandlerRoutingEntry(
                handler=ModelHandlerRef(name=handler_name, module=handler_module),
                event_model=event_model,
                operation=operation,
            ),
        ),
    )


def _make_contract(
    name: str = "node_test",
    package_name: str = "test-package",
    subscribe_topics: tuple[str, ...] = ("onex.evt.platform.test-input.v1",),
    publish_topics: tuple[str, ...] = (),
    handler_routing: ModelHandlerRouting | None = None,
) -> ModelDiscoveredContract:
    return ModelDiscoveredContract(
        name=name,
        node_type="EFFECT_GENERIC",
        contract_version=_make_contract_version(),
        contract_path=Path("/fake/contract.yaml"),
        entry_point_name=name,
        package_name=package_name,
        event_bus=ModelEventBusWiring(
            subscribe_topics=subscribe_topics,
            publish_topics=publish_topics,
        ),
        handler_routing=handler_routing,
    )


# ---------------------------------------------------------------------------
# Unit tests: helper functions
# ---------------------------------------------------------------------------


class TestDeriveTopicPattern:
    def test_five_segment_topic(self) -> None:
        result = _derive_topic_pattern_from_topic(
            "onex.evt.platform.node-introspection.v1"
        )
        assert result == "*.evt.platform.node-introspection.*"

    def test_short_topic_returns_exact(self) -> None:
        result = _derive_topic_pattern_from_topic("foo.bar")
        assert result == "foo.bar"


class TestDeriveMessageCategory:
    def test_evt(self) -> None:
        assert _derive_message_category("onex.evt.platform.test.v1") == "event"

    def test_cmd(self) -> None:
        assert _derive_message_category("onex.cmd.platform.test.v1") == "command"

    def test_intent(self) -> None:
        assert _derive_message_category("onex.intent.platform.test.v1") == "intent"

    def test_unknown_defaults_to_event(self) -> None:
        assert _derive_message_category("onex.unknown.platform.test.v1") == "event"


class TestDeriveIds:
    def test_route_id(self) -> None:
        assert (
            _derive_route_id("my_node", "my_handler", "onex.evt.platform.my-topic.v1")
            == "route.auto.my_node.my_handler.onex_evt_platform_my_topic_v1"
        )

    def test_dispatcher_id(self) -> None:
        assert (
            _derive_dispatcher_id("my_node", "my_handler")
            == "dispatcher.auto.my_node.my_handler"
        )


class TestMakeDispatchCallback:
    @pytest.mark.asyncio
    async def test_callback_delegates_to_handle(self) -> None:
        handler = MagicMock()
        handler.handle = AsyncMock(return_value=None)
        callback = _make_dispatch_callback(handler)
        envelope = MagicMock()
        result = await callback(envelope)
        handler.handle.assert_called_once_with(envelope)
        assert result is None


# ---------------------------------------------------------------------------
# Unit tests: duplicate detection
# ---------------------------------------------------------------------------


class TestDetectDuplicateTopics:
    def test_no_duplicates(self) -> None:
        manifest = ModelAutoWiringManifest(
            contracts=(
                _make_contract(name="a", subscribe_topics=("onex.evt.platform.a.v1",)),
                _make_contract(name="b", subscribe_topics=("onex.evt.platform.b.v1",)),
            ),
        )
        dups = _detect_duplicate_topics(manifest)
        assert len(dups) == 0

    def test_intra_package_duplicate(self) -> None:
        manifest = ModelAutoWiringManifest(
            contracts=(
                _make_contract(
                    name="a",
                    package_name="pkg1",
                    subscribe_topics=("onex.evt.platform.shared.v1",),
                ),
                _make_contract(
                    name="b",
                    package_name="pkg1",
                    subscribe_topics=("onex.evt.platform.shared.v1",),
                ),
            ),
        )
        dups = _detect_duplicate_topics(manifest)
        assert len(dups) == 1
        assert dups[0].level == "intra-package"
        assert dups[0].topic == "onex.evt.platform.shared.v1"
        assert set(dups[0].owners) == {"a", "b"}

    def test_cross_package_duplicate(self) -> None:
        manifest = ModelAutoWiringManifest(
            contracts=(
                _make_contract(
                    name="a",
                    package_name="pkg1",
                    subscribe_topics=("onex.evt.platform.shared.v1",),
                ),
                _make_contract(
                    name="b",
                    package_name="pkg2",
                    subscribe_topics=("onex.evt.platform.shared.v1",),
                ),
            ),
        )
        dups = _detect_duplicate_topics(manifest)
        assert len(dups) == 1
        assert dups[0].level == "package"


# ---------------------------------------------------------------------------
# Unit tests: wire_from_manifest
# ---------------------------------------------------------------------------


class TestWireFromManifest:
    @pytest.mark.asyncio
    async def test_skip_contract_without_handler_routing(self) -> None:
        contract = _make_contract(handler_routing=None)
        manifest = ModelAutoWiringManifest(contracts=(contract,))
        engine = MagicMock()
        report = await wire_from_manifest(manifest, engine)
        assert report.total_skipped == 1
        assert report.total_wired == 0
        assert report.results[0].outcome == EnumWiringOutcome.SKIPPED

    @pytest.mark.asyncio
    async def test_skip_contract_without_event_bus(self) -> None:
        contract = ModelDiscoveredContract(
            name="no_bus",
            node_type="EFFECT_GENERIC",
            contract_version=_make_contract_version(),
            contract_path=Path("/fake/contract.yaml"),
            entry_point_name="no_bus",
            package_name="test",
            event_bus=None,
            handler_routing=_make_handler_routing(),
        )
        manifest = ModelAutoWiringManifest(contracts=(contract,))
        engine = MagicMock()
        report = await wire_from_manifest(manifest, engine)
        assert report.total_skipped == 1

    @pytest.mark.asyncio
    async def test_wire_success(self) -> None:
        """Test successful wiring with a mock handler class."""
        from omnibase_infra.runtime.service_message_dispatch_engine import (
            MessageDispatchEngine,
        )

        handler_routing = _make_handler_routing(
            handler_name="FakeHandler",
            handler_module="fake.module",
        )
        contract = _make_contract(handler_routing=handler_routing)
        manifest = ModelAutoWiringManifest(contracts=(contract,))

        # Create a real engine instance
        engine = MessageDispatchEngine()

        # ModelHandlerResolverContext requires ``handler_cls: type`` (OMN-9201),
        # so use a real class rather than a MagicMock.
        class FakeHandler:
            async def handle(self, envelope: object) -> None:
                return None

        with patch(
            "omnibase_infra.runtime.auto_wiring.handler_wiring._import_handler_class",
            return_value=FakeHandler,
        ):
            report = await wire_from_manifest(manifest, engine)

        assert report.total_wired == 1
        assert report.total_failed == 0
        assert len(report.results[0].dispatchers_registered) == 1
        assert len(report.results[0].routes_registered) >= 1

    @pytest.mark.asyncio
    async def test_wire_failure_import_error_raises(self) -> None:
        """OMN-8735: import errors must raise, not be silently captured."""
        from omnibase_core.models.errors import ModelOnexError

        handler_routing = _make_handler_routing(
            handler_name="MissingHandler",
            handler_module="nonexistent.module",
        )
        contract = _make_contract(handler_routing=handler_routing)
        manifest = ModelAutoWiringManifest(contracts=(contract,))

        engine = MagicMock()
        with pytest.raises(ModelOnexError):
            await wire_from_manifest(manifest, engine)

    @pytest.mark.asyncio
    async def test_unsatisfiable_di_raises_on_startup(self) -> None:
        """OMN-8735 negative test: handler with unsatisfiable DI deps must raise."""
        from omnibase_core.models.errors import ModelOnexError
        from omnibase_infra.runtime.service_message_dispatch_engine import (
            MessageDispatchEngine,
        )

        handler_routing = _make_handler_routing(
            handler_name="HandlerWithDeps",
            handler_module="fake.module",
        )
        contract = _make_contract(handler_routing=handler_routing)
        manifest = ModelAutoWiringManifest(contracts=(contract,))
        engine = MessageDispatchEngine()

        class HandlerWithDeps:
            def __init__(self, required_service: object) -> None:
                self.required_service = required_service

            async def handle(self, envelope: object) -> None:
                pass

        with patch(
            "omnibase_infra.runtime.auto_wiring.handler_wiring._import_handler_class",
            return_value=HandlerWithDeps,
        ):
            with pytest.raises((ModelOnexError, TypeError)):
                await wire_from_manifest(manifest, engine)

    @pytest.mark.asyncio
    async def test_multi_handler_orchestrator_distinct_route_ids(self) -> None:
        """OMN-8735 positive test: N handlers x M topics produce N*M distinct route_ids."""
        from omnibase_infra.runtime.service_message_dispatch_engine import (
            MessageDispatchEngine,
        )

        topics = (
            "onex.evt.platform.topic-alpha.v1",
            "onex.evt.platform.topic-beta.v1",
        )
        handler_names = ["HandlerA", "HandlerB", "HandlerC"]

        handlers_entries = tuple(
            ModelHandlerRoutingEntry(
                handler=ModelHandlerRef(name=name, module="fake.module"),
                event_model=None,
                operation=None,
            )
            for name in handler_names
        )
        handler_routing = ModelHandlerRouting(
            routing_strategy="payload_type_match",
            handlers=handlers_entries,
        )
        contract = _make_contract(
            name="orchestrator_node",
            subscribe_topics=topics,
            handler_routing=handler_routing,
        )
        manifest = ModelAutoWiringManifest(contracts=(contract,))
        engine = MessageDispatchEngine()

        def make_fake_cls(name: str) -> type:
            class FakeCls:
                async def handle(self, envelope: object) -> None:
                    pass

            FakeCls.__name__ = name
            return FakeCls

        fake_classes = {name: make_fake_cls(name) for name in handler_names}

        def fake_import(module: str, cls_name: str) -> type:
            return fake_classes[cls_name]

        with patch(
            "omnibase_infra.runtime.auto_wiring.handler_wiring._import_handler_class",
            side_effect=fake_import,
        ):
            report = await wire_from_manifest(manifest, engine)

        all_route_ids: list[str] = []
        for r in report.results:
            all_route_ids.extend(r.routes_registered)

        expected_count = len(handler_names) * len(topics)
        assert len(all_route_ids) == expected_count, (
            f"Expected {expected_count} route_ids, got {len(all_route_ids)}: {all_route_ids}"
        )
        assert len(set(all_route_ids)) == expected_count, (
            f"Duplicate route_ids detected: {all_route_ids}"
        )

    @pytest.mark.asyncio
    async def test_report_bool_true_when_no_failures(self) -> None:
        report = ModelAutoWiringReport(
            results=(
                ModelContractWiringResult(
                    contract_name="a",
                    package_name="pkg",
                    outcome=EnumWiringOutcome.WIRED,
                ),
                ModelContractWiringResult(
                    contract_name="b",
                    package_name="pkg",
                    outcome=EnumWiringOutcome.SKIPPED,
                ),
            ),
        )
        assert bool(report) is True

    @pytest.mark.asyncio
    async def test_report_bool_false_when_failures(self) -> None:
        report = ModelAutoWiringReport(
            results=(
                ModelContractWiringResult(
                    contract_name="a",
                    package_name="pkg",
                    outcome=EnumWiringOutcome.FAILED,
                    reason="import error",
                ),
            ),
        )
        assert bool(report) is False

    @pytest.mark.asyncio
    async def test_duplicate_detection_in_report(self) -> None:
        """Test that wire_from_manifest includes duplicate detection."""
        contract_a = _make_contract(
            name="a",
            package_name="pkg1",
            subscribe_topics=("onex.evt.platform.shared.v1",),
            handler_routing=None,  # will be skipped
        )
        contract_b = _make_contract(
            name="b",
            package_name="pkg2",
            subscribe_topics=("onex.evt.platform.shared.v1",),
            handler_routing=None,
        )
        manifest = ModelAutoWiringManifest(contracts=(contract_a, contract_b))
        engine = MagicMock()
        report = await wire_from_manifest(manifest, engine)
        assert len(report.duplicates) == 1
        assert report.duplicates[0].level == "package"


# ---------------------------------------------------------------------------
# Model tests
# ---------------------------------------------------------------------------


class TestModelHandlerRouting:
    def test_frozen(self) -> None:
        routing = _make_handler_routing()
        with pytest.raises(Exception):
            routing.routing_strategy = "changed"  # type: ignore[misc]

    def test_handler_ref_fields(self) -> None:
        ref = ModelHandlerRef(name="TestHandler", module="test.module")
        assert ref.name == "TestHandler"
        assert ref.module == "test.module"


class TestModelAutoWiringReport:
    def test_aggregate_properties(self) -> None:
        report = ModelAutoWiringReport(
            results=(
                ModelContractWiringResult(
                    contract_name="a",
                    package_name="p",
                    outcome=EnumWiringOutcome.WIRED,
                ),
                ModelContractWiringResult(
                    contract_name="b",
                    package_name="p",
                    outcome=EnumWiringOutcome.SKIPPED,
                ),
                ModelContractWiringResult(
                    contract_name="c",
                    package_name="p",
                    outcome=EnumWiringOutcome.FAILED,
                    reason="err",
                ),
            ),
        )
        assert report.total_wired == 1
        assert report.total_skipped == 1
        assert report.total_failed == 1
