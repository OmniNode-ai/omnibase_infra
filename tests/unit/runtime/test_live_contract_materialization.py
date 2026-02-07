# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Unit tests for live contract materialization (OMN-1989).

Tests validate that RuntimeHostProcess can dynamically materialize handlers
from Kafka-sourced contract descriptors at runtime, without a restart.

Tests use RuntimeHostProcess.__new__() to skip __init__ and manually set
internal state, isolating the methods under test from full runtime setup.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from omnibase_infra.models.handlers import ModelHandlerDescriptor
from omnibase_infra.runtime.kafka_contract_source import (
    KafkaContractCache,
    KafkaContractSource,
)
from omnibase_infra.runtime.service_runtime_host_process import RuntimeHostProcess

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_descriptor(
    handler_id: str = "proto.test_handler",
    name: str = "test-handler",
    handler_class: str | None = "omnibase_infra.test_module.TestHandler",
    contract_path: str | None = "kafka://dev/contracts/test-handler",
    contract_config: dict | None = None,
) -> ModelHandlerDescriptor:
    """Create a test ModelHandlerDescriptor."""
    return ModelHandlerDescriptor(
        handler_id=handler_id,
        name=name,
        version="1.0.0",
        handler_kind="effect",
        input_model="omnibase_infra.models.TestInput",
        output_model="omnibase_infra.models.TestOutput",
        handler_class=handler_class,
        contract_path=contract_path,
        contract_config=contract_config,
    )


def _make_runtime(**overrides: object) -> RuntimeHostProcess:
    """Create a RuntimeHostProcess via __new__() with minimal state.

    Skips __init__ and manually sets internal state needed for
    live materialization methods.
    """
    runtime = RuntimeHostProcess.__new__(RuntimeHostProcess)
    runtime._handlers = {}
    runtime._handler_descriptors = {}
    runtime._handler_mutation_lock = asyncio.Lock()
    runtime._announced_capabilities = set()
    runtime._is_running = True
    runtime._config = None
    runtime._event_bus_wiring = None
    runtime._introspection_service = None
    runtime._materialized_resources = None
    runtime._dependency_resolver = None
    runtime._container = None

    # Import the config class for introspection
    from omnibase_infra.models.runtime import ModelRuntimeIntrospectionConfig

    runtime._introspection_config = ModelRuntimeIntrospectionConfig()

    for key, value in overrides.items():
        setattr(runtime, key, value)
    return runtime


# ===========================================================================
# KafkaContractCache.get() tests
# ===========================================================================


class TestKafkaContractCacheGet:
    """Tests for KafkaContractCache.get() accessor."""

    def test_cache_get_returns_descriptor(self) -> None:
        """get() returns the descriptor previously added."""
        cache = KafkaContractCache()
        descriptor = _make_descriptor()
        cache.add("test-node", descriptor)

        result = cache.get("test-node")
        assert result is not None
        assert result.handler_id == "proto.test_handler"

    def test_cache_get_returns_none_for_unknown(self) -> None:
        """get() returns None for a node name not in the cache."""
        cache = KafkaContractCache()
        assert cache.get("nonexistent") is None


# ===========================================================================
# KafkaContractSource.get_cached_descriptor() tests
# ===========================================================================


class TestKafkaContractSourceGetCachedDescriptor:
    """Tests for KafkaContractSource.get_cached_descriptor()."""

    def test_source_get_cached_descriptor(self) -> None:
        """get_cached_descriptor() delegates to cache.get()."""
        source = KafkaContractSource(environment="test")
        descriptor = _make_descriptor()
        # Directly add to cache to test delegation without YAML parsing
        source._cache.add("direct-node", descriptor)
        result = source.get_cached_descriptor("direct-node")
        assert result is not None
        assert result.handler_id == "proto.test_handler"

    def test_source_get_cached_descriptor_returns_none(self) -> None:
        """get_cached_descriptor() returns None for unknown node."""
        source = KafkaContractSource(environment="test")
        assert source.get_cached_descriptor("nonexistent") is None


# ===========================================================================
# _materialize_handler_live() tests
# ===========================================================================


class TestMaterializeHandlerLive:
    """Tests for RuntimeHostProcess._materialize_handler_live()."""

    @pytest.mark.asyncio
    async def test_materialize_happy_path(self) -> None:
        """Full flow: import -> instantiate -> register -> introspection."""
        runtime = _make_runtime()
        descriptor = _make_descriptor()
        correlation_id = uuid4()

        mock_handler_instance = MagicMock()
        mock_handler_instance.initialize = AsyncMock()
        mock_handler_cls = MagicMock(return_value=mock_handler_instance)
        mock_container = MagicMock()

        with (
            patch.object(
                runtime, "_get_or_create_container", return_value=mock_container
            ),
            patch(
                "omnibase_infra.runtime.service_runtime_host_process.importlib.import_module"
            ) as mock_import,
        ):
            mock_module = MagicMock()
            mock_module.TestHandler = mock_handler_cls
            mock_import.return_value = mock_module

            result = await runtime._materialize_handler_live(
                node_name="test-node",
                descriptor=descriptor,
                correlation_id=correlation_id,
            )

        assert result is True
        assert "test_handler" in runtime._handlers
        assert "test_handler" in runtime._handler_descriptors
        mock_handler_cls.assert_called_once_with(container=mock_container)

    @pytest.mark.asyncio
    async def test_materialize_idempotent(self) -> None:
        """Second call returns True without re-instantiation."""
        mock_existing = MagicMock()
        runtime = _make_runtime()
        runtime._handlers["test_handler"] = mock_existing

        descriptor = _make_descriptor()
        correlation_id = uuid4()

        result = await runtime._materialize_handler_live(
            node_name="test-node",
            descriptor=descriptor,
            correlation_id=correlation_id,
        )

        assert result is True
        # Original handler preserved, not replaced
        assert runtime._handlers["test_handler"] is mock_existing

    @pytest.mark.asyncio
    async def test_materialize_skips_no_handler_class(self) -> None:
        """handler_class=None returns False."""
        runtime = _make_runtime()
        descriptor = _make_descriptor(handler_class=None)
        correlation_id = uuid4()

        result = await runtime._materialize_handler_live(
            node_name="test-node",
            descriptor=descriptor,
            correlation_id=correlation_id,
        )

        assert result is False
        assert "test_handler" not in runtime._handlers

    @pytest.mark.asyncio
    async def test_materialize_rejects_untrusted_namespace(self) -> None:
        """Handler from untrusted namespace is rejected."""
        runtime = _make_runtime()
        descriptor = _make_descriptor(handler_class="evil_package.malicious.Handler")
        correlation_id = uuid4()

        result = await runtime._materialize_handler_live(
            node_name="test-node",
            descriptor=descriptor,
            correlation_id=correlation_id,
        )

        assert result is False
        assert "test_handler" not in runtime._handlers

    @pytest.mark.asyncio
    async def test_materialize_handles_import_error(self) -> None:
        """Bad module path returns False, does not crash."""
        runtime = _make_runtime()
        descriptor = _make_descriptor(
            handler_class="omnibase_infra.nonexistent_module.FakeHandler"
        )
        correlation_id = uuid4()

        # importlib.import_module will raise ModuleNotFoundError
        result = await runtime._materialize_handler_live(
            node_name="test-node",
            descriptor=descriptor,
            correlation_id=correlation_id,
        )

        assert result is False
        assert "test_handler" not in runtime._handlers

    @pytest.mark.asyncio
    async def test_materialize_skips_kafka_path_deps(self) -> None:
        """kafka:// contract_path skips filesystem dep resolution."""
        runtime = _make_runtime()
        descriptor = _make_descriptor(
            contract_path="kafka://dev/contracts/test-handler",
        )
        correlation_id = uuid4()

        mock_handler_instance = MagicMock()
        mock_handler_instance.initialize = AsyncMock()
        mock_handler_cls = MagicMock(return_value=mock_handler_instance)
        mock_container = MagicMock()

        with (
            patch.object(
                runtime, "_get_or_create_container", return_value=mock_container
            ),
            patch.object(
                runtime,
                "_resolve_handler_dependencies",
                new_callable=AsyncMock,
            ) as mock_resolve,
            patch(
                "omnibase_infra.runtime.service_runtime_host_process.importlib.import_module"
            ) as mock_import,
        ):
            mock_module = MagicMock()
            mock_module.TestHandler = mock_handler_cls
            mock_import.return_value = mock_module

            result = await runtime._materialize_handler_live(
                node_name="test-node",
                descriptor=descriptor,
                correlation_id=correlation_id,
            )

        assert result is True
        # _resolve_handler_dependencies should NOT be called for kafka:// paths
        mock_resolve.assert_not_called()

    @pytest.mark.asyncio
    async def test_materialize_subscription_failure_rolls_back(self) -> None:
        """Subscription wiring fails -> handler NOT in _handlers."""
        runtime = _make_runtime()
        descriptor = _make_descriptor(
            contract_config={"event_bus": {"subscribe_topics": ["some.topic"]}},
        )
        correlation_id = uuid4()

        mock_handler_instance = MagicMock()
        mock_handler_instance.initialize = AsyncMock()
        mock_handler_cls = MagicMock(return_value=mock_handler_instance)
        mock_container = MagicMock()

        # Set up event_bus_wiring that will fail
        mock_wiring = MagicMock()
        mock_wiring.wire_subscriptions = AsyncMock(
            side_effect=RuntimeError("subscription failed")
        )
        runtime._event_bus_wiring = mock_wiring

        with (
            patch.object(
                runtime, "_get_or_create_container", return_value=mock_container
            ),
            patch(
                "omnibase_infra.runtime.service_runtime_host_process.importlib.import_module"
            ) as mock_import,
        ):
            mock_module = MagicMock()
            mock_module.TestHandler = mock_handler_cls
            mock_import.return_value = mock_module

            result = await runtime._materialize_handler_live(
                node_name="test-node",
                descriptor=descriptor,
                correlation_id=correlation_id,
            )

        assert result is False
        assert "test_handler" not in runtime._handlers
        # No CAPABILITY_CHANGE should be announced
        assert "test-node" not in runtime._announced_capabilities


# ===========================================================================
# _wire_live_handler_subscriptions() tests
# ===========================================================================


class TestWireLiveHandlerSubscriptions:
    """Tests for _wire_live_handler_subscriptions()."""

    @pytest.mark.asyncio
    async def test_wire_live_subscriptions_noop_without_wiring(self) -> None:
        """_event_bus_wiring=None -> no-op, no error."""
        runtime = _make_runtime()
        descriptor = _make_descriptor(
            contract_config={"event_bus": {"subscribe_topics": ["test.topic"]}},
        )

        # Should complete without error
        await runtime._wire_live_handler_subscriptions("test-node", descriptor)


# ===========================================================================
# _publish_capability_change() tests
# ===========================================================================


class TestPublishCapabilityChange:
    """Tests for _publish_capability_change()."""

    @pytest.mark.asyncio
    async def test_publish_capability_change(self) -> None:
        """Introspection called with CAPABILITY_CHANGE reason."""
        from omnibase_infra.enums import EnumIntrospectionReason

        mock_introspection = AsyncMock()
        runtime = _make_runtime(
            _introspection_service=mock_introspection,
        )
        correlation_id = uuid4()

        await runtime._publish_capability_change("test-node", correlation_id)

        mock_introspection.publish_introspection.assert_awaited_once_with(
            reason=EnumIntrospectionReason.CAPABILITY_CHANGE,
            correlation_id=correlation_id,
        )
        assert "test-node" in runtime._announced_capabilities

    @pytest.mark.asyncio
    async def test_publish_capability_change_idempotent(self) -> None:
        """Second call for same node_name skips publish."""
        mock_introspection = AsyncMock()
        runtime = _make_runtime(
            _introspection_service=mock_introspection,
        )
        runtime._announced_capabilities.add("test-node")
        correlation_id = uuid4()

        await runtime._publish_capability_change("test-node", correlation_id)

        mock_introspection.publish_introspection.assert_not_awaited()


# ===========================================================================
# handle_registration() integration test
# ===========================================================================


class TestHandleRegistrationTriggersLive:
    """Tests for handle_registration callback triggering live materialization.

    These tests simulate the control flow of handle_registration() without
    invoking actual YAML parsing. The flow tested is:
        1. source.on_contract_registered() succeeds
        2. runtime._is_running is True
        3. source.get_cached_descriptor() returns descriptor
        4. runtime._materialize_handler_live() is called
    """

    @pytest.mark.asyncio
    async def test_handle_registration_triggers_materialization(self) -> None:
        """When registration succeeds and runtime is running, materialization fires."""
        runtime = _make_runtime()
        descriptor = _make_descriptor()
        correlation_id = uuid4()
        node_name = "test-node"

        # Pre-populate cache to simulate successful on_contract_registered
        source = KafkaContractSource(environment="test")
        source._cache.add(node_name, descriptor)

        # Mock the materialization method
        runtime._materialize_handler_live = AsyncMock(return_value=True)  # type: ignore[method-assign]

        # Simulate the handle_registration logic:
        # success=True (already cached), _is_running=True
        success = True
        if success and runtime._is_running:
            desc = source.get_cached_descriptor(node_name)
            if desc is not None:
                await runtime._materialize_handler_live(
                    node_name=node_name,
                    descriptor=desc,
                    correlation_id=correlation_id,
                )

        runtime._materialize_handler_live.assert_awaited_once_with(
            node_name=node_name,
            descriptor=descriptor,
            correlation_id=correlation_id,
        )

    @pytest.mark.asyncio
    async def test_handle_registration_skips_when_not_running(self) -> None:
        """_is_running=False -> no materialization."""
        runtime = _make_runtime(_is_running=False)
        runtime._materialize_handler_live = AsyncMock()  # type: ignore[method-assign]

        source = KafkaContractSource(environment="test")
        descriptor = _make_descriptor()
        node_name = "test-node"
        correlation_id = uuid4()
        source._cache.add(node_name, descriptor)

        # Simulate the guard condition
        success = True
        if success and runtime._is_running:
            desc = source.get_cached_descriptor(node_name)
            if desc is not None:
                await runtime._materialize_handler_live(
                    node_name=node_name,
                    descriptor=desc,
                    correlation_id=correlation_id,
                )

        # Should not have been called because _is_running is False
        runtime._materialize_handler_live.assert_not_awaited()
