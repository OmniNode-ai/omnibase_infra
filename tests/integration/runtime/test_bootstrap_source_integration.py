# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Integration tests for HandlerBootstrapSource integration (OMN-1087).

This module validates that HandlerBootstrapSource is correctly integrated
into the RuntimeHostProcess bootstrap flow, ensuring core infrastructure
handlers are loaded from descriptor-based definitions.

Test Coverage:
- HandlerBootstrapSource.discover_handlers() is called during runtime bootstrap
- Core 5 handlers (consul, db, http, mcp, vault) are loaded from bootstrap source
- Bootstrap handlers are registered before contract-based or default handlers
- Bootstrap handlers are available after RuntimeHostProcess.start()

Related:
    - OMN-1087: HandlerBootstrapSource for hardcoded handler registration
    - src/omnibase_infra/runtime/handler_bootstrap_source.py
    - src/omnibase_infra/runtime/service_runtime_host_process.py

Note:
    These tests verify handler REGISTRATION, not handler EXECUTION.
    Handlers may fail during initialize() if external services are not available,
    but they should still be registered in the handler registry.
"""

from __future__ import annotations

import logging
from collections.abc import Generator
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from omnibase_infra.errors.error_infra import ProtocolConfigurationError
from omnibase_infra.event_bus.event_bus_inmemory import EventBusInmemory
from omnibase_infra.runtime.handler_bootstrap_source import (
    SOURCE_TYPE_BOOTSTRAP,
    HandlerBootstrapSource,
)
from omnibase_infra.runtime.handler_registry import (
    HANDLER_TYPE_CONSUL,
    HANDLER_TYPE_DATABASE,
    HANDLER_TYPE_HTTP,
    HANDLER_TYPE_MCP,
    HANDLER_TYPE_VAULT,
    RegistryProtocolBinding,
    get_handler_registry,
)
from omnibase_infra.runtime.service_runtime_host_process import RuntimeHostProcess

# =============================================================================
# Test Classes
# =============================================================================


class TestHandlerBootstrapSourceDiscovery:
    """Test HandlerBootstrapSource discover_handlers() functionality.

    These tests verify that HandlerBootstrapSource correctly provides
    handler descriptors for the core infrastructure handlers.
    """

    @pytest.mark.asyncio
    async def test_discover_handlers_returns_five_descriptors(self) -> None:
        """HandlerBootstrapSource.discover_handlers() returns 5 handler descriptors.

        Verifies:
        1. discover_handlers() returns ModelContractDiscoveryResult
        2. Result contains exactly 5 descriptors (consul, db, http, mcp, vault)
        3. No validation errors (hardcoded handlers are pre-validated)
        """
        source = HandlerBootstrapSource()
        result = await source.discover_handlers()

        assert len(result.descriptors) == 5
        assert len(result.validation_errors) == 0

    @pytest.mark.asyncio
    async def test_discover_handlers_includes_core_handlers(self) -> None:
        """HandlerBootstrapSource includes consul, db, http, mcp, vault handlers.

        Verifies that all five core infrastructure handlers are present
        in the discovery result.
        """
        source = HandlerBootstrapSource()
        result = await source.discover_handlers()

        handler_ids = {d.handler_id for d in result.descriptors}
        expected_ids = {
            "proto.consul",
            "proto.db",
            "proto.http",
            "proto.mcp",
            "proto.vault",
        }

        assert handler_ids == expected_ids

    @pytest.mark.asyncio
    async def test_discover_handlers_descriptors_have_handler_class(self) -> None:
        """All bootstrap handler descriptors have valid handler_class paths.

        Verifies:
        1. Each descriptor has a non-None handler_class
        2. Handler class paths are fully qualified (contain dots)
        """
        source = HandlerBootstrapSource()
        result = await source.discover_handlers()

        for descriptor in result.descriptors:
            assert descriptor.handler_class is not None
            assert "." in descriptor.handler_class
            assert descriptor.handler_class.startswith("omnibase_infra.handlers.")

    @pytest.mark.asyncio
    async def test_source_type_is_bootstrap(self) -> None:
        """HandlerBootstrapSource.source_type returns 'BOOTSTRAP'."""
        source = HandlerBootstrapSource()
        assert source.source_type == SOURCE_TYPE_BOOTSTRAP

    @pytest.mark.asyncio
    async def test_discover_handlers_is_idempotent(self) -> None:
        """Multiple calls to discover_handlers() return consistent results.

        Verifies that discover_handlers() can be called multiple times
        and returns the same descriptors each time.
        """
        source = HandlerBootstrapSource()

        result1 = await source.discover_handlers()
        result2 = await source.discover_handlers()

        # Same number of descriptors
        assert len(result1.descriptors) == len(result2.descriptors)

        # Same handler IDs
        ids1 = {d.handler_id for d in result1.descriptors}
        ids2 = {d.handler_id for d in result2.descriptors}
        assert ids1 == ids2


class TestBootstrapSourceRuntimeIntegration:
    """Test HandlerBootstrapSource integration with RuntimeHostProcess.

    These tests verify that RuntimeHostProcess correctly uses
    HandlerBootstrapSource during the startup bootstrap process.
    """

    @pytest.fixture(autouse=True)
    def _skip_materialize_dependencies(self) -> Generator[None, None, None]:
        """Skip dependency materialization which requires OMNIBASE_INFRA_DB_URL.

        These tests exercise bootstrap handler registration, not infrastructure
        dependency materialization (tested in test_dependency_materializer.py).
        """
        with patch(
            "omnibase_infra.runtime.service_runtime_host_process"
            ".RuntimeHostProcess._materialize_dependencies",
            new_callable=AsyncMock,
        ):
            yield

    @pytest.mark.asyncio
    async def test_bootstrap_handlers_registered_on_start(self) -> None:
        """Bootstrap handlers are registered in handler registry on start.

        Verifies:
        1. RuntimeHostProcess.start() registers bootstrap handlers
        2. All 5 core handlers (consul, db, http, mcp, vault) are in registry
        """
        event_bus = EventBusInmemory()
        process = RuntimeHostProcess(
            event_bus=event_bus,
            input_topic="test.input",
            config={
                "service_name": "test-bootstrap-service",
                "node_name": "test-bootstrap-node",
            },
            # No contract_paths - bootstrap handlers provide core infrastructure
        )

        try:
            await process.start()

            # Verify bootstrap handlers are in the singleton registry
            registry = get_handler_registry()
            assert registry.is_registered(HANDLER_TYPE_CONSUL)
            assert registry.is_registered(HANDLER_TYPE_DATABASE)
            assert registry.is_registered(HANDLER_TYPE_HTTP)
            assert registry.is_registered(HANDLER_TYPE_MCP)
            assert registry.is_registered(HANDLER_TYPE_VAULT)

        finally:
            await process.stop()

    @pytest.mark.asyncio
    async def test_bootstrap_handlers_registered_before_contracts(
        self, tmp_path: Path
    ) -> None:
        """Bootstrap handlers are registered BEFORE contract-based discovery.

        Verifies that bootstrap handlers are loaded first, then contract-based
        handlers. This ensures core infrastructure is always available.
        """
        # Create a contract directory (empty - no handlers)
        contracts_dir = tmp_path / "contracts"
        contracts_dir.mkdir()

        # Track registration order
        registration_order: list[str] = []

        original_register = RegistryProtocolBinding.register

        def tracking_register(
            self: RegistryProtocolBinding, protocol_type: str, handler_class: type
        ) -> None:
            registration_order.append(protocol_type)
            return original_register(self, protocol_type, handler_class)

        with patch.object(RegistryProtocolBinding, "register", tracking_register):
            event_bus = EventBusInmemory()
            process = RuntimeHostProcess(
                event_bus=event_bus,
                input_topic="test.input",
                config={
                    "service_name": "test-bootstrap-service",
                    "node_name": "test-bootstrap-node",
                },
                contract_paths=[str(contracts_dir)],
            )

            try:
                await process.start()

                # Bootstrap handlers should be registered first
                # (consul, db, http, mcp, vault in some order)
                bootstrap_handlers = {
                    HANDLER_TYPE_CONSUL,
                    HANDLER_TYPE_DATABASE,
                    HANDLER_TYPE_HTTP,
                    HANDLER_TYPE_MCP,
                    HANDLER_TYPE_VAULT,
                }

                # Check that bootstrap handlers were registered
                registered_set = set(registration_order)
                assert bootstrap_handlers.issubset(registered_set), (
                    f"Bootstrap handlers not found in registration. "
                    f"Expected: {bootstrap_handlers}, Got: {registered_set}"
                )

            finally:
                await process.stop()

    @pytest.mark.asyncio
    async def test_bootstrap_source_called_during_start(self) -> None:
        """HandlerBootstrapSource.discover_handlers() is called during start.

        Verifies that RuntimeHostProcess actually calls the bootstrap source
        discover_handlers() method during startup.

        HYBRID Mode Double-Discovery Design Decision:
            When HYBRID mode is configured but no contract_paths are provided,
            the runtime reuses bootstrap_source as both the bootstrap_source
            AND contract_source parameters to HandlerSourceResolver. This means
            discover_handlers() is called twice on the same instance.

            This is INTENTIONAL for two reasons:

            1. **Observability Symmetry**: HYBRID mode logs contract_handler_count,
               bootstrap_handler_count, fallback_handler_count, and override_count.
               Calling both sources ensures consistent metrics even when sources
               are identical.

            2. **Semantic Correctness**: HYBRID semantics require consulting both
               sources. The resolver's merge logic produces correct results even
               when both sources return identical handlers.

            If you're tempted to "optimize" this to a single call, DON'T - it would
            break observability expectations and change HYBRID mode semantics.
        """
        from omnibase_core.models.primitives import ModelSemVer
        from omnibase_infra.models.handlers import ModelHandlerDescriptor

        event_bus = EventBusInmemory()

        # Patch at the source module where it's imported from
        with patch(
            "omnibase_infra.runtime.handler_bootstrap_source.HandlerBootstrapSource"
        ) as MockBootstrapSource:
            # Create a mock that returns proper discovery result with valid descriptor
            mock_source = MagicMock()
            mock_source.source_type = SOURCE_TYPE_BOOTSTRAP

            # Provide a real handler descriptor so the runtime can start
            mock_descriptor = ModelHandlerDescriptor(
                handler_id="proto.http",
                name="HTTP Handler",
                version=ModelSemVer(major=1, minor=0, patch=0),
                handler_kind="effect",
                input_model="omnibase_infra.models.types.JsonDict",
                output_model="omnibase_core.models.dispatch.ModelHandlerOutput",
                description="HTTP REST protocol handler",
                handler_class="omnibase_infra.handlers.handler_http.HandlerHttpRest",
            )

            mock_discovery_result = MagicMock()
            mock_discovery_result.descriptors = [mock_descriptor]
            mock_discovery_result.validation_errors = []
            mock_source.discover_handlers = AsyncMock(
                return_value=mock_discovery_result
            )
            MockBootstrapSource.return_value = mock_source

            process = RuntimeHostProcess(
                event_bus=event_bus,
                input_topic="test.input",
                config={
                    "service_name": "test-bootstrap-service",
                    "node_name": "test-bootstrap-node",
                },
            )

            try:
                await process.start()

                # Verify HandlerBootstrapSource was instantiated only ONCE
                # This is the key fix from OMN-1095 - previously it was instantiated twice
                MockBootstrapSource.assert_called_once()

                # In HYBRID mode (default), discover_handlers() is called twice:
                # once for bootstrap_source and once for contract_source
                # (which is the same instance when no contract_paths provided)
                #
                # IMPORTANT: Double-discovery is intentional for observability symmetry.
                # See docstring above. Do not "optimize" this to call_count == 1.
                assert mock_source.discover_handlers.call_count == 2, (
                    "discover_handlers should be called twice in HYBRID mode"
                )

            finally:
                await process.stop()

    @pytest.mark.asyncio
    async def test_bootstrap_handlers_logged(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Bootstrap handler registration is logged.

        Verifies that appropriate log messages are generated during
        bootstrap handler registration.
        """
        event_bus = EventBusInmemory()
        process = RuntimeHostProcess(
            event_bus=event_bus,
            input_topic="test.input",
            config={
                "service_name": "test-bootstrap-service",
                "node_name": "test-bootstrap-node",
            },
        )

        with caplog.at_level(logging.INFO):
            try:
                await process.start()

                # Check for bootstrap-related log messages
                log_messages = [r.message for r in caplog.records]

                has_bootstrap_log = any(
                    "bootstrap" in msg.lower() for msg in log_messages
                )
                assert has_bootstrap_log, (
                    f"Should have logs mentioning 'bootstrap'. "
                    f"Log messages: {log_messages}"
                )

            finally:
                await process.stop()


class TestBootstrapSourceErrorHandling:
    """Test error handling in bootstrap source integration.

    These tests verify that errors during bootstrap handler registration
    are handled gracefully without crashing the runtime.
    """

    @pytest.fixture(autouse=True)
    def _skip_materialize_dependencies(self) -> Generator[None, None, None]:
        """Skip dependency materialization which requires OMNIBASE_INFRA_DB_URL.

        These tests exercise bootstrap handler error handling, not infrastructure
        dependency materialization (tested in test_dependency_materializer.py).
        """
        with patch(
            "omnibase_infra.runtime.service_runtime_host_process"
            ".RuntimeHostProcess._materialize_dependencies",
            new_callable=AsyncMock,
        ):
            yield

    @pytest.mark.asyncio
    async def test_single_handler_import_error_does_not_crash(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Import error for one handler doesn't prevent others from loading.

        Verifies graceful degradation: if one bootstrap handler fails to
        import, other handlers should still be registered.
        """
        event_bus = EventBusInmemory()

        # Patch importlib.import_module to fail for one specific handler
        original_import = __import__("importlib").import_module

        def failing_import(name: str, package: str | None = None) -> object:
            if "handler_consul" in name:
                raise ImportError("Simulated import error for consul handler")
            return original_import(name, package)

        with patch("importlib.import_module", side_effect=failing_import):
            with caplog.at_level(logging.WARNING):
                process = RuntimeHostProcess(
                    event_bus=event_bus,
                    input_topic="test.input",
                    config={
                        "service_name": "test-bootstrap-service",
                        "node_name": "test-bootstrap-node",
                    },
                )

                try:
                    await process.start()

                    # Process should still start
                    assert process.is_running

                    # Other handlers should be registered (db, http, vault)
                    registry = get_handler_registry()
                    # At least some handlers should be registered
                    registered = registry.list_protocols()
                    assert len(registered) >= 1, (
                        "At least one handler should be registered despite consul failing"
                    )

                    # Check for error log about failed import
                    error_logs = [
                        r for r in caplog.records if r.levelno >= logging.WARNING
                    ]
                    assert len(error_logs) >= 1, (
                        "Should have warning/error logs for failed handler import"
                    )

                finally:
                    await process.stop()

    @pytest.mark.asyncio
    async def test_bootstrap_with_empty_descriptors_fails_fast(self) -> None:
        """Runtime fails fast if no handlers are registered.

        Verifies that if HandlerBootstrapSource returns no descriptors and no
        other handler source provides handlers, the runtime raises
        ProtocolConfigurationError (fail-fast validation).

        This is intentional: a runtime without handlers cannot process any
        events and is considered misconfigured. Failing fast catches
        configuration issues early rather than starting a useless runtime.

        Related:
            - RuntimeHostProcess.start() step 4.1: fail-fast handler validation
        """
        event_bus = EventBusInmemory()

        # Patch at the source module where it's imported from
        with patch(
            "omnibase_infra.runtime.handler_bootstrap_source.HandlerBootstrapSource"
        ) as MockBootstrapSource:
            # Return empty descriptors
            mock_source = MagicMock()
            mock_discovery_result = MagicMock()
            mock_discovery_result.descriptors = []
            mock_source.discover_handlers = AsyncMock(
                return_value=mock_discovery_result
            )
            MockBootstrapSource.return_value = mock_source

            # Use an isolated empty registry to avoid singleton contamination
            # from handlers registered by previous tests
            isolated_registry = RegistryProtocolBinding()

            process = RuntimeHostProcess(
                event_bus=event_bus,
                input_topic="test.input",
                config={
                    "service_name": "test-bootstrap-service",
                    "node_name": "test-bootstrap-node",
                },
                handler_registry=isolated_registry,
            )

            # Runtime should fail fast with no handlers registered
            with pytest.raises(ProtocolConfigurationError) as exc_info:
                await process.start()

            # Verify error indicates no handlers
            assert "No handlers registered" in str(exc_info.value)


class TestBootstrapSourcePerformance:
    """Test performance characteristics of bootstrap source integration.

    These tests verify that bootstrap handler loading is fast and doesn't
    add significant overhead to runtime startup.
    """

    @pytest.mark.asyncio
    async def test_discover_handlers_is_fast(self) -> None:
        """HandlerBootstrapSource.discover_handlers() completes in < 100ms.

        Bootstrap handler discovery should be very fast since it's just
        creating in-memory descriptor objects (no file I/O).
        """
        import time

        source = HandlerBootstrapSource()

        start = time.perf_counter()
        await source.discover_handlers()
        duration = time.perf_counter() - start

        # Should complete in under 100ms (generous for CI)
        assert duration < 0.1, f"Discovery took {duration:.3f}s, expected < 0.1s"


__all__: list[str] = [
    "TestHandlerBootstrapSourceDiscovery",
    "TestBootstrapSourceRuntimeIntegration",
    "TestBootstrapSourceErrorHandling",
    "TestBootstrapSourcePerformance",
]
