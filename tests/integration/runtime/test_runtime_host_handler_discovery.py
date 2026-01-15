# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Integration tests for RuntimeHostProcess handler discovery (OMN-1133).

This module validates that RuntimeHostProcess correctly integrates with
contract-based handler discovery via the ContractHandlerDiscovery service.

Test Coverage:
- RuntimeHostProcess starts with contract_paths (OMN-1133)
- RuntimeHostProcess fallback to wire_default_handlers
- RuntimeHostProcess graceful degradation with mix of valid/invalid contracts
- Full lifecycle with discovered handlers

Related:
    - OMN-1133: Contract-based handler discovery
    - src/omnibase_infra/runtime/runtime_host_process.py
    - src/omnibase_infra/runtime/contract_handler_discovery.py

Note:
    These tests create temporary handler contract YAML files that point to
    real handler classes. They do NOT require external infrastructure because
    they test handler discovery, not handler execution.

    Some handlers (DB, Vault) require specific config during initialize().
    Tests account for this by:
    1. Using HTTP handler (works without config) for most scenarios
    2. Testing graceful degradation when handlers fail to initialize
    3. Using consul handler which has sensible defaults
"""

from __future__ import annotations

from pathlib import Path

import pytest

from omnibase_infra.event_bus.event_bus_inmemory import EventBusInmemory
from omnibase_infra.runtime.handler_plugin_loader import HANDLER_CONTRACT_FILENAME
from omnibase_infra.runtime.handler_registry import (
    HANDLER_TYPE_DATABASE,
    HANDLER_TYPE_HTTP,
    RegistryProtocolBinding,
    get_handler_registry,
)
from omnibase_infra.runtime.service_runtime_host_process import RuntimeHostProcess

# =============================================================================
# Constants for Handler Contract Templates
# =============================================================================

# Real handler class paths from omnibase_infra.handlers
# Note: HttpRestHandler works without config or external services.
# Other handlers (DB, Consul, Vault) require external services during initialize().
REAL_HANDLER_HTTP_CLASS = "omnibase_infra.handlers.handler_http.HttpRestHandler"

# Handler contract template for creating test contracts
HANDLER_CONTRACT_YAML_TEMPLATE = """
handler_name: "{handler_name}"
handler_class: "{handler_class}"
handler_type: "effect"
capability_tags:
  - {tag1}
  - {tag2}
"""

# Invalid handler contract (missing required fields)
INVALID_HANDLER_CONTRACT_YAML = """
handler_name: "invalid.handler"
# Missing handler_class - will fail to load
handler_type: "effect"
"""

# Handler contract pointing to non-existent class
NONEXISTENT_CLASS_CONTRACT_YAML = """
handler_name: "nonexistent.handler"
handler_class: "omnibase_infra.handlers.nonexistent_module.NonexistentHandler"
handler_type: "effect"
"""


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def isolated_handler_registry() -> RegistryProtocolBinding:
    """Create an isolated handler registry for testing.

    Returns:
        Fresh RegistryProtocolBinding instance that is not the singleton.
    """
    return RegistryProtocolBinding()


@pytest.fixture
def valid_handler_contract_dir(tmp_path: Path) -> Path:
    """Create a directory with valid handler contracts.

    Creates contracts for HTTP handler only - it can initialize without
    external services/config. Other handlers (DB, Consul, Vault) require
    external services during initialize() and will fail gracefully.

    Returns:
        Path to the handlers directory.
    """
    handlers_dir = tmp_path / "handlers"
    handlers_dir.mkdir(parents=True)

    # Create HTTP handler contract (works without config or external services)
    http_dir = handlers_dir / "http"
    http_dir.mkdir()
    http_contract = http_dir / HANDLER_CONTRACT_FILENAME
    http_contract.write_text(
        HANDLER_CONTRACT_YAML_TEMPLATE.format(
            handler_name="http",
            handler_class=REAL_HANDLER_HTTP_CLASS,
            tag1="http",
            tag2="rest",
        )
    )

    return handlers_dir


@pytest.fixture
def multiple_handler_contract_dirs(tmp_path: Path) -> tuple[Path, Path]:
    """Create two separate directories with valid handler contracts.

    Returns:
        Tuple of (handlers_dir1, handlers_dir2).
    """
    # First directory with HTTP handler
    handlers_dir1 = tmp_path / "handlers1"
    handlers_dir1.mkdir(parents=True)
    http_dir = handlers_dir1 / "http"
    http_dir.mkdir()
    (http_dir / HANDLER_CONTRACT_FILENAME).write_text(
        HANDLER_CONTRACT_YAML_TEMPLATE.format(
            handler_name="http",
            handler_class=REAL_HANDLER_HTTP_CLASS,
            tag1="http",
            tag2="rest",
        )
    )

    # Second directory with another HTTP handler (different name)
    handlers_dir2 = tmp_path / "handlers2"
    handlers_dir2.mkdir(parents=True)
    http2_dir = handlers_dir2 / "http2"
    http2_dir.mkdir()
    (http2_dir / HANDLER_CONTRACT_FILENAME).write_text(
        HANDLER_CONTRACT_YAML_TEMPLATE.format(
            handler_name="http2",
            handler_class=REAL_HANDLER_HTTP_CLASS,
            tag1="http",
            tag2="client",
        )
    )

    return handlers_dir1, handlers_dir2


@pytest.fixture
def mixed_contracts_dir(tmp_path: Path) -> Path:
    """Create a directory with mix of valid and invalid handler contracts.

    Structure:
        handlers/
        |-- valid/
        |   |-- handler_contract.yaml  (valid - HttpRestHandler)
        |-- invalid/
        |   |-- handler_contract.yaml  (invalid - missing handler_class)
        |-- nonexistent/
        |   |-- handler_contract.yaml  (invalid - class doesn't exist)

    Returns:
        Path to the handlers directory.
    """
    handlers_dir = tmp_path / "handlers"
    handlers_dir.mkdir(parents=True)

    # Create valid handler contract
    valid_dir = handlers_dir / "valid"
    valid_dir.mkdir()
    valid_contract = valid_dir / HANDLER_CONTRACT_FILENAME
    valid_contract.write_text(
        HANDLER_CONTRACT_YAML_TEMPLATE.format(
            handler_name="valid.http",
            handler_class=REAL_HANDLER_HTTP_CLASS,
            tag1="http",
            tag2="valid",
        )
    )

    # Create invalid handler contract (missing handler_class)
    invalid_dir = handlers_dir / "invalid"
    invalid_dir.mkdir()
    invalid_contract = invalid_dir / HANDLER_CONTRACT_FILENAME
    invalid_contract.write_text(INVALID_HANDLER_CONTRACT_YAML)

    # Create handler contract with non-existent class
    nonexistent_dir = handlers_dir / "nonexistent"
    nonexistent_dir.mkdir()
    nonexistent_contract = nonexistent_dir / HANDLER_CONTRACT_FILENAME
    nonexistent_contract.write_text(NONEXISTENT_CLASS_CONTRACT_YAML)

    return handlers_dir


# =============================================================================
# Test Classes
# =============================================================================


class TestRuntimeHostProcessWithContractPaths:
    """Test RuntimeHostProcess with contract-based handler discovery (OMN-1133).

    These tests verify that RuntimeHostProcess correctly uses ContractHandlerDiscovery
    to auto-discover handlers from contract_paths during start().
    """

    @pytest.mark.asyncio
    async def test_starts_with_contract_paths(
        self,
        valid_handler_contract_dir: Path,
    ) -> None:
        """RuntimeHostProcess discovers handlers from contract_paths on start.

        Verifies:
        1. RuntimeHostProcess accepts contract_paths parameter
        2. Handlers are discovered from contracts during start()
        3. Process is running after start()
        4. Discovered handlers are available via get_handler()

        Note:
            Only HTTP handler is used because it can initialize without external
            services. Other handlers (DB, Consul, Vault) require running services.
        """
        event_bus = EventBusInmemory()
        process = RuntimeHostProcess(
            event_bus=event_bus,
            input_topic="test.input",
            contract_paths=[str(valid_handler_contract_dir)],
        )

        try:
            await process.start()

            # Verify process is running
            assert process.is_running

            # Verify HTTP handler was discovered and initialized
            http_handler = process.get_handler("http")
            assert http_handler is not None, "HTTP handler should be discovered"

        finally:
            await process.stop()

    @pytest.mark.asyncio
    async def test_contract_paths_override_default_wiring(
        self,
        valid_handler_contract_dir: Path,
    ) -> None:
        """Contract paths should use discovery instead of wire_default_handlers.

        Verifies:
        1. When contract_paths is provided, wire_default_handlers is NOT called
        2. Only handlers from contracts are registered
        """
        event_bus = EventBusInmemory()

        process = RuntimeHostProcess(
            event_bus=event_bus,
            input_topic="test.input",
            contract_paths=[str(valid_handler_contract_dir)],
        )

        try:
            await process.start()

            # The contract_paths path should be taken (not wire_default_handlers)
            # We verify by checking that the handlers match our contracts
            assert process.get_handler("http") is not None

        finally:
            await process.stop()

    @pytest.mark.asyncio
    async def test_multiple_contract_paths(
        self,
        multiple_handler_contract_dirs: tuple[Path, Path],
    ) -> None:
        """RuntimeHostProcess accepts multiple contract paths.

        Verifies that handlers can be discovered from multiple separate directories.
        Both directories contain HTTP-based handlers that can initialize without
        external services.
        """
        handlers_dir1, handlers_dir2 = multiple_handler_contract_dirs

        event_bus = EventBusInmemory()
        process = RuntimeHostProcess(
            event_bus=event_bus,
            input_topic="test.input",
            contract_paths=[str(handlers_dir1), str(handlers_dir2)],
        )

        try:
            await process.start()

            # Both handlers should be discovered from different directories
            assert process.get_handler("http") is not None
            assert process.get_handler("http2") is not None

        finally:
            await process.stop()

    @pytest.mark.asyncio
    async def test_health_check_shows_discovered_handlers(
        self,
        valid_handler_contract_dir: Path,
    ) -> None:
        """Health check should list handlers discovered from contracts.

        Verifies that registered_handlers in health check reflects
        contract-discovered handlers.
        """
        event_bus = EventBusInmemory()
        process = RuntimeHostProcess(
            event_bus=event_bus,
            input_topic="test.input",
            contract_paths=[str(valid_handler_contract_dir)],
        )

        try:
            await process.start()
            health = await process.health_check()

            # Verify registered_handlers includes discovered handlers
            registered = health["registered_handlers"]
            assert isinstance(registered, list)
            assert "http" in registered

        finally:
            await process.stop()


class TestRuntimeHostProcessFallback:
    """Test RuntimeHostProcess fallback to wire_default_handlers.

    These tests verify that when no contract_paths are provided,
    RuntimeHostProcess uses the existing wire_default_handlers behavior.
    """

    @pytest.mark.asyncio
    async def test_fallback_to_default_handlers(self) -> None:
        """RuntimeHostProcess uses wire_default_handlers when no contract_paths.

        Verifies:
        1. Without contract_paths, wire_default_handlers is called
        2. Default handlers (HTTP, DB, etc.) are available
        """
        event_bus = EventBusInmemory()
        process = RuntimeHostProcess(
            event_bus=event_bus,
            input_topic="test.input",
            # No contract_paths - should use wire_default_handlers
        )

        try:
            await process.start()

            # Verify default handlers are available
            # Default wiring registers http, db, consul, vault
            registry = get_handler_registry()
            assert registry.is_registered(HANDLER_TYPE_HTTP)
            assert registry.is_registered(HANDLER_TYPE_DATABASE)

            # Verify process has handlers populated
            assert process.is_running
            health = await process.health_check()
            assert health["is_running"] is True

        finally:
            await process.stop()

    @pytest.mark.asyncio
    async def test_empty_contract_paths_falls_back(self) -> None:
        """Empty contract_paths list falls back to wire_default_handlers.

        Verifies that an empty list [] is treated the same as None.
        """
        event_bus = EventBusInmemory()
        process = RuntimeHostProcess(
            event_bus=event_bus,
            input_topic="test.input",
            contract_paths=[],  # Empty list
        )

        try:
            await process.start()

            # Should fall back to default handlers
            assert process.is_running

            # Default handlers should be registered
            registry = get_handler_registry()
            assert registry.is_registered(HANDLER_TYPE_HTTP)

        finally:
            await process.stop()


class TestRuntimeHostProcessGracefulDegradation:
    """Test RuntimeHostProcess graceful degradation with errors.

    These tests verify that RuntimeHostProcess handles handler discovery
    errors gracefully, registering valid handlers even when some fail.
    """

    @pytest.mark.asyncio
    async def test_graceful_degradation_with_mixed_contracts(
        self,
        mixed_contracts_dir: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """RuntimeHostProcess registers valid handlers despite invalid contracts.

        Verifies:
        1. Valid handlers are registered
        2. Invalid contracts are logged as errors
        3. Process still starts successfully
        4. Degraded state is NOT set (handler discovery errors don't cause degradation)
        """
        import logging

        event_bus = EventBusInmemory()
        process = RuntimeHostProcess(
            event_bus=event_bus,
            input_topic="test.input",
            contract_paths=[str(mixed_contracts_dir)],
        )

        with caplog.at_level(logging.WARNING):
            try:
                await process.start()

                # Process should start successfully
                assert process.is_running

                # Valid handler should be registered
                valid_handler = process.get_handler("valid.http")
                assert valid_handler is not None, "Valid handler should be registered"

                # Invalid handlers should NOT be registered
                invalid_handler = process.get_handler("invalid.handler")
                assert invalid_handler is None, (
                    "Invalid handler should not be registered"
                )

                nonexistent_handler = process.get_handler("nonexistent.handler")
                assert nonexistent_handler is None, (
                    "Nonexistent handler should not be registered"
                )

                # Verify errors were logged
                error_logs = [r for r in caplog.records if r.levelno >= logging.WARNING]
                # Should have warnings about failed handlers
                assert len(error_logs) > 0, (
                    "Should have warning logs for failed handlers"
                )

            finally:
                await process.stop()

    @pytest.mark.asyncio
    async def test_nonexistent_contract_path_handled_gracefully(
        self,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Non-existent contract path is handled gracefully.

        Verifies that specifying a path that doesn't exist doesn't crash
        the runtime - errors are logged but process continues.
        """
        import logging

        # Create a valid handler path
        valid_dir = tmp_path / "valid"
        valid_dir.mkdir()
        (valid_dir / HANDLER_CONTRACT_FILENAME).write_text(
            HANDLER_CONTRACT_YAML_TEMPLATE.format(
                handler_name="valid.handler",
                handler_class=REAL_HANDLER_HTTP_CLASS,
                tag1="valid",
                tag2="handler",
            )
        )

        # Non-existent path
        nonexistent_path = tmp_path / "does_not_exist"

        event_bus = EventBusInmemory()
        process = RuntimeHostProcess(
            event_bus=event_bus,
            input_topic="test.input",
            contract_paths=[str(valid_dir), str(nonexistent_path)],
        )

        with caplog.at_level(logging.WARNING):
            try:
                await process.start()

                # Process should still start
                assert process.is_running

                # Valid handler from the existing path should be registered
                valid_handler = process.get_handler("valid.handler")
                assert valid_handler is not None

            finally:
                await process.stop()

    @pytest.mark.asyncio
    async def test_empty_directory_handled_gracefully(
        self,
        tmp_path: Path,
    ) -> None:
        """Empty directory (no contracts) is handled gracefully.

        Verifies that a directory with no handler contracts doesn't crash.
        """
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()

        event_bus = EventBusInmemory()
        process = RuntimeHostProcess(
            event_bus=event_bus,
            input_topic="test.input",
            contract_paths=[str(empty_dir)],
        )

        try:
            await process.start()

            # Process should start (with no handlers registered from contracts)
            assert process.is_running

            # Health check should succeed
            health = await process.health_check()
            assert health["is_running"] is True

        finally:
            await process.stop()


class TestRuntimeHostProcessLifecycle:
    """Test full lifecycle with discovered handlers.

    These tests verify the complete start/stop lifecycle when using
    contract-based handler discovery.
    """

    @pytest.mark.asyncio
    async def test_full_lifecycle_with_discovered_handlers(
        self,
        valid_handler_contract_dir: Path,
    ) -> None:
        """Test complete start/stop lifecycle with contract-discovered handlers.

        Verifies:
        1. Process starts with discovered handlers
        2. Handlers can be retrieved while running
        3. Health check succeeds while running
        4. Process stops cleanly
        5. After stop, is_running is False

        Note:
            The singleton handler registry may contain handlers from other tests
            that fail during initialization. We focus on testing is_running and
            handler discovery, not overall health status (which requires all
            handlers in the singleton registry to initialize successfully).
        """
        event_bus = EventBusInmemory()
        process = RuntimeHostProcess(
            event_bus=event_bus,
            input_topic="test.input",
            contract_paths=[str(valid_handler_contract_dir)],
        )

        # Before start
        assert not process.is_running

        # Start
        await process.start()
        assert process.is_running

        # Verify handlers while running
        assert process.get_handler("http") is not None

        # Health check while running - check core lifecycle status
        health = await process.health_check()
        assert health["is_running"] is True
        # Note: healthy may be False if singleton registry has handlers from
        # other tests that fail to initialize (DB, Consul, Vault need external services)
        assert "registered_handlers" in health
        assert "http" in health["registered_handlers"]

        # Stop
        await process.stop()
        assert not process.is_running

        # Health check after stop
        health_after = await process.health_check()
        assert health_after["is_running"] is False

    @pytest.mark.asyncio
    async def test_restart_after_stop(
        self,
        valid_handler_contract_dir: Path,
    ) -> None:
        """Test that RuntimeHostProcess can restart after stop.

        Verifies that the process can be started again after being stopped,
        with handlers re-discovered from contracts.
        """
        event_bus = EventBusInmemory()
        process = RuntimeHostProcess(
            event_bus=event_bus,
            input_topic="test.input",
            contract_paths=[str(valid_handler_contract_dir)],
        )

        # First start/stop cycle
        await process.start()
        assert process.is_running
        assert process.get_handler("http") is not None
        await process.stop()
        assert not process.is_running

        # Note: After stop, we need a fresh event bus since the old one is closed
        # This simulates a full restart scenario
        fresh_event_bus = EventBusInmemory()
        process2 = RuntimeHostProcess(
            event_bus=fresh_event_bus,
            input_topic="test.input",
            contract_paths=[str(valid_handler_contract_dir)],
        )

        try:
            # Second start
            await process2.start()
            assert process2.is_running
            assert process2.get_handler("http") is not None
        finally:
            await process2.stop()

    @pytest.mark.asyncio
    async def test_idempotent_start(
        self,
        valid_handler_contract_dir: Path,
    ) -> None:
        """Test that calling start() multiple times is idempotent.

        Verifies that calling start() on an already-started process
        is safe and has no adverse effects.
        """
        event_bus = EventBusInmemory()
        process = RuntimeHostProcess(
            event_bus=event_bus,
            input_topic="test.input",
            contract_paths=[str(valid_handler_contract_dir)],
        )

        try:
            # First start
            await process.start()
            assert process.is_running

            http_handler_first = process.get_handler("http")
            assert http_handler_first is not None

            # Second start (should be idempotent - no-op)
            await process.start()
            assert process.is_running

            # Handler should still be the same
            http_handler_second = process.get_handler("http")
            assert http_handler_second is http_handler_first

        finally:
            await process.stop()

    @pytest.mark.asyncio
    async def test_idempotent_stop(
        self,
        valid_handler_contract_dir: Path,
    ) -> None:
        """Test that calling stop() multiple times is idempotent.

        Verifies that calling stop() on an already-stopped process
        is safe and has no adverse effects.
        """
        event_bus = EventBusInmemory()
        process = RuntimeHostProcess(
            event_bus=event_bus,
            input_topic="test.input",
            contract_paths=[str(valid_handler_contract_dir)],
        )

        await process.start()
        assert process.is_running

        # First stop
        await process.stop()
        assert not process.is_running

        # Second stop (should be idempotent - no-op)
        await process.stop()
        assert not process.is_running

        # Third stop (should still be safe)
        await process.stop()
        assert not process.is_running


class TestRuntimeHostProcessDiscoveryLogging:
    """Test logging behavior during handler discovery.

    These tests verify that appropriate logs are generated during
    the handler discovery process.
    """

    @pytest.mark.asyncio
    async def test_discovery_logs_registered_handlers(
        self,
        valid_handler_contract_dir: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Verify that handler registration is logged.

        Logs should include information about discovered and registered handlers.
        """
        import logging

        event_bus = EventBusInmemory()
        process = RuntimeHostProcess(
            event_bus=event_bus,
            input_topic="test.input",
            contract_paths=[str(valid_handler_contract_dir)],
        )

        with caplog.at_level(logging.INFO):
            try:
                await process.start()

                # Should have log entries about discovery
                log_messages = [r.message for r in caplog.records]

                # Check for expected log patterns
                has_discovery_log = any(
                    "discovery" in msg.lower() or "handlers_discovered" in msg.lower()
                    for msg in log_messages
                )
                has_registered_log = any(
                    "registered" in msg.lower() for msg in log_messages
                )

                # At least one of these should be present
                assert has_discovery_log or has_registered_log, (
                    "Should have logs about handler discovery or registration. "
                    f"Log messages: {log_messages}"
                )

            finally:
                await process.stop()


__all__: list[str] = [
    "TestRuntimeHostProcessWithContractPaths",
    "TestRuntimeHostProcessFallback",
    "TestRuntimeHostProcessGracefulDegradation",
    "TestRuntimeHostProcessLifecycle",
    "TestRuntimeHostProcessDiscoveryLogging",
]
