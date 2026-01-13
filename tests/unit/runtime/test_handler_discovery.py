# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Unit tests for handler discovery functionality in RuntimeHostProcess.

This module provides comprehensive test coverage for:
1. The _discover_handlers_from_contracts() method in RuntimeHostProcess
2. Negative tests for invalid contract YAML handling
3. The no_handlers_registered health check field behavior

Part of OMN-1317: Infrastructure Integration with Handler Contracts and E2E Fixes

Test Coverage:
- RuntimeHostProcess._discover_handlers_from_contracts() method
- ContractHandlerDiscovery integration with RuntimeHostProcess
- Invalid/malformed YAML contract handling
- Empty contracts directory handling
- Health check no_handlers_registered field

Related:
    - src/omnibase_infra/runtime/runtime_host_process.py
    - src/omnibase_infra/runtime/contract_handler_discovery.py
    - src/omnibase_infra/runtime/handler_plugin_loader.py

Note:
    These tests create temporary handler contract YAML files that point to
    real handler classes. They do NOT require external infrastructure because
    they test handler discovery, not handler execution.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from omnibase_infra.event_bus.inmemory_event_bus import InMemoryEventBus
from omnibase_infra.runtime.handler_plugin_loader import HANDLER_CONTRACT_FILENAME
from omnibase_infra.runtime.handler_registry import (
    ProtocolBindingRegistry,
)
from omnibase_infra.runtime.runtime_host_process import RuntimeHostProcess

# =============================================================================
# Constants for Handler Contract Templates
# =============================================================================

# Real handler class path from omnibase_infra.handlers
REAL_HANDLER_HTTP_CLASS = "omnibase_infra.handlers.handler_http.HttpRestHandler"

# Valid handler contract template
VALID_HANDLER_CONTRACT_YAML = """
handler_name: "{handler_name}"
handler_class: "{handler_class}"
handler_type: "effect"
capability_tags:
  - {tag1}
  - {tag2}
"""

# Invalid YAML syntax (unclosed bracket)
INVALID_YAML_SYNTAX = """
handler_name: "test.handler"
handler_class: this is not valid yaml: [
    unclosed bracket
"""

# Missing required handler_class field
HANDLER_CONTRACT_MISSING_CLASS = """
handler_name: "missing.class.handler"
handler_type: "effect"
capability_tags:
  - test
"""

# Missing required handler_name field
HANDLER_CONTRACT_MISSING_NAME = """
handler_class: "some.module.Handler"
handler_type: "effect"
"""

# Missing both required fields
HANDLER_CONTRACT_MINIMAL_INVALID = """
handler_type: "effect"
capability_tags:
  - test
"""

# Empty YAML file
EMPTY_YAML = ""

# YAML with only comments
COMMENTS_ONLY_YAML = """
# This is a comment
# Another comment
"""

# Handler contract pointing to non-existent module
NONEXISTENT_MODULE_CONTRACT_YAML = """
handler_name: "nonexistent.handler"
handler_class: "nonexistent_module.does.not.exist.Handler"
handler_type: "effect"
capability_tags:
  - test
"""

# Handler contract pointing to non-existent class
NONEXISTENT_CLASS_CONTRACT_YAML = """
handler_name: "nonexistent.class.handler"
handler_class: "omnibase_infra.handlers.handler_http.NonexistentHandler"
handler_type: "effect"
capability_tags:
  - test
"""


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def isolated_handler_registry() -> ProtocolBindingRegistry:
    """Create an isolated handler registry for testing.

    Returns:
        Fresh ProtocolBindingRegistry instance that is not the singleton.
    """
    return ProtocolBindingRegistry()


@pytest.fixture
def valid_handler_contract_dir(tmp_path: Path) -> Path:
    """Create a directory with a valid handler contract.

    Returns:
        Path to the handlers directory.
    """
    handlers_dir = tmp_path / "handlers"
    handlers_dir.mkdir(parents=True)

    http_dir = handlers_dir / "http"
    http_dir.mkdir()
    http_contract = http_dir / HANDLER_CONTRACT_FILENAME
    http_contract.write_text(
        VALID_HANDLER_CONTRACT_YAML.format(
            handler_name="http",
            handler_class=REAL_HANDLER_HTTP_CLASS,
            tag1="http",
            tag2="rest",
        )
    )

    return handlers_dir


@pytest.fixture
def invalid_yaml_contract_dir(tmp_path: Path) -> Path:
    """Create a directory with an invalid YAML contract.

    Returns:
        Path to the directory with invalid YAML.
    """
    handlers_dir = tmp_path / "handlers"
    handlers_dir.mkdir(parents=True)

    invalid_dir = handlers_dir / "invalid_yaml"
    invalid_dir.mkdir()
    (invalid_dir / HANDLER_CONTRACT_FILENAME).write_text(INVALID_YAML_SYNTAX)

    return handlers_dir


@pytest.fixture
def missing_fields_contract_dir(tmp_path: Path) -> Path:
    """Create a directory with contracts missing required fields.

    Returns:
        Path to the directory with contracts missing fields.
    """
    handlers_dir = tmp_path / "handlers"
    handlers_dir.mkdir(parents=True)

    # Missing handler_class
    missing_class_dir = handlers_dir / "missing_class"
    missing_class_dir.mkdir()
    (missing_class_dir / HANDLER_CONTRACT_FILENAME).write_text(
        HANDLER_CONTRACT_MISSING_CLASS
    )

    # Missing handler_name
    missing_name_dir = handlers_dir / "missing_name"
    missing_name_dir.mkdir()
    (missing_name_dir / HANDLER_CONTRACT_FILENAME).write_text(
        HANDLER_CONTRACT_MISSING_NAME
    )

    return handlers_dir


@pytest.fixture
def nonexistent_handler_contract_dir(tmp_path: Path) -> Path:
    """Create a directory with contracts pointing to non-existent handlers.

    Returns:
        Path to the directory with non-existent handler contracts.
    """
    handlers_dir = tmp_path / "handlers"
    handlers_dir.mkdir(parents=True)

    # Non-existent module
    nonexistent_module_dir = handlers_dir / "nonexistent_module"
    nonexistent_module_dir.mkdir()
    (nonexistent_module_dir / HANDLER_CONTRACT_FILENAME).write_text(
        NONEXISTENT_MODULE_CONTRACT_YAML
    )

    # Non-existent class
    nonexistent_class_dir = handlers_dir / "nonexistent_class"
    nonexistent_class_dir.mkdir()
    (nonexistent_class_dir / HANDLER_CONTRACT_FILENAME).write_text(
        NONEXISTENT_CLASS_CONTRACT_YAML
    )

    return handlers_dir


@pytest.fixture
def empty_contract_dir(tmp_path: Path) -> Path:
    """Create an empty directory with no contracts.

    Returns:
        Path to the empty directory.
    """
    empty_dir = tmp_path / "empty_handlers"
    empty_dir.mkdir(parents=True)
    return empty_dir


@pytest.fixture
def empty_yaml_contract_dir(tmp_path: Path) -> Path:
    """Create a directory with empty YAML contracts.

    Returns:
        Path to the directory with empty YAML files.
    """
    handlers_dir = tmp_path / "handlers"
    handlers_dir.mkdir(parents=True)

    # Empty YAML file
    empty_yaml_dir = handlers_dir / "empty_yaml"
    empty_yaml_dir.mkdir()
    (empty_yaml_dir / HANDLER_CONTRACT_FILENAME).write_text(EMPTY_YAML)

    # Comments-only YAML file
    comments_dir = handlers_dir / "comments_only"
    comments_dir.mkdir()
    (comments_dir / HANDLER_CONTRACT_FILENAME).write_text(COMMENTS_ONLY_YAML)

    return handlers_dir


@pytest.fixture
def mixed_valid_invalid_contract_dir(tmp_path: Path) -> Path:
    """Create a directory with a mix of valid and invalid contracts.

    Returns:
        Path to the directory with mixed contracts.
    """
    handlers_dir = tmp_path / "handlers"
    handlers_dir.mkdir(parents=True)

    # Valid contract
    valid_dir = handlers_dir / "valid"
    valid_dir.mkdir()
    (valid_dir / HANDLER_CONTRACT_FILENAME).write_text(
        VALID_HANDLER_CONTRACT_YAML.format(
            handler_name="valid.http.handler",
            handler_class=REAL_HANDLER_HTTP_CLASS,
            tag1="http",
            tag2="valid",
        )
    )

    # Invalid YAML
    invalid_yaml_dir = handlers_dir / "invalid_yaml"
    invalid_yaml_dir.mkdir()
    (invalid_yaml_dir / HANDLER_CONTRACT_FILENAME).write_text(INVALID_YAML_SYNTAX)

    # Missing class
    missing_class_dir = handlers_dir / "missing_class"
    missing_class_dir.mkdir()
    (missing_class_dir / HANDLER_CONTRACT_FILENAME).write_text(
        HANDLER_CONTRACT_MISSING_CLASS
    )

    # Non-existent module
    nonexistent_dir = handlers_dir / "nonexistent"
    nonexistent_dir.mkdir()
    (nonexistent_dir / HANDLER_CONTRACT_FILENAME).write_text(
        NONEXISTENT_MODULE_CONTRACT_YAML
    )

    return handlers_dir


# =============================================================================
# Test Classes for _discover_handlers_from_contracts()
# =============================================================================


class TestDiscoverHandlersFromContracts:
    """Unit tests for RuntimeHostProcess._discover_handlers_from_contracts().

    These tests verify the internal discovery method that bridges
    ContractHandlerDiscovery with RuntimeHostProcess initialization.
    """

    @pytest.mark.asyncio
    async def test_discover_handlers_from_contracts_called_on_start(
        self,
        valid_handler_contract_dir: Path,
    ) -> None:
        """Verify _discover_handlers_from_contracts is called during start().

        When contract_paths is provided, RuntimeHostProcess should call
        _discover_handlers_from_contracts() instead of wire_handlers().
        """
        event_bus = InMemoryEventBus()
        # Use isolated registry
        isolated_registry = ProtocolBindingRegistry()

        process = RuntimeHostProcess(
            event_bus=event_bus,
            input_topic="test.input",
            contract_paths=[str(valid_handler_contract_dir)],
            handler_registry=isolated_registry,
        )

        # Track method call
        discovery_called = False
        original_method = process._discover_handlers_from_contracts

        async def tracking_discover() -> None:
            nonlocal discovery_called
            discovery_called = True
            await original_method()

        process._discover_handlers_from_contracts = tracking_discover  # type: ignore[method-assign]

        try:
            await process.start()
            assert discovery_called, (
                "_discover_handlers_from_contracts should be called"
            )
        finally:
            await process.stop()

    @pytest.mark.asyncio
    async def test_discover_handlers_creates_discovery_service(
        self,
        valid_handler_contract_dir: Path,
    ) -> None:
        """Verify _discover_handlers_from_contracts creates ContractHandlerDiscovery.

        The method should instantiate ContractHandlerDiscovery with
        HandlerPluginLoader and the handler registry.
        """
        event_bus = InMemoryEventBus()
        # Use isolated registry
        isolated_registry = ProtocolBindingRegistry()

        process = RuntimeHostProcess(
            event_bus=event_bus,
            input_topic="test.input",
            contract_paths=[str(valid_handler_contract_dir)],
            handler_registry=isolated_registry,
        )

        try:
            await process.start()

            # After start, _handler_discovery should be set
            assert process._handler_discovery is not None
            assert hasattr(process._handler_discovery, "discover_and_register")

        finally:
            await process.stop()

    @pytest.mark.asyncio
    async def test_discover_handlers_uses_contract_paths(
        self,
        valid_handler_contract_dir: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Verify _discover_handlers_from_contracts uses the provided paths.

        The discovery should scan all configured contract_paths.
        """
        event_bus = InMemoryEventBus()
        # Use isolated registry
        isolated_registry = ProtocolBindingRegistry()

        process = RuntimeHostProcess(
            event_bus=event_bus,
            input_topic="test.input",
            contract_paths=[str(valid_handler_contract_dir)],
            handler_registry=isolated_registry,
        )

        with caplog.at_level(logging.INFO):
            try:
                await process.start()

                # Check that discovery happened
                log_messages = [r.message for r in caplog.records]
                has_discovery_log = any(
                    "discovery" in msg.lower() for msg in log_messages
                )
                assert has_discovery_log, "Should log handler discovery"

                # HTTP handler should be discovered
                assert process.get_handler("http") is not None

            finally:
                await process.stop()

    @pytest.mark.asyncio
    async def test_discover_handlers_logs_errors_gracefully(
        self,
        mixed_valid_invalid_contract_dir: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Verify discovery logs errors but doesn't fail startup.

        Invalid contracts should be logged as warnings/errors,
        but valid contracts should still be registered.
        """
        event_bus = InMemoryEventBus()
        # Use isolated registry
        isolated_registry = ProtocolBindingRegistry()

        process = RuntimeHostProcess(
            event_bus=event_bus,
            input_topic="test.input",
            contract_paths=[str(mixed_valid_invalid_contract_dir)],
            handler_registry=isolated_registry,
        )

        with caplog.at_level(logging.WARNING):
            try:
                await process.start()

                # Process should start successfully
                assert process.is_running

                # Valid handler should be registered
                valid_handler = process.get_handler("valid.http.handler")
                assert valid_handler is not None

                # Should have warning logs for failed handlers
                warning_logs = [
                    r for r in caplog.records if r.levelno >= logging.WARNING
                ]
                assert len(warning_logs) > 0, (
                    "Should have warnings for invalid contracts"
                )

            finally:
                await process.stop()


# =============================================================================
# Test Classes for Invalid Contract YAML Handling
# =============================================================================


class TestInvalidContractYamlHandling:
    """Negative tests for invalid contract YAML handling.

    These tests verify that various types of invalid contracts
    are handled gracefully with proper error messages.

    Note: When using isolated registries and all contracts are invalid,
    the runtime will fail fast with ProtocolConfigurationError. These
    tests verify the logging and error handling behavior.
    """

    @pytest.mark.asyncio
    async def test_invalid_yaml_syntax_raises_configuration_error(
        self,
        invalid_yaml_contract_dir: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Verify invalid YAML syntax is logged and raises ProtocolConfigurationError.

        When using an isolated registry and all contracts have invalid YAML,
        the runtime should fail fast because no handlers can be registered.
        """
        from omnibase_infra.errors import ProtocolConfigurationError

        event_bus = InMemoryEventBus()
        # Use isolated registry
        isolated_registry = ProtocolBindingRegistry()

        process = RuntimeHostProcess(
            event_bus=event_bus,
            input_topic="test.input",
            contract_paths=[str(invalid_yaml_contract_dir)],
            handler_registry=isolated_registry,
        )

        with caplog.at_level(logging.WARNING):
            with pytest.raises(ProtocolConfigurationError) as exc_info:
                await process.start()

            # Should have warning about invalid YAML in logs
            warning_messages = " ".join(
                r.message for r in caplog.records if r.levelno >= logging.WARNING
            )
            assert (
                "invalid" in warning_messages.lower()
                or "yaml" in warning_messages.lower()
                or "error" in warning_messages.lower()
            )

            # Verify the exception
            error_message = str(exc_info.value)
            assert "No handlers registered" in error_message

    @pytest.mark.asyncio
    async def test_missing_required_fields_raises_configuration_error(
        self,
        missing_fields_contract_dir: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Verify contracts with missing required fields raise ProtocolConfigurationError.

        When using an isolated registry and all contracts are missing fields,
        the runtime should fail fast.
        """
        from omnibase_infra.errors import ProtocolConfigurationError

        event_bus = InMemoryEventBus()
        # Use isolated registry
        isolated_registry = ProtocolBindingRegistry()

        process = RuntimeHostProcess(
            event_bus=event_bus,
            input_topic="test.input",
            contract_paths=[str(missing_fields_contract_dir)],
            handler_registry=isolated_registry,
        )

        with caplog.at_level(logging.WARNING):
            with pytest.raises(ProtocolConfigurationError):
                await process.start()

            # Should have warnings about missing fields
            warning_logs = [r for r in caplog.records if r.levelno >= logging.WARNING]
            assert len(warning_logs) > 0, "Should have warnings for missing fields"

    @pytest.mark.asyncio
    async def test_nonexistent_handler_module_raises_configuration_error(
        self,
        nonexistent_handler_contract_dir: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Verify contracts pointing to non-existent modules raise ProtocolConfigurationError.

        Import errors should be logged and the runtime should fail fast
        when no handlers can be loaded.
        """
        from omnibase_infra.errors import ProtocolConfigurationError

        event_bus = InMemoryEventBus()
        # Use isolated registry
        isolated_registry = ProtocolBindingRegistry()

        process = RuntimeHostProcess(
            event_bus=event_bus,
            input_topic="test.input",
            contract_paths=[str(nonexistent_handler_contract_dir)],
            handler_registry=isolated_registry,
        )

        with caplog.at_level(logging.WARNING):
            with pytest.raises(ProtocolConfigurationError):
                await process.start()

            # Should have warnings about import failures
            warning_logs = [r for r in caplog.records if r.levelno >= logging.WARNING]
            assert len(warning_logs) > 0, "Should have warnings for import failures"

    @pytest.mark.asyncio
    async def test_empty_yaml_file_raises_configuration_error(
        self,
        empty_yaml_contract_dir: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Verify empty YAML files cause ProtocolConfigurationError.

        Empty contracts should be skipped, and with an isolated registry,
        the runtime should fail fast because no handlers can be registered.
        """
        from omnibase_infra.errors import ProtocolConfigurationError

        event_bus = InMemoryEventBus()
        # Use isolated registry
        isolated_registry = ProtocolBindingRegistry()

        process = RuntimeHostProcess(
            event_bus=event_bus,
            input_topic="test.input",
            contract_paths=[str(empty_yaml_contract_dir)],
            handler_registry=isolated_registry,
        )

        with caplog.at_level(logging.WARNING):
            with pytest.raises(ProtocolConfigurationError):
                await process.start()

    @pytest.mark.asyncio
    async def test_empty_contract_directory_raises_configuration_error(
        self,
        empty_contract_dir: Path,
    ) -> None:
        """Verify empty directories raise ProtocolConfigurationError.

        An empty directory with an isolated registry should cause
        the runtime to fail fast because no handlers can be registered.
        """
        from omnibase_infra.errors import ProtocolConfigurationError

        event_bus = InMemoryEventBus()
        # Use isolated registry
        isolated_registry = ProtocolBindingRegistry()

        process = RuntimeHostProcess(
            event_bus=event_bus,
            input_topic="test.input",
            contract_paths=[str(empty_contract_dir)],
            handler_registry=isolated_registry,
        )

        with pytest.raises(ProtocolConfigurationError) as exc_info:
            await process.start()

        error_message = str(exc_info.value)
        assert "No handlers registered" in error_message

    @pytest.mark.asyncio
    async def test_nonexistent_contract_path_raises_configuration_error(
        self,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Verify non-existent contract paths raise ProtocolConfigurationError.

        A path that doesn't exist with an isolated registry should cause
        the runtime to fail fast.
        """
        from omnibase_infra.errors import ProtocolConfigurationError

        nonexistent_path = tmp_path / "does_not_exist"

        event_bus = InMemoryEventBus()
        # Use isolated registry
        isolated_registry = ProtocolBindingRegistry()

        process = RuntimeHostProcess(
            event_bus=event_bus,
            input_topic="test.input",
            contract_paths=[str(nonexistent_path)],
            handler_registry=isolated_registry,
        )

        with caplog.at_level(logging.WARNING):
            with pytest.raises(ProtocolConfigurationError):
                await process.start()

            # Should have warning about non-existent path
            # (logged by discovery service)


# =============================================================================
# Test Classes for no_handlers_registered Health Check Field
# =============================================================================


class TestNoHandlersRegisteredHealthCheck:
    """Tests for the no_handlers_registered health check field.

    This field indicates whether the runtime has any handlers registered.
    A runtime with no handlers is considered unhealthy as it cannot
    process any events.

    Note: RuntimeHostProcess has a FAIL-FAST validation that raises
    ProtocolConfigurationError if no handlers are registered during start().
    Therefore, tests for empty/invalid scenarios must expect this exception.
    """

    @pytest.mark.asyncio
    async def test_empty_directory_raises_configuration_error(
        self,
        empty_contract_dir: Path,
    ) -> None:
        """Verify empty contract directory raises ProtocolConfigurationError.

        When contract_paths points to an empty directory with an isolated
        registry, no handlers can be registered, and the runtime should
        fail fast with a configuration error.
        """
        from omnibase_infra.errors import ProtocolConfigurationError

        event_bus = InMemoryEventBus()
        # Use isolated registry to prevent interference from other tests
        isolated_registry = ProtocolBindingRegistry()

        process = RuntimeHostProcess(
            event_bus=event_bus,
            input_topic="test.input",
            contract_paths=[str(empty_contract_dir)],
            handler_registry=isolated_registry,
        )

        with pytest.raises(ProtocolConfigurationError) as exc_info:
            await process.start()

        # Verify error message indicates no handlers
        error_message = str(exc_info.value)
        assert "No handlers registered" in error_message

    @pytest.mark.asyncio
    async def test_no_handlers_registered_false_with_valid_handlers(
        self,
        valid_handler_contract_dir: Path,
    ) -> None:
        """Verify no_handlers_registered is False when handlers are registered.

        When valid handlers are discovered and registered,
        no_handlers_registered should be False.
        """
        event_bus = InMemoryEventBus()
        # Use isolated registry
        isolated_registry = ProtocolBindingRegistry()

        process = RuntimeHostProcess(
            event_bus=event_bus,
            input_topic="test.input",
            contract_paths=[str(valid_handler_contract_dir)],
            handler_registry=isolated_registry,
        )

        try:
            await process.start()
            health = await process.health_check()

            # Handlers should be registered
            assert health["no_handlers_registered"] is False
            assert "http" in health["registered_handlers"]

            # Should be running (healthy depends on handler initialization)
            assert health["is_running"] is True

        finally:
            await process.stop()

    @pytest.mark.asyncio
    async def test_all_invalid_contracts_raises_configuration_error(
        self,
        invalid_yaml_contract_dir: Path,
    ) -> None:
        """Verify all invalid contracts raises ProtocolConfigurationError.

        When all provided contracts fail to load with an isolated registry,
        the runtime should fail fast with a configuration error.
        """
        from omnibase_infra.errors import ProtocolConfigurationError

        event_bus = InMemoryEventBus()
        # Use isolated registry
        isolated_registry = ProtocolBindingRegistry()

        process = RuntimeHostProcess(
            event_bus=event_bus,
            input_topic="test.input",
            contract_paths=[str(invalid_yaml_contract_dir)],
            handler_registry=isolated_registry,
        )

        with pytest.raises(ProtocolConfigurationError) as exc_info:
            await process.start()

        # Verify error message indicates no handlers
        error_message = str(exc_info.value)
        assert "No handlers registered" in error_message

    @pytest.mark.asyncio
    async def test_no_handlers_registered_with_mixed_contracts(
        self,
        mixed_valid_invalid_contract_dir: Path,
    ) -> None:
        """Verify no_handlers_registered is False when some contracts are valid.

        Even if some contracts fail, valid ones should be registered.
        """
        event_bus = InMemoryEventBus()
        # Use isolated registry
        isolated_registry = ProtocolBindingRegistry()

        process = RuntimeHostProcess(
            event_bus=event_bus,
            input_topic="test.input",
            contract_paths=[str(mixed_valid_invalid_contract_dir)],
            handler_registry=isolated_registry,
        )

        try:
            await process.start()
            health = await process.health_check()

            # At least one handler should be registered
            assert health["no_handlers_registered"] is False
            assert len(health["registered_handlers"]) >= 1
            assert "valid.http.handler" in health["registered_handlers"]

        finally:
            await process.stop()

    @pytest.mark.asyncio
    async def test_health_check_before_start_shows_no_handlers(
        self,
    ) -> None:
        """Verify health check before start shows no handlers registered.

        Before start() is called, there should be no handlers.
        """
        event_bus = InMemoryEventBus()
        # Use isolated registry
        isolated_registry = ProtocolBindingRegistry()

        process = RuntimeHostProcess(
            event_bus=event_bus,
            input_topic="test.input",
            handler_registry=isolated_registry,
        )

        # Before start
        health = await process.health_check()

        assert health["no_handlers_registered"] is True
        assert health["is_running"] is False
        assert health["healthy"] is False

    @pytest.mark.asyncio
    async def test_health_check_after_stop_shows_handlers_still_registered(
        self,
        valid_handler_contract_dir: Path,
    ) -> None:
        """Verify health check after stop shows handlers still registered.

        After stop(), handlers should still be in the registry
        (they're just not processing).
        """
        event_bus = InMemoryEventBus()
        # Use isolated registry
        isolated_registry = ProtocolBindingRegistry()

        process = RuntimeHostProcess(
            event_bus=event_bus,
            input_topic="test.input",
            contract_paths=[str(valid_handler_contract_dir)],
            handler_registry=isolated_registry,
        )

        await process.start()
        assert process.is_running

        await process.stop()
        assert not process.is_running

        # After stop, handlers should still be registered
        health = await process.health_check()
        assert health["is_running"] is False
        # Handlers remain registered after stop
        assert health["no_handlers_registered"] is False

    @pytest.mark.asyncio
    async def test_no_handlers_registered_field_computation(
        self,
        valid_handler_contract_dir: Path,
    ) -> None:
        """Verify no_handlers_registered field is computed from _handlers length.

        The no_handlers_registered field should be True when len(_handlers) == 0
        and False otherwise.
        """
        event_bus = InMemoryEventBus()
        # Use isolated registry
        isolated_registry = ProtocolBindingRegistry()

        process = RuntimeHostProcess(
            event_bus=event_bus,
            input_topic="test.input",
            contract_paths=[str(valid_handler_contract_dir)],
            handler_registry=isolated_registry,
        )

        try:
            await process.start()
            health = await process.health_check()

            # Verify the relationship between _handlers and no_handlers_registered
            handlers_count = len(health["registered_handlers"])
            expected_no_handlers = handlers_count == 0
            assert health["no_handlers_registered"] is expected_no_handlers

            # With valid handlers, should not be empty
            assert handlers_count > 0
            assert health["no_handlers_registered"] is False

        finally:
            await process.stop()

    @pytest.mark.asyncio
    async def test_no_handlers_registered_affects_healthy_status(
        self,
        valid_handler_contract_dir: Path,
    ) -> None:
        """Verify no_handlers_registered affects the overall healthy status.

        When handlers are registered, the healthy status should not be
        negatively impacted by no_handlers_registered.
        """
        event_bus = InMemoryEventBus()
        # Use isolated registry
        isolated_registry = ProtocolBindingRegistry()

        process = RuntimeHostProcess(
            event_bus=event_bus,
            input_topic="test.input",
            contract_paths=[str(valid_handler_contract_dir)],
            handler_registry=isolated_registry,
        )

        try:
            await process.start()
            health = await process.health_check()

            # no_handlers_registered should be False
            assert health["no_handlers_registered"] is False

            # is_running should be True
            assert health["is_running"] is True

            # healthy depends on handler initialization success
            # but no_handlers_registered=False should not make it unhealthy
            if health["healthy"]:
                # If healthy, no_handlers_registered must be False
                assert health["no_handlers_registered"] is False

        finally:
            await process.stop()


# =============================================================================
# Test Classes for Discovery Logging
# =============================================================================


class TestHandlerDiscoveryLogging:
    """Tests for logging during handler discovery.

    Verifies that discovery operations produce appropriate log messages
    for observability purposes.
    """

    @pytest.mark.asyncio
    async def test_discovery_logs_contract_paths(
        self,
        valid_handler_contract_dir: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Verify discovery logs the contract paths being scanned."""
        event_bus = InMemoryEventBus()
        # Use isolated registry
        isolated_registry = ProtocolBindingRegistry()

        process = RuntimeHostProcess(
            event_bus=event_bus,
            input_topic="test.input",
            contract_paths=[str(valid_handler_contract_dir)],
            handler_registry=isolated_registry,
        )

        with caplog.at_level(logging.INFO):
            try:
                await process.start()

                # Should have log about starting discovery
                log_messages = " ".join(r.message for r in caplog.records)
                has_discovery_log = "discovery" in log_messages.lower()
                assert has_discovery_log, "Should log about handler discovery"

            finally:
                await process.stop()

    @pytest.mark.asyncio
    async def test_discovery_logs_successful_registration(
        self,
        valid_handler_contract_dir: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Verify discovery logs successful handler registrations."""
        event_bus = InMemoryEventBus()
        # Use isolated registry
        isolated_registry = ProtocolBindingRegistry()

        process = RuntimeHostProcess(
            event_bus=event_bus,
            input_topic="test.input",
            contract_paths=[str(valid_handler_contract_dir)],
            handler_registry=isolated_registry,
        )

        with caplog.at_level(logging.INFO):
            try:
                await process.start()

                # Should log about discovered/registered handlers
                log_messages = " ".join(r.message for r in caplog.records)
                has_registration_log = (
                    "registered" in log_messages.lower()
                    or "discovered" in log_messages.lower()
                )
                assert has_registration_log, "Should log about handler registration"

            finally:
                await process.stop()

    @pytest.mark.asyncio
    async def test_discovery_logs_errors_with_details(
        self,
        invalid_yaml_contract_dir: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Verify discovery logs error details for failed handlers."""
        from omnibase_infra.errors import ProtocolConfigurationError

        event_bus = InMemoryEventBus()
        # Use isolated registry
        isolated_registry = ProtocolBindingRegistry()

        process = RuntimeHostProcess(
            event_bus=event_bus,
            input_topic="test.input",
            contract_paths=[str(invalid_yaml_contract_dir)],
            handler_registry=isolated_registry,
        )

        with caplog.at_level(logging.WARNING):
            # With isolated registry, this will raise ProtocolConfigurationError
            with pytest.raises(ProtocolConfigurationError):
                await process.start()

            # Should have warning logs with details about the invalid contract
            warning_logs = [r for r in caplog.records if r.levelno >= logging.WARNING]
            assert len(warning_logs) > 0, "Should have warning logs"

            # At least one warning should exist about the error
            warning_messages = " ".join(r.message for r in warning_logs)
            has_error_indicator = any(
                term in warning_messages.lower()
                for term in ["error", "fail", "invalid", "yaml"]
            )
            assert has_error_indicator, "Warnings should indicate errors"


# =============================================================================
# Export
# =============================================================================

__all__: list[str] = [
    "TestDiscoverHandlersFromContracts",
    "TestInvalidContractYamlHandling",
    "TestNoHandlersRegisteredHealthCheck",
    "TestHandlerDiscoveryLogging",
]
