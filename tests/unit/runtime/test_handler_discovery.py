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
import sys
import uuid
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

# Valid handler contract template (matches schema from contracts/handlers/)
# Uses canonical ONEX field names with required security metadata
# NOTE: Uses 'name' and 'tags' (canonical) not 'handler_name'/'capability_tags' (aliases)
VALID_HANDLER_CONTRACT_YAML = """
name: "{handler_name}"
handler_class: "{handler_class}"
handler_type: "effect"
tags:
  - {tag1}
  - {tag2}
security:
  trusted_namespace: omnibase_infra.handlers
  audit_logging: true
"""

# Invalid YAML syntax (unclosed bracket)
INVALID_YAML_SYNTAX = """
name: "test.handler"
handler_class: this is not valid yaml: [
    unclosed bracket
"""

# Missing required handler_class field
HANDLER_CONTRACT_MISSING_CLASS = """
name: "missing.class.handler"
handler_type: "effect"
tags:
  - test
security:
  trusted_namespace: omnibase_infra.handlers
  audit_logging: false
"""

# Missing required name field (no 'name' key)
HANDLER_CONTRACT_MISSING_NAME = """
handler_class: "some.module.Handler"
handler_type: "effect"
tags:
  - test
security:
  trusted_namespace: omnibase_infra.handlers
  audit_logging: false
"""

# Missing both required fields (name and handler_class)
HANDLER_CONTRACT_MINIMAL_INVALID = """
handler_type: "effect"
tags:
  - test
security:
  trusted_namespace: omnibase_infra.handlers
  audit_logging: false
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
name: "nonexistent.handler"
handler_class: "nonexistent_module.does.not.exist.Handler"
handler_type: "effect"
tags:
  - test
security:
  trusted_namespace: nonexistent_module
  audit_logging: false
"""

# Handler contract pointing to non-existent class
NONEXISTENT_CLASS_CONTRACT_YAML = """
name: "nonexistent.class.handler"
handler_class: "omnibase_infra.handlers.handler_http.NonexistentHandler"
handler_type: "effect"
tags:
  - test
security:
  trusted_namespace: omnibase_infra.handlers
  audit_logging: false
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

        try:
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
        finally:
            await process.stop()

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

        try:
            with caplog.at_level(logging.WARNING):
                with pytest.raises(ProtocolConfigurationError):
                    await process.start()

                # Should have warnings about missing fields
                warning_logs = [
                    r for r in caplog.records if r.levelno >= logging.WARNING
                ]
                assert len(warning_logs) > 0, "Should have warnings for missing fields"
        finally:
            await process.stop()

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

        try:
            with caplog.at_level(logging.WARNING):
                with pytest.raises(ProtocolConfigurationError):
                    await process.start()

                # Should have warnings about import failures
                warning_logs = [
                    r for r in caplog.records if r.levelno >= logging.WARNING
                ]
                assert len(warning_logs) > 0, "Should have warnings for import failures"
        finally:
            await process.stop()

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

        try:
            with caplog.at_level(logging.WARNING):
                with pytest.raises(ProtocolConfigurationError):
                    await process.start()
        finally:
            await process.stop()

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

        try:
            with pytest.raises(ProtocolConfigurationError) as exc_info:
                await process.start()

            error_message = str(exc_info.value)
            assert "No handlers registered" in error_message
        finally:
            await process.stop()

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

        try:
            with caplog.at_level(logging.WARNING):
                with pytest.raises(ProtocolConfigurationError):
                    await process.start()

                # Should have warning about non-existent path
                # (logged by discovery service)
        finally:
            await process.stop()


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

        try:
            with pytest.raises(ProtocolConfigurationError) as exc_info:
                await process.start()

            # Verify error message indicates no handlers
            error_message = str(exc_info.value)
            assert "No handlers registered" in error_message
        finally:
            await process.stop()

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

        try:
            with pytest.raises(ProtocolConfigurationError) as exc_info:
                await process.start()

            # Verify error message indicates no handlers
            error_message = str(exc_info.value)
            assert "No handlers registered" in error_message
        finally:
            await process.stop()

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

        try:
            await process.start()
            assert process.is_running

            await process.stop()
            assert not process.is_running

            # After stop, handlers should still be registered
            health = await process.health_check()
            assert health["is_running"] is False
            # Handlers remain registered after stop
            assert health["no_handlers_registered"] is False
        finally:
            # Ensure stop() is called even if assertions fail
            await process.stop()

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

        try:
            with caplog.at_level(logging.WARNING):
                # With isolated registry, this will raise ProtocolConfigurationError
                with pytest.raises(ProtocolConfigurationError):
                    await process.start()

                # Should have warning logs with details about the invalid contract
                warning_logs = [
                    r for r in caplog.records if r.levelno >= logging.WARNING
                ]
                assert len(warning_logs) > 0, "Should have warning logs"

                # At least one warning should exist about the error
                warning_messages = " ".join(r.message for r in warning_logs)
                has_error_indicator = any(
                    term in warning_messages.lower()
                    for term in ["error", "fail", "invalid", "yaml"]
                )
                assert has_error_indicator, "Warnings should indicate errors"
        finally:
            await process.stop()


# =============================================================================
# Export
# =============================================================================

__all__: list[str] = [
    "TestDiscoverHandlersFromContracts",
    "TestInvalidContractYamlHandling",
    "TestNoHandlersRegisteredHealthCheck",
    "TestHandlerDiscoveryLogging",
    "TestDiscoverHandlersFromContractsUnit",
    "TestDiscoverHandlersFromContractsErrorHandling",
    "TestDiscoverHandlersFromContractsFileSystemEdgeCases",
    "TestDiscoverHandlersFromContractsCorrelationTracking",
]


# =============================================================================
# Test Classes for Direct _discover_handlers_from_contracts() Unit Tests
# =============================================================================


class TestDiscoverHandlersFromContractsUnit:
    """Direct unit tests for RuntimeHostProcess._discover_handlers_from_contracts().

    These tests directly invoke the internal method without going through start(),
    providing more focused unit test coverage of the discovery behavior.
    """

    @pytest.mark.asyncio
    async def test_direct_call_creates_handler_discovery_service(
        self,
        valid_handler_contract_dir: Path,
    ) -> None:
        """Verify direct call creates ContractHandlerDiscovery with correct components.

        When _discover_handlers_from_contracts() is called directly, it should
        create a ContractHandlerDiscovery service with HandlerPluginLoader and
        the handler registry.
        """
        event_bus = InMemoryEventBus()
        isolated_registry = ProtocolBindingRegistry()

        process = RuntimeHostProcess(
            event_bus=event_bus,
            input_topic="test.input",
            contract_paths=[str(valid_handler_contract_dir)],
            handler_registry=isolated_registry,
        )

        # _handler_discovery should be None before calling the method
        assert process._handler_discovery is None

        # Call the method directly
        await process._discover_handlers_from_contracts()

        # _handler_discovery should now be set
        assert process._handler_discovery is not None

        # Verify it has the expected interface
        assert hasattr(process._handler_discovery, "discover_and_register")
        assert hasattr(process._handler_discovery, "last_discovery_result")

        # Verify discovery result was cached
        result = process._handler_discovery.last_discovery_result
        assert result is not None
        assert result.handlers_discovered >= 1

    @pytest.mark.asyncio
    async def test_direct_call_with_multiple_contract_paths(
        self,
        tmp_path: Path,
    ) -> None:
        """Verify discovery works with multiple contract paths.

        When multiple paths are provided, all paths should be scanned
        and handlers from each path should be discovered.
        """
        # Create two separate handler directories
        handlers_dir_1 = tmp_path / "handlers1"
        handlers_dir_1.mkdir(parents=True)
        http_dir = handlers_dir_1 / "http"
        http_dir.mkdir()
        (http_dir / HANDLER_CONTRACT_FILENAME).write_text(
            VALID_HANDLER_CONTRACT_YAML.format(
                handler_name="http",
                handler_class=REAL_HANDLER_HTTP_CLASS,
                tag1="http",
                tag2="rest",
            )
        )

        handlers_dir_2 = tmp_path / "handlers2"
        handlers_dir_2.mkdir(parents=True)
        http2_dir = handlers_dir_2 / "http2"
        http2_dir.mkdir()
        (http2_dir / HANDLER_CONTRACT_FILENAME).write_text(
            VALID_HANDLER_CONTRACT_YAML.format(
                handler_name="http2",
                handler_class=REAL_HANDLER_HTTP_CLASS,
                tag1="http",
                tag2="api",
            )
        )

        event_bus = InMemoryEventBus()
        isolated_registry = ProtocolBindingRegistry()

        process = RuntimeHostProcess(
            event_bus=event_bus,
            input_topic="test.input",
            contract_paths=[str(handlers_dir_1), str(handlers_dir_2)],
            handler_registry=isolated_registry,
        )

        await process._discover_handlers_from_contracts()

        # Verify both handlers were discovered
        result = process._handler_discovery.last_discovery_result
        assert result is not None
        assert result.handlers_discovered >= 2, (
            f"Expected at least 2 handlers, got {result.handlers_discovered}"
        )

    @pytest.mark.asyncio
    async def test_direct_call_with_nested_directory_structure(
        self,
        tmp_path: Path,
    ) -> None:
        """Verify discovery handles nested directory structures.

        When a contract path contains nested directories with contracts,
        all contracts should be discovered recursively.
        """
        # Create nested directory structure
        root_dir = tmp_path / "handlers"
        root_dir.mkdir(parents=True)

        # First level handler
        level1_dir = root_dir / "level1"
        level1_dir.mkdir()
        (level1_dir / HANDLER_CONTRACT_FILENAME).write_text(
            VALID_HANDLER_CONTRACT_YAML.format(
                handler_name="level1.handler",
                handler_class=REAL_HANDLER_HTTP_CLASS,
                tag1="level1",
                tag2="test",
            )
        )

        # Second level handler (nested)
        level2_dir = root_dir / "level1" / "level2"
        level2_dir.mkdir()
        (level2_dir / HANDLER_CONTRACT_FILENAME).write_text(
            VALID_HANDLER_CONTRACT_YAML.format(
                handler_name="level2.handler",
                handler_class=REAL_HANDLER_HTTP_CLASS,
                tag1="level2",
                tag2="nested",
            )
        )

        event_bus = InMemoryEventBus()
        isolated_registry = ProtocolBindingRegistry()

        process = RuntimeHostProcess(
            event_bus=event_bus,
            input_topic="test.input",
            contract_paths=[str(root_dir)],
            handler_registry=isolated_registry,
        )

        await process._discover_handlers_from_contracts()

        result = process._handler_discovery.last_discovery_result
        assert result is not None
        # Both handlers should be discovered from nested structure
        assert result.handlers_discovered >= 2

    @pytest.mark.asyncio
    async def test_direct_call_discovery_result_counts(
        self,
        mixed_valid_invalid_contract_dir: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Verify discovery result correctly tracks discovered vs registered counts.

        When some contracts are valid and some invalid:
        - Invalid contracts are filtered at the plugin loader level (logged as warnings)
        - Only successfully loaded handlers appear in handlers_discovered
        - handlers_registered equals handlers_discovered when all loaded handlers register
        """
        event_bus = InMemoryEventBus()
        isolated_registry = ProtocolBindingRegistry()

        process = RuntimeHostProcess(
            event_bus=event_bus,
            input_topic="test.input",
            contract_paths=[str(mixed_valid_invalid_contract_dir)],
            handler_registry=isolated_registry,
        )

        with caplog.at_level(logging.WARNING):
            await process._discover_handlers_from_contracts()

        result = process._handler_discovery.last_discovery_result
        assert result is not None

        # At least one handler should be registered (the valid one)
        assert result.handlers_registered >= 1

        # Handlers discovered equals handlers registered when all loaded handlers register
        # (invalid contracts are filtered at loader level, not tracked as errors)
        assert result.handlers_discovered == result.handlers_registered

        # Invalid contracts should be logged as warnings by the plugin loader
        warning_logs = [r for r in caplog.records if r.levelno >= logging.WARNING]
        assert len(warning_logs) > 0, "Invalid contracts should be logged as warnings"

    @pytest.mark.asyncio
    async def test_direct_call_with_directory_containing_non_contract_files(
        self,
        tmp_path: Path,
    ) -> None:
        """Verify discovery ignores non-contract files in directories.

        When a directory contains files that are not handler contracts,
        they should be silently ignored without errors.
        """
        handlers_dir = tmp_path / "handlers"
        handlers_dir.mkdir(parents=True)

        # Valid handler contract
        http_dir = handlers_dir / "http"
        http_dir.mkdir()
        (http_dir / HANDLER_CONTRACT_FILENAME).write_text(
            VALID_HANDLER_CONTRACT_YAML.format(
                handler_name="http",
                handler_class=REAL_HANDLER_HTTP_CLASS,
                tag1="http",
                tag2="rest",
            )
        )

        # Non-contract files that should be ignored
        (handlers_dir / "readme.txt").write_text("This is a readme file")
        (handlers_dir / "notes.md").write_text("# Some Notes")
        (handlers_dir / "config.json").write_text('{"key": "value"}')

        event_bus = InMemoryEventBus()
        isolated_registry = ProtocolBindingRegistry()

        process = RuntimeHostProcess(
            event_bus=event_bus,
            input_topic="test.input",
            contract_paths=[str(handlers_dir)],
            handler_registry=isolated_registry,
        )

        await process._discover_handlers_from_contracts()

        result = process._handler_discovery.last_discovery_result
        assert result is not None
        # Only the valid handler should be discovered
        assert result.handlers_discovered == 1
        assert result.handlers_registered == 1
        # No errors from non-contract files
        assert not result.has_errors

    @pytest.mark.asyncio
    async def test_direct_call_idempotent_discovery_service_creation(
        self,
        valid_handler_contract_dir: Path,
    ) -> None:
        """Verify calling _discover_handlers_from_contracts twice creates new service.

        When called multiple times, each call should create a new discovery
        service instance (not cached).
        """
        event_bus = InMemoryEventBus()
        isolated_registry = ProtocolBindingRegistry()

        process = RuntimeHostProcess(
            event_bus=event_bus,
            input_topic="test.input",
            contract_paths=[str(valid_handler_contract_dir)],
            handler_registry=isolated_registry,
        )

        # First call
        await process._discover_handlers_from_contracts()
        first_discovery = process._handler_discovery
        first_result = first_discovery.last_discovery_result

        # Second call
        await process._discover_handlers_from_contracts()
        second_discovery = process._handler_discovery
        second_result = second_discovery.last_discovery_result

        # A new discovery service is created each time
        assert first_discovery is not second_discovery
        # But results should be consistent
        assert first_result.handlers_discovered == second_result.handlers_discovered

    @pytest.mark.asyncio
    async def test_direct_call_with_empty_contract_paths_list(
        self,
    ) -> None:
        """Verify discovery handles empty contract_paths list gracefully.

        When contract_paths is an empty list, discovery should complete
        without errors but find no handlers.
        """
        event_bus = InMemoryEventBus()
        isolated_registry = ProtocolBindingRegistry()

        process = RuntimeHostProcess(
            event_bus=event_bus,
            input_topic="test.input",
            contract_paths=[],  # Empty list
            handler_registry=isolated_registry,
        )

        await process._discover_handlers_from_contracts()

        result = process._handler_discovery.last_discovery_result
        assert result is not None
        assert result.handlers_discovered == 0
        assert result.handlers_registered == 0
        # Empty paths list should not produce errors
        assert not result.has_errors


class TestDiscoverHandlersFromContractsErrorHandling:
    """Error handling tests for _discover_handlers_from_contracts().

    These tests verify that various error conditions are handled
    gracefully with proper error reporting.

    Note: Invalid contracts are filtered at the plugin loader level and logged
    as warnings. The discovery result only tracks errors when handler registration
    fails AFTER the contract is successfully loaded.
    """

    @pytest.mark.asyncio
    async def test_discovery_logs_error_codes_for_invalid_contracts(
        self,
        nonexistent_handler_contract_dir: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Verify discovery logs error codes for invalid contracts.

        When handlers fail to load, the plugin loader logs warnings with
        error codes (e.g., HANDLER_LOADER_010, HANDLER_LOADER_011).
        """
        event_bus = InMemoryEventBus()
        isolated_registry = ProtocolBindingRegistry()

        process = RuntimeHostProcess(
            event_bus=event_bus,
            input_topic="test.input",
            contract_paths=[str(nonexistent_handler_contract_dir)],
            handler_registry=isolated_registry,
        )

        with caplog.at_level(logging.WARNING):
            await process._discover_handlers_from_contracts()

        result = process._handler_discovery.last_discovery_result
        assert result is not None

        # Invalid contracts are filtered at loader level, not tracked as errors
        # in the discovery result
        assert result.handlers_discovered == 0
        assert result.handlers_registered == 0

        # But warnings should be logged with error codes
        warning_logs = [r for r in caplog.records if r.levelno >= logging.WARNING]
        assert len(warning_logs) > 0, "Should have WARNING logs for invalid contracts"

        # Verify error codes are in the log messages
        warning_messages = " ".join(r.message for r in warning_logs)
        assert "HANDLER_LOADER" in warning_messages, (
            "Warnings should include HANDLER_LOADER error codes"
        )

    @pytest.mark.asyncio
    async def test_discovery_logs_individual_failures_as_warnings(
        self,
        mixed_valid_invalid_contract_dir: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Verify discovery logs individual failures at WARNING level.

        When handlers fail during discovery, each failure should be
        logged as a warning by the plugin loader.
        """
        event_bus = InMemoryEventBus()
        isolated_registry = ProtocolBindingRegistry()

        process = RuntimeHostProcess(
            event_bus=event_bus,
            input_topic="test.input",
            contract_paths=[str(mixed_valid_invalid_contract_dir)],
            handler_registry=isolated_registry,
        )

        with caplog.at_level(logging.WARNING):
            await process._discover_handlers_from_contracts()

        # Should have warning logs for failed handlers
        warning_logs = [r for r in caplog.records if r.levelno >= logging.WARNING]
        assert len(warning_logs) > 0, (
            "Should have WARNING level logs for failed handlers"
        )

    @pytest.mark.asyncio
    async def test_discovery_continues_after_path_errors(
        self,
        tmp_path: Path,
    ) -> None:
        """Verify discovery continues processing after encountering path errors.

        When one path fails, subsequent paths should still be processed.
        """
        # Create one valid and one invalid path
        valid_dir = tmp_path / "valid_handlers"
        valid_dir.mkdir(parents=True)
        http_dir = valid_dir / "http"
        http_dir.mkdir()
        (http_dir / HANDLER_CONTRACT_FILENAME).write_text(
            VALID_HANDLER_CONTRACT_YAML.format(
                handler_name="http",
                handler_class=REAL_HANDLER_HTTP_CLASS,
                tag1="http",
                tag2="rest",
            )
        )

        nonexistent_path = tmp_path / "does_not_exist"

        event_bus = InMemoryEventBus()
        isolated_registry = ProtocolBindingRegistry()

        # Put invalid path first to ensure processing continues
        process = RuntimeHostProcess(
            event_bus=event_bus,
            input_topic="test.input",
            contract_paths=[str(nonexistent_path), str(valid_dir)],
            handler_registry=isolated_registry,
        )

        await process._discover_handlers_from_contracts()

        result = process._handler_discovery.last_discovery_result
        assert result is not None

        # Should have error for nonexistent path
        assert result.has_errors
        path_errors = [e for e in result.errors if e.error_code == "PATH_NOT_FOUND"]
        assert len(path_errors) >= 1

        # But valid handler should still be registered
        assert result.handlers_registered >= 1

    @pytest.mark.asyncio
    async def test_discovery_logs_include_contract_path(
        self,
        invalid_yaml_contract_dir: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Verify discovery logs include contract path for debugging.

        When a contract fails to load, the warning log should include the
        path to the problematic contract file.
        """
        event_bus = InMemoryEventBus()
        isolated_registry = ProtocolBindingRegistry()

        process = RuntimeHostProcess(
            event_bus=event_bus,
            input_topic="test.input",
            contract_paths=[str(invalid_yaml_contract_dir)],
            handler_registry=isolated_registry,
        )

        with caplog.at_level(logging.WARNING):
            await process._discover_handlers_from_contracts()

        result = process._handler_discovery.last_discovery_result
        assert result is not None

        # Invalid contracts are filtered at loader level, not tracked in result
        assert result.handlers_discovered == 0
        assert result.handlers_registered == 0

        # But warnings should include the contract path
        warning_logs = [r for r in caplog.records if r.levelno >= logging.WARNING]
        assert len(warning_logs) > 0, "Should have warnings for invalid contracts"

        # Check that at least one warning mentions the contract path
        warning_messages = " ".join(r.message for r in warning_logs)
        assert (
            "handler_contract.yaml" in warning_messages
            or "contract" in warning_messages.lower()
        )


class TestDiscoverHandlersFromContractsFileSystemEdgeCases:
    """File system edge case tests for _discover_handlers_from_contracts().

    These tests verify handling of unusual filesystem conditions.
    """

    @pytest.mark.asyncio
    @pytest.mark.skipif(
        sys.platform == "win32",
        reason="Symlinks require elevated privileges on Windows",
    )
    async def test_discovery_with_symlink_to_valid_directory(
        self,
        tmp_path: Path,
    ) -> None:
        """Verify discovery follows symlinks to valid directories.

        When a contract path is a symlink to a valid directory,
        discovery should follow it and find handlers.
        """
        # Create actual handler directory
        real_dir = tmp_path / "real_handlers"
        real_dir.mkdir(parents=True)
        http_dir = real_dir / "http"
        http_dir.mkdir()
        (http_dir / HANDLER_CONTRACT_FILENAME).write_text(
            VALID_HANDLER_CONTRACT_YAML.format(
                handler_name="http.symlink",
                handler_class=REAL_HANDLER_HTTP_CLASS,
                tag1="http",
                tag2="symlink",
            )
        )

        # Create symlink to the directory
        symlink_dir = tmp_path / "handlers_symlink"
        symlink_dir.symlink_to(real_dir)

        event_bus = InMemoryEventBus()
        isolated_registry = ProtocolBindingRegistry()

        process = RuntimeHostProcess(
            event_bus=event_bus,
            input_topic="test.input",
            contract_paths=[str(symlink_dir)],
            handler_registry=isolated_registry,
        )

        await process._discover_handlers_from_contracts()

        result = process._handler_discovery.last_discovery_result
        assert result is not None
        # Handler should be discovered via symlink
        assert result.handlers_discovered >= 1
        assert result.handlers_registered >= 1

    @pytest.mark.asyncio
    @pytest.mark.skipif(
        sys.platform == "win32",
        reason="Symlinks require elevated privileges on Windows",
    )
    async def test_discovery_with_symlink_to_file(
        self,
        tmp_path: Path,
    ) -> None:
        """Verify discovery handles symlinks to contract files.

        When a contract path is a symlink to a contract file,
        discovery should follow it and load the contract.
        """
        # Create actual contract file
        real_dir = tmp_path / "real_handlers" / "http"
        real_dir.mkdir(parents=True)
        contract_file = real_dir / HANDLER_CONTRACT_FILENAME
        contract_file.write_text(
            VALID_HANDLER_CONTRACT_YAML.format(
                handler_name="http.file.symlink",
                handler_class=REAL_HANDLER_HTTP_CLASS,
                tag1="http",
                tag2="file",
            )
        )

        # Create symlink to the file
        symlink_file = tmp_path / "contract_symlink.yaml"
        symlink_file.symlink_to(contract_file)

        event_bus = InMemoryEventBus()
        isolated_registry = ProtocolBindingRegistry()

        process = RuntimeHostProcess(
            event_bus=event_bus,
            input_topic="test.input",
            contract_paths=[str(symlink_file)],
            handler_registry=isolated_registry,
        )

        await process._discover_handlers_from_contracts()

        result = process._handler_discovery.last_discovery_result
        assert result is not None
        # Handler should be discovered via symlink to file
        assert result.handlers_discovered >= 1

    @pytest.mark.asyncio
    @pytest.mark.skipif(
        sys.platform == "win32",
        reason="Symlinks require elevated privileges on Windows",
    )
    async def test_discovery_with_broken_symlink(
        self,
        tmp_path: Path,
    ) -> None:
        """Verify discovery handles broken symlinks gracefully.

        When a contract path is a broken symlink (target doesn't exist),
        discovery should report an appropriate error.
        """
        # Create symlink to non-existent path
        broken_symlink = tmp_path / "broken_symlink"
        broken_symlink.symlink_to(tmp_path / "nonexistent_target")

        event_bus = InMemoryEventBus()
        isolated_registry = ProtocolBindingRegistry()

        process = RuntimeHostProcess(
            event_bus=event_bus,
            input_topic="test.input",
            contract_paths=[str(broken_symlink)],
            handler_registry=isolated_registry,
        )

        await process._discover_handlers_from_contracts()

        result = process._handler_discovery.last_discovery_result
        assert result is not None
        # Should have error for broken symlink
        assert result.has_errors
        assert result.handlers_registered == 0

    @pytest.mark.asyncio
    async def test_discovery_with_special_characters_in_path(
        self,
        tmp_path: Path,
    ) -> None:
        """Verify discovery handles paths with special characters.

        Paths containing spaces, unicode, or other special characters
        should be handled correctly.
        """
        # Create directory with special characters in name
        special_dir = tmp_path / "handlers with spaces"
        special_dir.mkdir(parents=True)
        http_dir = special_dir / "http handler"
        http_dir.mkdir()
        (http_dir / HANDLER_CONTRACT_FILENAME).write_text(
            VALID_HANDLER_CONTRACT_YAML.format(
                handler_name="http.special",
                handler_class=REAL_HANDLER_HTTP_CLASS,
                tag1="http",
                tag2="special",
            )
        )

        event_bus = InMemoryEventBus()
        isolated_registry = ProtocolBindingRegistry()

        process = RuntimeHostProcess(
            event_bus=event_bus,
            input_topic="test.input",
            contract_paths=[str(special_dir)],
            handler_registry=isolated_registry,
        )

        await process._discover_handlers_from_contracts()

        result = process._handler_discovery.last_discovery_result
        assert result is not None
        # Handler should be discovered despite special characters
        assert result.handlers_discovered >= 1
        assert result.handlers_registered >= 1

    @pytest.mark.asyncio
    async def test_discovery_with_deeply_nested_structure(
        self,
        tmp_path: Path,
    ) -> None:
        """Verify discovery handles deeply nested directory structures.

        Even with many levels of nesting, contracts should be discovered.
        """
        # Create deeply nested structure
        deep_dir = tmp_path
        for i in range(10):
            deep_dir = deep_dir / f"level{i}"
        deep_dir.mkdir(parents=True)

        # Add handler at the deepest level
        (deep_dir / HANDLER_CONTRACT_FILENAME).write_text(
            VALID_HANDLER_CONTRACT_YAML.format(
                handler_name="deep.handler",
                handler_class=REAL_HANDLER_HTTP_CLASS,
                tag1="deep",
                tag2="nested",
            )
        )

        event_bus = InMemoryEventBus()
        isolated_registry = ProtocolBindingRegistry()

        process = RuntimeHostProcess(
            event_bus=event_bus,
            input_topic="test.input",
            contract_paths=[str(tmp_path)],
            handler_registry=isolated_registry,
        )

        await process._discover_handlers_from_contracts()

        result = process._handler_discovery.last_discovery_result
        assert result is not None
        # Handler should be discovered even at deep nesting level
        assert result.handlers_discovered >= 1
        assert result.handlers_registered >= 1


class TestDiscoverHandlersFromContractsCorrelationTracking:
    """Correlation ID tracking tests for _discover_handlers_from_contracts().

    These tests verify that correlation IDs are properly propagated
    through the discovery process for observability.
    """

    @pytest.mark.asyncio
    async def test_discovery_generates_correlation_id(
        self,
        valid_handler_contract_dir: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Verify discovery generates correlation ID when not provided.

        The discovery process should auto-generate a correlation ID
        for tracing purposes.
        """
        event_bus = InMemoryEventBus()
        isolated_registry = ProtocolBindingRegistry()

        process = RuntimeHostProcess(
            event_bus=event_bus,
            input_topic="test.input",
            contract_paths=[str(valid_handler_contract_dir)],
            handler_registry=isolated_registry,
        )

        with caplog.at_level(logging.INFO):
            await process._discover_handlers_from_contracts()

        # Discovery should produce INFO level logs with correlation tracking
        info_logs = [r for r in caplog.records if r.levelno == logging.INFO]
        assert len(info_logs) > 0, "Should have INFO level logs during discovery"

        # Verify the discovery completed successfully
        discovery_result = process._handler_discovery.last_discovery_result
        assert discovery_result is not None, "Discovery result should be available"
        # Discovery result tracks handlers discovered/registered, not correlation_id directly
        # Correlation tracking happens in logs and error contexts, not on the result model
        # With valid_handler_contract_dir, at least one handler should be discovered
        assert discovery_result.handlers_discovered >= 1, (
            "Discovery should discover at least one handler from valid contract directory"
        )

        # Verify correlation ID tracking in logs
        # The contract handler discovery auto-generates correlation IDs and includes
        # them in structured log extra data (e.g., extra={"correlation_id": str(uuid)})
        #
        # We verify that:
        # 1. At least one log record contains a correlation_id field
        # 2. All correlation_id values are valid UUIDs (parseable by uuid.UUID)
        # 3. All correlation IDs within the same discovery call match (same correlation)
        correlation_ids_found: list[uuid.UUID] = []
        for log_record in caplog.records:
            # Check record __dict__ for correlation_id in extra data
            record_dict = log_record.__dict__
            if "correlation_id" in record_dict:
                corr_id = record_dict["correlation_id"]
                # Verify it's not None
                assert corr_id is not None, "Correlation ID in log should not be None"

                # Convert to string for UUID parsing
                corr_id_str = str(corr_id)

                # Parse as UUID to verify it's a valid UUID (raises ValueError if invalid)
                try:
                    parsed_uuid = uuid.UUID(corr_id_str)
                except ValueError as e:
                    pytest.fail(
                        f"Correlation ID '{corr_id_str}' is not a valid UUID: {e}"
                    )

                # Verify UUID version (should be version 4 for auto-generated UUIDs)
                assert parsed_uuid.version == 4, (
                    f"Correlation ID '{corr_id_str}' should be UUID version 4, "
                    f"got version {parsed_uuid.version}"
                )

                correlation_ids_found.append(parsed_uuid)

        # Assert that we found at least one correlation ID in the logs
        # The contract handler discovery MUST log with correlation IDs for observability
        assert len(correlation_ids_found) > 0, (
            "Discovery should log with correlation_id in structured log extra data. "
            "No correlation IDs found in log records. "
            "Ensure ContractHandlerDiscovery logs with extra={'correlation_id': ...}"
        )

        # Verify all correlation IDs from the same discovery call are consistent
        # (same correlation ID should be used throughout the discovery operation)
        unique_correlation_ids = set(correlation_ids_found)
        assert len(unique_correlation_ids) == 1, (
            f"All logs from same discovery call should use the same correlation ID. "
            f"Found {len(unique_correlation_ids)} different IDs: {unique_correlation_ids}"
        )

        # Verify the found correlation ID is a valid auto-generated UUID
        found_uuid = correlation_ids_found[0]
        assert found_uuid.variant == uuid.RFC_4122, (
            f"Correlation ID should be RFC 4122 variant, got {found_uuid.variant}"
        )
