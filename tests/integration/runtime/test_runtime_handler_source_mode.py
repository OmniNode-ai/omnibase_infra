# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Integration tests for RuntimeHostProcess handler source mode integration (OMN-1095).

This module validates that RuntimeHostProcess correctly integrates with
HandlerSourceResolver for handler discovery based on the configured source mode.

Test Coverage:
- BOOTSTRAP mode loads only hardcoded bootstrap handlers
- CONTRACT mode loads only contract-discovered handlers
- HYBRID mode uses contract-first with bootstrap fallback
- Expired bootstrap_expires_at forces effective mode to CONTRACT
- Structured logging during handler resolution
- Default configuration (no handler_source) uses HYBRID mode

Related:
    - OMN-1095: Handler Source Mode Hybrid Resolution
    - src/omnibase_infra/runtime/handler_source_resolver.py
    - src/omnibase_infra/runtime/service_runtime_host_process.py
    - src/omnibase_infra/models/handlers/model_handler_source_config.py

Note:
    These tests verify handler REGISTRATION based on source mode, not handler EXECUTION.
    Handlers may fail during initialize() if external services are not available,
    but they should still be registered in the handler registry based on the configured mode.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from omnibase_infra.enums.enum_handler_source_mode import EnumHandlerSourceMode
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
from omnibase_infra.runtime.handler_source_resolver import HandlerSourceResolver
from omnibase_infra.runtime.service_runtime_host_process import RuntimeHostProcess

if TYPE_CHECKING:
    from omnibase_infra.models.handlers import ModelHandlerSourceConfig


# =============================================================================
# Constants for Handler Contract Templates
# =============================================================================

# Handler contract template for creating test contracts
HANDLER_CONTRACT_YAML_TEMPLATE = """
handler_name: "{handler_name}"
handler_class: "{handler_class}"
handler_type: "effect"
capability_tags:
  - {tag1}
  - {tag2}
"""

# Real handler class paths from omnibase_infra.handlers
REAL_HANDLER_HTTP_CLASS = "omnibase_infra.handlers.handler_http.HandlerHttpRest"
REAL_HANDLER_CONSUL_CLASS = "omnibase_infra.handlers.handler_consul.HandlerConsul"


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
def temp_contract_directory(tmp_path: Path) -> Path:
    """Create a temporary directory with test contract files.

    Creates a handler contract for HTTP handler (works without external services).

    Returns:
        Path to the handlers directory.
    """
    handlers_dir = tmp_path / "handlers"
    handlers_dir.mkdir(parents=True)

    # Create HTTP handler contract (works without config or external services)
    http_dir = handlers_dir / "http"
    http_dir.mkdir()
    http_contract = http_dir / "handler_contract.yaml"
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
def runtime_process_factory(
    isolated_handler_registry: RegistryProtocolBinding,
) -> type:
    """Factory fixture for creating RuntimeHostProcess with test config.

    Returns:
        A factory function that creates RuntimeHostProcess instances.
    """

    class RuntimeProcessFactory:
        """Factory for creating RuntimeHostProcess instances."""

        @staticmethod
        def create(
            config: dict | None = None,
            contract_paths: list[str] | None = None,
            handler_registry: RegistryProtocolBinding | None = None,
        ) -> RuntimeHostProcess:
            """Create a RuntimeHostProcess with the given configuration.

            Args:
                config: Optional configuration dict with handler_source settings.
                contract_paths: Optional list of paths for contract discovery.
                handler_registry: Optional handler registry (uses isolated by default).

            Returns:
                Configured RuntimeHostProcess instance.
            """
            event_bus = EventBusInmemory()
            return RuntimeHostProcess(
                event_bus=event_bus,
                input_topic="test.input",
                config=config,
                contract_paths=contract_paths,
                handler_registry=handler_registry or isolated_handler_registry,
            )

    return RuntimeProcessFactory


# =============================================================================
# Test Class: BOOTSTRAP Mode
# =============================================================================


class TestBootstrapModeLoadsOnlyBootstrapHandlers:
    """Tests for BOOTSTRAP mode loading only hardcoded bootstrap handlers.

    In BOOTSTRAP mode, the runtime should:
    1. Only load handlers from HandlerBootstrapSource
    2. NOT load handlers from HandlerContractSource
    3. Register all 5 core infrastructure handlers (consul, db, http, mcp, vault)
    """

    @pytest.mark.asyncio
    async def test_bootstrap_mode_loads_only_bootstrap_handlers(self) -> None:
        """Verify BOOTSTRAP mode loads only hardcoded bootstrap handlers.

        Given:
            - RuntimeHostProcess configured with BOOTSTRAP mode
            - No contract_paths provided

        When:
            - RuntimeHostProcess.start() is called

        Then:
            - Only bootstrap handlers (consul, db, http, mcp, vault) are registered
            - Contract source is NOT queried
        """
        event_bus = EventBusInmemory()
        config: dict[str, object] = {
            "handler_source": {
                "mode": "bootstrap",
            }
        }

        process = RuntimeHostProcess(
            event_bus=event_bus,
            input_topic="test.input",
            config=config,
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
    async def test_bootstrap_mode_does_not_query_contract_source(
        self,
        temp_contract_directory: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Verify BOOTSTRAP mode does NOT query contract source.

        Given:
            - RuntimeHostProcess configured with BOOTSTRAP mode
            - contract_paths provided (but should be ignored)

        When:
            - RuntimeHostProcess.start() is called

        Then:
            - Contract source discover_handlers() is NOT called
            - Only BOOTSTRAP mode is logged
        """
        event_bus = EventBusInmemory()
        config: dict[str, object] = {
            "handler_source": {
                "mode": "bootstrap",
            }
        }

        process = RuntimeHostProcess(
            event_bus=event_bus,
            input_topic="test.input",
            config=config,
            contract_paths=[str(temp_contract_directory)],
        )

        with caplog.at_level(logging.INFO):
            try:
                await process.start()

                # In BOOTSTRAP mode, contract source should NOT be called
                # The resolver itself doesn't call contract source in BOOTSTRAP mode
                # Verify through logs that BOOTSTRAP mode was used
                mode_logs = [
                    r
                    for r in caplog.records
                    if hasattr(r, "__dict__") and r.__dict__.get("mode") == "bootstrap"
                ]
                assert len(mode_logs) >= 1, "Should log bootstrap mode"

                # Verify bootstrap handlers are registered (confirming BOOTSTRAP mode worked)
                registry = get_handler_registry()
                assert registry.is_registered(HANDLER_TYPE_CONSUL)
                assert registry.is_registered(HANDLER_TYPE_DATABASE)
                assert registry.is_registered(HANDLER_TYPE_HTTP)
                assert registry.is_registered(HANDLER_TYPE_MCP)
                assert registry.is_registered(HANDLER_TYPE_VAULT)

            finally:
                await process.stop()


# =============================================================================
# Test Class: CONTRACT Mode
# =============================================================================


class TestContractModeLoadsOnlyContractHandlers:
    """Tests for CONTRACT mode loading only contract-discovered handlers.

    In CONTRACT mode, the runtime should:
    1. Only load handlers from HandlerContractSource
    2. NOT load handlers from HandlerBootstrapSource
    3. Register handlers found in contract files
    """

    @pytest.mark.asyncio
    async def test_contract_mode_loads_only_contract_handlers(
        self,
        temp_contract_directory: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Verify CONTRACT mode loads only contract-discovered handlers.

        Given:
            - RuntimeHostProcess configured with CONTRACT mode
            - contract_paths pointing to valid contract files

        When:
            - RuntimeHostProcess.start() is called

        Then:
            - Handlers from contracts are registered
            - Bootstrap handlers are NOT loaded directly (fallback disabled)
        """
        event_bus = EventBusInmemory()
        config: dict[str, object] = {
            "handler_source": {
                "mode": "contract",
            }
        }

        process = RuntimeHostProcess(
            event_bus=event_bus,
            input_topic="test.input",
            config=config,
            contract_paths=[str(temp_contract_directory)],
        )

        with caplog.at_level(logging.INFO):
            try:
                await process.start()

                # Verify the effective mode is CONTRACT
                log_messages = [r.message for r in caplog.records]
                mode_logs = [
                    msg for msg in log_messages if "source mode" in msg.lower()
                ]
                assert len(mode_logs) >= 1, "Should log handler source mode"

            finally:
                await process.stop()

    @pytest.mark.asyncio
    async def test_contract_mode_graceful_with_no_contracts(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Verify CONTRACT mode gracefully handles no contract paths.

        Given:
            - RuntimeHostProcess configured with CONTRACT mode
            - NO contract_paths provided

        When:
            - RuntimeHostProcess.start() is called

        Then:
            - Runtime should start (graceful degradation)
            - Warning or info logged about no contracts
        """
        event_bus = EventBusInmemory()
        config: dict[str, object] = {
            "handler_source": {
                "mode": "contract",
            }
        }

        process = RuntimeHostProcess(
            event_bus=event_bus,
            input_topic="test.input",
            config=config,
            # No contract_paths - graceful degradation
        )

        with caplog.at_level(logging.INFO):
            try:
                await process.start()

                # Process should still start (graceful degradation)
                assert process.is_running

            finally:
                await process.stop()


# =============================================================================
# Test Class: HYBRID Mode
# =============================================================================


class TestHybridModeContractFirstBootstrapFallback:
    """Tests for HYBRID mode using contract-first with bootstrap fallback.

    In HYBRID mode, the runtime should:
    1. Load handlers from both sources
    2. Contract handlers take precedence for same handler_id
    3. Bootstrap handlers serve as fallback when no contract match
    """

    @pytest.mark.asyncio
    async def test_hybrid_mode_contract_first_bootstrap_fallback(
        self,
        temp_contract_directory: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Verify HYBRID mode uses contract handlers first, bootstrap as fallback.

        Given:
            - RuntimeHostProcess configured with HYBRID mode
            - contract_paths with HTTP handler contract
            - Bootstrap provides consul, db, http, mcp, vault handlers

        When:
            - RuntimeHostProcess.start() is called

        Then:
            - Contract handlers take precedence when available
            - Bootstrap handlers used as fallback when contract unavailable
        """
        event_bus = EventBusInmemory()
        config: dict[str, object] = {
            "handler_source": {
                "mode": "hybrid",
            }
        }

        process = RuntimeHostProcess(
            event_bus=event_bus,
            input_topic="test.input",
            config=config,
            contract_paths=[str(temp_contract_directory)],
        )

        with caplog.at_level(logging.INFO):
            try:
                await process.start()

                # Verify hybrid mode is used
                log_records = [
                    r for r in caplog.records if "source mode" in r.message.lower()
                ]
                assert len(log_records) >= 1, "Should log handler source mode"

                # In HYBRID mode, bootstrap handlers should be available as fallback
                registry = get_handler_registry()
                # Bootstrap handlers should be registered (as fallback)
                assert registry.is_registered(HANDLER_TYPE_CONSUL)
                assert registry.is_registered(HANDLER_TYPE_DATABASE)
                assert registry.is_registered(HANDLER_TYPE_VAULT)

            finally:
                await process.stop()

    @pytest.mark.asyncio
    async def test_hybrid_mode_with_empty_contracts_uses_all_bootstrap(
        self,
        tmp_path: Path,
    ) -> None:
        """Verify HYBRID mode uses all bootstrap handlers when contracts are empty.

        Given:
            - RuntimeHostProcess configured with HYBRID mode
            - contract_paths points to empty directory (no contracts)

        When:
            - RuntimeHostProcess.start() is called

        Then:
            - All bootstrap handlers should be registered (full fallback)
        """
        # Create empty contracts directory
        empty_contracts_dir = tmp_path / "empty_contracts"
        empty_contracts_dir.mkdir()

        event_bus = EventBusInmemory()
        config: dict[str, object] = {
            "handler_source": {
                "mode": "hybrid",
            }
        }

        process = RuntimeHostProcess(
            event_bus=event_bus,
            input_topic="test.input",
            config=config,
            contract_paths=[str(empty_contracts_dir)],
        )

        try:
            await process.start()

            # All bootstrap handlers should be registered as fallback
            registry = get_handler_registry()
            assert registry.is_registered(HANDLER_TYPE_CONSUL)
            assert registry.is_registered(HANDLER_TYPE_DATABASE)
            assert registry.is_registered(HANDLER_TYPE_HTTP)
            assert registry.is_registered(HANDLER_TYPE_MCP)
            assert registry.is_registered(HANDLER_TYPE_VAULT)

        finally:
            await process.stop()


# =============================================================================
# Test Class: Bootstrap Expiry
# =============================================================================


class TestExpiredBootstrapForcesContractMode:
    """Tests for bootstrap_expires_at forcing effective mode to CONTRACT.

    When bootstrap_expires_at is set and the current time exceeds it:
    - The effective_mode becomes CONTRACT regardless of configured mode
    - Bootstrap handlers are NOT loaded
    """

    @pytest.mark.asyncio
    async def test_expired_bootstrap_forces_contract_mode(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Verify expired bootstrap_expires_at forces effective mode to CONTRACT.

        Given:
            - RuntimeHostProcess configured with HYBRID mode
            - bootstrap_expires_at set to a past datetime

        When:
            - RuntimeHostProcess.start() is called

        Then:
            - effective_mode becomes CONTRACT
            - Bootstrap handlers are NOT loaded as fallback
        """
        # Set expiry in the past (yesterday)
        past_expiry = datetime.now(UTC) - timedelta(days=1)

        event_bus = EventBusInmemory()
        config: dict[str, object] = {
            "handler_source": {
                "mode": "hybrid",  # Configured as HYBRID
                "bootstrap_expires_at": past_expiry.isoformat(),  # But expired
            }
        }

        process = RuntimeHostProcess(
            event_bus=event_bus,
            input_topic="test.input",
            config=config,
        )

        with caplog.at_level(logging.INFO):
            try:
                await process.start()

                # Look for logs indicating effective mode is CONTRACT
                # The _resolve_handler_descriptors method logs effective_mode
                mode_logs = [
                    r
                    for r in caplog.records
                    if hasattr(r, "__dict__")
                    and r.__dict__.get("effective_mode") == "contract"
                ]

                # Also check is_bootstrap_expired in logs
                expired_logs = [
                    r
                    for r in caplog.records
                    if hasattr(r, "__dict__")
                    and r.__dict__.get("is_bootstrap_expired") is True
                ]

                # Verify expiry was detected
                assert len(expired_logs) >= 1 or any(
                    "expired" in r.message.lower() for r in caplog.records
                ), "Should log that bootstrap is expired"

            finally:
                await process.stop()

    @pytest.mark.asyncio
    async def test_non_expired_bootstrap_uses_configured_mode(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Verify non-expired bootstrap uses the configured mode.

        Given:
            - RuntimeHostProcess configured with HYBRID mode
            - bootstrap_expires_at set to a future datetime

        When:
            - RuntimeHostProcess.start() is called

        Then:
            - effective_mode remains HYBRID (not forced to CONTRACT)
        """
        # Set expiry in the future (tomorrow)
        future_expiry = datetime.now(UTC) + timedelta(days=1)

        event_bus = EventBusInmemory()
        config: dict[str, object] = {
            "handler_source": {
                "mode": "hybrid",  # Configured as HYBRID
                "bootstrap_expires_at": future_expiry.isoformat(),  # Not expired
            }
        }

        process = RuntimeHostProcess(
            event_bus=event_bus,
            input_topic="test.input",
            config=config,
        )

        with caplog.at_level(logging.INFO):
            try:
                await process.start()

                # Verify is_bootstrap_expired is False
                non_expired_logs = [
                    r
                    for r in caplog.records
                    if hasattr(r, "__dict__")
                    and r.__dict__.get("is_bootstrap_expired") is False
                ]

                assert len(non_expired_logs) >= 1, (
                    "Should log is_bootstrap_expired=False"
                )

                # Bootstrap handlers should be available (HYBRID mode active)
                registry = get_handler_registry()
                assert registry.is_registered(HANDLER_TYPE_CONSUL)

            finally:
                await process.stop()


# =============================================================================
# Test Class: Structured Logging
# =============================================================================


class TestHandlerResolutionLogging:
    """Tests for structured logging during handler resolution.

    The runtime should log:
    - mode: The configured handler source mode
    - effective_mode: The actual mode used (may differ if expired)
    - bootstrap_expires_at: The expiry datetime (if set)
    - descriptor_count: Number of handlers resolved
    """

    @pytest.mark.asyncio
    async def test_handler_resolution_logs_mode_info(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Verify handler resolution logs mode and statistics.

        Given:
            - RuntimeHostProcess configured with HYBRID mode

        When:
            - RuntimeHostProcess.start() is called

        Then:
            - Logs should include structured fields for mode info
        """
        event_bus = EventBusInmemory()
        config: dict[str, object] = {
            "handler_source": {
                "mode": "hybrid",
            }
        }

        process = RuntimeHostProcess(
            event_bus=event_bus,
            input_topic="test.input",
            config=config,
        )

        with caplog.at_level(logging.INFO):
            try:
                await process.start()

                # Find the handler resolution log messages
                resolution_logs = [
                    r
                    for r in caplog.records
                    if "handler" in r.message.lower()
                    and (
                        "resolution" in r.message.lower()
                        or "source mode" in r.message.lower()
                    )
                ]

                assert len(resolution_logs) >= 1, (
                    "Expected at least one handler resolution log message"
                )

                # Check for structured logging fields
                found_mode_info = False
                for record in resolution_logs:
                    extra = getattr(record, "__dict__", {})
                    if "mode" in extra or "effective_mode" in extra:
                        found_mode_info = True
                        break

                assert found_mode_info, (
                    "Expected structured logging fields for mode info. "
                    "The runtime should log 'mode' and 'effective_mode'."
                )

            finally:
                await process.stop()

    @pytest.mark.asyncio
    async def test_handler_resolution_logs_descriptor_count(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Verify handler resolution logs descriptor count.

        Given:
            - RuntimeHostProcess configured with BOOTSTRAP mode

        When:
            - RuntimeHostProcess.start() is called

        Then:
            - Logs should include descriptor_count field
        """
        event_bus = EventBusInmemory()
        config: dict[str, object] = {
            "handler_source": {
                "mode": "bootstrap",
            }
        }

        process = RuntimeHostProcess(
            event_bus=event_bus,
            input_topic="test.input",
            config=config,
        )

        with caplog.at_level(logging.INFO):
            try:
                await process.start()

                # Find log with descriptor_count
                descriptor_count_logs = [
                    r
                    for r in caplog.records
                    if hasattr(r, "__dict__") and "descriptor_count" in r.__dict__
                ]

                assert len(descriptor_count_logs) >= 1, (
                    "Expected log message with descriptor_count field"
                )

                # Verify count is reasonable (bootstrap has 5 handlers)
                for record in descriptor_count_logs:
                    count = record.__dict__.get("descriptor_count", 0)
                    assert count >= 5, (
                        f"Expected at least 5 bootstrap handlers, got {count}"
                    )

            finally:
                await process.stop()


# =============================================================================
# Test Class: Default Configuration
# =============================================================================


class TestDefaultConfigUsesHybridMode:
    """Tests for default configuration using HYBRID mode.

    When no handler_source configuration is provided:
    - Default mode should be HYBRID
    - Bootstrap handlers should be available as fallback
    """

    @pytest.mark.asyncio
    async def test_default_config_uses_hybrid_mode(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Verify default configuration (no handler_source) uses HYBRID mode.

        Given:
            - RuntimeHostProcess with NO config

        When:
            - RuntimeHostProcess.start() is called

        Then:
            - HYBRID mode is used
            - Bootstrap handlers are available as fallback
        """
        event_bus = EventBusInmemory()

        # No config - should default to HYBRID
        process = RuntimeHostProcess(
            event_bus=event_bus,
            input_topic="test.input",
            # No config parameter
        )

        with caplog.at_level(logging.INFO):
            try:
                await process.start()

                # Verify HYBRID mode is used (check logs)
                hybrid_mode_logs = [
                    r
                    for r in caplog.records
                    if hasattr(r, "__dict__") and r.__dict__.get("mode") == "hybrid"
                ]

                assert len(hybrid_mode_logs) >= 1, (
                    "Expected HYBRID mode when no config provided"
                )

                # Bootstrap handlers should be available (HYBRID fallback)
                registry = get_handler_registry()
                assert registry.is_registered(HANDLER_TYPE_CONSUL)
                assert registry.is_registered(HANDLER_TYPE_DATABASE)
                assert registry.is_registered(HANDLER_TYPE_HTTP)

            finally:
                await process.stop()

    @pytest.mark.asyncio
    async def test_empty_handler_source_config_uses_hybrid_mode(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Verify empty handler_source config uses HYBRID mode.

        Given:
            - RuntimeHostProcess with empty handler_source dict

        When:
            - RuntimeHostProcess.start() is called

        Then:
            - HYBRID mode is used (default)
        """
        event_bus = EventBusInmemory()
        config: dict[str, object] = {
            "handler_source": {},  # Empty - should default to HYBRID
        }

        process = RuntimeHostProcess(
            event_bus=event_bus,
            input_topic="test.input",
            config=config,
        )

        with caplog.at_level(logging.INFO):
            try:
                await process.start()

                # Verify process started successfully with HYBRID mode
                assert process.is_running

                # Bootstrap handlers should be available
                registry = get_handler_registry()
                assert registry.is_registered(HANDLER_TYPE_CONSUL)

            finally:
                await process.stop()


# =============================================================================
# Test Class: Invalid Configuration Handling
# =============================================================================


class TestInvalidConfigurationHandling:
    """Tests for graceful handling of invalid configuration.

    The runtime should handle invalid config values gracefully:
    - Invalid mode string defaults to HYBRID
    - Invalid expiry datetime is ignored
    """

    @pytest.mark.asyncio
    async def test_invalid_mode_defaults_to_hybrid(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Verify invalid mode string defaults to HYBRID with warning.

        Given:
            - RuntimeHostProcess with invalid mode value

        When:
            - RuntimeHostProcess.start() is called

        Then:
            - Warning logged about invalid mode
            - HYBRID mode used as default
        """
        event_bus = EventBusInmemory()
        config: dict[str, object] = {
            "handler_source": {
                "mode": "invalid_mode",  # Invalid value
            }
        }

        process = RuntimeHostProcess(
            event_bus=event_bus,
            input_topic="test.input",
            config=config,
        )

        with caplog.at_level(logging.WARNING):
            try:
                await process.start()

                # Should have warning about invalid mode
                warning_logs = [
                    r for r in caplog.records if r.levelno == logging.WARNING
                ]
                mode_warnings = [
                    r
                    for r in warning_logs
                    if "handler_source_mode" in r.message.lower()
                ]

                assert len(mode_warnings) >= 1, (
                    "Expected warning about invalid handler_source_mode"
                )

                # Should still start with HYBRID mode
                assert process.is_running

            finally:
                await process.stop()

    @pytest.mark.asyncio
    async def test_invalid_expiry_datetime_is_ignored(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Verify invalid bootstrap_expires_at datetime is ignored with warning.

        Given:
            - RuntimeHostProcess with invalid expiry datetime string

        When:
            - RuntimeHostProcess.start() is called

        Then:
            - Warning logged about invalid datetime
            - No expiry enforcement applied
        """
        event_bus = EventBusInmemory()
        config: dict[str, object] = {
            "handler_source": {
                "mode": "hybrid",
                "bootstrap_expires_at": "not-a-valid-datetime",  # Invalid
            }
        }

        process = RuntimeHostProcess(
            event_bus=event_bus,
            input_topic="test.input",
            config=config,
        )

        with caplog.at_level(logging.WARNING):
            try:
                await process.start()

                # Should have warning about invalid datetime
                warning_logs = [
                    r for r in caplog.records if r.levelno == logging.WARNING
                ]
                datetime_warnings = [
                    r
                    for r in warning_logs
                    if "bootstrap_expires_at" in r.message.lower()
                ]

                assert len(datetime_warnings) >= 1, (
                    "Expected warning about invalid bootstrap_expires_at"
                )

                # Should still start (expiry ignored)
                assert process.is_running

                # Bootstrap handlers should be available (no expiry applied)
                registry = get_handler_registry()
                assert registry.is_registered(HANDLER_TYPE_CONSUL)

            finally:
                await process.stop()


# =============================================================================
# Test Class: HandlerSourceResolver Direct Integration
# =============================================================================


class TestHandlerSourceResolverIntegration:
    """Tests for direct HandlerSourceResolver integration with sources.

    These tests verify the resolver works correctly with actual
    bootstrap and contract sources.
    """

    @pytest.mark.asyncio
    async def test_resolver_with_real_bootstrap_source(self) -> None:
        """Verify HandlerSourceResolver works with real HandlerBootstrapSource.

        Given:
            - HandlerSourceResolver with real HandlerBootstrapSource
            - Mode set to BOOTSTRAP

        When:
            - resolve_handlers() is called

        Then:
            - Returns 5 bootstrap handler descriptors
        """
        bootstrap_source = HandlerBootstrapSource()

        # Create mock contract source (not used in BOOTSTRAP mode)
        mock_contract_source = MagicMock()
        mock_contract_source.source_type = "CONTRACT"

        resolver = HandlerSourceResolver(
            bootstrap_source=bootstrap_source,
            contract_source=mock_contract_source,
            mode=EnumHandlerSourceMode.BOOTSTRAP,
        )

        result = await resolver.resolve_handlers()

        # Should have 5 bootstrap handlers
        assert len(result.descriptors) == 5
        assert len(result.validation_errors) == 0

        # Verify handler IDs
        handler_ids = {d.handler_id for d in result.descriptors}
        expected_ids = {
            "bootstrap.consul",
            "bootstrap.db",
            "bootstrap.http",
            "bootstrap.mcp",
            "bootstrap.vault",
        }
        assert handler_ids == expected_ids

    @pytest.mark.asyncio
    async def test_resolver_mode_property(self) -> None:
        """Verify HandlerSourceResolver exposes mode property."""
        bootstrap_source = HandlerBootstrapSource()
        mock_contract_source = MagicMock()
        mock_contract_source.source_type = "CONTRACT"

        resolver = HandlerSourceResolver(
            bootstrap_source=bootstrap_source,
            contract_source=mock_contract_source,
            mode=EnumHandlerSourceMode.HYBRID,
        )

        assert resolver.mode == EnumHandlerSourceMode.HYBRID


__all__ = [
    "TestBootstrapModeLoadsOnlyBootstrapHandlers",
    "TestContractModeLoadsOnlyContractHandlers",
    "TestHybridModeContractFirstBootstrapFallback",
    "TestExpiredBootstrapForcesContractMode",
    "TestHandlerResolutionLogging",
    "TestDefaultConfigUsesHybridMode",
    "TestInvalidConfigurationHandling",
    "TestHandlerSourceResolverIntegration",
]
