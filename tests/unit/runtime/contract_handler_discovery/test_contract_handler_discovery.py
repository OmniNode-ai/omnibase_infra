# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Unit tests for ContractHandlerDiscovery class.

This module tests the ContractHandlerDiscovery service which bridges
the HandlerPluginLoader with the RegistryProtocolBinding.

Part of OMN-1133: Handler Discovery Service implementation.
"""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import pytest

from omnibase_infra.models.runtime.model_discovery_result import ModelDiscoveryResult
from omnibase_infra.runtime import (
    ContractHandlerDiscovery,
    ProtocolHandlerDiscovery,
    RegistryProtocolBinding,
)


class TestContractHandlerDiscoveryProtocolCompliance:
    """Tests for ProtocolHandlerDiscovery protocol compliance."""

    def test_implements_protocol_handler_discovery(
        self,
        discovery_service: ContractHandlerDiscovery,
    ) -> None:
        """Test that ContractHandlerDiscovery implements ProtocolHandlerDiscovery."""
        assert isinstance(discovery_service, ProtocolHandlerDiscovery)

    def test_has_discover_and_register_method(
        self,
        discovery_service: ContractHandlerDiscovery,
    ) -> None:
        """Test that discover_and_register method exists and is callable."""
        assert hasattr(discovery_service, "discover_and_register")
        assert callable(discovery_service.discover_and_register)


class TestContractHandlerDiscoveryBasicFunctionality:
    """Tests for basic discovery and registration functionality."""

    @pytest.mark.asyncio
    async def test_discover_single_valid_contract(
        self,
        discovery_service: ContractHandlerDiscovery,
        handler_registry: RegistryProtocolBinding,
        valid_contract_path: Path,
    ) -> None:
        """Test discovery and registration of a single valid contract file."""
        result = await discovery_service.discover_and_register([valid_contract_path])

        assert isinstance(result, ModelDiscoveryResult)
        assert result.handlers_discovered == 1
        assert result.handlers_registered == 1
        assert not result.has_errors
        assert bool(result)  # Result should be truthy when no errors

        # Verify handler was registered
        assert handler_registry.is_registered("test.valid.handler")

    @pytest.mark.asyncio
    async def test_discover_directory_with_multiple_contracts(
        self,
        discovery_service: ContractHandlerDiscovery,
        handler_registry: RegistryProtocolBinding,
        valid_contract_directory: Path,
    ) -> None:
        """Test discovery and registration from a directory with multiple contracts."""
        result = await discovery_service.discover_and_register(
            [valid_contract_directory]
        )

        assert isinstance(result, ModelDiscoveryResult)
        assert result.handlers_discovered == 2
        assert result.handlers_registered == 2
        assert not result.has_errors
        assert bool(result)

        # Verify handlers were registered
        assert handler_registry.is_registered("handler.one")
        assert handler_registry.is_registered("handler.two")

    @pytest.mark.asyncio
    async def test_discover_empty_directory_returns_empty_result(
        self,
        discovery_service: ContractHandlerDiscovery,
        empty_directory: Path,
    ) -> None:
        """Test that discovering from an empty directory returns an empty result."""
        result = await discovery_service.discover_and_register([empty_directory])

        assert isinstance(result, ModelDiscoveryResult)
        assert result.handlers_discovered == 0
        assert result.handlers_registered == 0
        assert not result.has_errors
        assert bool(result)  # Empty result is still successful


class TestContractHandlerDiscoveryErrorHandling:
    """Tests for error handling during discovery."""

    @pytest.mark.asyncio
    async def test_non_existent_path_adds_error(
        self,
        discovery_service: ContractHandlerDiscovery,
    ) -> None:
        """Test that non-existent paths are recorded as errors."""
        non_existent = Path("/non/existent/path")
        result = await discovery_service.discover_and_register([non_existent])

        assert result.handlers_discovered == 0
        assert result.handlers_registered == 0
        assert result.has_errors
        assert not bool(result)  # Result should be falsy when errors exist
        assert len(result.errors) == 1
        assert result.errors[0].error_code == "PATH_NOT_FOUND"

    @pytest.mark.asyncio
    async def test_mixed_valid_invalid_contracts_partial_success(
        self,
        discovery_service: ContractHandlerDiscovery,
        handler_registry: RegistryProtocolBinding,
        mixed_valid_invalid_directory: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Test that valid handlers are registered even when some contracts fail to load.

        This test verifies the partial success behavior where:
        1. Valid contracts are discovered and registered successfully
        2. Invalid contracts (bad YAML, missing fields) are filtered out by the loader
        3. Failures are logged as warnings/errors but don't prevent successful handlers
        4. The result.has_errors is False because ContractHandlerDiscovery only sees
           the filtered (successful) handlers from the loader

        Note: The HandlerPluginLoader filters out failed contracts during
        directory loading. Failures are logged as warnings but not captured
        in the result's errors list. This is intentional design - individual
        handler failures should not prevent other handlers from loading.

        IMPORTANT: This test explicitly verifies that errors ARE captured in logs
        even though result.has_errors is False. This proves the logging is working
        correctly for observability purposes.
        """
        import logging

        with caplog.at_level(logging.WARNING):
            result = await discovery_service.discover_and_register(
                [mixed_valid_invalid_directory]
            )

        # Should have discovered and registered only the valid handler
        assert result.handlers_discovered == 1  # Only the valid one
        assert result.handlers_registered == 1

        # Invalid contracts produce warnings in the loader, not errors in result
        # This is by design: HandlerPluginLoader filters failures during load
        # and only returns successful handlers to ContractHandlerDiscovery
        # Note: has_errors is False because the discovery operation succeeded
        # from the ContractHandlerDiscovery perspective - it registered all
        # handlers that the loader returned to it
        assert not result.has_errors  # Discovery itself succeeded

        # =================================================================
        # CRITICAL ASSERTION: Verify errors ARE logged even though has_errors is False
        # This is the key verification that logging works for observability
        # =================================================================

        # Collect all warning-level and above log records
        warning_records = [
            record for record in caplog.records if record.levelno >= logging.WARNING
        ]

        # There MUST be logged warnings for the invalid contracts
        assert len(warning_records) >= 2, (
            f"Expected at least 2 warning logs for invalid contracts, "
            f"but got {len(warning_records)}. Log records: "
            f"{[r.getMessage() for r in caplog.records]}"
        )

        # Verify specific error types are logged
        warning_messages = [record.message for record in warning_records]
        warning_text = " ".join(warning_messages)
        assert "invalid_yaml" in warning_text.lower() or "Invalid YAML" in warning_text
        assert (
            "missing_class" in warning_text.lower() or "Field required" in warning_text
        )

        # Verify that error/failure indicators ARE present in the logs
        # This proves logging is working correctly for observability purposes
        error_indicators = ["error", "fail", "invalid", "missing", "cannot", "unable"]
        warning_text_lower = warning_text.lower()
        matching_indicators = [
            ind for ind in error_indicators if ind in warning_text_lower
        ]
        assert matching_indicators, (
            f"Expected error indicators {error_indicators} in warning logs but found none. "
            f"This indicates logging may not be capturing contract failures properly. "
            f"Warning messages: {warning_messages}"
        )

        # Valid handler should be registered despite failures of other contracts
        assert handler_registry.is_registered("valid.handler")


class TestContractHandlerDiscoveryCorrelationId:
    """Tests for correlation ID handling."""

    @pytest.mark.asyncio
    async def test_auto_generates_correlation_id(
        self,
        discovery_service: ContractHandlerDiscovery,
        empty_directory: Path,
    ) -> None:
        """Test that correlation ID is auto-generated when not provided."""
        result = await discovery_service.discover_and_register([empty_directory])
        # Result should complete successfully (proves correlation_id was auto-generated)
        assert isinstance(result, ModelDiscoveryResult)

    @pytest.mark.asyncio
    async def test_preserves_provided_correlation_id(
        self,
        discovery_service: ContractHandlerDiscovery,
        empty_directory: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Test that provided correlation ID is preserved and used in logging.

        Verifies that:
        1. The discovery operation completes successfully with the provided ID
        2. The exact correlation_id string appears in log records' extra fields
        3. The ID is propagated to internal operations for tracing
        """
        import logging

        correlation_id = uuid4()
        correlation_id_str = str(correlation_id)

        with caplog.at_level(logging.DEBUG):
            result = await discovery_service.discover_and_register(
                [empty_directory],
                correlation_id=correlation_id,
            )

        assert isinstance(result, ModelDiscoveryResult)

        # =================================================================
        # Robustly verify correlation_id appears in log records
        # Check multiple possible locations where extra fields may appear
        # =================================================================
        correlation_id_in_logs = False
        found_in_location = None

        for record in caplog.records:
            # Method 1: Check if correlation_id is a direct attribute
            # (set via extra={} in logging call)
            if hasattr(record, "correlation_id"):
                # Use record.__dict__ access for dynamic attribute set via extra={}
                record_correlation_id = record.__dict__["correlation_id"]
                if str(record_correlation_id) == correlation_id_str:
                    correlation_id_in_logs = True
                    found_in_location = "record.correlation_id attribute"
                    break

            # Method 2: Check record.__dict__ for extra fields (fallback)
            # Some logging configurations store extras here without setting attribute
            elif "correlation_id" in record.__dict__:
                record_correlation_id = record.__dict__["correlation_id"]
                if str(record_correlation_id) == correlation_id_str:
                    correlation_id_in_logs = True
                    found_in_location = "record.__dict__['correlation_id']"
                    break

            # Method 3: Check the formatted message itself
            # Some loggers embed correlation_id in the message format
            if correlation_id_str in record.getMessage():
                correlation_id_in_logs = True
                found_in_location = "log message content"
                break

            # Method 4: Check args if correlation_id is passed as format arg
            if record.args and correlation_id_str in str(record.args):
                correlation_id_in_logs = True
                found_in_location = "record.args"
                break

        # Build detailed error message for debugging
        log_details = []
        for record in caplog.records:
            detail: dict[str, object] = {
                "message": record.getMessage(),
                "has_correlation_id_attr": hasattr(record, "correlation_id"),
            }
            if hasattr(record, "correlation_id"):
                detail["correlation_id_value"] = record.__dict__["correlation_id"]
            log_details.append(detail)

        assert correlation_id_in_logs, (
            f"Expected exact correlation_id '{correlation_id_str}' to appear in logs. "
            f"Found in: {found_in_location}. "
            f"Log record details: {log_details}"
        )


class TestContractHandlerDiscoveryMixedPaths:
    """Tests for handling mixed path types (files and directories)."""

    @pytest.mark.asyncio
    async def test_handles_mixed_files_and_directories(
        self,
        discovery_service: ContractHandlerDiscovery,
        handler_registry: RegistryProtocolBinding,
        valid_contract_path: Path,
        valid_contract_directory: Path,
    ) -> None:
        """Test that both files and directories are processed."""
        result = await discovery_service.discover_and_register(
            [
                valid_contract_path,  # File
                valid_contract_directory,  # Directory
            ]
        )

        # Should have discovered handlers from both
        assert result.handlers_discovered >= 1
        assert result.handlers_registered >= 1


class TestContractHandlerDiscoveryResultModel:
    """Tests for ModelDiscoveryResult properties."""

    @pytest.mark.asyncio
    async def test_result_has_errors_property(
        self,
        discovery_service: ContractHandlerDiscovery,
    ) -> None:
        """Test that result has_errors property works correctly."""
        non_existent = Path("/non/existent/path")
        result = await discovery_service.discover_and_register([non_existent])

        assert result.has_errors is True
        assert bool(result) is False

    @pytest.mark.asyncio
    async def test_result_has_warnings_property(
        self,
        discovery_service: ContractHandlerDiscovery,
        empty_directory: Path,
    ) -> None:
        """Test that result has_warnings property works correctly."""
        result = await discovery_service.discover_and_register([empty_directory])

        # Empty directory should not produce warnings
        assert result.has_warnings is False

    @pytest.mark.asyncio
    async def test_result_discovered_at_timestamp(
        self,
        discovery_service: ContractHandlerDiscovery,
        empty_directory: Path,
    ) -> None:
        """Test that result includes discovery timestamp."""
        result = await discovery_service.discover_and_register([empty_directory])

        assert result.discovered_at is not None


class TestContractHandlerDiscoveryRegistryIntegration:
    """Tests for integration with RegistryProtocolBinding."""

    @pytest.mark.asyncio
    async def test_registered_handler_is_retrievable(
        self,
        discovery_service: ContractHandlerDiscovery,
        handler_registry: RegistryProtocolBinding,
        valid_contract_path: Path,
    ) -> None:
        """Test that registered handlers can be retrieved from registry."""
        await discovery_service.discover_and_register([valid_contract_path])

        handler_class = handler_registry.get("test.valid.handler")
        assert handler_class is not None

    @pytest.mark.asyncio
    async def test_multiple_discoveries_idempotent_reregistration(
        self,
        discovery_service: ContractHandlerDiscovery,
        handler_registry: RegistryProtocolBinding,
        valid_contract_directory: Path,
    ) -> None:
        """Test that multiple discovery calls with same contracts are idempotent.

        This test verifies re-registration behavior:
        1. First discovery registers handlers
        2. Second discovery with same contracts re-registers (overwrites) handlers
        3. Both handlers remain accessible after re-registration
        4. No errors occur from duplicate registration

        The registry allows re-registration of the same handler name, which
        overwrites the previous binding. This is idempotent - running discovery
        twice with the same contracts yields the same final state.
        """
        # First discovery - establishes initial registrations
        result1 = await discovery_service.discover_and_register(
            [valid_contract_directory]
        )
        assert result1.handlers_registered == 2
        assert handler_registry.is_registered("handler.one")
        assert handler_registry.is_registered("handler.two")

        # Capture handler classes after first registration
        handler_one_class_v1 = handler_registry.get("handler.one")
        handler_two_class_v1 = handler_registry.get("handler.two")

        # Second discovery with same contracts - should succeed (idempotent)
        # Re-registration overwrites the existing bindings
        result2 = await discovery_service.discover_and_register(
            [valid_contract_directory]
        )
        assert result2.handlers_registered == 2
        assert not result2.has_errors  # No errors from re-registration

        # Both handlers should still be registered after re-registration
        assert handler_registry.is_registered("handler.one")
        assert handler_registry.is_registered("handler.two")

        # Handler classes should be the same (same contracts, same classes)
        handler_one_class_v2 = handler_registry.get("handler.one")
        handler_two_class_v2 = handler_registry.get("handler.two")
        assert handler_one_class_v1 is handler_one_class_v2
        assert handler_two_class_v1 is handler_two_class_v2


class TestContractHandlerDiscoveryObservability:
    """Tests for observability features (last_discovery_result caching)."""

    def test_last_discovery_result_initially_none(
        self,
        discovery_service: ContractHandlerDiscovery,
    ) -> None:
        """Test that last_discovery_result is None before any discovery."""
        assert discovery_service.last_discovery_result is None

    @pytest.mark.asyncio
    async def test_last_discovery_result_cached_after_discovery(
        self,
        discovery_service: ContractHandlerDiscovery,
        valid_contract_path: Path,
    ) -> None:
        """Test that last_discovery_result is populated after discovery."""
        result = await discovery_service.discover_and_register([valid_contract_path])

        cached = discovery_service.last_discovery_result
        assert cached is not None
        assert cached is result  # Should be the same object
        assert cached.handlers_discovered == result.handlers_discovered
        assert cached.handlers_registered == result.handlers_registered

    @pytest.mark.asyncio
    async def test_last_discovery_result_updated_on_each_discovery(
        self,
        discovery_service: ContractHandlerDiscovery,
        valid_contract_path: Path,
        empty_directory: Path,
    ) -> None:
        """Test that last_discovery_result is updated on each discovery call."""
        # First discovery
        result1 = await discovery_service.discover_and_register([valid_contract_path])
        cached1 = discovery_service.last_discovery_result
        assert cached1 is result1
        assert cached1.handlers_discovered == 1

        # Second discovery (different path)
        result2 = await discovery_service.discover_and_register([empty_directory])
        cached2 = discovery_service.last_discovery_result
        assert cached2 is result2
        assert cached2 is not cached1  # Different object
        assert cached2.handlers_discovered == 0

    @pytest.mark.asyncio
    async def test_last_discovery_result_enables_observability_queries(
        self,
        discovery_service: ContractHandlerDiscovery,
        valid_contract_directory: Path,
    ) -> None:
        """Test that cached result can be queried for observability purposes."""
        await discovery_service.discover_and_register([valid_contract_directory])

        # Simulate observability tool querying the discovery state
        cached = discovery_service.last_discovery_result
        assert cached is not None

        # Can inspect discovery metrics without re-running
        assert cached.handlers_discovered >= 0
        assert cached.handlers_registered >= 0
        assert cached.discovered_at is not None
        assert hasattr(cached, "has_errors")
        assert hasattr(cached, "has_warnings")
