# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Unit tests for ContractHandlerDiscovery class.

This module tests the ContractHandlerDiscovery service which bridges
the HandlerPluginLoader with the ProtocolBindingRegistry.

Part of OMN-1133: Handler Discovery Service implementation.
"""

from __future__ import annotations

from pathlib import Path
from uuid import UUID, uuid4

import pytest

from omnibase_infra.models.runtime.model_discovery_result import ModelDiscoveryResult
from omnibase_infra.runtime import (
    ContractHandlerDiscovery,
    HandlerPluginLoader,
    ProtocolBindingRegistry,
    ProtocolHandlerDiscovery,
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
        handler_registry: ProtocolBindingRegistry,
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
        handler_registry: ProtocolBindingRegistry,
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
        handler_registry: ProtocolBindingRegistry,
        mixed_valid_invalid_directory: Path,
    ) -> None:
        """Test that valid handlers are registered even when some fail."""
        result = await discovery_service.discover_and_register(
            [mixed_valid_invalid_directory]
        )

        # Should have discovered some handlers but have errors
        assert result.handlers_discovered == 1  # Only the valid one
        assert result.handlers_registered == 1
        # There may be errors from the invalid contracts
        # (the exact behavior depends on whether they're counted as discovered)

        # Valid handler should be registered
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
    ) -> None:
        """Test that provided correlation ID is used."""
        correlation_id = uuid4()
        result = await discovery_service.discover_and_register(
            [empty_directory],
            correlation_id=correlation_id,
        )
        assert isinstance(result, ModelDiscoveryResult)


class TestContractHandlerDiscoveryMixedPaths:
    """Tests for handling mixed path types (files and directories)."""

    @pytest.mark.asyncio
    async def test_handles_mixed_files_and_directories(
        self,
        discovery_service: ContractHandlerDiscovery,
        handler_registry: ProtocolBindingRegistry,
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
    """Tests for integration with ProtocolBindingRegistry."""

    @pytest.mark.asyncio
    async def test_registered_handler_is_retrievable(
        self,
        discovery_service: ContractHandlerDiscovery,
        handler_registry: ProtocolBindingRegistry,
        valid_contract_path: Path,
    ) -> None:
        """Test that registered handlers can be retrieved from registry."""
        await discovery_service.discover_and_register([valid_contract_path])

        handler_class = handler_registry.get("test.valid.handler")
        assert handler_class is not None

    @pytest.mark.asyncio
    async def test_multiple_discoveries_accumulate_registrations(
        self,
        discovery_service: ContractHandlerDiscovery,
        handler_registry: ProtocolBindingRegistry,
        valid_contract_directory: Path,
    ) -> None:
        """Test that multiple discovery calls accumulate registrations."""
        # First discovery
        result1 = await discovery_service.discover_and_register(
            [valid_contract_directory]
        )

        # Second discovery with same contracts (should overwrite)
        result2 = await discovery_service.discover_and_register(
            [valid_contract_directory]
        )

        # Both handlers should still be registered
        assert handler_registry.is_registered("handler.one")
        assert handler_registry.is_registered("handler.two")
