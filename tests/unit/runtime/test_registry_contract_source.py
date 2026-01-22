# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Unit tests for RegistryContractSource Consul KV discovery.

Tests the RegistryContractSource functionality including:
- Discovery of handler contracts from Consul KV storage
- Transformation of contracts to ModelHandlerDescriptor instances
- Contract validation during discovery
- Error handling for connection failures and malformed contracts
- Environment variable configuration

Related:
    - OMN-1100: Registry-Based Handler Contract Discovery
    - src/omnibase_infra/runtime/registry_contract_source.py

Expected Behavior:
    RegistryContractSource implements ProtocolContractSource from omnibase_infra.
    It discovers handler contracts from Consul KV by fetching all keys under
    a configured prefix, parsing the YAML values, and transforming them into
    ModelHandlerDescriptor instances.

    The source_type property returns "REGISTRY" as per the protocol.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch
from uuid import UUID

import pytest

from omnibase_infra.enums import EnumHandlerErrorType
from omnibase_infra.models.handlers import (
    ModelContractDiscoveryResult,
    ModelHandlerDescriptor,
)
from omnibase_infra.runtime.protocol_contract_source import ProtocolContractSource
from omnibase_infra.runtime.registry_contract_source import (
    DEFAULT_CONTRACT_PREFIX,
    MAX_CONTRACT_SIZE,
    RegistryContractSource,
)

# =============================================================================
# Test Constants
# =============================================================================

VALID_HANDLER_CONTRACT_YAML = """
handler_id: "effect.test.handler"
name: "Test Effect Handler"
contract_version:
  major: 1
  minor: 0
  patch: 0
description: "A test handler for unit tests"
descriptor:
  handler_kind: "effect"
input_model: "omnibase_infra.models.types.JsonDict"
output_model: "omnibase_core.models.dispatch.ModelHandlerOutput"
metadata:
  handler_class: "omnibase_infra.handlers.TestHandler"
"""

MINIMAL_HANDLER_CONTRACT_YAML = """
handler_id: "{handler_id}"
name: "{name}"
contract_version:
  major: 1
  minor: 0
  patch: 0
descriptor:
  handler_kind: "compute"
input_model: "test.models.Input"
output_model: "test.models.Output"
"""


# =============================================================================
# Source Type Tests
# =============================================================================


class TestRegistryContractSourceType:
    """Tests for source_type property."""

    def test_source_type_returns_registry(self) -> None:
        """source_type should return 'REGISTRY' to identify the source type.

        The source_type is used for observability and debugging purposes only.
        The runtime MUST NOT branch on this value.
        """
        with patch("consul.Consul"):
            source = RegistryContractSource()

        assert source.source_type == "REGISTRY"

    def test_implements_protocol_contract_source(self) -> None:
        """RegistryContractSource should implement ProtocolContractSource.

        The implementation must satisfy ProtocolContractSource with:
        - source_type property returning "REGISTRY"
        - async discover_handlers() method returning ModelContractDiscoveryResult
        """
        with patch("consul.Consul"):
            source = RegistryContractSource()

        # Protocol compliance check via duck typing (ONEX convention)
        assert hasattr(source, "source_type")
        assert hasattr(source, "discover_handlers")
        assert callable(source.discover_handlers)

        # Runtime checkable protocol verification
        assert isinstance(source, ProtocolContractSource)


# =============================================================================
# Discovery Tests
# =============================================================================


class TestRegistryContractSourceDiscovery:
    """Tests for discover_handlers method."""

    @pytest.mark.asyncio
    async def test_discover_handlers_empty_registry(self) -> None:
        """discover_handlers should return empty result when no keys exist.

        When Consul KV returns None (no keys under prefix), the source should
        return an empty ModelContractDiscoveryResult with no descriptors and
        no validation errors.
        """
        mock_client = MagicMock()
        mock_client.kv.get.return_value = (0, None)  # (index, data)

        with patch("consul.Consul", return_value=mock_client):
            source = RegistryContractSource()
            result = await source.discover_handlers()

        assert isinstance(result, ModelContractDiscoveryResult)
        assert result.descriptors == []
        assert result.validation_errors == []
        mock_client.kv.get.assert_called_once_with(
            DEFAULT_CONTRACT_PREFIX, recurse=True
        )

    @pytest.mark.asyncio
    async def test_discover_handlers_success(self) -> None:
        """discover_handlers should create descriptors from valid contracts.

        When Consul KV returns valid contract YAML, the source should parse it
        and create ModelHandlerDescriptor instances with correct attributes.
        """
        mock_client = MagicMock()
        mock_client.kv.get.return_value = (
            1,  # index
            [
                {
                    "Key": f"{DEFAULT_CONTRACT_PREFIX}effect.test.handler",
                    "Value": VALID_HANDLER_CONTRACT_YAML.encode("utf-8"),
                },
            ],
        )

        with patch("consul.Consul", return_value=mock_client):
            source = RegistryContractSource(host="localhost", port=8500)
            result = await source.discover_handlers()

        assert isinstance(result, ModelContractDiscoveryResult)
        assert len(result.descriptors) == 1
        assert len(result.validation_errors) == 0

        descriptor = result.descriptors[0]
        assert isinstance(descriptor, ModelHandlerDescriptor)
        assert descriptor.handler_id == "effect.test.handler"
        assert descriptor.name == "Test Effect Handler"
        # version is a ModelSemVer object; compare string representation
        assert str(descriptor.version) == "1.0.0"
        assert descriptor.handler_kind == "effect"
        assert descriptor.input_model == "omnibase_infra.models.types.JsonDict"
        assert (
            descriptor.output_model
            == "omnibase_core.models.dispatch.ModelHandlerOutput"
        )
        assert descriptor.handler_class == "omnibase_infra.handlers.TestHandler"
        assert "consul://" in descriptor.contract_path

    @pytest.mark.asyncio
    async def test_discover_handlers_multiple_contracts(self) -> None:
        """discover_handlers should handle multiple contracts from registry.

        When Consul KV returns multiple contract entries, all should be parsed
        and returned as separate descriptors.
        """
        contract1 = MINIMAL_HANDLER_CONTRACT_YAML.format(
            handler_id="handler.one",
            name="Handler One",
        )
        contract2 = MINIMAL_HANDLER_CONTRACT_YAML.format(
            handler_id="handler.two",
            name="Handler Two",
        )

        mock_client = MagicMock()
        mock_client.kv.get.return_value = (
            2,
            [
                {
                    "Key": f"{DEFAULT_CONTRACT_PREFIX}handler.one",
                    "Value": contract1.encode("utf-8"),
                },
                {
                    "Key": f"{DEFAULT_CONTRACT_PREFIX}handler.two",
                    "Value": contract2.encode("utf-8"),
                },
            ],
        )

        with patch("consul.Consul", return_value=mock_client):
            source = RegistryContractSource()
            result = await source.discover_handlers()

        assert len(result.descriptors) == 2
        assert len(result.validation_errors) == 0

        handler_ids = {d.handler_id for d in result.descriptors}
        assert handler_ids == {"handler.one", "handler.two"}

    @pytest.mark.asyncio
    async def test_discover_handlers_skips_prefix_key(self) -> None:
        """discover_handlers should skip the prefix directory key itself.

        Consul may return the prefix key in results; it should be skipped.
        """
        mock_client = MagicMock()
        mock_client.kv.get.return_value = (
            1,
            [
                {
                    "Key": DEFAULT_CONTRACT_PREFIX,  # The prefix itself
                    "Value": None,
                },
                {
                    # Key must match contract's handler_id (strict mode enforces this)
                    "Key": f"{DEFAULT_CONTRACT_PREFIX}effect.test.handler",
                    "Value": VALID_HANDLER_CONTRACT_YAML.encode("utf-8"),
                },
            ],
        )

        with patch("consul.Consul", return_value=mock_client):
            source = RegistryContractSource()
            result = await source.discover_handlers()

        assert len(result.descriptors) == 1
        assert result.descriptors[0].handler_id == "effect.test.handler"

    @pytest.mark.asyncio
    async def test_discover_handlers_skips_none_values(self) -> None:
        """discover_handlers should skip entries with None values."""
        mock_client = MagicMock()
        mock_client.kv.get.return_value = (
            1,
            [
                {
                    "Key": f"{DEFAULT_CONTRACT_PREFIX}empty.handler",
                    "Value": None,  # Empty/deleted key
                },
                {
                    # Key must match contract's handler_id (strict mode enforces this)
                    "Key": f"{DEFAULT_CONTRACT_PREFIX}effect.test.handler",
                    "Value": VALID_HANDLER_CONTRACT_YAML.encode("utf-8"),
                },
            ],
        )

        with patch("consul.Consul", return_value=mock_client):
            source = RegistryContractSource()
            result = await source.discover_handlers()

        assert len(result.descriptors) == 1
        assert result.descriptors[0].handler_id == "effect.test.handler"


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestRegistryContractSourceErrorHandling:
    """Tests for error handling in RegistryContractSource."""

    @pytest.mark.asyncio
    async def test_discover_handlers_connection_error_graceful_mode(self) -> None:
        """Connection errors in graceful mode should collect errors, not raise.

        When graceful_mode=True and Consul connection fails, the error should
        be collected in validation_errors rather than raised as an exception.
        """
        import consul

        mock_client = MagicMock()
        mock_client.kv.get.side_effect = consul.ConsulException("Connection refused")

        with patch("consul.Consul", return_value=mock_client):
            source = RegistryContractSource(graceful_mode=True)
            result = await source.discover_handlers()

        assert isinstance(result, ModelContractDiscoveryResult)
        assert result.descriptors == []
        assert len(result.validation_errors) == 1

        error = result.validation_errors[0]
        assert error.error_type == EnumHandlerErrorType.CONTRACT_PARSE_ERROR
        assert error.rule_id == "REGISTRY-002"
        assert "Connection refused" in error.message
        assert error.correlation_id is not None
        assert isinstance(error.correlation_id, UUID)

    @pytest.mark.asyncio
    async def test_discover_handlers_connection_error_strict_mode(self) -> None:
        """Connection errors in strict mode should raise exception.

        When graceful_mode=False (default) and Consul connection fails, the
        exception should be propagated to the caller.
        """
        import consul

        mock_client = MagicMock()
        mock_client.kv.get.side_effect = consul.ConsulException("Connection refused")

        with patch("consul.Consul", return_value=mock_client):
            source = RegistryContractSource(graceful_mode=False)

            with pytest.raises(consul.ConsulException, match="Connection refused"):
                await source.discover_handlers()

    @pytest.mark.asyncio
    async def test_parse_contract_size_limit(self) -> None:
        """Contracts exceeding MAX_CONTRACT_SIZE should be rejected.

        Large contracts may indicate misconfiguration or malicious content.
        They should be rejected with an appropriate error.
        """
        # Create a contract that exceeds the size limit
        oversized_content = "x" * (MAX_CONTRACT_SIZE + 1)

        mock_client = MagicMock()
        mock_client.kv.get.return_value = (
            1,
            [
                {
                    "Key": f"{DEFAULT_CONTRACT_PREFIX}oversized.handler",
                    "Value": oversized_content.encode("utf-8"),
                },
            ],
        )

        with patch("consul.Consul", return_value=mock_client):
            source = RegistryContractSource(graceful_mode=True)
            result = await source.discover_handlers()

        assert len(result.descriptors) == 0
        assert len(result.validation_errors) == 1

        error = result.validation_errors[0]
        assert error.error_type == EnumHandlerErrorType.CONTRACT_PARSE_ERROR
        assert error.rule_id == "REGISTRY-001"
        assert "exceeds size limit" in error.message

    @pytest.mark.asyncio
    async def test_parse_contract_invalid_yaml(self) -> None:
        """Invalid YAML content should produce a validation error.

        Malformed YAML should be caught and reported with a clear error
        message and remediation hint.
        """
        invalid_yaml = """
this is not valid yaml: [
    unclosed bracket
handler_id: "missing"
"""
        mock_client = MagicMock()
        mock_client.kv.get.return_value = (
            1,
            [
                {
                    "Key": f"{DEFAULT_CONTRACT_PREFIX}invalid.handler",
                    "Value": invalid_yaml.encode("utf-8"),
                },
            ],
        )

        with patch("consul.Consul", return_value=mock_client):
            source = RegistryContractSource(graceful_mode=True)
            result = await source.discover_handlers()

        assert len(result.descriptors) == 0
        assert len(result.validation_errors) == 1

        error = result.validation_errors[0]
        assert error.error_type == EnumHandlerErrorType.CONTRACT_PARSE_ERROR
        assert error.rule_id == "REGISTRY-001"
        assert "invalid.handler" in error.message
        assert error.remediation_hint is not None
        assert "YAML syntax" in error.remediation_hint

    @pytest.mark.asyncio
    async def test_parse_contract_invalid_yaml_strict_mode(self) -> None:
        """Invalid YAML in strict mode should raise exception."""
        invalid_yaml = "not: [valid: yaml"

        mock_client = MagicMock()
        mock_client.kv.get.return_value = (
            1,
            [
                {
                    "Key": f"{DEFAULT_CONTRACT_PREFIX}invalid.handler",
                    "Value": invalid_yaml.encode("utf-8"),
                },
            ],
        )

        with patch("consul.Consul", return_value=mock_client):
            source = RegistryContractSource(graceful_mode=False)

            with pytest.raises(Exception):  # yaml.YAMLError or similar
                await source.discover_handlers()

    @pytest.mark.asyncio
    async def test_parse_contract_missing_required_fields(self) -> None:
        """Contracts missing required fields should produce validation error.

        ModelHandlerContract requires certain fields (handler_id, name, version).
        Missing fields should be caught by Pydantic validation.
        """
        incomplete_yaml = """
name: "Incomplete Handler"
version: "1.0.0"
"""  # Missing handler_id, input_model, output_model

        mock_client = MagicMock()
        mock_client.kv.get.return_value = (
            1,
            [
                {
                    "Key": f"{DEFAULT_CONTRACT_PREFIX}incomplete.handler",
                    "Value": incomplete_yaml.encode("utf-8"),
                },
            ],
        )

        with patch("consul.Consul", return_value=mock_client):
            source = RegistryContractSource(graceful_mode=True)
            result = await source.discover_handlers()

        assert len(result.descriptors) == 0
        assert len(result.validation_errors) == 1

        error = result.validation_errors[0]
        assert error.error_type == EnumHandlerErrorType.CONTRACT_PARSE_ERROR

    @pytest.mark.asyncio
    async def test_handler_id_mismatch_strict_mode_raises_error(self) -> None:
        """handler_id mismatch in strict mode should raise ValueError.

        When graceful_mode=False (default) and the Consul key's handler_id
        doesn't match the contract's handler_id, a ValueError should be raised.
        """
        mock_client = MagicMock()
        mock_client.kv.get.return_value = (
            1,
            [
                {
                    # Key has different handler_id than contract content
                    "Key": f"{DEFAULT_CONTRACT_PREFIX}mismatched.handler",
                    "Value": VALID_HANDLER_CONTRACT_YAML.encode("utf-8"),
                },
            ],
        )

        with patch("consul.Consul", return_value=mock_client):
            source = RegistryContractSource(graceful_mode=False)

            with pytest.raises(ValueError, match="handler_id mismatch"):
                await source.discover_handlers()

    @pytest.mark.asyncio
    async def test_handler_id_mismatch_graceful_mode_logs_warning(self) -> None:
        """handler_id mismatch in graceful mode should log warning and continue.

        When graceful_mode=True and the Consul key's handler_id doesn't match
        the contract's handler_id, a warning should be logged but processing
        continues using the contract's handler_id as authoritative.
        """
        mock_client = MagicMock()
        mock_client.kv.get.return_value = (
            1,
            [
                {
                    # Key has different handler_id than contract content
                    "Key": f"{DEFAULT_CONTRACT_PREFIX}mismatched.handler",
                    "Value": VALID_HANDLER_CONTRACT_YAML.encode("utf-8"),
                },
            ],
        )

        with patch("consul.Consul", return_value=mock_client):
            source = RegistryContractSource(graceful_mode=True)
            result = await source.discover_handlers()

        # Should succeed in graceful mode, using contract's handler_id
        assert len(result.descriptors) == 1
        assert result.descriptors[0].handler_id == "effect.test.handler"
        assert len(result.validation_errors) == 0


# =============================================================================
# Configuration Tests
# =============================================================================


class TestRegistryContractSourceConfiguration:
    """Tests for configuration from constructor params and env vars."""

    def test_env_var_configuration_consul_host(self) -> None:
        """CONSUL_HOST env var should be used when host not provided."""
        with (
            patch.dict("os.environ", {"CONSUL_HOST": "consul.example.com"}),
            patch("consul.Consul") as mock_consul_class,
        ):
            source = RegistryContractSource()

            # The stored _host attribute should use env var
            assert source._host == "consul.example.com"

    def test_env_var_configuration_consul_port(self) -> None:
        """CONSUL_PORT env var should be used when port not provided."""
        with (
            patch.dict("os.environ", {"CONSUL_PORT": "28500"}),
            patch("consul.Consul") as mock_consul_class,
        ):
            source = RegistryContractSource()

            assert source._port == 28500

    def test_env_var_configuration_consul_token(self) -> None:
        """CONSUL_TOKEN env var should be used when token not provided."""
        with (
            patch.dict("os.environ", {"CONSUL_TOKEN": "secret-token"}),
            patch("consul.Consul") as mock_consul_class,
        ):
            source = RegistryContractSource()

            assert source._token == "secret-token"

    def test_env_var_configuration_consul_scheme(self) -> None:
        """CONSUL_SCHEME env var should be used when scheme not provided."""
        with (
            patch.dict("os.environ", {"CONSUL_SCHEME": "https"}),
            patch("consul.Consul") as mock_consul_class,
        ):
            source = RegistryContractSource()

            assert source._scheme == "https"

    def test_explicit_params_override_env_vars(self) -> None:
        """Explicit constructor params should override env vars."""
        with (
            patch.dict(
                "os.environ",
                {
                    "CONSUL_HOST": "env.example.com",
                    "CONSUL_PORT": "9999",
                    "CONSUL_TOKEN": "env-token",
                    "CONSUL_SCHEME": "https",
                },
            ),
            patch("consul.Consul") as mock_consul_class,
        ):
            source = RegistryContractSource(
                host="explicit.example.com",
                port=8500,
                token="explicit-token",  # noqa: S106  # Test value, not a real token
                scheme="http",
            )

            # Explicit params should be stored
            assert source._host == "explicit.example.com"
            assert source._port == 8500
            assert source._token == "explicit-token"
            assert source._scheme == "http"

    def test_default_values_when_no_env_vars(self) -> None:
        """Default values should be used when no env vars or params provided."""
        with (
            patch.dict("os.environ", {}, clear=True),
            patch("consul.Consul") as mock_consul_class,
        ):
            # Clear relevant env vars
            import os

            for var in ["CONSUL_HOST", "CONSUL_PORT", "CONSUL_TOKEN", "CONSUL_SCHEME"]:
                os.environ.pop(var, None)

            source = RegistryContractSource()

            assert source._host == "localhost"
            assert source._port == 8500
            assert source._token is None
            assert source._scheme == "http"

    def test_custom_prefix_configuration(self) -> None:
        """Custom prefix should be used for KV lookups."""
        custom_prefix = "my/custom/prefix/"

        mock_client = MagicMock()
        mock_client.kv.get.return_value = (0, None)

        with patch("consul.Consul", return_value=mock_client):
            source = RegistryContractSource(prefix=custom_prefix)

            assert source._prefix == custom_prefix

    @pytest.mark.asyncio
    async def test_custom_prefix_used_in_discovery(self) -> None:
        """Custom prefix should be passed to Consul KV get."""
        custom_prefix = "my/custom/prefix/"

        mock_client = MagicMock()
        mock_client.kv.get.return_value = (0, None)

        with patch("consul.Consul", return_value=mock_client):
            source = RegistryContractSource(prefix=custom_prefix)
            await source.discover_handlers()

        mock_client.kv.get.assert_called_once_with(custom_prefix, recurse=True)

    def test_correlation_id_generated(self) -> None:
        """A unique correlation_id should be generated on initialization."""
        with patch("consul.Consul"):
            source1 = RegistryContractSource()
            source2 = RegistryContractSource()

        assert source1._correlation_id is not None
        assert source2._correlation_id is not None
        assert isinstance(source1._correlation_id, UUID)
        assert isinstance(source2._correlation_id, UUID)
        # Each instance should have unique correlation ID
        assert source1._correlation_id != source2._correlation_id


# =============================================================================
# Contract Path Tests
# =============================================================================


class TestRegistryContractSourceContractPath:
    """Tests for contract_path generation in descriptors."""

    @pytest.mark.asyncio
    async def test_contract_path_format(self) -> None:
        """contract_path should use consul:// URI format.

        The contract_path should indicate the source location using
        consul://{host}:{port}/{key} format for traceability.
        """
        mock_client = MagicMock()
        mock_client.kv.get.return_value = (
            1,
            [
                {
                    # Key must match contract's handler_id (strict mode enforces this)
                    "Key": f"{DEFAULT_CONTRACT_PREFIX}effect.test.handler",
                    "Value": VALID_HANDLER_CONTRACT_YAML.encode("utf-8"),
                },
            ],
        )

        with patch("consul.Consul", return_value=mock_client):
            source = RegistryContractSource(host="consul.local", port=8500)
            result = await source.discover_handlers()

        assert len(result.descriptors) == 1
        contract_path = result.descriptors[0].contract_path
        assert contract_path.startswith("consul://")
        assert "consul.local" in contract_path
        assert "8500" in contract_path
        assert "effect.test.handler" in contract_path


# =============================================================================
# Utility Function Tests
# =============================================================================


class TestRegistryContractSourceUtilities:
    """Tests for utility functions in the module."""

    def test_store_contract_in_consul(self) -> None:
        """store_contract_in_consul should put contract in Consul KV."""
        from omnibase_infra.runtime.registry_contract_source import (
            store_contract_in_consul,
        )

        mock_client = MagicMock()
        mock_client.kv.put.return_value = True

        with patch(
            "omnibase_infra.runtime.registry_contract_source._create_consul_client_from_env",
            return_value=mock_client,
        ):
            result = store_contract_in_consul(
                contract_yaml=VALID_HANDLER_CONTRACT_YAML,
                handler_id="effect.test.handler",
            )

        assert result is True
        mock_client.kv.put.assert_called_once()
        call_args = mock_client.kv.put.call_args
        assert call_args[0][0] == f"{DEFAULT_CONTRACT_PREFIX}effect.test.handler"
        assert call_args[0][1] == VALID_HANDLER_CONTRACT_YAML

    def test_list_contracts_in_consul_empty(self) -> None:
        """list_contracts_in_consul should return empty list when no contracts."""
        from omnibase_infra.runtime.registry_contract_source import (
            list_contracts_in_consul,
        )

        mock_client = MagicMock()
        mock_client.kv.get.return_value = (0, None)

        with patch(
            "omnibase_infra.runtime.registry_contract_source._create_consul_client_from_env",
            return_value=mock_client,
        ):
            result = list_contracts_in_consul()

        assert result == []

    def test_list_contracts_in_consul_with_contracts(self) -> None:
        """list_contracts_in_consul should return handler IDs."""
        from omnibase_infra.runtime.registry_contract_source import (
            list_contracts_in_consul,
        )

        mock_client = MagicMock()
        mock_client.kv.get.return_value = (
            1,
            [
                f"{DEFAULT_CONTRACT_PREFIX}handler.one",
                f"{DEFAULT_CONTRACT_PREFIX}handler.two",
            ],
        )

        with patch(
            "omnibase_infra.runtime.registry_contract_source._create_consul_client_from_env",
            return_value=mock_client,
        ):
            result = list_contracts_in_consul()

        assert result == ["handler.one", "handler.two"]


# =============================================================================
# Default Exports Tests
# =============================================================================


class TestRegistryContractSourceExports:
    """Tests for module exports."""

    def test_default_contract_prefix_exported(self) -> None:
        """DEFAULT_CONTRACT_PREFIX should be exported."""
        assert DEFAULT_CONTRACT_PREFIX == "onex/contracts/handlers/"

    def test_max_contract_size_exported(self) -> None:
        """MAX_CONTRACT_SIZE should be exported (10MB)."""
        assert MAX_CONTRACT_SIZE == 10 * 1024 * 1024
