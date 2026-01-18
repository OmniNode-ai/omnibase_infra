# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Integration tests for SecretResolver <-> HandlerVault envelope format contract.

These tests verify the envelope format contract between SecretResolver and HandlerVault.
The contract defines:
    - Request envelope structure (operation, payload, correlation_id)
    - Payload field requirements (path, mount_point)
    - Response format (status, payload.data, payload.metadata)

These tests catch interface drift if either component changes its expected format.

Contract Locations:
    - SecretResolver creates envelope: secret_resolver.py lines 1550-1557
    - HandlerVault validates envelope: handler_vault.py execute() method
    - HandlerVault response format: mixin_vault_secrets.py _read_secret()

Test Strategy:
    - Use MockVaultHandler that validates envelope format and returns mock responses
    - Tests do NOT connect to real Vault (unit-level contract testing)
    - Tests verify both directions: SecretResolver -> Handler AND Handler -> SecretResolver
"""

from __future__ import annotations

import uuid
from uuid import UUID

import pytest

from omnibase_core.models.dispatch import ModelHandlerOutput
from omnibase_infra.runtime.models.model_secret_mapping import ModelSecretMapping
from omnibase_infra.runtime.models.model_secret_resolver_config import (
    ModelSecretResolverConfig,
)
from omnibase_infra.runtime.models.model_secret_source_spec import ModelSecretSourceSpec
from omnibase_infra.runtime.secret_resolver import SecretResolver

# =============================================================================
# Contract Constants - These define the expected envelope format
# =============================================================================

# Required envelope fields for vault.read_secret operation
ENVELOPE_REQUIRED_FIELDS = frozenset({"operation", "payload", "correlation_id"})

# Required payload fields for vault.read_secret operation
PAYLOAD_REQUIRED_FIELDS = frozenset({"path"})

# Optional payload fields with their defaults
PAYLOAD_OPTIONAL_FIELDS = {"mount_point": "secret"}

# Expected operation value for secret reading
EXPECTED_OPERATION = "vault.read_secret"

# Expected response structure from HandlerVault
RESPONSE_REQUIRED_FIELDS = frozenset({"status", "payload", "correlation_id"})


# =============================================================================
# Mock Vault Handler for Contract Testing
# =============================================================================


class MockVaultHandler:
    """Mock HandlerVault that validates envelope format and returns mock responses.

    This mock is used to verify the contract between SecretResolver and HandlerVault
    without requiring a real Vault connection.

    Contract Validation:
        - Validates all required envelope fields are present
        - Validates all required payload fields are present
        - Validates field types match expected types
        - Raises AssertionError on contract violations

    Response Generation:
        - Returns properly formatted responses matching HandlerVault's format
        - Allows injection of custom response data for testing
    """

    def __init__(
        self,
        secret_data: dict[str, str] | None = None,
        should_return_success: bool = True,
        should_return_empty_data: bool = False,
        should_return_malformed: bool = False,
        malformed_response: object = None,
    ) -> None:
        """Initialize mock handler.

        Args:
            secret_data: Secret data to return (default: {"value": "test-secret"})
            should_return_success: Whether to return success status
            should_return_empty_data: Whether to return empty data dict
            should_return_malformed: Whether to return malformed response
            malformed_response: Custom malformed response to return
        """
        self._secret_data = secret_data or {"value": "test-secret"}
        self._should_return_success = should_return_success
        self._should_return_empty_data = should_return_empty_data
        self._should_return_malformed = should_return_malformed
        self._malformed_response = malformed_response
        self._last_envelope: dict[str, object] | None = None
        self._call_count: int = 0

    @property
    def last_envelope(self) -> dict[str, object] | None:
        """Return the last envelope received for inspection."""
        return self._last_envelope

    @property
    def call_count(self) -> int:
        """Return the number of times execute was called."""
        return self._call_count

    async def execute(
        self, envelope: dict[str, object]
    ) -> ModelHandlerOutput[dict[str, object]]:
        """Execute vault operation and validate envelope format.

        This method validates the envelope matches the contract, then returns
        a mock response matching HandlerVault's actual response format.

        Args:
            envelope: Request envelope from SecretResolver

        Returns:
            ModelHandlerOutput with mock secret data

        Raises:
            ContractViolationError: If envelope format doesn't match contract
        """
        self._last_envelope = envelope
        self._call_count += 1

        # Contract validation: verify all required fields present
        self._validate_envelope_structure(envelope)

        # Contract validation: verify payload structure
        payload = envelope.get("payload")
        assert isinstance(payload, dict), "payload must be a dict"
        self._validate_payload_structure(payload)

        # Generate response
        return self._generate_response(envelope)

    def _validate_envelope_structure(self, envelope: dict[str, object]) -> None:
        """Validate envelope has all required fields with correct types.

        Contract Requirements:
            - operation: str, must be "vault.read_secret"
            - payload: dict, contains path and optional mount_point
            - correlation_id: str (UUID format)
        """
        # Check all required fields are present
        missing_fields = ENVELOPE_REQUIRED_FIELDS - set(envelope.keys())
        assert not missing_fields, f"Envelope missing required fields: {missing_fields}"

        # Validate operation field
        operation = envelope.get("operation")
        assert isinstance(operation, str), "operation must be a string"
        assert operation == EXPECTED_OPERATION, (
            f"operation must be '{EXPECTED_OPERATION}', got '{operation}'"
        )

        # Validate payload field type
        payload = envelope.get("payload")
        assert isinstance(payload, dict), "payload must be a dict"

        # Validate correlation_id field
        correlation_id = envelope.get("correlation_id")
        assert isinstance(correlation_id, str), (
            "correlation_id must be a string (UUID format)"
        )
        # Verify it's a valid UUID format
        try:
            UUID(correlation_id)
        except ValueError:
            raise AssertionError(
                f"correlation_id must be valid UUID format, got: {correlation_id}"
            )

    def _validate_payload_structure(self, payload: dict[str, object]) -> None:
        """Validate payload has all required fields with correct types.

        Contract Requirements:
            - path: str, required, non-empty
            - mount_point: str, optional, defaults to "secret"
        """
        # Check all required payload fields are present
        missing_fields = PAYLOAD_REQUIRED_FIELDS - set(payload.keys())
        assert not missing_fields, f"Payload missing required fields: {missing_fields}"

        # Validate path field
        path = payload.get("path")
        assert isinstance(path, str), "path must be a string"
        # Note: path can be empty in edge cases (see _parse_vault_path_components)

        # Validate mount_point if present
        mount_point = payload.get("mount_point")
        if mount_point is not None:
            assert isinstance(mount_point, str), "mount_point must be a string"

    def _generate_response(
        self, envelope: dict[str, object]
    ) -> ModelHandlerOutput[dict[str, object]]:
        """Generate mock response matching HandlerVault's format."""
        correlation_id_str = envelope.get("correlation_id", str(uuid.uuid4()))

        # Handle malformed response case
        if self._should_return_malformed:
            return ModelHandlerOutput.for_compute(
                input_envelope_id=uuid.uuid4(),
                correlation_id=UUID(str(correlation_id_str)),
                handler_id="vault-handler",
                result=self._malformed_response,
            )

        # Build response matching actual HandlerVault format
        if self._should_return_success:
            status = "success"
            if self._should_return_empty_data:
                payload_data: dict[str, object] = {
                    "data": {},
                    "metadata": {},
                }
            else:
                payload_data = {
                    "data": self._secret_data,
                    "metadata": {
                        "created_time": "2025-01-18T00:00:00.000000Z",
                        "version": 1,
                    },
                }
        else:
            status = "error"
            payload_data = {"error": "Secret not found"}

        result: dict[str, object] = {
            "status": status,
            "payload": payload_data,
            "correlation_id": str(correlation_id_str),
        }

        return ModelHandlerOutput.for_compute(
            input_envelope_id=uuid.uuid4(),
            correlation_id=UUID(str(correlation_id_str)),
            handler_id="vault-handler",
            result=result,
        )


# =============================================================================
# Contract Test Fixtures
# =============================================================================


@pytest.fixture
def mock_vault_handler() -> MockVaultHandler:
    """Provide a mock vault handler for contract testing."""
    return MockVaultHandler(
        secret_data={
            "password": "test-secret-value",
            "username": "test-user",
            "api_key": "test-api-key-value",
        }
    )


@pytest.fixture
def secret_resolver_with_mock(
    mock_vault_handler: MockVaultHandler,
) -> SecretResolver:
    """Provide a SecretResolver configured with mock vault handler."""
    config = ModelSecretResolverConfig(
        mappings=[
            ModelSecretMapping(
                logical_name="test.secret",
                source=ModelSecretSourceSpec(
                    source_type="vault",
                    source_path="secret/myapp/db#password",
                ),
            ),
            ModelSecretMapping(
                logical_name="test.no.field",
                source=ModelSecretSourceSpec(
                    source_type="vault",
                    source_path="secret/myapp/db",
                ),
            ),
            ModelSecretMapping(
                logical_name="test.custom.mount",
                source=ModelSecretSourceSpec(
                    source_type="vault",
                    source_path="kv/myapp/config#api_key",
                ),
            ),
        ],
        enable_convention_fallback=False,
    )
    return SecretResolver(config=config, vault_handler=mock_vault_handler)


# =============================================================================
# Envelope Structure Contract Tests
# =============================================================================


class TestSecretResolverEnvelopeStructure:
    """Tests verifying SecretResolver creates correctly structured envelopes."""

    @pytest.mark.asyncio
    async def test_envelope_contains_required_operation_field(
        self,
        mock_vault_handler: MockVaultHandler,
        secret_resolver_with_mock: SecretResolver,
    ) -> None:
        """Envelope must contain operation field with value 'vault.read_secret'."""
        await secret_resolver_with_mock.get_secret_async("test.secret")

        envelope = mock_vault_handler.last_envelope
        assert envelope is not None
        assert "operation" in envelope
        assert envelope["operation"] == "vault.read_secret"

    @pytest.mark.asyncio
    async def test_envelope_contains_required_payload_field(
        self,
        mock_vault_handler: MockVaultHandler,
        secret_resolver_with_mock: SecretResolver,
    ) -> None:
        """Envelope must contain payload field as a dict."""
        await secret_resolver_with_mock.get_secret_async("test.secret")

        envelope = mock_vault_handler.last_envelope
        assert envelope is not None
        assert "payload" in envelope
        assert isinstance(envelope["payload"], dict)

    @pytest.mark.asyncio
    async def test_envelope_contains_required_correlation_id_field(
        self,
        mock_vault_handler: MockVaultHandler,
        secret_resolver_with_mock: SecretResolver,
    ) -> None:
        """Envelope must contain correlation_id field as UUID string."""
        await secret_resolver_with_mock.get_secret_async("test.secret")

        envelope = mock_vault_handler.last_envelope
        assert envelope is not None
        assert "correlation_id" in envelope
        assert isinstance(envelope["correlation_id"], str)

        # Verify it's a valid UUID
        correlation_id = envelope["correlation_id"]
        UUID(correlation_id)  # Raises ValueError if invalid

    @pytest.mark.asyncio
    async def test_envelope_correlation_id_propagates_from_caller(
        self,
        mock_vault_handler: MockVaultHandler,
        secret_resolver_with_mock: SecretResolver,
    ) -> None:
        """Correlation ID from caller should be propagated to envelope."""
        test_correlation_id = uuid.uuid4()
        await secret_resolver_with_mock.get_secret_async(
            "test.secret", correlation_id=test_correlation_id
        )

        envelope = mock_vault_handler.last_envelope
        assert envelope is not None
        assert envelope["correlation_id"] == str(test_correlation_id)


# =============================================================================
# Payload Structure Contract Tests
# =============================================================================


class TestSecretResolverPayloadStructure:
    """Tests verifying SecretResolver creates correctly structured payloads."""

    @pytest.mark.asyncio
    async def test_payload_contains_required_path_field(
        self,
        mock_vault_handler: MockVaultHandler,
        secret_resolver_with_mock: SecretResolver,
    ) -> None:
        """Payload must contain path field as string."""
        await secret_resolver_with_mock.get_secret_async("test.secret")

        envelope = mock_vault_handler.last_envelope
        assert envelope is not None
        payload = envelope.get("payload")
        assert isinstance(payload, dict)
        assert "path" in payload
        assert isinstance(payload["path"], str)

    @pytest.mark.asyncio
    async def test_payload_contains_mount_point_field(
        self,
        mock_vault_handler: MockVaultHandler,
        secret_resolver_with_mock: SecretResolver,
    ) -> None:
        """Payload must contain mount_point field as string."""
        await secret_resolver_with_mock.get_secret_async("test.secret")

        envelope = mock_vault_handler.last_envelope
        assert envelope is not None
        payload = envelope.get("payload")
        assert isinstance(payload, dict)
        assert "mount_point" in payload
        assert isinstance(payload["mount_point"], str)

    @pytest.mark.asyncio
    async def test_payload_path_excludes_mount_point(
        self,
        mock_vault_handler: MockVaultHandler,
        secret_resolver_with_mock: SecretResolver,
    ) -> None:
        """Payload path should be the path without mount_point prefix.

        Given source_path "secret/myapp/db#password":
            - mount_point should be "secret"
            - path should be "myapp/db" (NOT "secret/myapp/db")
        """
        await secret_resolver_with_mock.get_secret_async("test.secret")

        envelope = mock_vault_handler.last_envelope
        assert envelope is not None
        payload = envelope.get("payload")
        assert isinstance(payload, dict)

        # Path should be without mount_point prefix
        assert payload["path"] == "myapp/db"
        assert payload["mount_point"] == "secret"

    @pytest.mark.asyncio
    async def test_payload_path_excludes_field_specifier(
        self,
        mock_vault_handler: MockVaultHandler,
        secret_resolver_with_mock: SecretResolver,
    ) -> None:
        """Payload path should not contain field specifier (#field).

        Given source_path "secret/myapp/db#password":
            - path should be "myapp/db" (NOT "myapp/db#password")
        """
        await secret_resolver_with_mock.get_secret_async("test.secret")

        envelope = mock_vault_handler.last_envelope
        assert envelope is not None
        payload = envelope.get("payload")
        assert isinstance(payload, dict)

        # Path should not contain field specifier
        assert "#" not in str(payload["path"])

    @pytest.mark.asyncio
    async def test_payload_custom_mount_point(
        self,
        mock_vault_handler: MockVaultHandler,
        secret_resolver_with_mock: SecretResolver,
    ) -> None:
        """Payload should correctly parse custom mount points.

        Given source_path "kv/myapp/config#api_key":
            - mount_point should be "kv"
            - path should be "myapp/config"
        """
        await secret_resolver_with_mock.get_secret_async("test.custom.mount")

        envelope = mock_vault_handler.last_envelope
        assert envelope is not None
        payload = envelope.get("payload")
        assert isinstance(payload, dict)

        assert payload["mount_point"] == "kv"
        assert payload["path"] == "myapp/config"


# =============================================================================
# Response Parsing Contract Tests
# =============================================================================


class TestSecretResolverResponseParsing:
    """Tests verifying SecretResolver correctly parses HandlerVault responses."""

    @pytest.mark.asyncio
    async def test_parses_success_response_with_data(self) -> None:
        """SecretResolver should extract secret value from success response."""
        mock_handler = MockVaultHandler(
            secret_data={"password": "super-secret-password"}
        )
        config = ModelSecretResolverConfig(
            mappings=[
                ModelSecretMapping(
                    logical_name="test.password",
                    source=ModelSecretSourceSpec(
                        source_type="vault",
                        source_path="secret/db#password",
                    ),
                ),
            ],
            enable_convention_fallback=False,
        )
        resolver = SecretResolver(config=config, vault_handler=mock_handler)

        result = await resolver.get_secret_async("test.password")

        assert result is not None
        assert result.get_secret_value() == "super-secret-password"

    @pytest.mark.asyncio
    async def test_parses_response_first_value_when_no_field(self) -> None:
        """SecretResolver should return first value when no field specified."""
        mock_handler = MockVaultHandler(
            secret_data={"first_key": "first-value", "second_key": "second-value"}
        )
        config = ModelSecretResolverConfig(
            mappings=[
                ModelSecretMapping(
                    logical_name="test.any",
                    source=ModelSecretSourceSpec(
                        source_type="vault",
                        source_path="secret/db",  # No #field
                    ),
                ),
            ],
            enable_convention_fallback=False,
        )
        resolver = SecretResolver(config=config, vault_handler=mock_handler)

        result = await resolver.get_secret_async("test.any")

        assert result is not None
        # Should get one of the values (first in dict iteration order)
        assert result.get_secret_value() in ["first-value", "second-value"]

    @pytest.mark.asyncio
    async def test_returns_none_for_non_success_status(self) -> None:
        """SecretResolver should return None when status is not 'success'."""
        mock_handler = MockVaultHandler(should_return_success=False)
        config = ModelSecretResolverConfig(
            mappings=[
                ModelSecretMapping(
                    logical_name="test.error",
                    source=ModelSecretSourceSpec(
                        source_type="vault",
                        source_path="secret/db#password",
                    ),
                ),
            ],
            enable_convention_fallback=False,
        )
        resolver = SecretResolver(config=config, vault_handler=mock_handler)

        result = await resolver.get_secret_async("test.error", required=False)

        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_for_empty_secret_data(self) -> None:
        """SecretResolver should return None when secret data is empty."""
        mock_handler = MockVaultHandler(should_return_empty_data=True)
        config = ModelSecretResolverConfig(
            mappings=[
                ModelSecretMapping(
                    logical_name="test.empty",
                    source=ModelSecretSourceSpec(
                        source_type="vault",
                        source_path="secret/db#password",
                    ),
                ),
            ],
            enable_convention_fallback=False,
        )
        resolver = SecretResolver(config=config, vault_handler=mock_handler)

        result = await resolver.get_secret_async("test.empty", required=False)

        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_for_missing_field_in_data(self) -> None:
        """SecretResolver should return None when requested field doesn't exist."""
        mock_handler = MockVaultHandler(
            secret_data={"other_field": "some-value"}  # No "password" field
        )
        config = ModelSecretResolverConfig(
            mappings=[
                ModelSecretMapping(
                    logical_name="test.missing",
                    source=ModelSecretSourceSpec(
                        source_type="vault",
                        source_path="secret/db#password",  # Requests "password"
                    ),
                ),
            ],
            enable_convention_fallback=False,
        )
        resolver = SecretResolver(config=config, vault_handler=mock_handler)

        result = await resolver.get_secret_async("test.missing", required=False)

        assert result is None


# =============================================================================
# Malformed Response Handling Tests
# =============================================================================


class TestSecretResolverMalformedResponseHandling:
    """Tests verifying SecretResolver gracefully handles malformed responses."""

    @pytest.mark.asyncio
    async def test_handles_non_dict_result(self) -> None:
        """SecretResolver should return None when result is not a dict."""
        mock_handler = MockVaultHandler(
            should_return_malformed=True,
            malformed_response="not a dict",
        )
        config = ModelSecretResolverConfig(
            mappings=[
                ModelSecretMapping(
                    logical_name="test.malformed",
                    source=ModelSecretSourceSpec(
                        source_type="vault",
                        source_path="secret/db#password",
                    ),
                ),
            ],
            enable_convention_fallback=False,
        )
        resolver = SecretResolver(config=config, vault_handler=mock_handler)

        result = await resolver.get_secret_async("test.malformed", required=False)

        assert result is None

    @pytest.mark.asyncio
    async def test_handles_missing_status_field(self) -> None:
        """SecretResolver should return None when status field is missing."""
        mock_handler = MockVaultHandler(
            should_return_malformed=True,
            malformed_response={
                # Missing "status" field
                "payload": {"data": {"password": "value"}},
                "correlation_id": str(uuid.uuid4()),
            },
        )
        config = ModelSecretResolverConfig(
            mappings=[
                ModelSecretMapping(
                    logical_name="test.no.status",
                    source=ModelSecretSourceSpec(
                        source_type="vault",
                        source_path="secret/db#password",
                    ),
                ),
            ],
            enable_convention_fallback=False,
        )
        resolver = SecretResolver(config=config, vault_handler=mock_handler)

        result = await resolver.get_secret_async("test.no.status", required=False)

        assert result is None

    @pytest.mark.asyncio
    async def test_handles_missing_payload_field(self) -> None:
        """SecretResolver should return None when payload field is missing."""
        mock_handler = MockVaultHandler(
            should_return_malformed=True,
            malformed_response={
                "status": "success",
                # Missing "payload" field
                "correlation_id": str(uuid.uuid4()),
            },
        )
        config = ModelSecretResolverConfig(
            mappings=[
                ModelSecretMapping(
                    logical_name="test.no.payload",
                    source=ModelSecretSourceSpec(
                        source_type="vault",
                        source_path="secret/db#password",
                    ),
                ),
            ],
            enable_convention_fallback=False,
        )
        resolver = SecretResolver(config=config, vault_handler=mock_handler)

        result = await resolver.get_secret_async("test.no.payload", required=False)

        assert result is None

    @pytest.mark.asyncio
    async def test_handles_non_dict_payload(self) -> None:
        """SecretResolver should return None when payload is not a dict."""
        mock_handler = MockVaultHandler(
            should_return_malformed=True,
            malformed_response={
                "status": "success",
                "payload": "not a dict",
                "correlation_id": str(uuid.uuid4()),
            },
        )
        config = ModelSecretResolverConfig(
            mappings=[
                ModelSecretMapping(
                    logical_name="test.bad.payload",
                    source=ModelSecretSourceSpec(
                        source_type="vault",
                        source_path="secret/db#password",
                    ),
                ),
            ],
            enable_convention_fallback=False,
        )
        resolver = SecretResolver(config=config, vault_handler=mock_handler)

        result = await resolver.get_secret_async("test.bad.payload", required=False)

        assert result is None

    @pytest.mark.asyncio
    async def test_handles_missing_data_in_payload(self) -> None:
        """SecretResolver should return None when data is missing from payload."""
        mock_handler = MockVaultHandler(
            should_return_malformed=True,
            malformed_response={
                "status": "success",
                "payload": {
                    # Missing "data" field
                    "metadata": {},
                },
                "correlation_id": str(uuid.uuid4()),
            },
        )
        config = ModelSecretResolverConfig(
            mappings=[
                ModelSecretMapping(
                    logical_name="test.no.data",
                    source=ModelSecretSourceSpec(
                        source_type="vault",
                        source_path="secret/db#password",
                    ),
                ),
            ],
            enable_convention_fallback=False,
        )
        resolver = SecretResolver(config=config, vault_handler=mock_handler)

        result = await resolver.get_secret_async("test.no.data", required=False)

        assert result is None


# =============================================================================
# Contract Stability Tests
# =============================================================================


class TestContractStability:
    """Tests ensuring the contract is stable and well-defined."""

    def test_envelope_required_fields_documented(self) -> None:
        """Verify the documented envelope required fields."""
        expected = {"operation", "payload", "correlation_id"}
        assert expected == ENVELOPE_REQUIRED_FIELDS

    def test_payload_required_fields_documented(self) -> None:
        """Verify the documented payload required fields."""
        expected = {"path"}
        assert expected == PAYLOAD_REQUIRED_FIELDS

    def test_expected_operation_documented(self) -> None:
        """Verify the documented operation value."""
        assert EXPECTED_OPERATION == "vault.read_secret"

    @pytest.mark.asyncio
    async def test_multiple_calls_generate_unique_correlation_ids(self) -> None:
        """Each call without explicit correlation_id should generate unique ID."""
        # Create mock handler with password field
        mock_handler = MockVaultHandler(secret_data={"password": "secret1"})
        config = ModelSecretResolverConfig(
            mappings=[
                ModelSecretMapping(
                    logical_name="test.secret1",
                    source=ModelSecretSourceSpec(
                        source_type="vault",
                        source_path="secret/myapp/db1#password",
                    ),
                ),
                ModelSecretMapping(
                    logical_name="test.secret2",
                    source=ModelSecretSourceSpec(
                        source_type="vault",
                        source_path="secret/myapp/db2#password",
                    ),
                ),
            ],
            enable_convention_fallback=False,
            default_ttl_vault_seconds=0,  # Disable cache
        )
        resolver = SecretResolver(config=config, vault_handler=mock_handler)

        # First call
        await resolver.get_secret_async("test.secret1")
        first_correlation_id = mock_handler.last_envelope.get("correlation_id")

        # Second call (different logical name to avoid cache)
        await resolver.get_secret_async("test.secret2")
        second_correlation_id = mock_handler.last_envelope.get("correlation_id")

        # Both should be valid UUIDs but different
        assert first_correlation_id != second_correlation_id
        UUID(str(first_correlation_id))
        UUID(str(second_correlation_id))


__all__: list[str] = [
    "TestSecretResolverEnvelopeStructure",
    "TestSecretResolverPayloadStructure",
    "TestSecretResolverResponseParsing",
    "TestSecretResolverMalformedResponseHandling",
    "TestContractStability",
    "MockVaultHandler",
]
