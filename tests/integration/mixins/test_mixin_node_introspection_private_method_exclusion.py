# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Integration tests for MixinNodeIntrospection private method exclusion.

This module validates a critical security requirement of the ONEX introspection system:

**Private Method Exclusion**: Methods prefixed with `_` (single underscore) MUST be
excluded from introspection results to prevent exposure of internal implementation
details.

Security Context
----------------
The MixinNodeIntrospection uses Python reflection to discover node capabilities for
service discovery. This has security implications as documented in CLAUDE.md:

- **Threat**: Introspection data could reveal attack vectors through method names
- **Mitigation**: Private methods (`_` prefix) are excluded from discovery
- **This Test**: Validates the mitigation is correctly implemented

What Gets Tested
----------------
1. **Private method exclusion from method_signatures**: Methods starting with `_`
   should not appear in the `method_signatures` dict returned by `get_capabilities()`

2. **Private method exclusion from operations**: Private methods should not be
   classified as operations even if they contain operation keywords

3. **Public method inclusion**: Non-private methods should be properly discovered

4. **Edge cases**: Double-underscore methods, methods with underscore in name
   (but not as prefix), etc.

Related Documentation:
    - CLAUDE.md: "Node Introspection Security Considerations" section
    - mixin_node_introspection.py: Module docstring security documentation

Test Strategy:
    These tests create mock nodes with explicit public and private methods,
    then verify the introspection output correctly excludes private methods
    while including public ones.
"""

from __future__ import annotations

from uuid import UUID

import pytest

from omnibase_core.enums import EnumNodeKind
from omnibase_infra.mixins import MixinNodeIntrospection
from omnibase_infra.models.discovery import ModelIntrospectionConfig

# Module-level markers
pytestmark = [
    pytest.mark.asyncio,
    pytest.mark.integration,
]

# Test UUIDs - use deterministic values for reproducible tests
TEST_NODE_UUID_1 = UUID("00000000-0000-0000-0000-000000000001")
TEST_NODE_UUID_2 = UUID("00000000-0000-0000-0000-000000000002")
TEST_NODE_UUID_3 = UUID("00000000-0000-0000-0000-000000000003")


# =============================================================================
# Mock Nodes for Testing
# =============================================================================


class NodeWithPublicAndPrivateMethods(MixinNodeIntrospection):
    """Mock node with explicit public and private methods for testing.

    This node has a mix of:
    - Public methods (no underscore prefix)
    - Private methods (single underscore prefix)
    - Dunder methods (double underscore prefix)
    - Methods with underscores in the middle of the name
    """

    def __init__(self, node_id: UUID) -> None:
        """Initialize node with introspection.

        Args:
            node_id: Unique identifier for this node.
        """
        self._state = "initialized"
        self._internal_counter = 0

        config = ModelIntrospectionConfig(
            node_id=node_id,
            node_type=EnumNodeKind.EFFECT,
            node_name="private_method_test_node",
        )
        self.initialize_introspection(config)

    # Public methods - SHOULD be discovered
    async def execute_query(self, query: str) -> dict[str, str]:
        """Public execute method - should be discovered as operation.

        Args:
            query: Query to execute.

        Returns:
            Query result.
        """
        return {"result": query}

    async def handle_request(self, data: dict[str, str]) -> dict[str, bool]:
        """Public handle method - should be discovered as operation.

        Args:
            data: Request data.

        Returns:
            Processing result.
        """
        return {"handled": True}

    async def process_data(self, payload: bytes) -> bytes:
        """Public process method - should be discovered as operation.

        Args:
            payload: Data to process.

        Returns:
            Processed data.
        """
        return payload

    def validate_input(self, data: dict[str, str]) -> bool:
        """Public validation method - should be discovered but NOT as operation.

        Args:
            data: Data to validate.

        Returns:
            Validation result.
        """
        return True

    def method_with_underscore_in_name(self, value: str) -> str:
        """Public method with underscore in middle - should be discovered.

        Args:
            value: Input value.

        Returns:
            Processed value.
        """
        return value

    # Private methods - MUST NOT be discovered
    def _internal_helper(self, data: dict[str, str]) -> dict[str, str]:
        """Private helper method - MUST NOT be discovered.

        Args:
            data: Internal data.

        Returns:
            Processed internal data.
        """
        return data

    async def _execute_internal(self, query: str) -> str:
        """Private execute method - MUST NOT be discovered despite 'execute' keyword.

        Args:
            query: Internal query.

        Returns:
            Internal result.
        """
        return query

    def _handle_internal_event(self, event: dict[str, str]) -> None:
        """Private handle method - MUST NOT be discovered despite 'handle' keyword.

        Args:
            event: Internal event.
        """

    def _process_sensitive_data(self, data: bytes) -> bytes:
        """Private process method - MUST NOT be discovered despite 'process' keyword.

        Args:
            data: Sensitive data.

        Returns:
            Processed sensitive data.
        """
        return data

    def _validate_credentials(self, creds: dict[str, str]) -> bool:
        """Private validation with sensitive name - MUST NOT be discovered.

        Args:
            creds: Credentials to validate.

        Returns:
            Validation result.
        """
        return True

    def _decrypt_payload(self, encrypted: bytes) -> bytes:
        """Private decryption method - MUST NOT be discovered.

        Args:
            encrypted: Encrypted data.

        Returns:
            Decrypted data.
        """
        return encrypted


class NodeWithManyPrivateMethods(MixinNodeIntrospection):
    """Mock node with many private methods to test comprehensive exclusion.

    This node simulates a realistic scenario where internal implementation
    has many private helpers that must remain hidden.
    """

    def __init__(self, node_id: UUID) -> None:
        """Initialize node with introspection.

        Args:
            node_id: Unique identifier for this node.
        """
        self._connection = None
        self._cache: dict[str, str] = {}

        config = ModelIntrospectionConfig(
            node_id=node_id,
            node_type=EnumNodeKind.COMPUTE,
            node_name="minimal_method_node",
        )
        self.initialize_introspection(config)

    # Single public method
    async def execute(self, operation: str) -> dict[str, str]:
        """The only public operation method.

        Args:
            operation: Operation to execute.

        Returns:
            Operation result.
        """
        return {"status": "ok", "operation": operation}

    # Many private methods - ALL must be excluded
    def _connect_to_database(self) -> None:
        """Private database connection."""

    def _read_api_key(self) -> str:
        """Private API key reader - sensitive."""
        return "secret"

    def _fetch_credentials(self) -> dict[str, str]:
        """Private credential fetcher - sensitive."""
        return {}

    def _build_auth_header(self) -> str:
        """Private auth header builder - sensitive."""
        return "Bearer token"

    def _parse_response(self, data: bytes) -> dict[str, str]:
        """Private response parser."""
        return {}

    def _serialize_request(self, data: dict[str, str]) -> bytes:
        """Private request serializer."""
        return b""

    def _log_internal_state(self) -> None:
        """Private logging method."""

    def _cleanup_resources(self) -> None:
        """Private cleanup method."""


class NodeWithOnlyPrivateMethods(MixinNodeIntrospection):
    """Mock node with only private methods (no public operations).

    This edge case tests that introspection handles nodes where
    all implementation is private.
    """

    def __init__(self, node_id: UUID) -> None:
        """Initialize node with introspection.

        Args:
            node_id: Unique identifier for this node.
        """
        config = ModelIntrospectionConfig(
            node_id=node_id,
            node_type=EnumNodeKind.REDUCER,
            node_name="all_private_node",
        )
        self.initialize_introspection(config)

    def _private_method_1(self) -> None:
        """Private method 1."""

    def _private_method_2(self) -> None:
        """Private method 2."""

    def _execute_private(self) -> None:
        """Private execute - has keyword but should not be discovered."""


# =============================================================================
# Private Method Exclusion Tests
# =============================================================================


class TestPrivateMethodExclusionFromMethodSignatures:
    """Test that private methods are excluded from method_signatures."""

    async def test_private_methods_not_in_method_signatures(self) -> None:
        """Verify private methods do not appear in method_signatures dict."""
        node = NodeWithPublicAndPrivateMethods(node_id=TEST_NODE_UUID_1)
        capabilities = await node.get_capabilities()

        # Verify no method in signatures starts with underscore
        for method_name in capabilities.method_signatures:
            assert not method_name.startswith("_"), (
                f"Private method '{method_name}' should not be in method_signatures. "
                f"This is a security violation - private methods must be excluded."
            )

    async def test_specific_private_methods_excluded(self) -> None:
        """Verify specific known private methods are not discovered."""
        node = NodeWithPublicAndPrivateMethods(node_id=TEST_NODE_UUID_1)
        capabilities = await node.get_capabilities()

        # List of private methods that MUST NOT be discovered
        private_methods = [
            "_internal_helper",
            "_execute_internal",
            "_handle_internal_event",
            "_process_sensitive_data",
            "_validate_credentials",
            "_decrypt_payload",
        ]

        for private_method in private_methods:
            assert private_method not in capabilities.method_signatures, (
                f"Private method '{private_method}' was discovered in method_signatures. "
                f"This is a security violation."
            )

    async def test_public_methods_are_in_method_signatures(self) -> None:
        """Verify public methods ARE discovered in method_signatures."""
        node = NodeWithPublicAndPrivateMethods(node_id=TEST_NODE_UUID_1)
        capabilities = await node.get_capabilities()

        # Public methods that SHOULD be discovered
        # Note: Some may be filtered by exclude_prefixes (get_, set_, etc.)
        # but method_with_underscore_in_name should definitely be there
        assert "method_with_underscore_in_name" in capabilities.method_signatures, (
            "Public method 'method_with_underscore_in_name' should be discovered. "
            "Underscores in the middle of method names are allowed."
        )

    async def test_method_with_underscore_in_middle_is_discovered(self) -> None:
        """Verify methods with underscore in middle (not prefix) are discovered."""
        node = NodeWithPublicAndPrivateMethods(node_id=TEST_NODE_UUID_1)
        capabilities = await node.get_capabilities()

        # This is a public method - underscore is in the middle, not prefix
        assert "method_with_underscore_in_name" in capabilities.method_signatures


class TestPrivateMethodExclusionFromOperations:
    """Test that private methods are excluded from operations list."""

    async def test_private_methods_not_in_operations(self) -> None:
        """Verify private methods do not appear in operations tuple."""
        node = NodeWithPublicAndPrivateMethods(node_id=TEST_NODE_UUID_1)
        capabilities = await node.get_capabilities()

        # Verify no operation starts with underscore
        for operation in capabilities.operations:
            assert not operation.startswith("_"), (
                f"Private method '{operation}' should not be in operations. "
                f"This is a security violation - private methods must be excluded."
            )

    async def test_private_execute_method_not_in_operations(self) -> None:
        """Verify _execute_internal is not discovered despite 'execute' keyword."""
        node = NodeWithPublicAndPrivateMethods(node_id=TEST_NODE_UUID_1)
        capabilities = await node.get_capabilities()

        # _execute_internal has the 'execute' keyword but is private
        assert "_execute_internal" not in capabilities.operations, (
            "Private method '_execute_internal' was discovered as an operation. "
            "Private methods must be excluded even if they contain operation keywords."
        )

    async def test_private_handle_method_not_in_operations(self) -> None:
        """Verify _handle_internal_event is not discovered despite 'handle' keyword."""
        node = NodeWithPublicAndPrivateMethods(node_id=TEST_NODE_UUID_1)
        capabilities = await node.get_capabilities()

        assert "_handle_internal_event" not in capabilities.operations, (
            "Private method '_handle_internal_event' was discovered as an operation. "
            "Private methods must be excluded even if they contain operation keywords."
        )

    async def test_private_process_method_not_in_operations(self) -> None:
        """Verify _process_sensitive_data is not discovered despite 'process' keyword."""
        node = NodeWithPublicAndPrivateMethods(node_id=TEST_NODE_UUID_1)
        capabilities = await node.get_capabilities()

        assert "_process_sensitive_data" not in capabilities.operations, (
            "Private method '_process_sensitive_data' was discovered as an operation. "
            "Private methods must be excluded even if they contain operation keywords."
        )

    async def test_public_operations_are_discovered(self) -> None:
        """Verify public operation methods ARE discovered."""
        node = NodeWithPublicAndPrivateMethods(node_id=TEST_NODE_UUID_1)
        capabilities = await node.get_capabilities()

        # Public operation methods that SHOULD be discovered
        expected_operations = [
            "execute_query",
            "handle_request",
            "process_data",
        ]

        for expected_op in expected_operations:
            assert expected_op in capabilities.operations, (
                f"Public operation '{expected_op}' should be discovered in operations."
            )


class TestPrivateMethodExclusionInIntrospectionData:
    """Test that private methods are excluded from full introspection data."""

    async def test_introspection_data_excludes_private_methods(self) -> None:
        """Verify get_introspection_data() excludes private methods."""
        node = NodeWithPublicAndPrivateMethods(node_id=TEST_NODE_UUID_1)
        introspection_data = await node.get_introspection_data()

        # Check discovered_capabilities
        for method_name in introspection_data.discovered_capabilities.method_signatures:
            assert not method_name.startswith("_"), (
                f"Private method '{method_name}' found in introspection data. "
                f"This is a security violation."
            )

        for operation in introspection_data.discovered_capabilities.operations:
            assert not operation.startswith("_"), (
                f"Private operation '{operation}' found in introspection data. "
                f"This is a security violation."
            )

    async def test_introspection_data_includes_public_methods(self) -> None:
        """Verify get_introspection_data() includes public methods."""
        node = NodeWithPublicAndPrivateMethods(node_id=TEST_NODE_UUID_1)
        introspection_data = await node.get_introspection_data()

        # Should have at least some public operations
        assert len(introspection_data.discovered_capabilities.operations) > 0, (
            "Introspection should discover at least one public operation."
        )

        # Verify specific public operations
        assert "execute_query" in introspection_data.discovered_capabilities.operations


class TestManyPrivateMethodsExclusion:
    """Test comprehensive exclusion with many private methods."""

    async def test_all_private_methods_excluded(self) -> None:
        """Verify all private methods are excluded when node has many."""
        node = NodeWithManyPrivateMethods(node_id=TEST_NODE_UUID_2)
        capabilities = await node.get_capabilities()

        # ALL methods in signatures should be public
        for method_name in capabilities.method_signatures:
            assert not method_name.startswith("_"), (
                f"Private method '{method_name}' leaked into method_signatures."
            )

        # ALL operations should be public
        for operation in capabilities.operations:
            assert not operation.startswith("_"), (
                f"Private method '{operation}' leaked into operations."
            )

    async def test_only_public_method_discovered(self) -> None:
        """Verify only the public 'execute' method is discovered as operation."""
        node = NodeWithManyPrivateMethods(node_id=TEST_NODE_UUID_2)
        capabilities = await node.get_capabilities()

        # Should have exactly 'execute' as an operation
        assert "execute" in capabilities.operations, (
            "Public method 'execute' should be discovered."
        )


class TestOnlyPrivateMethodsNode:
    """Test edge case: node with only private methods."""

    async def test_no_operations_when_all_private(self) -> None:
        """Verify no operations are discovered when all methods are private."""
        node = NodeWithOnlyPrivateMethods(node_id=TEST_NODE_UUID_3)
        capabilities = await node.get_capabilities()

        # No private methods should leak
        for operation in capabilities.operations:
            assert not operation.startswith("_"), (
                f"Private method '{operation}' leaked into operations."
            )

    async def test_empty_or_inherited_only_method_signatures(self) -> None:
        """Verify method_signatures has no private methods (may have inherited)."""
        node = NodeWithOnlyPrivateMethods(node_id=TEST_NODE_UUID_3)
        capabilities = await node.get_capabilities()

        # No private methods in signatures
        for method_name in capabilities.method_signatures:
            assert not method_name.startswith("_"), (
                f"Private method '{method_name}' leaked into method_signatures."
            )


class TestPrivateMethodExclusionWithCaching:
    """Test that caching doesn't affect private method exclusion."""

    async def test_cache_hit_still_excludes_private_methods(self) -> None:
        """Verify private methods are excluded even on cache hit."""
        node = NodeWithPublicAndPrivateMethods(node_id=TEST_NODE_UUID_1)

        # First call - cache miss
        data1 = await node.get_introspection_data()

        # Verify no private methods
        for method_name in data1.discovered_capabilities.method_signatures:
            assert not method_name.startswith("_")

        # Second call - cache hit
        data2 = await node.get_introspection_data()

        # Verify still no private methods
        for method_name in data2.discovered_capabilities.method_signatures:
            assert not method_name.startswith("_"), (
                f"Private method '{method_name}' leaked on cache hit."
            )

    async def test_cache_invalidation_still_excludes_private_methods(self) -> None:
        """Verify private methods are excluded after cache invalidation."""
        node = NodeWithPublicAndPrivateMethods(node_id=TEST_NODE_UUID_1)

        # Populate cache
        await node.get_introspection_data()

        # Invalidate cache
        node.invalidate_introspection_cache()

        # Fresh call after invalidation
        data = await node.get_introspection_data()

        # Verify no private methods
        for method_name in data.discovered_capabilities.method_signatures:
            assert not method_name.startswith("_"), (
                f"Private method '{method_name}' leaked after cache invalidation."
            )


# =============================================================================
# Security Verification Tests
# =============================================================================


class TestSecuritySensitiveMethodExclusion:
    """Test that security-sensitive private methods are never exposed."""

    async def test_credential_methods_not_exposed(self) -> None:
        """Verify methods with 'credential' in name are not exposed if private."""
        node = NodeWithPublicAndPrivateMethods(node_id=TEST_NODE_UUID_1)
        capabilities = await node.get_capabilities()

        # _validate_credentials should not be exposed
        assert "_validate_credentials" not in capabilities.method_signatures
        assert "_validate_credentials" not in capabilities.operations

    async def test_decrypt_methods_not_exposed(self) -> None:
        """Verify methods with 'decrypt' in name are not exposed if private."""
        node = NodeWithPublicAndPrivateMethods(node_id=TEST_NODE_UUID_1)
        capabilities = await node.get_capabilities()

        # _decrypt_payload should not be exposed
        assert "_decrypt_payload" not in capabilities.method_signatures
        assert "_decrypt_payload" not in capabilities.operations

    async def test_api_key_methods_not_exposed(self) -> None:
        """Verify methods with 'api_key' in name are not exposed if private."""
        node = NodeWithManyPrivateMethods(node_id=TEST_NODE_UUID_2)
        capabilities = await node.get_capabilities()

        # _read_api_key should not be exposed
        assert "_read_api_key" not in capabilities.method_signatures
        assert "_read_api_key" not in capabilities.operations

    async def test_auth_methods_not_exposed(self) -> None:
        """Verify methods with 'auth' in name are not exposed if private."""
        node = NodeWithManyPrivateMethods(node_id=TEST_NODE_UUID_2)
        capabilities = await node.get_capabilities()

        # _build_auth_header should not be exposed
        assert "_build_auth_header" not in capabilities.method_signatures
        assert "_build_auth_header" not in capabilities.operations
