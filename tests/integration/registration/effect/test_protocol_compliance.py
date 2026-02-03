# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Protocol compliance tests for test doubles.

This module verifies that test doubles implement the same protocols as the
real infrastructure clients they replace. This ensures type safety and
behavioral contract adherence in integration tests.

Protocol Compliance Strategy:
    - Uses @runtime_checkable protocols for isinstance() verification
    - Tests method signatures match protocol definitions
    - Verifies return types are compatible
    - Documents behavioral contracts via test assertions

Why Protocol Compliance Matters:
    - Ensures test doubles are drop-in replacements for real clients
    - Catches interface drift between protocols and implementations
    - Provides compile-time-like safety for dynamically typed Python
    - Documents the contract relationship explicitly

Related:
    - test_doubles.py: Contains StubConsulClient, StubPostgresAdapter
    - protocol_consul_client.py: ProtocolConsulClient definition
    - protocol_postgres_adapter.py: ProtocolPostgresAdapter definition
    - OMN-915: Registration workflow integration testing
"""

from __future__ import annotations

import asyncio
import inspect
from typing import get_type_hints
from uuid import uuid4

import pytest

from omnibase_core.enums.enum_node_kind import EnumNodeKind
from omnibase_core.models.primitives.model_semver import ModelSemVer
from omnibase_infra.models import ModelBackendResult
from omnibase_infra.nodes.effects.protocol_consul_client import ProtocolConsulClient
from omnibase_infra.nodes.effects.protocol_postgres_adapter import (
    ProtocolPostgresAdapter,
)
from tests.integration.registration.effect.test_doubles import (
    StubConsulClient,
    StubPostgresAdapter,
)

# -----------------------------------------------------------------------------
# Protocol Compliance Tests - isinstance() verification
# -----------------------------------------------------------------------------


@pytest.mark.unit
class TestStubConsulClientProtocolCompliance:
    """Verify StubConsulClient implements ProtocolConsulClient.

    Protocol Contract:
        - Must have async register_service() method
        - Method signature must match protocol definition
        - Return type must be ModelBackendResult
        - Must be thread-safe for concurrent async calls
    """

    def test_isinstance_protocol_check(self) -> None:
        """Verify StubConsulClient passes isinstance check for protocol.

        This test uses @runtime_checkable to verify structural subtyping.
        The stub must implement the same method signatures as the protocol.
        """
        stub = StubConsulClient()
        assert isinstance(stub, ProtocolConsulClient), (
            "StubConsulClient must implement ProtocolConsulClient. "
            "Check that all required methods are present with correct signatures."
        )

    def test_register_service_method_exists(self) -> None:
        """Verify register_service method exists and is async."""
        stub = StubConsulClient()
        assert hasattr(stub, "register_service"), (
            "StubConsulClient missing required method: register_service"
        )
        assert asyncio.iscoroutinefunction(stub.register_service), (
            "register_service must be an async method"
        )

    def test_register_service_signature_matches_protocol(self) -> None:
        """Verify register_service method signature matches protocol.

        Protocol defines:
            async def register_service(
                self,
                service_id: str,
                service_name: str,
                tags: list[str],
                health_check: dict[str, str] | None = None,
            ) -> ModelBackendResult
        """
        stub = StubConsulClient()
        sig = inspect.signature(stub.register_service)
        params = list(sig.parameters.keys())

        # Verify required parameters
        expected_params = ["service_id", "service_name", "tags", "health_check"]
        assert params == expected_params, (
            f"register_service signature mismatch. "
            f"Expected params: {expected_params}, got: {params}"
        )

        # Verify health_check has default value of None
        health_check_param = sig.parameters["health_check"]
        assert health_check_param.default is None, (
            "health_check parameter must have default value of None"
        )

    @pytest.mark.asyncio
    async def test_register_service_returns_model_backend_result(self) -> None:
        """Verify register_service returns ModelBackendResult instance."""
        stub = StubConsulClient()
        result = await stub.register_service(
            service_id="test-service-id",
            service_name="test-service",
            tags=["test", "integration"],
            health_check=None,
        )
        assert isinstance(result, ModelBackendResult), (
            f"register_service must return ModelBackendResult, got {type(result)}"
        )

    @pytest.mark.asyncio
    async def test_success_result_has_correct_fields(self) -> None:
        """Verify successful result has expected fields set."""
        stub = StubConsulClient()
        result = await stub.register_service(
            service_id="test-service-id",
            service_name="test-service",
            tags=["test"],
        )
        assert result.success is True
        assert result.error is None

    @pytest.mark.asyncio
    async def test_failure_result_has_error_message(self) -> None:
        """Verify failed result has error message set."""
        stub = StubConsulClient(should_fail=True, failure_error="Test error")
        result = await stub.register_service(
            service_id="test-service-id",
            service_name="test-service",
            tags=["test"],
        )
        assert result.success is False
        assert result.error == "Test error"


@pytest.mark.unit
class TestStubPostgresAdapterProtocolCompliance:
    """Verify StubPostgresAdapter implements ProtocolPostgresAdapter.

    Protocol Contract:
        - Must have async upsert() method
        - Method signature must match protocol definition
        - node_type parameter must accept EnumNodeKind
        - Return type must be ModelBackendResult
        - Must be thread-safe for concurrent async calls
    """

    def test_isinstance_protocol_check(self) -> None:
        """Verify StubPostgresAdapter passes isinstance check for protocol.

        This test uses @runtime_checkable to verify structural subtyping.
        The stub must implement the same method signatures as the protocol.
        """
        stub = StubPostgresAdapter()
        assert isinstance(stub, ProtocolPostgresAdapter), (
            "StubPostgresAdapter must implement ProtocolPostgresAdapter. "
            "Check that all required methods are present with correct signatures."
        )

    def test_upsert_method_exists(self) -> None:
        """Verify upsert method exists and is async."""
        stub = StubPostgresAdapter()
        assert hasattr(stub, "upsert"), (
            "StubPostgresAdapter missing required method: upsert"
        )
        assert asyncio.iscoroutinefunction(stub.upsert), (
            "upsert must be an async method"
        )

    def test_upsert_signature_matches_protocol(self) -> None:
        """Verify upsert method signature matches protocol.

        Protocol defines:
            async def upsert(
                self,
                node_id: UUID,
                node_type: EnumNodeKind,
                node_version: str,
                endpoints: dict[str, str],
                metadata: dict[str, str],
            ) -> ModelBackendResult
        """
        stub = StubPostgresAdapter()
        sig = inspect.signature(stub.upsert)
        params = list(sig.parameters.keys())

        # Verify required parameters
        expected_params = [
            "node_id",
            "node_type",
            "node_version",
            "endpoints",
            "metadata",
        ]
        assert params == expected_params, (
            f"upsert signature mismatch. "
            f"Expected params: {expected_params}, got: {params}"
        )

    def test_upsert_node_type_annotation(self) -> None:
        """Verify node_type parameter accepts EnumNodeKind.

        This ensures type safety when passing node types from
        ModelRegistryRequest to the adapter.
        """
        stub = StubPostgresAdapter()
        hints = get_type_hints(stub.upsert)

        # The node_type should be annotated as EnumNodeKind
        # Note: get_type_hints resolves forward references
        assert "node_type" in hints, "node_type parameter must have type annotation"
        assert hints["node_type"] is EnumNodeKind, (
            f"node_type must be annotated as EnumNodeKind, got {hints['node_type']}"
        )

    @pytest.mark.asyncio
    async def test_upsert_returns_model_backend_result(self) -> None:
        """Verify upsert returns ModelBackendResult instance."""
        stub = StubPostgresAdapter()
        result = await stub.upsert(
            node_id=uuid4(),
            node_type=EnumNodeKind.EFFECT,
            node_version=ModelSemVer.parse("1.0.0"),
            endpoints={"health": "http://localhost:8080/health"},
            metadata={"environment": "test"},
        )
        assert isinstance(result, ModelBackendResult), (
            f"upsert must return ModelBackendResult, got {type(result)}"
        )

    @pytest.mark.asyncio
    async def test_upsert_accepts_all_node_kinds(self) -> None:
        """Verify upsert accepts all EnumNodeKind values.

        This ensures the adapter can handle any ONEX node type.
        """
        stub = StubPostgresAdapter()
        node_kinds = [
            EnumNodeKind.EFFECT,
            EnumNodeKind.COMPUTE,
            EnumNodeKind.REDUCER,
            EnumNodeKind.ORCHESTRATOR,
        ]

        for node_kind in node_kinds:
            result = await stub.upsert(
                node_id=uuid4(),
                node_type=node_kind,
                node_version=ModelSemVer.parse("1.0.0"),
                endpoints={},
                metadata={},
            )
            assert result.success is True, (
                f"upsert should succeed for node_type={node_kind}"
            )

    @pytest.mark.asyncio
    async def test_success_result_has_correct_fields(self) -> None:
        """Verify successful result has expected fields set."""
        stub = StubPostgresAdapter()
        result = await stub.upsert(
            node_id=uuid4(),
            node_type=EnumNodeKind.EFFECT,
            node_version=ModelSemVer.parse("1.0.0"),
            endpoints={},
            metadata={},
        )
        assert result.success is True
        assert result.error is None

    @pytest.mark.asyncio
    async def test_failure_result_has_error_message(self) -> None:
        """Verify failed result has error message set."""
        stub = StubPostgresAdapter(should_fail=True, failure_error="DB error")
        result = await stub.upsert(
            node_id=uuid4(),
            node_type=EnumNodeKind.EFFECT,
            node_version=ModelSemVer.parse("1.0.0"),
            endpoints={},
            metadata={},
        )
        assert result.success is False
        assert result.error == "DB error"


# -----------------------------------------------------------------------------
# Cross-Protocol Consistency Tests
# -----------------------------------------------------------------------------


@pytest.mark.unit
class TestProtocolConsistency:
    """Verify both protocols and stubs follow consistent patterns.

    These tests ensure that both backend protocols (Consul, PostgreSQL)
    follow the same result pattern, making them interchangeable in
    the dual-registration workflow.
    """

    @pytest.mark.asyncio
    async def test_both_stubs_return_same_result_type(self) -> None:
        """Verify both stubs return ModelBackendResult.

        This consistency is required for ModelRegistryResponse to
        aggregate results from both backends uniformly.
        """
        consul_stub = StubConsulClient()
        postgres_stub = StubPostgresAdapter()

        consul_result = await consul_stub.register_service(
            service_id="test-id",
            service_name="test-service",
            tags=["test"],
        )
        postgres_result = await postgres_stub.upsert(
            node_id=uuid4(),
            node_type=EnumNodeKind.EFFECT,
            node_version=ModelSemVer.parse("1.0.0"),
            endpoints={},
            metadata={},
        )

        assert type(consul_result) is type(postgres_result), (
            "Both stubs must return the same result type (ModelBackendResult)"
        )

    def test_both_protocols_are_runtime_checkable(self) -> None:
        """Verify both protocols can be used with isinstance().

        The @runtime_checkable decorator must be present on both
        protocols for structural subtyping verification.
        """
        # Create instances to test isinstance
        consul_stub = StubConsulClient()
        postgres_stub = StubPostgresAdapter()

        # These should not raise TypeError
        isinstance(consul_stub, ProtocolConsulClient)
        isinstance(postgres_stub, ProtocolPostgresAdapter)

        # Additional check: verify we can check arbitrary objects
        # (would fail without @runtime_checkable)
        class NotAClient:
            pass

        assert not isinstance(NotAClient(), ProtocolConsulClient)
        assert not isinstance(NotAClient(), ProtocolPostgresAdapter)


__all__ = [
    "TestStubConsulClientProtocolCompliance",
    "TestStubPostgresAdapterProtocolCompliance",
    "TestProtocolConsistency",
]
