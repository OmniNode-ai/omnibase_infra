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
    - protocol_postgres_adapter.py: ProtocolPostgresAdapter definition
    - OMN-915: Registration workflow integration testing
    - OMN-3540: Consul removed; TestStubConsulClientProtocolCompliance and
      Consul-related cross-protocol tests have been removed
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
from omnibase_infra.nodes.node_registry_effect.protocols.protocol_postgres_adapter import (
    ProtocolPostgresAdapter,
)
from tests.integration.registration.effect.test_doubles import (
    StubPostgresAdapter,
)

# -----------------------------------------------------------------------------
# Protocol Compliance Tests - isinstance() verification
# -----------------------------------------------------------------------------


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


__all__ = [
    "TestStubPostgresAdapterProtocolCompliance",
]
