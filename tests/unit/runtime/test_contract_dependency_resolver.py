# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Unit tests for ContractDependencyResolver.

Part of OMN-1732: Runtime dependency injection for zero-code nodes.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from omnibase_infra.errors import ProtocolDependencyResolutionError
from omnibase_infra.models.runtime.model_resolved_dependencies import (
    ModelResolvedDependencies,
)
from omnibase_infra.runtime.contract_dependency_resolver import (
    ContractDependencyResolver,
)


class TestContractDependencyResolver:
    """Tests for ContractDependencyResolver."""

    @pytest.fixture
    def mock_container(self) -> MagicMock:
        """Create a mock ONEX container with service_registry."""
        container = MagicMock()
        container.service_registry = MagicMock()
        container.service_registry.resolve_service = AsyncMock()
        return container

    @pytest.fixture
    def mock_contract_with_protocols(self) -> MagicMock:
        """Create a mock contract with protocol dependencies."""
        contract = MagicMock()
        contract.name = "test_node"

        # Mock dependency with protocol type using is_protocol() method
        dep1 = MagicMock()
        dep1.name = "protocol_postgres"
        dep1.class_name = "ProtocolPostgresAdapter"
        dep1.module = "omnibase_infra.adapters.protocol_postgres_adapter"
        dep1.is_protocol = MagicMock(return_value=True)

        dep2 = MagicMock()
        dep2.name = "protocol_circuit_breaker"
        dep2.class_name = "ProtocolCircuitBreakerAware"
        dep2.module = "omnibase_infra.mixins.protocol_circuit_breaker_aware"
        dep2.is_protocol = MagicMock(return_value=True)

        contract.dependencies = [dep1, dep2]
        return contract

    @pytest.fixture
    def mock_contract_no_protocols(self) -> MagicMock:
        """Create a mock contract with no dependencies."""
        contract = MagicMock()
        contract.name = "test_node_no_deps"
        contract.dependencies = []
        return contract

    @pytest.fixture
    def mock_contract_with_type_field(self) -> MagicMock:
        """Create a mock contract using type='protocol' field style."""
        contract = MagicMock()
        contract.name = "test_node_type_field"

        dep = MagicMock()
        dep.name = "protocol_test"
        dep.class_name = "ProtocolTestAdapter"
        dep.module = "test.module"
        dep.type = "protocol"
        # No is_protocol method - uses type field instead
        del dep.is_protocol

        contract.dependencies = [dep]
        return contract

    @pytest.fixture
    def mock_contract_with_dependency_type_enum(self) -> MagicMock:
        """Create a mock contract using dependency_type enum."""
        contract = MagicMock()
        contract.name = "test_node_enum"

        dep = MagicMock()
        dep.name = "protocol_enum"
        dep.class_name = "ProtocolEnumAdapter"
        dep.module = "test.module.enum"
        dep.dependency_type = MagicMock(value="PROTOCOL")
        # No is_protocol method - uses dependency_type attribute instead
        del dep.is_protocol

        contract.dependencies = [dep]
        return contract

    @pytest.mark.asyncio
    async def test_resolve_empty_dependencies_returns_empty(
        self,
        mock_container: MagicMock,
        mock_contract_no_protocols: MagicMock,
    ) -> None:
        """Test that contracts with no dependencies return empty result."""
        resolver = ContractDependencyResolver(mock_container)

        result = await resolver.resolve(mock_contract_no_protocols)

        assert isinstance(result, ModelResolvedDependencies)
        assert len(result) == 0
        mock_container.service_registry.resolve_service.assert_not_called()

    @pytest.mark.asyncio
    async def test_resolve_extracts_protocol_dependencies_is_protocol_method(
        self,
        mock_container: MagicMock,
        mock_contract_with_protocols: MagicMock,
    ) -> None:
        """Test that protocol dependencies are extracted using is_protocol() method."""
        resolver = ContractDependencyResolver(mock_container)

        deps = resolver._extract_protocol_dependencies(mock_contract_with_protocols)

        assert len(deps) == 2
        assert deps[0]["class_name"] == "ProtocolPostgresAdapter"
        assert deps[1]["class_name"] == "ProtocolCircuitBreakerAware"

    @pytest.mark.asyncio
    async def test_resolve_extracts_protocol_dependencies_type_field(
        self,
        mock_container: MagicMock,
        mock_contract_with_type_field: MagicMock,
    ) -> None:
        """Test that protocol dependencies are extracted using type='protocol' field."""
        resolver = ContractDependencyResolver(mock_container)

        deps = resolver._extract_protocol_dependencies(mock_contract_with_type_field)

        assert len(deps) == 1
        assert deps[0]["class_name"] == "ProtocolTestAdapter"

    @pytest.mark.asyncio
    async def test_resolve_extracts_protocol_dependencies_enum_type(
        self,
        mock_container: MagicMock,
        mock_contract_with_dependency_type_enum: MagicMock,
    ) -> None:
        """Test that protocol dependencies are extracted using dependency_type enum."""
        resolver = ContractDependencyResolver(mock_container)

        deps = resolver._extract_protocol_dependencies(
            mock_contract_with_dependency_type_enum
        )

        assert len(deps) == 1
        assert deps[0]["class_name"] == "ProtocolEnumAdapter"

    @pytest.mark.asyncio
    async def test_resolve_success_returns_resolved_dependencies(
        self,
        mock_container: MagicMock,
        mock_contract_with_protocols: MagicMock,
    ) -> None:
        """Test successful resolution returns ModelResolvedDependencies."""
        # Setup mock instances
        mock_postgres = MagicMock()
        mock_circuit_breaker = MagicMock()

        mock_container.service_registry.resolve_service = AsyncMock(
            side_effect=[mock_postgres, mock_circuit_breaker]
        )

        resolver = ContractDependencyResolver(mock_container)

        # Mock the import to avoid actual module import
        with patch.object(resolver, "_import_protocol_class") as mock_import:
            mock_import.side_effect = [
                MagicMock(__name__="ProtocolPostgresAdapter"),
                MagicMock(__name__="ProtocolCircuitBreakerAware"),
            ]

            result = await resolver.resolve(mock_contract_with_protocols)

        assert isinstance(result, ModelResolvedDependencies)
        assert len(result) == 2
        assert result.get("ProtocolPostgresAdapter") is mock_postgres
        assert result.get("ProtocolCircuitBreakerAware") is mock_circuit_breaker

    @pytest.mark.asyncio
    async def test_resolve_missing_protocol_raises_error(
        self,
        mock_container: MagicMock,
        mock_contract_with_protocols: MagicMock,
    ) -> None:
        """Test that missing protocols raise ProtocolDependencyResolutionError."""
        # Setup container to fail resolution
        mock_container.service_registry.resolve_service = AsyncMock(
            side_effect=Exception("Protocol not registered")
        )

        resolver = ContractDependencyResolver(mock_container)

        with patch.object(resolver, "_import_protocol_class") as mock_import:
            mock_import.return_value = MagicMock()

            with pytest.raises(ProtocolDependencyResolutionError) as exc_info:
                await resolver.resolve(mock_contract_with_protocols)

        assert "missing required protocols" in str(exc_info.value).lower()
        # Check that error context contains missing_protocols
        context = exc_info.value.model.context
        assert isinstance(context, dict)
        assert context.get("missing_protocols") is not None
        missing = context["missing_protocols"]
        assert isinstance(missing, list)
        assert "ProtocolPostgresAdapter" in missing
        assert "ProtocolCircuitBreakerAware" in missing

    @pytest.mark.asyncio
    async def test_resolve_allow_missing_skips_failures(
        self,
        mock_container: MagicMock,
        mock_contract_with_protocols: MagicMock,
    ) -> None:
        """Test that allow_missing=True skips failed resolutions."""
        # Setup container to fail resolution
        mock_container.service_registry.resolve_service = AsyncMock(
            side_effect=Exception("Protocol not registered")
        )

        resolver = ContractDependencyResolver(mock_container)

        with patch.object(resolver, "_import_protocol_class") as mock_import:
            mock_import.return_value = MagicMock()

            # Should not raise with allow_missing=True
            result = await resolver.resolve(
                mock_contract_with_protocols,
                allow_missing=True,
            )

        assert isinstance(result, ModelResolvedDependencies)
        assert len(result) == 0  # All failed, but no error raised

    @pytest.mark.asyncio
    async def test_resolve_none_service_registry_raises_error(
        self,
        mock_contract_with_protocols: MagicMock,
    ) -> None:
        """Test that None service_registry raises ProtocolDependencyResolutionError.

        Note: The RuntimeError is caught internally and wrapped in
        ProtocolDependencyResolutionError as part of the fail-fast behavior.
        """
        container = MagicMock()
        container.service_registry = None

        resolver = ContractDependencyResolver(container)

        with patch.object(resolver, "_import_protocol_class") as mock_import:
            mock_import.return_value = MagicMock()

            with pytest.raises(ProtocolDependencyResolutionError) as exc_info:
                await resolver.resolve(mock_contract_with_protocols)

        # The RuntimeError message is included in the wrapped error
        assert "service_registry is None" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_resolve_partial_success_raises_error(
        self,
        mock_container: MagicMock,
        mock_contract_with_protocols: MagicMock,
    ) -> None:
        """Test partial resolution (some succeed, some fail) raises error."""
        mock_postgres = MagicMock()

        # First succeeds, second fails
        mock_container.service_registry.resolve_service = AsyncMock(
            side_effect=[mock_postgres, Exception("Second protocol not found")]
        )

        resolver = ContractDependencyResolver(mock_container)

        with patch.object(resolver, "_import_protocol_class") as mock_import:
            mock_import.return_value = MagicMock()

            with pytest.raises(ProtocolDependencyResolutionError) as exc_info:
                await resolver.resolve(mock_contract_with_protocols)

        # Should list only the failed protocol
        context = exc_info.value.model.context
        assert isinstance(context, dict)
        missing = context["missing_protocols"]
        assert isinstance(missing, list)
        assert len(missing) == 1
        assert "ProtocolCircuitBreakerAware" in missing

    @pytest.mark.asyncio
    async def test_resolve_skips_dependency_without_class_name(
        self,
        mock_container: MagicMock,
    ) -> None:
        """Test that dependencies without class_name are skipped."""
        contract = MagicMock()
        contract.name = "test_node"

        # Dependency with protocol type but no class_name
        dep = MagicMock()
        dep.name = "no_class"
        dep.class_name = None  # Missing class_name
        dep.module = "test.module"
        dep.is_protocol = MagicMock(return_value=True)

        contract.dependencies = [dep]

        resolver = ContractDependencyResolver(mock_container)

        result = await resolver.resolve(contract)

        assert len(result) == 0
        mock_container.service_registry.resolve_service.assert_not_called()

    @pytest.mark.asyncio
    async def test_import_protocol_class_no_module_raises(
        self,
        mock_container: MagicMock,
    ) -> None:
        """Test _import_protocol_class raises ImportError when no module path."""
        resolver = ContractDependencyResolver(mock_container)

        with pytest.raises(ImportError) as exc_info:
            resolver._import_protocol_class("TestProtocol", None)

        assert "no module path specified" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_import_protocol_class_module_not_found(
        self,
        mock_container: MagicMock,
    ) -> None:
        """Test _import_protocol_class raises ImportError for nonexistent module."""
        resolver = ContractDependencyResolver(mock_container)

        with pytest.raises(ImportError) as exc_info:
            resolver._import_protocol_class(
                "TestProtocol", "nonexistent.module.path.that.does.not.exist"
            )

        assert "not found" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_import_protocol_class_class_not_in_module(
        self,
        mock_container: MagicMock,
    ) -> None:
        """Test _import_protocol_class raises ImportError for missing class."""
        resolver = ContractDependencyResolver(mock_container)

        # Use a real module but request a class that doesn't exist
        with pytest.raises(ImportError) as exc_info:
            resolver._import_protocol_class("NonExistentClass", "omnibase_infra.errors")

        assert "not found" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_is_protocol_dependency_false_for_non_protocol(
        self,
        mock_container: MagicMock,
    ) -> None:
        """Test _is_protocol_dependency returns False for non-protocol deps."""
        resolver = ContractDependencyResolver(mock_container)

        # Dependency with no protocol indicators
        dep = MagicMock(spec=["name", "class_name"])
        dep.name = "regular_dep"
        dep.class_name = "SomeClass"

        result = resolver._is_protocol_dependency(dep)

        assert result is False

    @pytest.mark.asyncio
    async def test_extract_handles_missing_dependencies_attr(
        self,
        mock_container: MagicMock,
    ) -> None:
        """Test _extract_protocol_dependencies handles contract without dependencies."""
        contract = MagicMock(spec=["name"])
        contract.name = "no_deps_contract"

        resolver = ContractDependencyResolver(mock_container)

        deps = resolver._extract_protocol_dependencies(contract)

        assert deps == []

    @pytest.mark.asyncio
    async def test_error_includes_node_name(
        self,
        mock_container: MagicMock,
        mock_contract_with_protocols: MagicMock,
    ) -> None:
        """Test that error includes node name in context."""
        mock_container.service_registry.resolve_service = AsyncMock(
            side_effect=Exception("Resolution failed")
        )

        resolver = ContractDependencyResolver(mock_container)

        with patch.object(resolver, "_import_protocol_class") as mock_import:
            mock_import.return_value = MagicMock()

            with pytest.raises(ProtocolDependencyResolutionError) as exc_info:
                await resolver.resolve(mock_contract_with_protocols)

        assert exc_info.value.model.context.get("node_name") == "test_node"


class TestModelResolvedDependencies:
    """Tests for ModelResolvedDependencies model."""

    def test_get_returns_protocol(self) -> None:
        """Test get() returns the protocol instance."""
        mock_adapter = MagicMock()
        resolved = ModelResolvedDependencies(
            protocols={"ProtocolPostgresAdapter": mock_adapter}
        )

        result = resolved.get("ProtocolPostgresAdapter")

        assert result is mock_adapter

    def test_get_raises_keyerror_for_missing(self) -> None:
        """Test get() raises KeyError for missing protocol."""
        resolved = ModelResolvedDependencies(protocols={})

        with pytest.raises(KeyError) as exc_info:
            resolved.get("NonExistentProtocol")

        assert "NonExistentProtocol" in str(exc_info.value)

    def test_get_keyerror_message_includes_available(self) -> None:
        """Test KeyError message lists available protocols."""
        resolved = ModelResolvedDependencies(
            protocols={"ProtocolA": MagicMock(), "ProtocolB": MagicMock()}
        )

        with pytest.raises(KeyError) as exc_info:
            resolved.get("NonExistent")

        error_msg = str(exc_info.value)
        assert "ProtocolA" in error_msg
        assert "ProtocolB" in error_msg

    def test_get_optional_returns_value(self) -> None:
        """Test get_optional() returns value when present."""
        mock_adapter = MagicMock()
        resolved = ModelResolvedDependencies(protocols={"ProtocolTest": mock_adapter})

        result = resolved.get_optional("ProtocolTest")

        assert result is mock_adapter

    def test_get_optional_returns_default(self) -> None:
        """Test get_optional() returns default for missing protocol."""
        resolved = ModelResolvedDependencies(protocols={})

        result = resolved.get_optional("NonExistentProtocol", default="fallback")

        assert result == "fallback"

    def test_get_optional_returns_none_by_default(self) -> None:
        """Test get_optional() returns None when default not specified."""
        resolved = ModelResolvedDependencies(protocols={})

        result = resolved.get_optional("NonExistent")

        assert result is None

    def test_has_returns_true_for_existing(self) -> None:
        """Test has() returns True for existing protocol."""
        resolved = ModelResolvedDependencies(
            protocols={"ProtocolPostgresAdapter": MagicMock()}
        )

        assert resolved.has("ProtocolPostgresAdapter") is True

    def test_has_returns_false_for_missing(self) -> None:
        """Test has() returns False for missing protocol."""
        resolved = ModelResolvedDependencies(protocols={})

        assert resolved.has("NonExistent") is False

    def test_len_returns_protocol_count(self) -> None:
        """Test __len__ returns number of protocols."""
        resolved = ModelResolvedDependencies(
            protocols={
                "Protocol1": MagicMock(),
                "Protocol2": MagicMock(),
            }
        )

        assert len(resolved) == 2

    def test_len_returns_zero_for_empty(self) -> None:
        """Test __len__ returns 0 for empty protocols."""
        resolved = ModelResolvedDependencies(protocols={})

        assert len(resolved) == 0

    def test_bool_true_when_protocols_present(self) -> None:
        """Test __bool__ returns True when protocols are present."""
        resolved = ModelResolvedDependencies(protocols={"Protocol1": MagicMock()})

        assert bool(resolved) is True

    def test_bool_false_when_empty(self) -> None:
        """Test __bool__ returns False when no protocols."""
        resolved = ModelResolvedDependencies(protocols={})

        assert bool(resolved) is False

    def test_default_factory_creates_empty_dict(self) -> None:
        """Test that ModelResolvedDependencies() creates empty protocols dict."""
        resolved = ModelResolvedDependencies()

        assert resolved.protocols == {}
        assert len(resolved) == 0
        assert bool(resolved) is False

    def test_model_is_immutable(self) -> None:
        """Test that the model is frozen (immutable)."""
        resolved = ModelResolvedDependencies(protocols={"Test": MagicMock()})

        with pytest.raises(Exception):  # ValidationError or AttributeError
            resolved.protocols = {}  # type: ignore[misc]

    def test_extra_fields_forbidden(self) -> None:
        """Test that extra fields are not allowed."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            ModelResolvedDependencies(
                protocols={},
                extra_field="not allowed",  # type: ignore[call-arg]
            )
