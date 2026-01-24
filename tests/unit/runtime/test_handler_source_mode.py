# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Unit Tests for Handler Source Mode Hybrid Resolution.

This module contains unit tests for OMN-1095: Handler Source Mode hybrid
resolution functionality. These tests verify the behavior of the
HandlerSourceResolver class.

The HandlerSourceResolver is responsible for:
1. Resolving handlers from multiple sources (Bootstrap, Contract) based on mode
2. Implementing per-handler identity resolution in HYBRID mode
3. Contract precedence: contract handlers override bootstrap handlers with same identity
4. Bootstrap fallback: bootstrap handlers used when no matching contract handler exists

Test Categories:
    - Hybrid Mode Resolution: Contract wins over bootstrap by handler_id
    - Bootstrap Fallback: Uses bootstrap when contract missing
    - Bootstrap Only Mode: Ignores contract handlers
    - Contract Only Mode: Ignores bootstrap handlers
    - Structured Logging: Proper handler count logging

Related:
    - OMN-1095: Handler Source Mode Hybrid Resolution
    - HandlerBootstrapSource: Provides hardcoded bootstrap handlers
    - HandlerContractSource: Provides contract-discovered handlers
    - EnumHandlerSourceMode: Defines BOOTSTRAP, CONTRACT, HYBRID modes

Implementation:
    src/omnibase_infra/runtime/handler_source_resolver.py

See Also:
    - test_handler_bootstrap_source.py: Tests for bootstrap source
    - test_handler_contract_source.py: Tests for contract source
    - docs/architecture/HANDLER_PROTOCOL_DRIVEN_ARCHITECTURE.md
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from omnibase_core.models.primitives import ModelSemVer
from omnibase_infra.enums.enum_handler_source_mode import EnumHandlerSourceMode

# Import ModelHandlerValidationError BEFORE ModelContractDiscoveryResult
# to ensure forward reference is available for model_rebuild()
from omnibase_infra.models.errors import ModelHandlerValidationError
from omnibase_infra.models.handlers import (
    ModelContractDiscoveryResult,
    ModelHandlerDescriptor,
)

# Rebuild to resolve forward reference to ModelHandlerValidationError
# This is required because ModelContractDiscoveryResult uses TYPE_CHECKING
# to defer the import of ModelHandlerValidationError for circular import avoidance.
# See: handler_contract_source.py for the same pattern.
ModelContractDiscoveryResult.model_rebuild()

if TYPE_CHECKING:
    from omnibase_infra.runtime.handler_bootstrap_source import HandlerBootstrapSource
    from omnibase_infra.runtime.handler_contract_source import HandlerContractSource


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def sample_version() -> ModelSemVer:
    """Create a sample version for test descriptors."""
    return ModelSemVer(major=1, minor=0, patch=0)


@pytest.fixture
def bootstrap_consul_descriptor(sample_version: ModelSemVer) -> ModelHandlerDescriptor:
    """Create a bootstrap handler descriptor for Consul.

    This represents a handler loaded from HandlerBootstrapSource.
    The handler_id follows the bootstrap naming convention: "bootstrap.consul"
    """
    return ModelHandlerDescriptor(
        handler_id="bootstrap.consul",
        name="Consul Handler (Bootstrap)",
        version=sample_version,
        handler_kind="effect",
        input_model="omnibase_infra.models.types.JsonDict",
        output_model="omnibase_core.models.dispatch.ModelHandlerOutput",
        description="Bootstrap Consul handler for service discovery",
        handler_class="omnibase_infra.handlers.handler_consul.HandlerConsul",
        contract_path="contracts/handlers/consul/handler_contract.yaml",
    )


@pytest.fixture
def bootstrap_vault_descriptor(sample_version: ModelSemVer) -> ModelHandlerDescriptor:
    """Create a bootstrap handler descriptor for Vault.

    This represents a handler loaded from HandlerBootstrapSource.
    The handler_id follows the bootstrap naming convention: "bootstrap.vault"
    """
    return ModelHandlerDescriptor(
        handler_id="bootstrap.vault",
        name="Vault Handler (Bootstrap)",
        version=sample_version,
        handler_kind="effect",
        input_model="omnibase_infra.models.types.JsonDict",
        output_model="omnibase_core.models.dispatch.ModelHandlerOutput",
        description="Bootstrap Vault handler for secret management",
        handler_class="omnibase_infra.handlers.handler_vault.HandlerVault",
        contract_path="contracts/handlers/vault/handler_contract.yaml",
    )


@pytest.fixture
def contract_consul_descriptor(sample_version: ModelSemVer) -> ModelHandlerDescriptor:
    """Create a contract handler descriptor for Consul with SAME identity.

    This represents a handler loaded from HandlerContractSource with the SAME
    handler_id as the bootstrap Consul handler. In HYBRID mode, this should
    WIN over the bootstrap handler because contract takes precedence.
    """
    return ModelHandlerDescriptor(
        handler_id="bootstrap.consul",  # Same identity as bootstrap
        name="Consul Handler (Contract)",  # Different name to verify which wins
        version=sample_version,
        handler_kind="effect",
        input_model="omnibase_infra.models.types.JsonDict",
        output_model="omnibase_core.models.dispatch.ModelHandlerOutput",
        description="Contract-defined Consul handler (should override bootstrap)",
        handler_class="omnibase_infra.handlers.handler_consul.HandlerConsul",
        contract_path="nodes/consul/handler_contract.yaml",
    )


@pytest.fixture
def contract_custom_descriptor(sample_version: ModelSemVer) -> ModelHandlerDescriptor:
    """Create a contract handler descriptor for a custom handler.

    This represents a handler that only exists in contracts, not in bootstrap.
    This handler should be registered in both CONTRACT and HYBRID modes.
    """
    return ModelHandlerDescriptor(
        handler_id="contract.custom",
        name="Custom Handler (Contract Only)",
        version=sample_version,
        handler_kind="compute",
        input_model="myapp.models.CustomInput",
        output_model="myapp.models.CustomOutput",
        description="Custom contract-only handler",
        handler_class="myapp.handlers.handler_custom.HandlerCustom",
        contract_path="nodes/custom/handler_contract.yaml",
    )


@pytest.fixture
def mock_bootstrap_source(
    bootstrap_consul_descriptor: ModelHandlerDescriptor,
    bootstrap_vault_descriptor: ModelHandlerDescriptor,
) -> MagicMock:
    """Create a mock HandlerBootstrapSource.

    Returns a mock that provides:
    - bootstrap.consul handler
    - bootstrap.vault handler
    """
    mock_source = MagicMock()
    mock_source.source_type = "BOOTSTRAP"
    mock_source.discover_handlers = AsyncMock(
        return_value=ModelContractDiscoveryResult(
            descriptors=[bootstrap_consul_descriptor, bootstrap_vault_descriptor],
            validation_errors=[],
        )
    )
    return mock_source


@pytest.fixture
def mock_contract_source(
    contract_consul_descriptor: ModelHandlerDescriptor,
    contract_custom_descriptor: ModelHandlerDescriptor,
) -> MagicMock:
    """Create a mock HandlerContractSource.

    Returns a mock that provides:
    - bootstrap.consul handler (same identity as bootstrap, should override)
    - contract.custom handler (unique to contract)
    """
    mock_source = MagicMock()
    mock_source.source_type = "CONTRACT"
    mock_source.discover_handlers = AsyncMock(
        return_value=ModelContractDiscoveryResult(
            descriptors=[contract_consul_descriptor, contract_custom_descriptor],
            validation_errors=[],
        )
    )
    return mock_source


@pytest.fixture
def empty_contract_source() -> MagicMock:
    """Create a mock HandlerContractSource with no handlers."""
    mock_source = MagicMock()
    mock_source.source_type = "CONTRACT"
    mock_source.discover_handlers = AsyncMock(
        return_value=ModelContractDiscoveryResult(
            descriptors=[],
            validation_errors=[],
        )
    )
    return mock_source


# =============================================================================
# Test Class: Hybrid Mode Resolution - Contract Wins Over Bootstrap
# =============================================================================


class TestHybridModeContractWins:
    """Tests for HYBRID mode where contract handlers override bootstrap handlers.

    In HYBRID mode, when both bootstrap and contract sources provide a handler
    with the same handler_id (identity), the contract handler should win.
    This enables gradual migration from bootstrap to contract-based handlers.

    The resolution algorithm should:
    1. Discover handlers from both sources
    2. Build a handler map keyed by handler_id
    3. Contract handlers override bootstrap handlers with same handler_id
    4. Return the merged set of unique handlers
    """

    @pytest.mark.asyncio
    async def test_hybrid_mode_contract_wins_over_bootstrap_when_same_identity(
        self,
        mock_bootstrap_source: MagicMock,
        mock_contract_source: MagicMock,
    ) -> None:
        """When HYBRID mode, contract handler should override bootstrap handler with same handler_id.

        Given:
            - Bootstrap source provides handler with handler_id="bootstrap.consul"
            - Contract source provides handler with handler_id="bootstrap.consul" (same identity)

        When:
            - Resolve handlers in HYBRID mode

        Then:
            - Only ONE handler with handler_id="bootstrap.consul" should be registered
            - That handler should be the CONTRACT version (name="Consul Handler (Contract)")
            - The bootstrap version should be discarded
        """
        from omnibase_infra.runtime.handler_source_resolver import HandlerSourceResolver

        resolver = HandlerSourceResolver(
            bootstrap_source=mock_bootstrap_source,
            contract_source=mock_contract_source,
            mode=EnumHandlerSourceMode.HYBRID,
        )

        result = await resolver.resolve_handlers()

        # Get handler by ID
        consul_handlers = [
            h for h in result.descriptors if h.handler_id == "bootstrap.consul"
        ]

        # Should have exactly one Consul handler (not two)
        assert len(consul_handlers) == 1, (
            f"Expected exactly 1 handler with id 'bootstrap.consul', "
            f"got {len(consul_handlers)}. "
            "Contract should override bootstrap when same identity."
        )

        # The winning handler should be from contract (verify by name)
        consul_handler = consul_handlers[0]
        assert consul_handler.name == "Consul Handler (Contract)", (
            f"Expected contract handler to win, got name='{consul_handler.name}'. "
            "In HYBRID mode, contract takes precedence over bootstrap."
        )

    @pytest.mark.asyncio
    async def test_hybrid_mode_includes_contract_only_handlers(
        self,
        mock_bootstrap_source: MagicMock,
        mock_contract_source: MagicMock,
    ) -> None:
        """Hybrid mode should include handlers that only exist in contracts.

        Given:
            - Bootstrap source provides: bootstrap.consul, bootstrap.vault
            - Contract source provides: bootstrap.consul (override), contract.custom (unique)

        When:
            - Resolve handlers in HYBRID mode

        Then:
            - contract.custom should be in the result (contract-only handler)
        """
        from omnibase_infra.runtime.handler_source_resolver import HandlerSourceResolver

        resolver = HandlerSourceResolver(
            bootstrap_source=mock_bootstrap_source,
            contract_source=mock_contract_source,
            mode=EnumHandlerSourceMode.HYBRID,
        )

        result = await resolver.resolve_handlers()

        custom_handlers = [
            h for h in result.descriptors if h.handler_id == "contract.custom"
        ]

        assert len(custom_handlers) == 1, (
            "Expected contract-only handler 'contract.custom' to be included"
        )

    @pytest.mark.asyncio
    async def test_hybrid_mode_total_handler_count(
        self,
        mock_bootstrap_source: MagicMock,
        mock_contract_source: MagicMock,
    ) -> None:
        """Hybrid mode should return merged unique handlers.

        Given:
            - Bootstrap: bootstrap.consul, bootstrap.vault (2 handlers)
            - Contract: bootstrap.consul (override), contract.custom (1 unique)

        When:
            - Resolve handlers in HYBRID mode

        Then:
            - Total should be 3 unique handlers:
              - bootstrap.consul (from contract, overrides bootstrap)
              - bootstrap.vault (from bootstrap, no contract override)
              - contract.custom (from contract, unique)
        """
        from omnibase_infra.runtime.handler_source_resolver import HandlerSourceResolver

        resolver = HandlerSourceResolver(
            bootstrap_source=mock_bootstrap_source,
            contract_source=mock_contract_source,
            mode=EnumHandlerSourceMode.HYBRID,
        )

        result = await resolver.resolve_handlers()

        assert len(result.descriptors) == 3, (
            f"Expected 3 unique handlers in HYBRID mode, got {len(result.descriptors)}. "
            "Handlers: bootstrap.consul (contract), bootstrap.vault (bootstrap), "
            "contract.custom (contract)"
        )


# =============================================================================
# Test Class: Hybrid Mode Fallback - Bootstrap When Contract Missing
# =============================================================================


class TestHybridModeFallback:
    """Tests for HYBRID mode fallback to bootstrap handlers.

    When a handler exists in bootstrap but NOT in contracts, the bootstrap
    handler should be used as a fallback. This enables core handlers to
    remain available even when contracts don't define them.
    """

    @pytest.mark.asyncio
    async def test_hybrid_mode_uses_bootstrap_fallback_when_contract_missing(
        self,
        mock_bootstrap_source: MagicMock,
        mock_contract_source: MagicMock,
    ) -> None:
        """When HYBRID mode, bootstrap handler should be used if no contract handler with same identity.

        Given:
            - Bootstrap source provides handler with handler_id="bootstrap.vault"
            - Contract source does NOT provide handler with handler_id="bootstrap.vault"

        When:
            - Resolve handlers in HYBRID mode

        Then:
            - bootstrap.vault handler should be registered (from bootstrap as fallback)
        """
        from omnibase_infra.runtime.handler_source_resolver import HandlerSourceResolver

        resolver = HandlerSourceResolver(
            bootstrap_source=mock_bootstrap_source,
            contract_source=mock_contract_source,
            mode=EnumHandlerSourceMode.HYBRID,
        )

        result = await resolver.resolve_handlers()

        vault_handlers = [
            h for h in result.descriptors if h.handler_id == "bootstrap.vault"
        ]

        assert len(vault_handlers) == 1, (
            "Expected bootstrap.vault to be included as fallback"
        )

        vault_handler = vault_handlers[0]
        assert vault_handler.name == "Vault Handler (Bootstrap)", (
            f"Expected bootstrap handler to be used as fallback, "
            f"got name='{vault_handler.name}'"
        )

    @pytest.mark.asyncio
    async def test_hybrid_mode_with_empty_contract_uses_all_bootstrap(
        self,
        mock_bootstrap_source: MagicMock,
        empty_contract_source: MagicMock,
    ) -> None:
        """When contracts are empty, HYBRID mode should use all bootstrap handlers.

        Given:
            - Bootstrap source provides: bootstrap.consul, bootstrap.vault
            - Contract source provides: (empty)

        When:
            - Resolve handlers in HYBRID mode

        Then:
            - All bootstrap handlers should be included as fallback
        """
        from omnibase_infra.runtime.handler_source_resolver import HandlerSourceResolver

        resolver = HandlerSourceResolver(
            bootstrap_source=mock_bootstrap_source,
            contract_source=empty_contract_source,
            mode=EnumHandlerSourceMode.HYBRID,
        )

        result = await resolver.resolve_handlers()

        assert len(result.descriptors) == 2, (
            "Expected all bootstrap handlers when contracts are empty"
        )

        handler_ids = {h.handler_id for h in result.descriptors}
        assert handler_ids == {"bootstrap.consul", "bootstrap.vault"}, (
            f"Expected bootstrap handler IDs, got {handler_ids}"
        )


# =============================================================================
# Test Class: Bootstrap Only Mode
# =============================================================================


class TestBootstrapOnlyMode:
    """Tests for BOOTSTRAP mode where only bootstrap handlers are loaded.

    In BOOTSTRAP mode, the resolver should:
    1. Only call discover_handlers() on the bootstrap source
    2. NOT call discover_handlers() on the contract source
    3. Return only bootstrap handlers
    """

    @pytest.mark.asyncio
    async def test_bootstrap_only_mode_ignores_contract_handlers(
        self,
        mock_bootstrap_source: MagicMock,
        mock_contract_source: MagicMock,
    ) -> None:
        """When BOOTSTRAP mode, only bootstrap handlers should be loaded.

        Given:
            - Both bootstrap and contract sources are available
            - Mode is BOOTSTRAP

        When:
            - Resolve handlers

        Then:
            - Only bootstrap handlers should be registered
            - Contract source should NOT be called
        """
        from omnibase_infra.runtime.handler_source_resolver import HandlerSourceResolver

        resolver = HandlerSourceResolver(
            bootstrap_source=mock_bootstrap_source,
            contract_source=mock_contract_source,
            mode=EnumHandlerSourceMode.BOOTSTRAP,
        )

        result = await resolver.resolve_handlers()

        # Bootstrap source should be called
        mock_bootstrap_source.discover_handlers.assert_called_once()

        # Contract source should NOT be called in BOOTSTRAP mode
        mock_contract_source.discover_handlers.assert_not_called()

        # Only bootstrap handlers should be in result
        assert len(result.descriptors) == 2, (
            f"Expected 2 bootstrap handlers, got {len(result.descriptors)}"
        )

        handler_ids = {h.handler_id for h in result.descriptors}
        assert handler_ids == {"bootstrap.consul", "bootstrap.vault"}, (
            f"Expected only bootstrap handler IDs, got {handler_ids}"
        )

    @pytest.mark.asyncio
    async def test_bootstrap_only_mode_handler_names(
        self,
        mock_bootstrap_source: MagicMock,
        mock_contract_source: MagicMock,
    ) -> None:
        """Bootstrap mode should return bootstrap handlers with correct metadata.

        Verify that the handlers returned have bootstrap source metadata,
        not contract source metadata.
        """
        from omnibase_infra.runtime.handler_source_resolver import HandlerSourceResolver

        resolver = HandlerSourceResolver(
            bootstrap_source=mock_bootstrap_source,
            contract_source=mock_contract_source,
            mode=EnumHandlerSourceMode.BOOTSTRAP,
        )

        result = await resolver.resolve_handlers()

        # All handlers should have bootstrap-style names
        for handler in result.descriptors:
            assert (
                "(Bootstrap)" in handler.name or "Bootstrap" in handler.description
            ), f"Handler {handler.handler_id} doesn't appear to be from bootstrap"


# =============================================================================
# Test Class: Contract Only Mode
# =============================================================================


class TestContractOnlyMode:
    """Tests for CONTRACT mode where only contract handlers are loaded.

    In CONTRACT mode, the resolver should:
    1. Only call discover_handlers() on the contract source
    2. NOT call discover_handlers() on the bootstrap source
    3. Return only contract handlers
    """

    @pytest.mark.asyncio
    async def test_contract_only_mode_ignores_bootstrap_handlers(
        self,
        mock_bootstrap_source: MagicMock,
        mock_contract_source: MagicMock,
    ) -> None:
        """When CONTRACT mode, only contract handlers should be loaded.

        Given:
            - Both bootstrap and contract sources are available
            - Mode is CONTRACT

        When:
            - Resolve handlers

        Then:
            - Only contract handlers should be registered
            - Bootstrap source should NOT be called
        """
        from omnibase_infra.runtime.handler_source_resolver import HandlerSourceResolver

        resolver = HandlerSourceResolver(
            bootstrap_source=mock_bootstrap_source,
            contract_source=mock_contract_source,
            mode=EnumHandlerSourceMode.CONTRACT,
        )

        result = await resolver.resolve_handlers()

        # Contract source should be called
        mock_contract_source.discover_handlers.assert_called_once()

        # Bootstrap source should NOT be called in CONTRACT mode
        mock_bootstrap_source.discover_handlers.assert_not_called()

        # Only contract handlers should be in result
        assert len(result.descriptors) == 2, (
            f"Expected 2 contract handlers, got {len(result.descriptors)}"
        )

        handler_ids = {h.handler_id for h in result.descriptors}
        assert handler_ids == {"bootstrap.consul", "contract.custom"}, (
            f"Expected only contract handler IDs, got {handler_ids}"
        )

    @pytest.mark.asyncio
    async def test_contract_only_mode_handler_names(
        self,
        mock_bootstrap_source: MagicMock,
        mock_contract_source: MagicMock,
    ) -> None:
        """Contract mode should return contract handlers with correct metadata.

        Verify that the handlers returned have contract source metadata,
        not bootstrap source metadata.
        """
        from omnibase_infra.runtime.handler_source_resolver import HandlerSourceResolver

        resolver = HandlerSourceResolver(
            bootstrap_source=mock_bootstrap_source,
            contract_source=mock_contract_source,
            mode=EnumHandlerSourceMode.CONTRACT,
        )

        result = await resolver.resolve_handlers()

        # Find the consul handler - should be from contract
        consul_handlers = [
            h for h in result.descriptors if h.handler_id == "bootstrap.consul"
        ]
        assert len(consul_handlers) == 1

        consul_handler = consul_handlers[0]
        assert "(Contract)" in consul_handler.name, (
            f"Handler should be from contract, got name='{consul_handler.name}'"
        )


# =============================================================================
# Test Class: Structured Logging
# =============================================================================


class TestHybridModeStructuredLogging:
    """Tests for structured logging of handler counts in HYBRID mode.

    The resolver should log structured fields for observability:
    - contract_handler_count: Number of handlers from contract source
    - bootstrap_handler_count: Number of handlers from bootstrap source
    - fallback_handler_count: Number of bootstrap handlers used as fallback
    - override_count: Number of bootstrap handlers overridden by contract
    """

    @pytest.mark.asyncio
    async def test_hybrid_mode_logs_handler_counts(
        self,
        mock_bootstrap_source: MagicMock,
        mock_contract_source: MagicMock,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Hybrid mode should log structured fields for handler counts.

        Given:
            - Bootstrap: 2 handlers (consul, vault)
            - Contract: 2 handlers (consul override, custom unique)

        When:
            - Resolve handlers in HYBRID mode

        Then:
            - Logs should include structured fields:
              - contract_handler_count: 2
              - bootstrap_handler_count: 2
              - fallback_handler_count: 1 (vault from bootstrap)
              - resolved_handler_count: 3 (total unique)
        """
        from omnibase_infra.runtime.handler_source_resolver import HandlerSourceResolver

        resolver = HandlerSourceResolver(
            bootstrap_source=mock_bootstrap_source,
            contract_source=mock_contract_source,
            mode=EnumHandlerSourceMode.HYBRID,
        )

        with caplog.at_level(logging.INFO):
            await resolver.resolve_handlers()

        # Find the resolution completion log message
        resolution_logs = [
            record
            for record in caplog.records
            if "handler" in record.message.lower()
            and "resolution" in record.message.lower()
        ]

        assert len(resolution_logs) >= 1, (
            "Expected at least one handler resolution log message"
        )

        # Check for structured logging fields in extra
        found_counts = False
        for record in resolution_logs:
            extra = getattr(record, "__dict__", {})
            if "contract_handler_count" in extra or "bootstrap_handler_count" in extra:
                found_counts = True
                # Verify expected counts
                if "contract_handler_count" in extra:
                    assert extra["contract_handler_count"] == 2
                if "bootstrap_handler_count" in extra:
                    assert extra["bootstrap_handler_count"] == 2
                if "fallback_handler_count" in extra:
                    assert extra["fallback_handler_count"] == 1
                if "resolved_handler_count" in extra:
                    assert extra["resolved_handler_count"] == 3

        assert found_counts, (
            "Expected structured logging fields for handler counts. "
            "The resolver should log contract_handler_count, bootstrap_handler_count, "
            "fallback_handler_count, and resolved_handler_count."
        )

    @pytest.mark.asyncio
    async def test_hybrid_mode_logs_override_count(
        self,
        mock_bootstrap_source: MagicMock,
        mock_contract_source: MagicMock,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Hybrid mode should log the number of handlers overridden by contract.

        Given:
            - Bootstrap: consul, vault
            - Contract: consul (overrides bootstrap.consul)

        When:
            - Resolve handlers in HYBRID mode

        Then:
            - Logs should include override_count: 1
        """
        from omnibase_infra.runtime.handler_source_resolver import HandlerSourceResolver

        resolver = HandlerSourceResolver(
            bootstrap_source=mock_bootstrap_source,
            contract_source=mock_contract_source,
            mode=EnumHandlerSourceMode.HYBRID,
        )

        with caplog.at_level(logging.INFO):
            await resolver.resolve_handlers()

        # Check for override count in logs
        found_override = False
        for record in caplog.records:
            extra = getattr(record, "__dict__", {})
            if "override_count" in extra:
                found_override = True
                assert extra["override_count"] == 1, (
                    f"Expected override_count=1, got {extra['override_count']}"
                )

        assert found_override, (
            "Expected structured logging field 'override_count' for handlers "
            "that were overridden by contract. bootstrap.consul should be overridden."
        )


# =============================================================================
# Test Class: Edge Cases and Error Handling
# =============================================================================


class TestHandlerSourceResolverEdgeCases:
    """Tests for edge cases and error handling in handler resolution."""

    @pytest.mark.asyncio
    async def test_resolver_handles_empty_bootstrap_source(
        self,
        mock_contract_source: MagicMock,
    ) -> None:
        """Resolver should handle empty bootstrap source gracefully.

        Given:
            - Bootstrap source returns no handlers
            - Contract source returns handlers

        When:
            - Resolve handlers in HYBRID mode

        Then:
            - Only contract handlers should be returned
            - No errors should be raised
        """
        from omnibase_infra.runtime.handler_source_resolver import HandlerSourceResolver

        empty_bootstrap = MagicMock()
        empty_bootstrap.source_type = "BOOTSTRAP"
        empty_bootstrap.discover_handlers = AsyncMock(
            return_value=ModelContractDiscoveryResult(
                descriptors=[],
                validation_errors=[],
            )
        )

        resolver = HandlerSourceResolver(
            bootstrap_source=empty_bootstrap,
            contract_source=mock_contract_source,
            mode=EnumHandlerSourceMode.HYBRID,
        )

        result = await resolver.resolve_handlers()

        assert len(result.descriptors) == 2, (
            "Expected contract handlers when bootstrap is empty"
        )

    @pytest.mark.asyncio
    async def test_resolver_handles_empty_both_sources(self) -> None:
        """Resolver should handle both sources being empty.

        Given:
            - Both bootstrap and contract sources return no handlers

        When:
            - Resolve handlers in HYBRID mode

        Then:
            - Empty result should be returned
            - No errors should be raised
        """
        from omnibase_infra.runtime.handler_source_resolver import HandlerSourceResolver

        empty_bootstrap = MagicMock()
        empty_bootstrap.source_type = "BOOTSTRAP"
        empty_bootstrap.discover_handlers = AsyncMock(
            return_value=ModelContractDiscoveryResult(
                descriptors=[],
                validation_errors=[],
            )
        )

        empty_contract = MagicMock()
        empty_contract.source_type = "CONTRACT"
        empty_contract.discover_handlers = AsyncMock(
            return_value=ModelContractDiscoveryResult(
                descriptors=[],
                validation_errors=[],
            )
        )

        resolver = HandlerSourceResolver(
            bootstrap_source=empty_bootstrap,
            contract_source=empty_contract,
            mode=EnumHandlerSourceMode.HYBRID,
        )

        result = await resolver.resolve_handlers()

        assert len(result.descriptors) == 0, (
            "Expected empty result when both sources are empty"
        )
        assert result.validation_errors == [], "Expected no validation errors"

    @pytest.mark.asyncio
    async def test_resolver_validation_errors_are_passed_through(
        self,
        mock_bootstrap_source: MagicMock,
        sample_version: ModelSemVer,
    ) -> None:
        """Resolver should pass through validation errors from sources.

        Given:
            - Contract source has validation errors

        When:
            - Resolve handlers in HYBRID mode

        Then:
            - Validation errors should be included in result
        """
        from omnibase_infra.enums import EnumHandlerErrorType, EnumHandlerSourceType
        from omnibase_infra.models.handlers import ModelHandlerIdentifier
        from omnibase_infra.runtime.handler_source_resolver import HandlerSourceResolver

        # Create a validation error
        validation_error = ModelHandlerValidationError(
            error_type=EnumHandlerErrorType.CONTRACT_PARSE_ERROR,
            rule_id="CONTRACT-001",
            handler_identity=ModelHandlerIdentifier.from_handler_id("broken.handler"),
            source_type=EnumHandlerSourceType.CONTRACT,
            message="Failed to parse contract",
            remediation_hint="Fix YAML syntax",
            file_path="broken/handler_contract.yaml",
        )

        contract_with_errors = MagicMock()
        contract_with_errors.source_type = "CONTRACT"
        contract_with_errors.discover_handlers = AsyncMock(
            return_value=ModelContractDiscoveryResult(
                descriptors=[
                    ModelHandlerDescriptor(
                        handler_id="contract.valid",
                        name="Valid Handler",
                        version=sample_version,
                        handler_kind="compute",
                        input_model="myapp.models.Input",
                        output_model="myapp.models.Output",
                    )
                ],
                validation_errors=[validation_error],
            )
        )

        resolver = HandlerSourceResolver(
            bootstrap_source=mock_bootstrap_source,
            contract_source=contract_with_errors,
            mode=EnumHandlerSourceMode.HYBRID,
        )

        result = await resolver.resolve_handlers()

        assert len(result.validation_errors) == 1, (
            "Expected validation errors to be passed through"
        )
        assert result.validation_errors[0].message == "Failed to parse contract"


# =============================================================================
# Test Class: Protocol Compliance
# =============================================================================


class TestHandlerSourceResolverProtocol:
    """Tests for HandlerSourceResolver protocol compliance and interface."""

    @pytest.mark.asyncio
    async def test_resolver_returns_model_contract_discovery_result(
        self,
        mock_bootstrap_source: MagicMock,
        mock_contract_source: MagicMock,
    ) -> None:
        """Resolver should return ModelContractDiscoveryResult.

        The result type should be consistent with the individual sources
        to enable unified handling by the runtime.
        """
        from omnibase_infra.runtime.handler_source_resolver import HandlerSourceResolver

        resolver = HandlerSourceResolver(
            bootstrap_source=mock_bootstrap_source,
            contract_source=mock_contract_source,
            mode=EnumHandlerSourceMode.HYBRID,
        )

        result = await resolver.resolve_handlers()

        assert isinstance(result, ModelContractDiscoveryResult), (
            f"Expected ModelContractDiscoveryResult, got {type(result).__name__}"
        )
        assert hasattr(result, "descriptors")
        assert hasattr(result, "validation_errors")

    @pytest.mark.asyncio
    async def test_resolver_has_mode_property(
        self,
        mock_bootstrap_source: MagicMock,
        mock_contract_source: MagicMock,
    ) -> None:
        """Resolver should expose mode property for introspection."""
        from omnibase_infra.runtime.handler_source_resolver import HandlerSourceResolver

        resolver = HandlerSourceResolver(
            bootstrap_source=mock_bootstrap_source,
            contract_source=mock_contract_source,
            mode=EnumHandlerSourceMode.HYBRID,
        )

        assert hasattr(resolver, "mode")
        assert resolver.mode == EnumHandlerSourceMode.HYBRID


__all__ = [
    "TestBootstrapOnlyMode",
    "TestContractOnlyMode",
    "TestHandlerSourceResolverEdgeCases",
    "TestHandlerSourceResolverProtocol",
    "TestHybridModeContractWins",
    "TestHybridModeFallback",
    "TestHybridModeStructuredLogging",
]
