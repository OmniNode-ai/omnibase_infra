# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Unit tests for KafkaContractSource cache-based discovery.

Tests the KafkaContractSource functionality including:
- Cache management for contract registration/deregistration events
- Transformation of contract YAML to ModelHandlerDescriptor instances
- Error handling for malformed contracts
- Graceful mode vs strict mode behavior

Related:
    - OMN-1654: KafkaContractSource (cache + discovery)
    - src/omnibase_infra/runtime/kafka_contract_source.py

Expected Behavior:
    KafkaContractSource implements ProtocolContractSource from omnibase_infra.
    It maintains an in-memory cache of handler descriptors derived from contract
    registration events received via Kafka. The discover_handlers() method
    returns the current cache state without performing I/O.

    The source_type property returns "KAFKA_EVENTS" as per the protocol.
"""

from __future__ import annotations

from uuid import UUID, uuid4

import pytest

from omnibase_core.models.errors import ModelOnexError
from omnibase_core.models.events import (
    ModelContractDeregisteredEvent,
    ModelContractRegisteredEvent,
)
from omnibase_core.models.primitives import ModelSemVer
from omnibase_infra.enums import EnumHandlerErrorType
from omnibase_infra.models.handlers import (
    ModelContractDiscoveryResult,
    ModelHandlerDescriptor,
)
from omnibase_infra.runtime.kafka_contract_source import (
    MAX_CONTRACT_SIZE,
    KafkaContractSource,
)
from omnibase_infra.runtime.protocol_contract_source import ProtocolContractSource

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
  node_archetype: "effect"
input_model: "omnibase_infra.models.types.JsonDict"
output_model: "omnibase_core.models.dispatch.ModelHandlerOutput"
metadata:
  handler_class: "omnibase_infra.handlers.TestHandler"
"""

VALID_COMPUTE_CONTRACT_YAML = """
handler_id: "compute.validation.handler"
name: "Validation Compute Handler"
contract_version:
  major: 2
  minor: 1
  patch: 3
description: "A compute handler for validation"
descriptor:
  node_archetype: "compute"
input_model: "test.models.ValidationInput"
output_model: "test.models.ValidationOutput"
metadata:
  handler_class: "test.handlers.ValidationHandler"
"""

MINIMAL_HANDLER_CONTRACT_YAML = """
handler_id: "{handler_id}"
name: "{name}"
contract_version:
  major: 1
  minor: 0
  patch: 0
descriptor:
  node_archetype: "compute"
input_model: "test.models.Input"
output_model: "test.models.Output"
"""


# =============================================================================
# Source Type Tests
# =============================================================================


class TestKafkaContractSourceType:
    """Tests for source_type property."""

    def test_source_type_returns_kafka_events(self) -> None:
        """source_type should return 'KAFKA_EVENTS' to identify the source type.

        The source_type is used for observability and debugging purposes only.
        The runtime MUST NOT branch on this value.
        """
        source = KafkaContractSource()

        assert source.source_type == "KAFKA_EVENTS"

    def test_implements_protocol_contract_source(self) -> None:
        """KafkaContractSource should implement ProtocolContractSource.

        The implementation must satisfy ProtocolContractSource with:
        - source_type property returning "KAFKA_EVENTS"
        - async discover_handlers() method returning ModelContractDiscoveryResult

        This test uses two complementary verification approaches:
        1. isinstance() check - enabled by @runtime_checkable decorator on protocol
        2. Duck typing checks - ONEX convention for explicit attribute verification
        """
        source = KafkaContractSource()

        # Protocol compliance via runtime_checkable (structural subtyping)
        assert isinstance(source, ProtocolContractSource)

        # Protocol compliance check via duck typing (ONEX convention)
        assert hasattr(source, "source_type")
        assert hasattr(source, "discover_handlers")
        assert callable(source.discover_handlers)


# =============================================================================
# Discovery Tests
# =============================================================================


class TestKafkaContractSourceDiscovery:
    """Tests for discover_handlers method."""

    @pytest.mark.asyncio
    async def test_discover_handlers_empty_cache(self) -> None:
        """discover_handlers should return empty result when cache is empty.

        When no contracts have been registered, the source should return
        an empty ModelContractDiscoveryResult with no descriptors and
        no validation errors.
        """
        source = KafkaContractSource()
        result = await source.discover_handlers()

        assert isinstance(result, ModelContractDiscoveryResult)
        assert result.descriptors == []
        assert result.validation_errors == []

    @pytest.mark.asyncio
    async def test_discover_handlers_returns_cached_descriptors(self) -> None:
        """discover_handlers should return all cached descriptors.

        After registering contracts via on_contract_registered(), the
        discover_handlers() method should return all cached descriptors.
        """
        source = KafkaContractSource()

        # Register a contract
        success = source.on_contract_registered(
            node_name="effect.test.handler",
            contract_yaml=VALID_HANDLER_CONTRACT_YAML,
        )
        assert success is True

        # Discover should return the cached descriptor
        result = await source.discover_handlers()

        assert len(result.descriptors) == 1
        assert len(result.validation_errors) == 0

        descriptor = result.descriptors[0]
        assert isinstance(descriptor, ModelHandlerDescriptor)
        assert descriptor.handler_id == "effect.test.handler"
        assert descriptor.name == "Test Effect Handler"
        assert str(descriptor.version) == "1.0.0"
        assert descriptor.handler_kind == "effect"
        assert descriptor.handler_class == "omnibase_infra.handlers.TestHandler"

    @pytest.mark.asyncio
    async def test_discover_handlers_multiple_contracts(self) -> None:
        """discover_handlers should return all registered contracts.

        Multiple contracts registered should all be returned.
        """
        source = KafkaContractSource()

        # Register multiple contracts
        source.on_contract_registered(
            node_name="effect.test.handler",
            contract_yaml=VALID_HANDLER_CONTRACT_YAML,
        )
        source.on_contract_registered(
            node_name="compute.validation.handler",
            contract_yaml=VALID_COMPUTE_CONTRACT_YAML,
        )

        result = await source.discover_handlers()

        assert len(result.descriptors) == 2
        assert len(result.validation_errors) == 0

        handler_ids = {d.handler_id for d in result.descriptors}
        assert handler_ids == {"effect.test.handler", "compute.validation.handler"}

    @pytest.mark.asyncio
    async def test_discover_handlers_is_idempotent(self) -> None:
        """discover_handlers should return consistent results on multiple calls.

        Calling discover_handlers() multiple times should return the same
        descriptors (idempotent behavior).
        """
        source = KafkaContractSource()

        source.on_contract_registered(
            node_name="effect.test.handler",
            contract_yaml=VALID_HANDLER_CONTRACT_YAML,
        )

        result1 = await source.discover_handlers()
        result2 = await source.discover_handlers()

        assert len(result1.descriptors) == len(result2.descriptors)
        assert result1.descriptors[0].handler_id == result2.descriptors[0].handler_id


# =============================================================================
# Registration Tests
# =============================================================================


class TestKafkaContractSourceRegistration:
    """Tests for on_contract_registered method."""

    def test_on_contract_registered_success(self) -> None:
        """on_contract_registered should cache valid contracts.

        A valid contract YAML should be parsed and cached successfully,
        returning True.
        """
        source = KafkaContractSource()

        success = source.on_contract_registered(
            node_name="effect.test.handler",
            contract_yaml=VALID_HANDLER_CONTRACT_YAML,
        )

        assert success is True
        assert source.cached_count == 1

    def test_on_contract_registered_with_correlation_id(self) -> None:
        """on_contract_registered should accept optional correlation_id.

        The correlation_id should be used for tracing and logging.
        """
        source = KafkaContractSource()
        correlation_id = uuid4()

        success = source.on_contract_registered(
            node_name="effect.test.handler",
            contract_yaml=VALID_HANDLER_CONTRACT_YAML,
            correlation_id=correlation_id,
        )

        assert success is True

    def test_on_contract_registered_overwrites_existing(self) -> None:
        """on_contract_registered should overwrite existing cache entry.

        Registering a contract with the same node_name should replace
        the previous entry.
        """
        source = KafkaContractSource()

        # Register initial contract
        source.on_contract_registered(
            node_name="test.node",
            contract_yaml=MINIMAL_HANDLER_CONTRACT_YAML.format(
                handler_id="handler.v1",
                name="Handler V1",
            ),
        )
        assert source.cached_count == 1

        # Register updated contract with same node_name
        source.on_contract_registered(
            node_name="test.node",
            contract_yaml=MINIMAL_HANDLER_CONTRACT_YAML.format(
                handler_id="handler.v2",
                name="Handler V2",
            ),
        )

        # Should still be 1 entry (overwritten)
        assert source.cached_count == 1

    def test_on_contract_registered_invalid_yaml_graceful_mode(self) -> None:
        """Invalid YAML in graceful mode should return False, not raise.

        When graceful_mode=True (default) and contract parsing fails,
        the error should be collected and False returned.
        """
        source = KafkaContractSource(graceful_mode=True)

        invalid_yaml = "this is: [not valid: yaml"

        success = source.on_contract_registered(
            node_name="invalid.handler",
            contract_yaml=invalid_yaml,
        )

        assert success is False
        assert source.cached_count == 0

    def test_on_contract_registered_invalid_yaml_strict_mode(self) -> None:
        """Invalid YAML in strict mode should raise exception.

        When graceful_mode=False and contract parsing fails, an exception
        should be raised.
        """
        source = KafkaContractSource(graceful_mode=False)

        invalid_yaml = "this is: [not valid: yaml"

        with pytest.raises(ModelOnexError, match="KAFKA_CONTRACT_001"):
            source.on_contract_registered(
                node_name="invalid.handler",
                contract_yaml=invalid_yaml,
            )

    def test_on_contract_registered_missing_fields(self) -> None:
        """Contracts missing required fields should fail validation."""
        source = KafkaContractSource(graceful_mode=True)

        incomplete_yaml = """
name: "Incomplete Handler"
version: "1.0.0"
"""  # Missing handler_id, input_model, output_model

        success = source.on_contract_registered(
            node_name="incomplete.handler",
            contract_yaml=incomplete_yaml,
        )

        assert success is False
        assert source.cached_count == 0

    def test_on_contract_registered_empty_yaml(self) -> None:
        """Empty YAML content should fail validation."""
        source = KafkaContractSource(graceful_mode=True)

        success = source.on_contract_registered(
            node_name="empty.handler",
            contract_yaml="",
        )

        assert success is False
        assert source.cached_count == 0

    def test_on_contract_registered_size_limit(self) -> None:
        """Contracts exceeding MAX_CONTRACT_SIZE should be rejected."""
        source = KafkaContractSource(graceful_mode=True)

        oversized_yaml = "x" * (MAX_CONTRACT_SIZE + 1)

        success = source.on_contract_registered(
            node_name="oversized.handler",
            contract_yaml=oversized_yaml,
        )

        assert success is False
        assert source.cached_count == 0


# =============================================================================
# Deregistration Tests
# =============================================================================


class TestKafkaContractSourceDeregistration:
    """Tests for on_contract_deregistered method."""

    def test_on_contract_deregistered_removes_from_cache(self) -> None:
        """on_contract_deregistered should remove descriptor from cache.

        After deregistration, the descriptor should no longer be returned
        by discover_handlers().
        """
        source = KafkaContractSource()

        # Register then deregister
        source.on_contract_registered(
            node_name="effect.test.handler",
            contract_yaml=VALID_HANDLER_CONTRACT_YAML,
        )
        assert source.cached_count == 1

        removed = source.on_contract_deregistered(node_name="effect.test.handler")

        assert removed is True
        assert source.cached_count == 0

    def test_on_contract_deregistered_unknown_node(self) -> None:
        """on_contract_deregistered should return False for unknown nodes.

        Attempting to deregister a node that isn't cached should return False.
        """
        source = KafkaContractSource()

        removed = source.on_contract_deregistered(node_name="unknown.handler")

        assert removed is False

    def test_on_contract_deregistered_with_correlation_id(self) -> None:
        """on_contract_deregistered should accept optional correlation_id."""
        source = KafkaContractSource()
        correlation_id = uuid4()

        source.on_contract_registered(
            node_name="effect.test.handler",
            contract_yaml=VALID_HANDLER_CONTRACT_YAML,
        )

        removed = source.on_contract_deregistered(
            node_name="effect.test.handler",
            correlation_id=correlation_id,
        )

        assert removed is True

    @pytest.mark.asyncio
    async def test_deregistration_removes_from_discover_results(self) -> None:
        """Deregistered contracts should not appear in discover_handlers()."""
        source = KafkaContractSource()

        # Register two contracts
        source.on_contract_registered(
            node_name="handler.one",
            contract_yaml=MINIMAL_HANDLER_CONTRACT_YAML.format(
                handler_id="handler.one",
                name="Handler One",
            ),
        )
        source.on_contract_registered(
            node_name="handler.two",
            contract_yaml=MINIMAL_HANDLER_CONTRACT_YAML.format(
                handler_id="handler.two",
                name="Handler Two",
            ),
        )

        # Deregister one
        source.on_contract_deregistered(node_name="handler.one")

        # Should only return remaining contract
        result = await source.discover_handlers()

        assert len(result.descriptors) == 1
        assert result.descriptors[0].handler_id == "handler.two"


# =============================================================================
# Error Collection Tests
# =============================================================================


class TestKafkaContractSourceErrorCollection:
    """Tests for error collection in graceful mode."""

    @pytest.mark.asyncio
    async def test_errors_collected_during_registration(self) -> None:
        """Registration errors should be collected and returned on discover.

        In graceful mode, parsing errors should be collected in pending_errors
        and returned on the next discover_handlers() call.
        """
        source = KafkaContractSource(graceful_mode=True)

        # Register invalid contract
        source.on_contract_registered(
            node_name="invalid.handler",
            contract_yaml="invalid: yaml: content",
        )

        result = await source.discover_handlers()

        assert len(result.descriptors) == 0
        assert len(result.validation_errors) == 1

        error = result.validation_errors[0]
        assert error.error_type == EnumHandlerErrorType.CONTRACT_PARSE_ERROR
        assert error.rule_id == "KAFKA-001"
        assert "invalid.handler" in error.message

    @pytest.mark.asyncio
    async def test_errors_cleared_after_discover(self) -> None:
        """Pending errors should be cleared after discover_handlers().

        Errors are only returned once - subsequent discover calls should
        not return previously reported errors.
        """
        source = KafkaContractSource(graceful_mode=True)

        # Register invalid contract
        source.on_contract_registered(
            node_name="invalid.handler",
            contract_yaml="invalid: yaml: content",
        )

        # First discover returns the error
        result1 = await source.discover_handlers()
        assert len(result1.validation_errors) == 1

        # Second discover returns empty errors
        result2 = await source.discover_handlers()
        assert len(result2.validation_errors) == 0

    @pytest.mark.asyncio
    async def test_mixed_success_and_failures(self) -> None:
        """Successful and failed registrations should both be tracked.

        Valid contracts should be cached while invalid ones generate errors.
        """
        source = KafkaContractSource(graceful_mode=True)

        # Register valid contract
        source.on_contract_registered(
            node_name="valid.handler",
            contract_yaml=VALID_HANDLER_CONTRACT_YAML,
        )

        # Register invalid contract
        source.on_contract_registered(
            node_name="invalid.handler",
            contract_yaml="not valid yaml",
        )

        result = await source.discover_handlers()

        # Should have one valid descriptor and one error
        assert len(result.descriptors) == 1
        assert result.descriptors[0].handler_id == "effect.test.handler"
        assert len(result.validation_errors) == 1


# =============================================================================
# Contract Path Tests
# =============================================================================


class TestKafkaContractSourceContractPath:
    """Tests for contract_path generation in descriptors."""

    @pytest.mark.asyncio
    async def test_contract_path_format(self) -> None:
        """contract_path should use kafka:// URI format.

        The contract_path should indicate the source location using
        kafka://{environment}/contracts/{node_name} format for traceability.
        """
        source = KafkaContractSource(environment="prod")

        source.on_contract_registered(
            node_name="effect.test.handler",
            contract_yaml=VALID_HANDLER_CONTRACT_YAML,
        )

        result = await source.discover_handlers()

        assert len(result.descriptors) == 1
        contract_path = result.descriptors[0].contract_path
        assert contract_path.startswith("kafka://")
        assert "prod" in contract_path
        assert "effect.test.handler" in contract_path


# =============================================================================
# Clear Cache Tests
# =============================================================================


class TestKafkaContractSourceClearCache:
    """Tests for clear_cache utility method."""

    def test_clear_cache_removes_all_descriptors(self) -> None:
        """clear_cache should remove all cached descriptors.

        After clearing, discover_handlers() should return empty results.
        """
        source = KafkaContractSource()

        # Register some contracts
        source.on_contract_registered(
            node_name="handler.one",
            contract_yaml=MINIMAL_HANDLER_CONTRACT_YAML.format(
                handler_id="handler.one",
                name="Handler One",
            ),
        )
        source.on_contract_registered(
            node_name="handler.two",
            contract_yaml=MINIMAL_HANDLER_CONTRACT_YAML.format(
                handler_id="handler.two",
                name="Handler Two",
            ),
        )
        assert source.cached_count == 2

        # Clear cache
        cleared = source.clear_cache()

        assert cleared == 2
        assert source.cached_count == 0

    def test_clear_cache_empty_returns_zero(self) -> None:
        """clear_cache on empty cache should return 0."""
        source = KafkaContractSource()

        cleared = source.clear_cache()

        assert cleared == 0

    @pytest.mark.asyncio
    async def test_clear_cache_also_clears_pending_errors(self) -> None:
        """clear_cache should also clear pending validation errors."""
        source = KafkaContractSource(graceful_mode=True)

        # Generate an error
        source.on_contract_registered(
            node_name="invalid",
            contract_yaml="invalid yaml",
        )

        # Clear cache (also clears pending errors)
        source.clear_cache()

        # Discover should return no errors
        result = await source.discover_handlers()
        assert len(result.validation_errors) == 0


# =============================================================================
# Configuration Tests
# =============================================================================


class TestKafkaContractSourceConfiguration:
    """Tests for configuration options."""

    def test_default_environment(self) -> None:
        """Default environment should be 'dev'."""
        source = KafkaContractSource()

        assert source._environment == "dev"

    def test_custom_environment(self) -> None:
        """Custom environment should be stored."""
        source = KafkaContractSource(environment="staging")

        assert source._environment == "staging"

    def test_default_graceful_mode(self) -> None:
        """Default graceful_mode should be True."""
        source = KafkaContractSource()

        assert source._graceful_mode is True

    def test_correlation_id_generated(self) -> None:
        """A unique correlation_id should be generated on initialization."""
        source1 = KafkaContractSource()
        source2 = KafkaContractSource()

        assert source1._correlation_id is not None
        assert source2._correlation_id is not None
        assert isinstance(source1._correlation_id, UUID)
        assert isinstance(source2._correlation_id, UUID)
        # Each instance should have unique correlation ID
        assert source1._correlation_id != source2._correlation_id


# =============================================================================
# Handler Class Extraction Tests
# =============================================================================


class TestKafkaContractSourceHandlerClass:
    """Tests for handler_class extraction from contracts."""

    @pytest.mark.asyncio
    async def test_handler_class_from_metadata(self) -> None:
        """handler_class should be extracted from metadata section."""
        source = KafkaContractSource()

        source.on_contract_registered(
            node_name="effect.test.handler",
            contract_yaml=VALID_HANDLER_CONTRACT_YAML,
        )

        result = await source.discover_handlers()

        assert (
            result.descriptors[0].handler_class == "omnibase_infra.handlers.TestHandler"
        )

    @pytest.mark.asyncio
    async def test_handler_class_from_compute_contract(self) -> None:
        """handler_class should be extracted from metadata in compute contracts.

        Note: ModelHandlerContract uses extra='forbid', so handler_class must
        be in the metadata section, not at the root level.
        """
        source = KafkaContractSource()

        source.on_contract_registered(
            node_name="compute.validation.handler",
            contract_yaml=VALID_COMPUTE_CONTRACT_YAML,
        )

        result = await source.discover_handlers()

        assert result.descriptors[0].handler_class == "test.handlers.ValidationHandler"

    @pytest.mark.asyncio
    async def test_handler_class_missing_is_none(self) -> None:
        """handler_class should be None when not specified."""
        source = KafkaContractSource()

        source.on_contract_registered(
            node_name="handler.one",
            contract_yaml=MINIMAL_HANDLER_CONTRACT_YAML.format(
                handler_id="handler.one",
                name="Handler One",
            ),
        )

        result = await source.discover_handlers()

        assert result.descriptors[0].handler_class is None


# =============================================================================
# Typed Event Handler Tests
# =============================================================================


class TestKafkaContractSourceTypedEvents:
    """Tests for typed event handler methods using omnibase_core event models."""

    @pytest.mark.asyncio
    async def test_handle_registered_event_success(self) -> None:
        """handle_registered_event should process typed registration events.

        The typed event handler extracts fields from ModelContractRegisteredEvent
        and delegates to on_contract_registered().
        """
        source = KafkaContractSource()

        event = ModelContractRegisteredEvent(
            node_name="effect.test.handler",
            node_version=ModelSemVer(major=1, minor=0, patch=0),
            contract_hash="abc123",
            contract_yaml=VALID_HANDLER_CONTRACT_YAML,
        )

        success = source.handle_registered_event(event)

        assert success is True
        assert source.cached_count == 1

        result = await source.discover_handlers()
        assert result.descriptors[0].handler_id == "effect.test.handler"

    @pytest.mark.asyncio
    async def test_handle_registered_event_uses_correlation_id(self) -> None:
        """handle_registered_event should propagate correlation_id from event."""
        source = KafkaContractSource()
        correlation_id = uuid4()

        event = ModelContractRegisteredEvent(
            node_name="effect.test.handler",
            node_version=ModelSemVer(major=1, minor=0, patch=0),
            contract_hash="abc123",
            contract_yaml=VALID_HANDLER_CONTRACT_YAML,
            correlation_id=correlation_id,
        )

        success = source.handle_registered_event(event)

        assert success is True

    def test_handle_deregistered_event_removes_from_cache(self) -> None:
        """handle_deregistered_event should remove descriptor from cache."""
        source = KafkaContractSource()

        # First register a contract
        source.on_contract_registered(
            node_name="effect.test.handler",
            contract_yaml=VALID_HANDLER_CONTRACT_YAML,
        )
        assert source.cached_count == 1

        # Then deregister using typed event
        event = ModelContractDeregisteredEvent(
            node_name="effect.test.handler",
            node_version=ModelSemVer(major=1, minor=0, patch=0),
            reason="shutdown",
        )

        removed = source.handle_deregistered_event(event)

        assert removed is True
        assert source.cached_count == 0

    def test_handle_deregistered_event_unknown_node(self) -> None:
        """handle_deregistered_event should return False for unknown nodes."""
        source = KafkaContractSource()

        event = ModelContractDeregisteredEvent(
            node_name="unknown.handler",
            node_version=ModelSemVer(major=1, minor=0, patch=0),
            reason="shutdown",
        )

        removed = source.handle_deregistered_event(event)

        assert removed is False

    def test_handle_registered_event_invalid_yaml(self) -> None:
        """handle_registered_event should handle invalid YAML gracefully."""
        source = KafkaContractSource(graceful_mode=True)

        event = ModelContractRegisteredEvent(
            node_name="invalid.handler",
            node_version=ModelSemVer(major=1, minor=0, patch=0),
            contract_hash="abc123",
            contract_yaml="invalid: [yaml content",
        )

        success = source.handle_registered_event(event)

        assert success is False
        assert source.cached_count == 0


# =============================================================================
# Export Tests
# =============================================================================


class TestKafkaContractSourceExports:
    """Tests for module exports."""

    def test_kafka_contract_source_exported_from_runtime(self) -> None:
        """KafkaContractSource should be exported from runtime module."""
        from omnibase_infra.runtime import KafkaContractSource as ExportedSource

        assert ExportedSource is KafkaContractSource

    def test_max_contract_size_exported(self) -> None:
        """MAX_CONTRACT_SIZE should be exported (10MB)."""
        assert MAX_CONTRACT_SIZE == 10 * 1024 * 1024

    def test_topic_constants_exported(self) -> None:
        """Topic suffix constants should be re-exported from omnibase_core."""
        # Verify they match omnibase_core canonical values
        from omnibase_core.constants import (
            TOPIC_SUFFIX_CONTRACT_DEREGISTERED as CORE_DEREG,
        )
        from omnibase_core.constants import (
            TOPIC_SUFFIX_CONTRACT_REGISTERED as CORE_REG,
        )
        from omnibase_infra.runtime.kafka_contract_source import (
            TOPIC_SUFFIX_CONTRACT_DEREGISTERED,
            TOPIC_SUFFIX_CONTRACT_REGISTERED,
        )

        assert TOPIC_SUFFIX_CONTRACT_REGISTERED == CORE_REG
        assert TOPIC_SUFFIX_CONTRACT_DEREGISTERED == CORE_DEREG
        assert "contract-registered" in TOPIC_SUFFIX_CONTRACT_REGISTERED
        assert "contract-deregistered" in TOPIC_SUFFIX_CONTRACT_DEREGISTERED
