# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Unit Tests for HandlerSourceResolver KAFKA_EVENTS Mode.

This module contains unit tests for the KAFKA_EVENTS resolution path in
HandlerSourceResolver, added as part of OMN-1654.

The KAFKA_EVENTS mode delegates handler resolution to a KafkaContractSource
instance that returns cached descriptors from contract registration events.
This is a beta cache-only implementation where discovered contracts take
effect on the next runtime restart.

Test Categories:
    - KAFKA_EVENTS Mode Resolution: Verifies descriptors returned from cache
    - Empty Cache Handling: Verifies empty cache returns empty list
    - Validation Error Propagation: Verifies errors are passed through
    - Structured Logging: Verifies proper log fields

Related:
    - OMN-1654: KafkaContractSource (cache + discovery)
    - HandlerSourceResolver: src/omnibase_infra/runtime/handler_source_resolver.py
    - KafkaContractSource: src/omnibase_infra/runtime/kafka_contract_source.py

See Also:
    - test_handler_source_mode.py: Tests for BOOTSTRAP, CONTRACT, HYBRID modes
    - test_kafka_contract_source.py: Tests for KafkaContractSource cache operations
"""

from __future__ import annotations

import logging
from unittest.mock import AsyncMock, MagicMock

import pytest

from omnibase_core.models.primitives import ModelSemVer
from omnibase_infra.enums import EnumHandlerErrorType, EnumHandlerSourceType
from omnibase_infra.enums.enum_handler_source_mode import EnumHandlerSourceMode
from omnibase_infra.models.errors import ModelHandlerValidationError
from omnibase_infra.models.handlers import (
    ModelContractDiscoveryResult,
    ModelHandlerDescriptor,
    ModelHandlerIdentifier,
)

# Forward Reference Resolution:
# ModelContractDiscoveryResult uses a forward reference to ModelHandlerValidationError.
# Since we import ModelHandlerValidationError above, we can call model_rebuild() here
# to resolve the forward reference. This call is idempotent - multiple calls are harmless.
ModelContractDiscoveryResult.model_rebuild()


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def sample_version() -> ModelSemVer:
    """Create a sample version for test descriptors."""
    return ModelSemVer(major=1, minor=0, patch=0)


@pytest.fixture
def kafka_effect_descriptor(sample_version: ModelSemVer) -> ModelHandlerDescriptor:
    """Create a handler descriptor representing a Kafka-cached effect handler.

    This represents a handler discovered via KafkaContractSource from
    contract registration events.
    """
    return ModelHandlerDescriptor(
        handler_id="effect.kafka.handler",
        name="Kafka Effect Handler",
        version=sample_version,
        handler_kind="effect",
        input_model="omnibase_infra.models.types.JsonDict",
        output_model="omnibase_core.models.dispatch.ModelHandlerOutput",
        description="Handler discovered via Kafka contract events",
        handler_class="omnibase_infra.handlers.KafkaEffectHandler",
        contract_path="kafka://dev/contracts/effect.kafka.handler",
    )


@pytest.fixture
def kafka_compute_descriptor(sample_version: ModelSemVer) -> ModelHandlerDescriptor:
    """Create a handler descriptor representing a Kafka-cached compute handler."""
    return ModelHandlerDescriptor(
        handler_id="compute.kafka.handler",
        name="Kafka Compute Handler",
        version=sample_version,
        handler_kind="compute",
        input_model="test.models.ComputeInput",
        output_model="test.models.ComputeOutput",
        description="Compute handler discovered via Kafka contract events",
        handler_class="test.handlers.KafkaComputeHandler",
        contract_path="kafka://dev/contracts/compute.kafka.handler",
    )


@pytest.fixture
def mock_kafka_contract_source(
    kafka_effect_descriptor: ModelHandlerDescriptor,
    kafka_compute_descriptor: ModelHandlerDescriptor,
) -> MagicMock:
    """Create a mock KafkaContractSource with cached descriptors.

    Simulates a KafkaContractSource that has received and cached
    contract registration events for two handlers.
    """
    mock_source = MagicMock()
    mock_source.source_type = "KAFKA_EVENTS"
    mock_source.discover_handlers = AsyncMock(
        return_value=ModelContractDiscoveryResult(
            descriptors=[kafka_effect_descriptor, kafka_compute_descriptor],
            validation_errors=[],
        )
    )
    return mock_source


@pytest.fixture
def mock_empty_kafka_source() -> MagicMock:
    """Create a mock KafkaContractSource with empty cache.

    Simulates a KafkaContractSource that has not received any
    contract registration events.
    """
    mock_source = MagicMock()
    mock_source.source_type = "KAFKA_EVENTS"
    mock_source.discover_handlers = AsyncMock(
        return_value=ModelContractDiscoveryResult(
            descriptors=[],
            validation_errors=[],
        )
    )
    return mock_source


@pytest.fixture
def mock_bootstrap_source_minimal() -> MagicMock:
    """Create a minimal mock bootstrap source.

    In KAFKA_EVENTS mode, the bootstrap source is not used, but the
    resolver constructor still requires it. This provides a minimal mock.
    """
    mock_source = MagicMock()
    mock_source.source_type = "BOOTSTRAP"
    mock_source.discover_handlers = AsyncMock(
        return_value=ModelContractDiscoveryResult(
            descriptors=[],
            validation_errors=[],
        )
    )
    return mock_source


# =============================================================================
# Test Class: KAFKA_EVENTS Mode Resolution
# =============================================================================


class TestKafkaEventsModeResolution:
    """Tests for KAFKA_EVENTS mode handler resolution.

    In KAFKA_EVENTS mode, the resolver delegates entirely to the contract_source
    (expected to be a KafkaContractSource) and returns its cached descriptors.
    The bootstrap source is not used in this mode.
    """

    @pytest.mark.asyncio
    async def test_kafka_events_mode_returns_cached_descriptors(
        self,
        mock_bootstrap_source_minimal: MagicMock,
        mock_kafka_contract_source: MagicMock,
    ) -> None:
        """KAFKA_EVENTS mode should return descriptors from the Kafka cache.

        Given:
            - KafkaContractSource has cached two handler descriptors

        When:
            - Resolve handlers in KAFKA_EVENTS mode

        Then:
            - Both cached descriptors should be returned
            - Bootstrap source should NOT be called
        """
        from omnibase_infra.runtime.handler_source_resolver import HandlerSourceResolver

        resolver = HandlerSourceResolver(
            bootstrap_source=mock_bootstrap_source_minimal,
            contract_source=mock_kafka_contract_source,
            mode=EnumHandlerSourceMode.KAFKA_EVENTS,
        )

        result = await resolver.resolve_handlers()

        # Contract source (Kafka) should be called
        mock_kafka_contract_source.discover_handlers.assert_called_once()

        # Bootstrap source should NOT be called
        mock_bootstrap_source_minimal.discover_handlers.assert_not_called()

        # Should return both cached descriptors
        assert len(result.descriptors) == 2, (
            f"Expected 2 cached descriptors, got {len(result.descriptors)}"
        )

        handler_ids = {h.handler_id for h in result.descriptors}
        assert handler_ids == {"effect.kafka.handler", "compute.kafka.handler"}, (
            f"Expected Kafka-cached handler IDs, got {handler_ids}"
        )

    @pytest.mark.asyncio
    async def test_kafka_events_mode_descriptor_metadata(
        self,
        mock_bootstrap_source_minimal: MagicMock,
        mock_kafka_contract_source: MagicMock,
    ) -> None:
        """KAFKA_EVENTS mode should preserve descriptor metadata from cache.

        Verify that handler descriptors returned have the correct metadata
        including contract_path with kafka:// URI scheme.
        """
        from omnibase_infra.runtime.handler_source_resolver import HandlerSourceResolver

        resolver = HandlerSourceResolver(
            bootstrap_source=mock_bootstrap_source_minimal,
            contract_source=mock_kafka_contract_source,
            mode=EnumHandlerSourceMode.KAFKA_EVENTS,
        )

        result = await resolver.resolve_handlers()

        # Find the effect handler
        effect_handlers = [
            h for h in result.descriptors if h.handler_id == "effect.kafka.handler"
        ]
        assert len(effect_handlers) == 1

        handler = effect_handlers[0]
        assert handler.name == "Kafka Effect Handler"
        assert handler.handler_kind == "effect"
        assert handler.contract_path is not None
        assert handler.contract_path.startswith("kafka://"), (
            f"Expected kafka:// URI scheme, got {handler.contract_path}"
        )

    @pytest.mark.asyncio
    async def test_kafka_events_mode_exposes_mode_property(
        self,
        mock_bootstrap_source_minimal: MagicMock,
        mock_kafka_contract_source: MagicMock,
    ) -> None:
        """Resolver should expose KAFKA_EVENTS mode via property."""
        from omnibase_infra.runtime.handler_source_resolver import HandlerSourceResolver

        resolver = HandlerSourceResolver(
            bootstrap_source=mock_bootstrap_source_minimal,
            contract_source=mock_kafka_contract_source,
            mode=EnumHandlerSourceMode.KAFKA_EVENTS,
        )

        assert resolver.mode == EnumHandlerSourceMode.KAFKA_EVENTS


# =============================================================================
# Test Class: KAFKA_EVENTS Empty Cache Handling
# =============================================================================


class TestKafkaEventsModeEmptyCache:
    """Tests for KAFKA_EVENTS mode with empty cache.

    When no contracts have been registered via Kafka, the source returns
    an empty result. This tests graceful handling of that scenario.
    """

    @pytest.mark.asyncio
    async def test_kafka_events_empty_cache_returns_empty_list(
        self,
        mock_bootstrap_source_minimal: MagicMock,
        mock_empty_kafka_source: MagicMock,
    ) -> None:
        """KAFKA_EVENTS mode with empty cache should return empty descriptors.

        Given:
            - KafkaContractSource cache is empty (no contracts registered)

        When:
            - Resolve handlers in KAFKA_EVENTS mode

        Then:
            - Empty descriptors list should be returned
            - No validation errors should be present
        """
        from omnibase_infra.runtime.handler_source_resolver import HandlerSourceResolver

        resolver = HandlerSourceResolver(
            bootstrap_source=mock_bootstrap_source_minimal,
            contract_source=mock_empty_kafka_source,
            mode=EnumHandlerSourceMode.KAFKA_EVENTS,
        )

        result = await resolver.resolve_handlers()

        assert len(result.descriptors) == 0, (
            "Expected empty descriptors list for empty Kafka cache"
        )
        assert len(result.validation_errors) == 0, (
            "Expected no validation errors for empty cache"
        )

    @pytest.mark.asyncio
    async def test_kafka_events_empty_cache_source_still_called(
        self,
        mock_bootstrap_source_minimal: MagicMock,
        mock_empty_kafka_source: MagicMock,
    ) -> None:
        """KAFKA_EVENTS mode should call discover_handlers even on empty cache.

        The source should still be queried to check for cached descriptors,
        even if the result is empty.
        """
        from omnibase_infra.runtime.handler_source_resolver import HandlerSourceResolver

        resolver = HandlerSourceResolver(
            bootstrap_source=mock_bootstrap_source_minimal,
            contract_source=mock_empty_kafka_source,
            mode=EnumHandlerSourceMode.KAFKA_EVENTS,
        )

        await resolver.resolve_handlers()

        # Kafka source should be called
        mock_empty_kafka_source.discover_handlers.assert_called_once()


# =============================================================================
# Test Class: KAFKA_EVENTS Validation Error Propagation
# =============================================================================


class TestKafkaEventsModeValidationErrors:
    """Tests for validation error propagation in KAFKA_EVENTS mode.

    When the KafkaContractSource encounters validation errors (e.g., malformed
    contract YAML in graceful mode), these errors should be propagated through
    the resolver to the caller.
    """

    @pytest.mark.asyncio
    async def test_kafka_events_propagates_validation_errors(
        self,
        mock_bootstrap_source_minimal: MagicMock,
        sample_version: ModelSemVer,
    ) -> None:
        """KAFKA_EVENTS mode should propagate validation errors from source.

        Given:
            - KafkaContractSource has one valid descriptor and one validation error

        When:
            - Resolve handlers in KAFKA_EVENTS mode

        Then:
            - Valid descriptors should be returned
            - Validation errors should be included in result
        """
        from omnibase_infra.runtime.handler_source_resolver import HandlerSourceResolver

        # Create a validation error from the Kafka source
        validation_error = ModelHandlerValidationError(
            error_type=EnumHandlerErrorType.CONTRACT_PARSE_ERROR,
            rule_id="KAFKA-001",
            handler_identity=ModelHandlerIdentifier.from_handler_id(
                "invalid.kafka.handler"
            ),
            source_type=EnumHandlerSourceType.CONTRACT,
            message="Failed to parse contract YAML from Kafka event",
            remediation_hint="Check contract YAML syntax in registration event",
            file_path="kafka://dev/contracts/invalid.kafka.handler",
        )

        # Create mock source with both valid descriptor and error
        valid_descriptor = ModelHandlerDescriptor(
            handler_id="valid.kafka.handler",
            name="Valid Kafka Handler",
            version=sample_version,
            handler_kind="effect",
            input_model="test.models.Input",
            output_model="test.models.Output",
        )

        mock_kafka_with_errors = MagicMock()
        mock_kafka_with_errors.source_type = "KAFKA_EVENTS"
        mock_kafka_with_errors.discover_handlers = AsyncMock(
            return_value=ModelContractDiscoveryResult(
                descriptors=[valid_descriptor],
                validation_errors=[validation_error],
            )
        )

        resolver = HandlerSourceResolver(
            bootstrap_source=mock_bootstrap_source_minimal,
            contract_source=mock_kafka_with_errors,
            mode=EnumHandlerSourceMode.KAFKA_EVENTS,
        )

        result = await resolver.resolve_handlers()

        # Should have valid descriptor
        assert len(result.descriptors) == 1
        assert result.descriptors[0].handler_id == "valid.kafka.handler"

        # Should propagate validation error
        assert len(result.validation_errors) == 1
        error = result.validation_errors[0]
        assert error.error_type == EnumHandlerErrorType.CONTRACT_PARSE_ERROR
        assert error.rule_id == "KAFKA-001"
        assert error.handler_identity.handler_id == "invalid.kafka.handler"

    @pytest.mark.asyncio
    async def test_kafka_events_multiple_validation_errors(
        self,
        mock_bootstrap_source_minimal: MagicMock,
    ) -> None:
        """KAFKA_EVENTS mode should propagate multiple validation errors.

        Given:
            - KafkaContractSource has multiple validation errors

        When:
            - Resolve handlers in KAFKA_EVENTS mode

        Then:
            - All validation errors should be included in result
        """
        from omnibase_infra.runtime.handler_source_resolver import HandlerSourceResolver

        # Create multiple validation errors
        error1 = ModelHandlerValidationError(
            error_type=EnumHandlerErrorType.CONTRACT_PARSE_ERROR,
            rule_id="KAFKA-001",
            handler_identity=ModelHandlerIdentifier.from_handler_id("error1.handler"),
            source_type=EnumHandlerSourceType.CONTRACT,
            message="Parse error for error1",
            remediation_hint="Fix YAML syntax",
        )

        error2 = ModelHandlerValidationError(
            error_type=EnumHandlerErrorType.CONTRACT_VALIDATION_ERROR,
            rule_id="KAFKA-002",
            handler_identity=ModelHandlerIdentifier.from_handler_id("error2.handler"),
            source_type=EnumHandlerSourceType.CONTRACT,
            message="Validation error for error2",
            remediation_hint="Add required fields",
        )

        mock_kafka_multi_errors = MagicMock()
        mock_kafka_multi_errors.source_type = "KAFKA_EVENTS"
        mock_kafka_multi_errors.discover_handlers = AsyncMock(
            return_value=ModelContractDiscoveryResult(
                descriptors=[],
                validation_errors=[error1, error2],
            )
        )

        resolver = HandlerSourceResolver(
            bootstrap_source=mock_bootstrap_source_minimal,
            contract_source=mock_kafka_multi_errors,
            mode=EnumHandlerSourceMode.KAFKA_EVENTS,
        )

        result = await resolver.resolve_handlers()

        assert len(result.validation_errors) == 2
        error_rule_ids = {e.rule_id for e in result.validation_errors}
        assert error_rule_ids == {"KAFKA-001", "KAFKA-002"}


# =============================================================================
# Test Class: KAFKA_EVENTS Structured Logging
# =============================================================================


class TestKafkaEventsModeStructuredLogging:
    """Tests for structured logging in KAFKA_EVENTS mode.

    The resolver should log structured fields for observability:
    - mode: The resolution mode ("kafka_events")
    - kafka_handler_count: Number of handlers from Kafka cache
    - resolved_handler_count: Total handlers resolved (same as kafka_handler_count)
    """

    @pytest.mark.asyncio
    async def test_kafka_events_logs_handler_counts(
        self,
        mock_bootstrap_source_minimal: MagicMock,
        mock_kafka_contract_source: MagicMock,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """KAFKA_EVENTS mode should log structured handler counts.

        Given:
            - KafkaContractSource has 2 cached handlers

        When:
            - Resolve handlers in KAFKA_EVENTS mode

        Then:
            - Logs should include structured fields:
              - mode: "kafka_events"
              - kafka_handler_count: 2
              - resolved_handler_count: 2
        """
        from omnibase_infra.runtime.handler_source_resolver import HandlerSourceResolver

        resolver = HandlerSourceResolver(
            bootstrap_source=mock_bootstrap_source_minimal,
            contract_source=mock_kafka_contract_source,
            mode=EnumHandlerSourceMode.KAFKA_EVENTS,
        )

        with caplog.at_level(logging.INFO):
            await resolver.resolve_handlers()

        # Find the resolution completion log message
        resolution_logs = [
            record
            for record in caplog.records
            if "handler" in record.message.lower()
            and "resolution" in record.message.lower()
            and "KAFKA_EVENTS" in record.message
        ]

        assert len(resolution_logs) >= 1, (
            "Expected at least one handler resolution log message for KAFKA_EVENTS mode"
        )

        # Check for structured logging fields in extra
        found_counts = False
        for record in resolution_logs:
            extra = getattr(record, "__dict__", {})
            if "kafka_handler_count" in extra:
                found_counts = True
                assert extra["mode"] == "kafka_events", (
                    f"Expected mode='kafka_events', got {extra.get('mode')}"
                )
                assert extra["kafka_handler_count"] == 2, (
                    f"Expected kafka_handler_count=2, got {extra.get('kafka_handler_count')}"
                )
                assert extra["resolved_handler_count"] == 2, (
                    f"Expected resolved_handler_count=2, got {extra.get('resolved_handler_count')}"
                )

        assert found_counts, (
            "Expected structured logging fields for handler counts. "
            "The resolver should log mode, kafka_handler_count, "
            "and resolved_handler_count for KAFKA_EVENTS mode."
        )

    @pytest.mark.asyncio
    async def test_kafka_events_logs_empty_cache_counts(
        self,
        mock_bootstrap_source_minimal: MagicMock,
        mock_empty_kafka_source: MagicMock,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """KAFKA_EVENTS mode should log zero counts for empty cache.

        Given:
            - KafkaContractSource cache is empty

        When:
            - Resolve handlers in KAFKA_EVENTS mode

        Then:
            - Logs should include kafka_handler_count: 0
        """
        from omnibase_infra.runtime.handler_source_resolver import HandlerSourceResolver

        resolver = HandlerSourceResolver(
            bootstrap_source=mock_bootstrap_source_minimal,
            contract_source=mock_empty_kafka_source,
            mode=EnumHandlerSourceMode.KAFKA_EVENTS,
        )

        with caplog.at_level(logging.INFO):
            await resolver.resolve_handlers()

        # Check for zero counts in logs
        found_empty_log = False
        for record in caplog.records:
            extra = getattr(record, "__dict__", {})
            if "kafka_handler_count" in extra and extra["kafka_handler_count"] == 0:
                found_empty_log = True
                assert extra["resolved_handler_count"] == 0

        assert found_empty_log, (
            "Expected log entry with kafka_handler_count=0 for empty cache"
        )


# =============================================================================
# Test Class: KAFKA_EVENTS Mode Returns ModelContractDiscoveryResult
# =============================================================================


class TestKafkaEventsModeReturnType:
    """Tests for KAFKA_EVENTS mode return type compliance."""

    @pytest.mark.asyncio
    async def test_kafka_events_returns_model_contract_discovery_result(
        self,
        mock_bootstrap_source_minimal: MagicMock,
        mock_kafka_contract_source: MagicMock,
    ) -> None:
        """KAFKA_EVENTS mode should return ModelContractDiscoveryResult.

        The result type should be consistent with other modes to enable
        unified handling by the runtime.
        """
        from omnibase_infra.runtime.handler_source_resolver import HandlerSourceResolver

        resolver = HandlerSourceResolver(
            bootstrap_source=mock_bootstrap_source_minimal,
            contract_source=mock_kafka_contract_source,
            mode=EnumHandlerSourceMode.KAFKA_EVENTS,
        )

        result = await resolver.resolve_handlers()

        assert isinstance(result, ModelContractDiscoveryResult), (
            f"Expected ModelContractDiscoveryResult, got {type(result).__name__}"
        )
        assert hasattr(result, "descriptors")
        assert hasattr(result, "validation_errors")

    @pytest.mark.asyncio
    async def test_kafka_events_descriptors_are_model_handler_descriptor(
        self,
        mock_bootstrap_source_minimal: MagicMock,
        mock_kafka_contract_source: MagicMock,
    ) -> None:
        """KAFKA_EVENTS mode descriptors should be ModelHandlerDescriptor instances."""
        from omnibase_infra.runtime.handler_source_resolver import HandlerSourceResolver

        resolver = HandlerSourceResolver(
            bootstrap_source=mock_bootstrap_source_minimal,
            contract_source=mock_kafka_contract_source,
            mode=EnumHandlerSourceMode.KAFKA_EVENTS,
        )

        result = await resolver.resolve_handlers()

        for descriptor in result.descriptors:
            assert isinstance(descriptor, ModelHandlerDescriptor), (
                f"Expected ModelHandlerDescriptor, got {type(descriptor).__name__}"
            )


# =============================================================================
# Test Class: KAFKA_EVENTS Mode Idempotency
# =============================================================================


class TestKafkaEventsModeIdempotency:
    """Tests for idempotent behavior in KAFKA_EVENTS mode."""

    @pytest.mark.asyncio
    async def test_kafka_events_multiple_calls_idempotent(
        self,
        mock_bootstrap_source_minimal: MagicMock,
        mock_kafka_contract_source: MagicMock,
    ) -> None:
        """Multiple resolve_handlers calls should return consistent results.

        Given:
            - KafkaContractSource has cached handlers

        When:
            - Call resolve_handlers() multiple times

        Then:
            - Same descriptors should be returned each time
        """
        from omnibase_infra.runtime.handler_source_resolver import HandlerSourceResolver

        resolver = HandlerSourceResolver(
            bootstrap_source=mock_bootstrap_source_minimal,
            contract_source=mock_kafka_contract_source,
            mode=EnumHandlerSourceMode.KAFKA_EVENTS,
        )

        result1 = await resolver.resolve_handlers()
        result2 = await resolver.resolve_handlers()

        assert len(result1.descriptors) == len(result2.descriptors)

        ids1 = {d.handler_id for d in result1.descriptors}
        ids2 = {d.handler_id for d in result2.descriptors}
        assert ids1 == ids2, "Handler IDs should be consistent across calls"


__all__ = [
    "TestKafkaEventsModeEmptyCache",
    "TestKafkaEventsModeIdempotency",
    "TestKafkaEventsModeResolution",
    "TestKafkaEventsModeReturnType",
    "TestKafkaEventsModeStructuredLogging",
    "TestKafkaEventsModeValidationErrors",
]
