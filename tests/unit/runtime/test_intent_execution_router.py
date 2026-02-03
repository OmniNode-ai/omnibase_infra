# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Unit tests for IntentExecutionRouter and ModelIntentExecutionSummary.

This module tests the IntentExecutionRouter class which routes intents from
the ContractRegistryReducer to PostgreSQL persistence handlers, and the
ModelIntentExecutionSummary model for batch execution results.

Test Coverage:
- IntentExecutionRouter initialization
- Intent execution with success, failure, and partial success scenarios
- Single intent execution routing
- Error handling for unknown intent types and missing payloads
- ModelIntentExecutionSummary property methods

Related:
- OMN-1869: Intent Execution Router implementation
- IntentExecutionRouter source implementation
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock
from uuid import UUID, uuid4

import pytest

from omnibase_infra.nodes.effects.models.model_backend_result import ModelBackendResult
from omnibase_infra.runtime.intent_execution_router import (
    INTENT_CLEANUP_TOPIC_REFERENCES,
    INTENT_DEACTIVATE_CONTRACT,
    INTENT_MARK_STALE,
    INTENT_UPDATE_HEARTBEAT,
    INTENT_UPDATE_TOPIC,
    INTENT_UPSERT_CONTRACT,
    IntentExecutionRouter,
)
from omnibase_infra.runtime.models.model_intent_execution_summary import (
    ModelIntentExecutionSummary,
)


class TestIntentExecutionRouterInit:
    """Tests for IntentExecutionRouter initialization."""

    def test_init_with_valid_pool(self) -> None:
        """Should initialize successfully with valid postgres pool."""
        mock_pool = MagicMock()

        router = IntentExecutionRouter(container=None, postgres_pool=mock_pool)

        assert router._pool is mock_pool
        assert len(router._handlers) == 6

    def test_init_raises_value_error_when_pool_is_none(self) -> None:
        """Should raise ValueError when postgres_pool is None."""
        with pytest.raises(
            ValueError, match="postgres_pool is required for IntentExecutionRouter"
        ):
            IntentExecutionRouter(container=None, postgres_pool=None)

    def test_init_with_container(self) -> None:
        """Should accept optional container parameter."""
        mock_pool = MagicMock()
        mock_container = MagicMock()

        router = IntentExecutionRouter(
            container=mock_container, postgres_pool=mock_pool
        )

        assert router._container is mock_container


class TestSupportedIntentTypes:
    """Tests for supported_intent_types property."""

    def test_supported_intent_types_returns_all_registered_types(self) -> None:
        """Should return tuple of all registered intent types."""
        mock_pool = MagicMock()
        router = IntentExecutionRouter(container=None, postgres_pool=mock_pool)

        supported = router.supported_intent_types

        assert isinstance(supported, tuple)
        assert INTENT_UPSERT_CONTRACT in supported
        assert INTENT_UPDATE_TOPIC in supported
        assert INTENT_MARK_STALE in supported
        assert INTENT_UPDATE_HEARTBEAT in supported
        assert INTENT_DEACTIVATE_CONTRACT in supported
        assert INTENT_CLEANUP_TOPIC_REFERENCES in supported
        assert len(supported) == 6


class TestExecuteIntents:
    """Tests for execute_intents async method."""

    @pytest.fixture
    def mock_pool(self) -> MagicMock:
        """Create a mock asyncpg pool."""
        return MagicMock()

    @pytest.fixture
    def router(self, mock_pool: MagicMock) -> IntentExecutionRouter:
        """Create a router instance for testing."""
        return IntentExecutionRouter(container=None, postgres_pool=mock_pool)

    def _create_mock_intent(
        self, intent_type: str = INTENT_UPSERT_CONTRACT
    ) -> MagicMock:
        """Create a mock intent with given intent_type."""
        intent = MagicMock()
        intent.intent_id = uuid4()
        intent.payload = MagicMock()
        intent.payload.intent_type = intent_type
        return intent

    @pytest.mark.asyncio
    async def test_execute_intents_with_empty_tuple_returns_zero_intents(
        self, router: IntentExecutionRouter
    ) -> None:
        """Should return summary with 0 intents when given empty tuple."""
        correlation_id = uuid4()

        summary = await router.execute_intents((), correlation_id)

        assert summary.total_intents == 0
        assert summary.successful_count == 0
        assert summary.failed_count == 0
        assert summary.correlation_id == correlation_id
        assert len(summary.results) == 0

    @pytest.mark.asyncio
    async def test_execute_intents_handles_successful_intent_execution(
        self, router: IntentExecutionRouter
    ) -> None:
        """Should increment successful_count for successful intent execution."""
        correlation_id = uuid4()
        intent = self._create_mock_intent()

        # Mock the handler to return success
        mock_handler = MagicMock()
        mock_handler.handle = AsyncMock(
            return_value=ModelBackendResult(
                success=True,
                duration_ms=10.0,
                backend_id="test",
                correlation_id=correlation_id,
            )
        )
        router._handlers[INTENT_UPSERT_CONTRACT] = mock_handler

        summary = await router.execute_intents((intent,), correlation_id)

        assert summary.total_intents == 1
        assert summary.successful_count == 1
        assert summary.failed_count == 0
        assert summary.all_successful is True

    @pytest.mark.asyncio
    async def test_execute_intents_handles_failed_intent_execution(
        self, router: IntentExecutionRouter
    ) -> None:
        """Should increment failed_count for failed intent execution."""
        correlation_id = uuid4()
        intent = self._create_mock_intent()

        # Mock the handler to return failure
        mock_handler = MagicMock()
        mock_handler.handle = AsyncMock(
            return_value=ModelBackendResult(
                success=False,
                error="Test error",
                error_code="TEST_ERROR",
                duration_ms=10.0,
                backend_id="test",
                correlation_id=correlation_id,
            )
        )
        router._handlers[INTENT_UPSERT_CONTRACT] = mock_handler

        summary = await router.execute_intents((intent,), correlation_id)

        assert summary.total_intents == 1
        assert summary.successful_count == 0
        assert summary.failed_count == 1
        assert summary.all_failed is True

    @pytest.mark.asyncio
    async def test_execute_intents_handles_partial_success(
        self, router: IntentExecutionRouter
    ) -> None:
        """Should correctly track partial success with mixed results."""
        correlation_id = uuid4()
        intent1 = self._create_mock_intent(INTENT_UPSERT_CONTRACT)
        intent2 = self._create_mock_intent(INTENT_UPDATE_TOPIC)

        # Mock handlers with different results
        mock_handler_success = MagicMock()
        mock_handler_success.handle = AsyncMock(
            return_value=ModelBackendResult(
                success=True,
                duration_ms=10.0,
                backend_id="test",
                correlation_id=correlation_id,
            )
        )

        mock_handler_failure = MagicMock()
        mock_handler_failure.handle = AsyncMock(
            return_value=ModelBackendResult(
                success=False,
                error="Failed",
                error_code="FAILED",
                duration_ms=10.0,
                backend_id="test",
                correlation_id=correlation_id,
            )
        )

        router._handlers[INTENT_UPSERT_CONTRACT] = mock_handler_success
        router._handlers[INTENT_UPDATE_TOPIC] = mock_handler_failure

        summary = await router.execute_intents((intent1, intent2), correlation_id)

        assert summary.total_intents == 2
        assert summary.successful_count == 1
        assert summary.failed_count == 1
        assert summary.partial_success is True

    @pytest.mark.asyncio
    async def test_execute_intents_handles_handler_exception(
        self, router: IntentExecutionRouter
    ) -> None:
        """Should catch handler exceptions and count as failure."""
        correlation_id = uuid4()
        intent = self._create_mock_intent()

        # Mock the handler to raise an exception
        mock_handler = MagicMock()
        mock_handler.handle = AsyncMock(side_effect=RuntimeError("Unexpected error"))
        router._handlers[INTENT_UPSERT_CONTRACT] = mock_handler

        summary = await router.execute_intents((intent,), correlation_id)

        assert summary.total_intents == 1
        assert summary.successful_count == 0
        assert summary.failed_count == 1
        # The result should contain error info
        assert len(summary.results) == 1
        assert summary.results[0].success is False


class TestExecuteSingleIntent:
    """Tests for _execute_single_intent method."""

    @pytest.fixture
    def mock_pool(self) -> MagicMock:
        """Create a mock asyncpg pool."""
        return MagicMock()

    @pytest.fixture
    def router(self, mock_pool: MagicMock) -> IntentExecutionRouter:
        """Create a router instance for testing."""
        return IntentExecutionRouter(container=None, postgres_pool=mock_pool)

    @pytest.mark.asyncio
    async def test_routes_to_correct_handler(
        self, router: IntentExecutionRouter
    ) -> None:
        """Should route intent to the handler matching its intent_type."""
        correlation_id = uuid4()
        intent = MagicMock()
        intent.intent_id = uuid4()
        intent.payload = MagicMock()
        intent.payload.intent_type = INTENT_UPDATE_HEARTBEAT

        # Mock the specific handler
        mock_handler = MagicMock()
        expected_result = ModelBackendResult(
            success=True,
            duration_ms=5.0,
            backend_id="postgres",
            correlation_id=correlation_id,
        )
        mock_handler.handle = AsyncMock(return_value=expected_result)
        router._handlers[INTENT_UPDATE_HEARTBEAT] = mock_handler

        result = await router._execute_single_intent(intent, correlation_id)

        mock_handler.handle.assert_called_once()
        assert result.success is True

    @pytest.mark.asyncio
    async def test_returns_error_for_unknown_intent_type(
        self, router: IntentExecutionRouter
    ) -> None:
        """Should return error result for unknown intent_type."""
        correlation_id = uuid4()
        intent = MagicMock()
        intent.intent_id = uuid4()
        intent.payload = MagicMock()
        intent.payload.intent_type = "unknown.intent.type"

        result = await router._execute_single_intent(intent, correlation_id)

        assert result.success is False
        assert result.error_code == "INTENT_TYPE_UNKNOWN"
        assert "No handler for intent type" in result.error

    @pytest.mark.asyncio
    async def test_returns_error_for_missing_payload(
        self, router: IntentExecutionRouter
    ) -> None:
        """Should return error result when intent has no payload."""
        correlation_id = uuid4()
        intent = MagicMock()
        intent.intent_id = uuid4()
        intent.payload = None

        result = await router._execute_single_intent(intent, correlation_id)

        assert result.success is False
        assert result.error_code == "INTENT_NO_PAYLOAD"
        assert "Intent has no payload" in result.error

    @pytest.mark.asyncio
    async def test_returns_error_for_missing_intent_type_field(
        self, router: IntentExecutionRouter
    ) -> None:
        """Should return error when payload has no intent_type field."""
        correlation_id = uuid4()
        intent = MagicMock()
        intent.intent_id = uuid4()
        intent.payload = MagicMock(spec=[])  # No intent_type attribute

        result = await router._execute_single_intent(intent, correlation_id)

        assert result.success is False
        assert result.error_code == "INTENT_TYPE_MISSING"
        assert "Payload has no intent_type field" in result.error

    @pytest.mark.asyncio
    async def test_returns_error_when_handler_raises_exception(
        self, router: IntentExecutionRouter
    ) -> None:
        """Should catch handler exceptions and return error result."""
        correlation_id = uuid4()
        intent = MagicMock()
        intent.intent_id = uuid4()
        intent.payload = MagicMock()
        intent.payload.intent_type = INTENT_UPSERT_CONTRACT

        # Mock handler to raise exception
        mock_handler = MagicMock()
        mock_handler.handle = AsyncMock(side_effect=ValueError("Database error"))
        router._handlers[INTENT_UPSERT_CONTRACT] = mock_handler

        result = await router._execute_single_intent(intent, correlation_id)

        assert result.success is False
        assert result.error_code == "HANDLER_EXECUTION_ERROR"


class TestModelIntentExecutionSummaryProperties:
    """Tests for ModelIntentExecutionSummary computed properties."""

    def test_all_successful_true_when_no_failures_and_has_intents(self) -> None:
        """Should return True when failed_count=0 and total_intents>0."""
        summary = ModelIntentExecutionSummary(
            total_intents=3,
            successful_count=3,
            failed_count=0,
            total_duration_ms=100.0,
            results=(),
            correlation_id=uuid4(),
        )

        assert summary.all_successful is True

    def test_all_successful_false_when_has_failures(self) -> None:
        """Should return False when any intent failed."""
        summary = ModelIntentExecutionSummary(
            total_intents=3,
            successful_count=2,
            failed_count=1,
            total_duration_ms=100.0,
            results=(),
            correlation_id=uuid4(),
        )

        assert summary.all_successful is False

    def test_all_successful_false_when_zero_intents(self) -> None:
        """Should return False when total_intents is 0."""
        summary = ModelIntentExecutionSummary(
            total_intents=0,
            successful_count=0,
            failed_count=0,
            total_duration_ms=0.0,
            results=(),
            correlation_id=uuid4(),
        )

        assert summary.all_successful is False

    def test_partial_success_true_when_both_counts_positive(self) -> None:
        """Should return True when both successful_count and failed_count > 0."""
        summary = ModelIntentExecutionSummary(
            total_intents=4,
            successful_count=2,
            failed_count=2,
            total_duration_ms=100.0,
            results=(),
            correlation_id=uuid4(),
        )

        assert summary.partial_success is True

    def test_partial_success_false_when_all_succeeded(self) -> None:
        """Should return False when all intents succeeded."""
        summary = ModelIntentExecutionSummary(
            total_intents=3,
            successful_count=3,
            failed_count=0,
            total_duration_ms=100.0,
            results=(),
            correlation_id=uuid4(),
        )

        assert summary.partial_success is False

    def test_partial_success_false_when_all_failed(self) -> None:
        """Should return False when all intents failed."""
        summary = ModelIntentExecutionSummary(
            total_intents=3,
            successful_count=0,
            failed_count=3,
            total_duration_ms=100.0,
            results=(),
            correlation_id=uuid4(),
        )

        assert summary.partial_success is False

    def test_all_failed_true_when_no_successes_and_has_intents(self) -> None:
        """Should return True when successful_count=0 and total_intents>0."""
        summary = ModelIntentExecutionSummary(
            total_intents=3,
            successful_count=0,
            failed_count=3,
            total_duration_ms=100.0,
            results=(),
            correlation_id=uuid4(),
        )

        assert summary.all_failed is True

    def test_all_failed_false_when_has_successes(self) -> None:
        """Should return False when any intent succeeded."""
        summary = ModelIntentExecutionSummary(
            total_intents=3,
            successful_count=1,
            failed_count=2,
            total_duration_ms=100.0,
            results=(),
            correlation_id=uuid4(),
        )

        assert summary.all_failed is False

    def test_all_failed_false_when_zero_intents(self) -> None:
        """Should return False when total_intents is 0."""
        summary = ModelIntentExecutionSummary(
            total_intents=0,
            successful_count=0,
            failed_count=0,
            total_duration_ms=0.0,
            results=(),
            correlation_id=uuid4(),
        )

        assert summary.all_failed is False

    def test_model_is_frozen(self) -> None:
        """Should be immutable (frozen model)."""
        summary = ModelIntentExecutionSummary(
            total_intents=1,
            successful_count=1,
            failed_count=0,
            total_duration_ms=10.0,
            results=(),
            correlation_id=uuid4(),
        )

        with pytest.raises(Exception):  # ValidationError or AttributeError
            summary.total_intents = 5  # type: ignore[misc]

    def test_model_with_results_tuple(self) -> None:
        """Should accept tuple of ModelBackendResult."""
        correlation_id = uuid4()
        results = (
            ModelBackendResult(
                success=True,
                duration_ms=10.0,
                backend_id="test",
                correlation_id=correlation_id,
            ),
            ModelBackendResult(
                success=False,
                error="Failed",
                duration_ms=5.0,
                backend_id="test",
                correlation_id=correlation_id,
            ),
        )

        summary = ModelIntentExecutionSummary(
            total_intents=2,
            successful_count=1,
            failed_count=1,
            total_duration_ms=15.0,
            results=results,
            correlation_id=correlation_id,
        )

        assert len(summary.results) == 2
        assert summary.results[0].success is True
        assert summary.results[1].success is False
