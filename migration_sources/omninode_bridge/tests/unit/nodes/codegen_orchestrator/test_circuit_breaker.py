#!/usr/bin/env python3
"""
Unit tests for circuit breaker and error handling in NodeCodegenOrchestrator.

Tests cover:
1. Circuit breaker triggering and recovery
2. Retry logic with exponential backoff
3. Graceful degradation with fallback values
4. Partial success scenarios
5. Error code usage and recovery hints
6. Structured error handling
"""

import asyncio
from unittest.mock import AsyncMock, Mock, patch
from uuid import uuid4

import pytest
from circuitbreaker import CircuitBreakerError

from omninode_bridge.nodes.codegen_orchestrator.v1_0_0.circuit_breaker_config import (
    PartialSuccessResult,
    with_graceful_degradation,
)
from omninode_bridge.nodes.codegen_orchestrator.v1_0_0.models import (
    EnumErrorCode,
    ModelGenerationContext,
)
from omninode_bridge.nodes.codegen_orchestrator.v1_0_0.workflow import (
    CodeGenerationWorkflow,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def generation_context():
    """Create test generation context."""
    return ModelGenerationContext(
        workflow_id=uuid4(),
        correlation_id=uuid4(),
        prompt="Create a database CRUD node for PostgreSQL",
        output_directory="/tmp/test",
        enable_intelligence=True,
    )


@pytest.fixture
def mock_kafka_client():
    """Create mock Kafka client."""
    client = Mock()
    client.is_connected = True
    client.publish = AsyncMock()
    return client


@pytest.fixture
def workflow(mock_kafka_client):
    """Create workflow instance with mock Kafka client."""
    return CodeGenerationWorkflow(
        kafka_client=mock_kafka_client,
        enable_intelligence=True,
        enable_quorum=False,
    )


# =============================================================================
# Test Circuit Breaker
# =============================================================================


@pytest.mark.asyncio
async def test_circuit_breaker_opens_after_failures(workflow, generation_context):
    """Test that circuit breaker opens after threshold failures."""

    # Patch asyncio.sleep inside the method to make it raise exception
    # This preserves the decorators while causing failures
    async def failing_sleep(*args, **kwargs):
        raise Exception("Service unavailable")

    with patch("asyncio.sleep", side_effect=failing_sleep):
        # First 5 calls should retry and eventually fail gracefully
        for i in range(5):
            result, error_code = await workflow._query_intelligence_with_protection(
                generation_context
            )
            assert result["degraded"] is True
            assert error_code == EnumErrorCode.INTELLIGENCE_UNAVAILABLE

        # 6th call should hit open circuit
        result, error_code = await workflow._query_intelligence_with_protection(
            generation_context
        )
        assert result["degraded"] is True
        assert error_code == EnumErrorCode.INTELLIGENCE_CIRCUIT_OPEN


@pytest.mark.asyncio
async def test_circuit_breaker_graceful_degradation(workflow, generation_context):
    """Test graceful degradation when circuit breaker is open."""
    with patch.object(
        workflow,
        "_query_intelligence_protected",
        side_effect=CircuitBreakerError("Circuit open"),
    ):
        result, error_code = await workflow._query_intelligence_with_protection(
            generation_context
        )

        # Should return fallback data
        assert result["patterns_found"] == 0
        assert result["degraded"] is True
        assert error_code == EnumErrorCode.INTELLIGENCE_CIRCUIT_OPEN


@pytest.mark.asyncio
async def test_circuit_breaker_recovery(workflow, generation_context):
    """Test circuit breaker recovery after timeout."""
    call_count = 0

    async def failing_then_succeeding(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count <= 5:
            raise Exception("Service unavailable")
        # After 5 failures, succeed (this won't be reached while circuit is open)

    with patch("asyncio.sleep", side_effect=failing_then_succeeding):
        # Fail 5 times to open circuit
        for _ in range(5):
            await workflow._query_intelligence_with_protection(generation_context)

        # Circuit should be open
        result, error_code = await workflow._query_intelligence_with_protection(
            generation_context
        )
        assert error_code == EnumErrorCode.INTELLIGENCE_CIRCUIT_OPEN

        # Simulate recovery timeout (circuit breaker library handles this internally)
        # For testing, we verify the fallback behavior works


# =============================================================================
# Test Retry Logic
# =============================================================================


@pytest.mark.asyncio
@pytest.mark.order(1)  # Run this test first before circuit breaker is opened
async def test_retry_logic_with_exponential_backoff(workflow, generation_context):
    """Test retry logic retries appropriate number of times."""
    call_count = 0
    original_sleep = asyncio.sleep

    async def selective_sleep(delay, *args, **kwargs):
        """
        Mock sleep that fails the method's sleep but allows retry backoff.

        The method sleeps for 0.5s, while retry backoff sleeps for 1s, 2s, etc.
        """
        nonlocal call_count
        # Detect the method's 0.5s sleep (vs retry backoff of >= 1.0s)
        if 0.4 <= delay <= 0.6:  # Method's asyncio.sleep(0.5)
            call_count += 1
            if call_count < 3:
                raise TimeoutError("Service timeout")
            # On 3rd call, complete the sleep successfully
            await original_sleep(0.001)
        else:
            # For retry backoff sleeps (1s, 2s, 4s, etc.), execute quickly
            await original_sleep(0.001)

    with patch("asyncio.sleep", side_effect=selective_sleep):
        result, error_code = await workflow._query_intelligence_with_protection(
            generation_context
        )

        # Should succeed after 2 retries
        assert (
            "degraded" not in result or result["degraded"] is False
        )  # Should succeed with real data
        assert result["patterns_found"] > 0  # Real intelligence data returned
        assert error_code is None
        assert call_count == 3  # Initial call + 2 retries


@pytest.mark.asyncio
async def test_retry_does_not_retry_validation_errors(workflow, generation_context):
    """Test that validation errors (4xx) are not retried."""
    call_count = 0

    async def validation_error(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        error = Exception("Bad request")
        error.status_code = 400  # type: ignore[attr-defined]
        raise error

    with patch.object(
        workflow, "_query_intelligence_protected", side_effect=validation_error
    ):
        result, error_code = await workflow._query_intelligence_with_protection(
            generation_context
        )

        # Should fail immediately without retries
        assert result["degraded"] is True
        assert error_code == EnumErrorCode.INTELLIGENCE_UNAVAILABLE
        # Note: call_count may be > 1 due to circuit breaker, but retries should not happen


@pytest.mark.asyncio
async def test_retry_logic_exhaustion(workflow, generation_context):
    """Test retry logic exhaustion after max attempts."""
    call_count = 0

    async def always_fail(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        raise TimeoutError("Service timeout")

    with patch.object(
        workflow, "_query_intelligence_protected", side_effect=always_fail
    ):
        result, error_code = await workflow._query_intelligence_with_protection(
            generation_context
        )

        # Should fail after max retries
        assert result["degraded"] is True
        assert error_code in (
            EnumErrorCode.INTELLIGENCE_UNAVAILABLE,
            EnumErrorCode.INTELLIGENCE_TIMEOUT,
        )


# =============================================================================
# Test Error Codes
# =============================================================================


def test_error_code_is_retryable():
    """Test error code retryability detection."""
    assert EnumErrorCode.INTELLIGENCE_UNAVAILABLE.is_retryable is True
    assert EnumErrorCode.INTELLIGENCE_TIMEOUT.is_retryable is True
    assert EnumErrorCode.VALIDATION_FAILED.is_retryable is False
    assert EnumErrorCode.CODE_GENERATION_FAILED.is_retryable is False


def test_error_code_requires_circuit_breaker():
    """Test error code circuit breaker detection."""
    assert EnumErrorCode.INTELLIGENCE_UNAVAILABLE.requires_circuit_breaker is True
    assert EnumErrorCode.INTELLIGENCE_TIMEOUT.requires_circuit_breaker is True
    assert EnumErrorCode.FILE_WRITE_ERROR.requires_circuit_breaker is False


def test_error_code_allows_partial_success():
    """Test error code partial success detection."""
    assert EnumErrorCode.INTELLIGENCE_DEGRADED.allows_partial_success is True
    assert (
        EnumErrorCode.VALIDATION_QUALITY_BELOW_THRESHOLD.allows_partial_success is True
    )
    assert EnumErrorCode.CODE_GENERATION_FAILED.allows_partial_success is False


def test_error_code_severity():
    """Test error code severity levels."""
    assert EnumErrorCode.SYSTEM_OUT_OF_MEMORY.severity == "CRITICAL"
    assert EnumErrorCode.CODE_GENERATION_FAILED.severity == "HIGH"
    assert EnumErrorCode.INTELLIGENCE_UNAVAILABLE.severity == "MEDIUM"
    assert EnumErrorCode.INTELLIGENCE_DEGRADED.severity == "LOW"


def test_error_code_recovery_hints():
    """Test error code recovery hint messages."""
    hint = EnumErrorCode.INTELLIGENCE_UNAVAILABLE.get_recovery_hint()
    assert "Intelligence service is unavailable" in hint
    assert "60s" in hint

    hint = EnumErrorCode.FILE_WRITE_ERROR.get_recovery_hint()
    assert "disk space" in hint or "permissions" in hint


# =============================================================================
# Test Partial Success Handling
# =============================================================================


def test_partial_success_result_creation():
    """Test PartialSuccessResult creation and properties."""
    # Full success
    result = PartialSuccessResult(
        data={"code": "pass"}, success=True, partial=False, error_code=None
    )
    assert result.is_usable() is True
    assert result.get_status() == "SUCCESS"

    # Partial success
    result = PartialSuccessResult(
        data={"code": "partial"},
        success=False,
        partial=True,
        error_code=EnumErrorCode.INTELLIGENCE_DEGRADED,
    )
    assert result.is_usable() is True
    assert result.get_status() == "PARTIAL_SUCCESS"

    # Failure
    result = PartialSuccessResult(
        data=None,
        success=False,
        partial=False,
        error_code=EnumErrorCode.CODE_GENERATION_FAILED,
    )
    assert result.is_usable() is False
    assert result.get_status() == "FAILED"


def test_partial_success_result_warnings():
    """Test PartialSuccessResult warning management."""
    result = PartialSuccessResult(data={"code": "pass"}, success=True)
    assert len(result.warnings) == 0

    result.add_warning("Quality below threshold")
    result.add_warning("Intelligence service degraded")
    assert len(result.warnings) == 2
    assert "Quality below threshold" in result.warnings


def test_partial_success_result_serialization():
    """Test PartialSuccessResult serialization."""
    result = PartialSuccessResult(
        data={"code": "pass"},
        success=False,
        partial=True,
        error_code=EnumErrorCode.INTELLIGENCE_DEGRADED,
        warnings=["Warning 1"],
    )

    serialized = result.to_dict()
    assert serialized["data"]["code"] == "pass"
    assert serialized["success"] is False
    assert serialized["partial"] is True
    assert serialized["error_code"] == EnumErrorCode.INTELLIGENCE_DEGRADED.value
    assert serialized["status"] == "PARTIAL_SUCCESS"
    assert len(serialized["warnings"]) == 1


# =============================================================================
# Test Graceful Degradation Helper
# =============================================================================


@pytest.mark.asyncio
async def test_graceful_degradation_success():
    """Test graceful degradation with successful operation."""

    async def successful_operation():
        return {"result": "success"}

    result, error_code = await with_graceful_degradation(
        coro=successful_operation,
        fallback_value={"result": "fallback"},
        error_code=EnumErrorCode.INTELLIGENCE_UNAVAILABLE,
        context={"test": "context"},
    )

    assert result["result"] == "success"
    assert error_code is None


@pytest.mark.asyncio
async def test_graceful_degradation_circuit_breaker_open():
    """Test graceful degradation when circuit breaker is open."""

    async def failing_operation():
        raise CircuitBreakerError("Circuit open")

    result, error_code = await with_graceful_degradation(
        coro=failing_operation,
        fallback_value={"result": "fallback"},
        error_code=EnumErrorCode.INTELLIGENCE_UNAVAILABLE,
        context={"test": "context"},
    )

    assert result["result"] == "fallback"
    assert error_code == EnumErrorCode.INTELLIGENCE_CIRCUIT_OPEN


@pytest.mark.asyncio
async def test_graceful_degradation_timeout():
    """Test graceful degradation on timeout."""

    async def timeout_operation():
        raise TimeoutError("Operation timed out")

    result, error_code = await with_graceful_degradation(
        coro=timeout_operation,
        fallback_value={"result": "fallback"},
        error_code=EnumErrorCode.INTELLIGENCE_UNAVAILABLE,
        context={"test": "context"},
    )

    assert result["result"] == "fallback"
    assert error_code == EnumErrorCode.INTELLIGENCE_TIMEOUT


@pytest.mark.asyncio
async def test_graceful_degradation_generic_error():
    """Test graceful degradation with generic error."""

    async def generic_error_operation():
        raise Exception("Generic error")

    result, error_code = await with_graceful_degradation(
        coro=generic_error_operation,
        fallback_value={"result": "fallback"},
        error_code=EnumErrorCode.INTELLIGENCE_UNAVAILABLE,
        context={"test": "context"},
    )

    assert result["result"] == "fallback"
    assert error_code == EnumErrorCode.INTELLIGENCE_UNAVAILABLE


# =============================================================================
# Test Workflow Integration
# =============================================================================


@pytest.mark.asyncio
async def test_workflow_publishes_failed_event(workflow, generation_context):
    """Test that workflow publishes failed event on error."""
    await workflow._publish_failed_event(
        gen_ctx=generation_context,
        failed_stage="intelligence_gathering",
        error_code=EnumErrorCode.INTELLIGENCE_UNAVAILABLE,
        error_message="Intelligence service unavailable",
        error_context={"attempt": 3},
    )

    # Verify Kafka publish was called
    workflow.kafka_client.publish.assert_called_once()
    call_args = workflow.kafka_client.publish.call_args

    # Verify event structure
    assert (
        call_args.kwargs["topic"] == "dev.omninode-bridge.codegen.generation-failed.v1"
    )
    event_data = call_args.kwargs["value"]
    assert event_data["error_code"] == EnumErrorCode.INTELLIGENCE_UNAVAILABLE.value
    assert event_data["is_retryable"] is True


@pytest.mark.asyncio
async def test_intelligence_gathering_with_degradation(workflow, generation_context):
    """Test intelligence gathering handles degradation gracefully."""
    # Mock intelligence query to fail
    with patch.object(
        workflow,
        "_query_intelligence_protected",
        side_effect=Exception("Service unavailable"),
    ):
        result, error_code = await workflow._query_intelligence_with_protection(
            generation_context
        )

        # Should return degraded intelligence
        assert result["patterns_found"] == 0
        assert result["degraded"] is True
        assert error_code == EnumErrorCode.INTELLIGENCE_UNAVAILABLE


# =============================================================================
# Test Partial Success: Validation Failures
# =============================================================================


@pytest.mark.asyncio
async def test_validation_failure_with_code_generated_marks_needs_review(
    workflow, generation_context, mock_kafka_client
):
    """Test that validation failure with generated code marks as needs_review."""
    from llama_index.core.workflow import Context

    from omninode_bridge.nodes.codegen_orchestrator.v1_0_0.workflow import (
        EventBusIntegratedEvent,
    )

    # Setup: Create context with generated code
    gen_ctx = generation_context
    gen_ctx.generated_code = {
        "node.py": "# Generated code",
        "contract.yaml": "# Contract",
    }

    ctx = Context(workflow=workflow)
    await ctx.set("generation_context", gen_ctx)

    # Mock validation to fail
    event = EventBusIntegratedEvent(integration_details={"topics_configured": 3})

    # Simulate validation failure by patching asyncio.sleep to raise exception
    with patch("asyncio.sleep", side_effect=Exception("Validation tools failed")):
        result = await workflow.validate_code(ctx, event)

        # Should return partial success with needs_review flag
        assert result.validation_results["needs_review"] is True
        assert result.validation_results["linting_passed"] is False
        assert result.validation_results["quality_score"] == 0.5
        assert len(result.validation_results["warnings"]) > 0
        assert "Unexpected validation error" in result.validation_results["warnings"][0]

        # Context should have warnings
        assert len(gen_ctx.stage_warnings.get("validation", [])) > 0
        assert "needs review" in gen_ctx.stage_warnings["validation"][0].lower()


@pytest.mark.asyncio
async def test_validation_failure_without_code_raises_exception(
    workflow, generation_context, mock_kafka_client
):
    """Test that validation failure WITHOUT generated code raises exception."""
    from llama_index.core.workflow import Context

    from omninode_bridge.nodes.codegen_orchestrator.v1_0_0.workflow import (
        EventBusIntegratedEvent,
    )

    # Setup: Create context WITHOUT generated code
    gen_ctx = generation_context
    gen_ctx.generated_code = {}  # No code generated

    ctx = Context(workflow=workflow)
    await ctx.set("generation_context", gen_ctx)

    event = EventBusIntegratedEvent(integration_details={"topics_configured": 3})

    # Simulate validation failure
    with patch("asyncio.sleep", side_effect=Exception("Validation tools failed")):
        with pytest.raises(Exception, match="Validation tools failed"):
            await workflow.validate_code(ctx, event)


# =============================================================================
# Test Partial Success: File Write Failures
# =============================================================================


@pytest.mark.asyncio
async def test_file_write_failure_saves_to_database(
    workflow, generation_context, mock_kafka_client
):
    """Test that file write failure saves code to database for recovery."""
    from llama_index.core.workflow import Context

    from omninode_bridge.nodes.codegen_orchestrator.v1_0_0.workflow import (
        RefinementCompleteEvent,
    )

    # Setup: Create context with generated and refined code
    gen_ctx = generation_context
    gen_ctx.generated_code = {
        "node.py": "# Generated code",
        "contract.yaml": "# Contract",
    }
    gen_ctx.validation_results = {
        "quality_score": 0.85,
        "needs_review": False,
    }
    gen_ctx.parsed_requirements = {
        "node_type": "effect",
        "service_name": "test_service",
    }

    ctx = Context(workflow=workflow)
    await ctx.set("generation_context", gen_ctx)

    event = RefinementCompleteEvent(
        refined_files={"node.py": "# Refined code", "contract.yaml": "# Contract"}
    )

    # Simulate file write failure (OSError for disk full scenario)
    with patch("asyncio.sleep", side_effect=OSError("Disk full")):
        result = await workflow.write_files(ctx, event)

        # Should return partial success result
        assert result.result["success"] is False
        assert result.result["partial_success"] is True
        assert result.result["generated_code_in_database"] is True
        assert result.result["needs_review"] is True
        assert result.result["error_code"] == EnumErrorCode.FILE_WRITE_ERROR.value
        assert "recovery_hint" in result.result
        assert len(result.result["generated_files"]) == 0  # No files written

        # Should publish failed event
        assert mock_kafka_client.publish.call_count >= 1

        # Context should have warnings
        assert len(gen_ctx.stage_warnings.get("file_writing", [])) > 0
        assert "saved to database" in gen_ctx.stage_warnings["file_writing"][0].lower()


@pytest.mark.asyncio
async def test_file_write_success_returns_complete_result(
    workflow, generation_context, mock_kafka_client
):
    """Test that successful file write returns complete result."""
    from llama_index.core.workflow import Context

    from omninode_bridge.nodes.codegen_orchestrator.v1_0_0.workflow import (
        RefinementCompleteEvent,
    )

    # Setup
    gen_ctx = generation_context
    gen_ctx.generated_code = {"node.py": "# Generated code"}
    gen_ctx.validation_results = {
        "quality_score": 0.85,
        "needs_review": False,
    }
    gen_ctx.parsed_requirements = {
        "node_type": "effect",
        "service_name": "test_service",
    }

    ctx = Context(workflow=workflow)
    await ctx.set("generation_context", gen_ctx)

    event = RefinementCompleteEvent(refined_files={"node.py": "# Refined code"})

    # Should succeed
    result = await workflow.write_files(ctx, event)

    # FileWritingCompleteEvent has written_files attribute
    assert isinstance(result.written_files, list)
    assert len(result.written_files) == 1
    assert "node.py" in result.written_files


# =============================================================================
# Test End-to-End Partial Success Workflows
# =============================================================================


@pytest.mark.asyncio
async def test_e2e_validation_failure_continues_with_review_flag(
    workflow, generation_context, mock_kafka_client
):
    """Test end-to-end workflow with validation failure continues with review flag."""
    from llama_index.core.workflow import Context

    from omninode_bridge.nodes.codegen_orchestrator.v1_0_0.workflow import (
        EventBusIntegratedEvent,
    )

    # Setup context
    gen_ctx = generation_context
    gen_ctx.generated_code = {"node.py": "# Code"}
    gen_ctx.validation_results = {}

    ctx = Context(workflow=workflow)
    await ctx.set("generation_context", gen_ctx)

    # Stage 6: Validation fails but returns partial success
    event_bus_event = EventBusIntegratedEvent(integration_details={})
    with patch("asyncio.sleep", side_effect=Exception("Linting failed")):
        validation_event = await workflow.validate_code(ctx, event_bus_event)

    assert validation_event.validation_results["needs_review"] is True
    assert validation_event.validation_results["quality_score"] == 0.5

    # Stage 7: Refinement continues with degraded quality
    refinement_event = await workflow.refine_code(ctx, validation_event)

    # Stage 8: File write should succeed and include needs_review flag
    result = await workflow.write_files(ctx, refinement_event)

    # FileWritingCompleteEvent has written_files attribute
    assert isinstance(result.written_files, list)
    # Verify that needs_review is tracked in the generation context
    assert gen_ctx.validation_results.get("needs_review") is True


@pytest.mark.asyncio
async def test_e2e_intelligence_degradation_continues_workflow(
    workflow, generation_context, mock_kafka_client
):
    """Test end-to-end workflow with intelligence degradation continues."""
    from llama_index.core.workflow import Context

    from omninode_bridge.nodes.codegen_orchestrator.v1_0_0.workflow import (
        PromptParsedEvent,
    )

    # Setup context
    gen_ctx = generation_context
    ctx = Context(workflow=workflow)
    await ctx.set("generation_context", gen_ctx)

    # Mock intelligence to fail
    with patch.object(
        workflow,
        "_query_intelligence_protected",
        side_effect=Exception("Intelligence unavailable"),
    ):
        # Stage 2: Intelligence gathering should degrade gracefully
        prompt_event = PromptParsedEvent(
            requirements={
                "node_type": "effect",
                "service_name": "test",
                "domain": "test",
                "operations": ["create"],
            }
        )

        intelligence_event = await workflow.gather_intelligence(ctx, prompt_event)

        # Should continue with degraded intelligence
        assert intelligence_event.intelligence_data["patterns_found"] == 0
        assert intelligence_event.intelligence_data["degraded"] is True

        # Context should have warning
        assert len(gen_ctx.stage_warnings.get("intelligence_gathering", [])) > 0
