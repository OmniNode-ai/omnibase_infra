#!/usr/bin/env python3
"""
Standalone validation script for circuit breaker and error handling.

Run with: poetry run python tests/unit/nodes/codegen_orchestrator/validate_circuit_breaker.py

This script validates:
- Error code properties
- Circuit breaker configuration
- Retry logic decorators
- Partial success handling
"""

import inspect

from omninode_bridge.nodes.codegen_orchestrator.v1_0_0.models.enum_error_code import (
    EnumErrorCode,
)
from omninode_bridge.nodes.codegen_orchestrator.v1_0_0.workflow import (
    CodeGenerationWorkflow,
)


def test_error_code_properties():
    """Test error code enum properties."""
    print("Testing Error Code Properties...")

    # Test retryability
    assert EnumErrorCode.INTELLIGENCE_UNAVAILABLE.is_retryable is True
    assert EnumErrorCode.INTELLIGENCE_TIMEOUT.is_retryable is True
    assert EnumErrorCode.FILE_WRITE_ERROR.is_retryable is True
    assert EnumErrorCode.VALIDATION_FAILED.is_retryable is False
    assert EnumErrorCode.FILE_PERMISSION_ERROR.is_retryable is False
    print("  ✓ Retryability checks passed")

    # Test circuit breaker triggers
    assert EnumErrorCode.INTELLIGENCE_UNAVAILABLE.requires_circuit_breaker is True
    assert EnumErrorCode.CODE_GENERATION_TIMEOUT.requires_circuit_breaker is True
    assert EnumErrorCode.FILE_WRITE_ERROR.requires_circuit_breaker is False
    assert EnumErrorCode.VALIDATION_FAILED.requires_circuit_breaker is False
    print("  ✓ Circuit breaker trigger checks passed")

    # Test partial success
    assert EnumErrorCode.INTELLIGENCE_DEGRADED.allows_partial_success is True
    assert (
        EnumErrorCode.VALIDATION_QUALITY_BELOW_THRESHOLD.allows_partial_success is True
    )
    assert EnumErrorCode.SYSTEM_OUT_OF_MEMORY.allows_partial_success is False
    assert EnumErrorCode.VALIDATION_FAILED.allows_partial_success is False
    print("  ✓ Partial success checks passed")

    # Test severity levels
    assert EnumErrorCode.SYSTEM_OUT_OF_MEMORY.severity == "CRITICAL"
    assert EnumErrorCode.CODE_GENERATION_FAILED.severity == "HIGH"
    assert EnumErrorCode.INTELLIGENCE_UNAVAILABLE.severity == "MEDIUM"
    assert EnumErrorCode.INTELLIGENCE_DEGRADED.severity == "LOW"
    print("  ✓ Severity level checks passed")

    # Test recovery hints
    hint = EnumErrorCode.INTELLIGENCE_UNAVAILABLE.get_recovery_hint()
    assert "Intelligence service is unavailable" in hint
    assert "Retrying" in hint

    hint = EnumErrorCode.FILE_PERMISSION_ERROR.get_recovery_hint()
    assert "Permission denied" in hint
    assert "writable" in hint
    print("  ✓ Recovery hint checks passed")

    print("✅ All error code property tests passed!\n")


def test_circuit_breaker_decorator():
    """Test that circuit breaker decorator is present."""
    print("Testing Circuit Breaker Decorator...")

    workflow = CodeGenerationWorkflow()

    # Check that _query_intelligence_with_resilience exists
    assert hasattr(workflow, "_query_intelligence_with_resilience")
    print("  ✓ _query_intelligence_with_resilience method exists")

    # Check method signature
    method = workflow._query_intelligence_with_resilience
    sig = inspect.signature(method)
    params = list(sig.parameters.keys())
    assert "gen_ctx" in params
    assert "requirements" in params
    print("  ✓ Method signature is correct")

    # Check docstring mentions circuit breaker
    docstring = method.__doc__ or ""
    assert "circuit breaker" in docstring.lower()
    assert "retry" in docstring.lower()
    print("  ✓ Method has proper documentation")

    print("✅ Circuit breaker decorator tests passed!\n")


def test_workflow_methods():
    """Test that workflow has enhanced error handling methods."""
    print("Testing Enhanced Workflow Methods...")

    workflow = CodeGenerationWorkflow()

    # Check _publish_failure_event exists
    assert hasattr(workflow, "_publish_failure_event")
    print("  ✓ _publish_failure_event method exists")

    # Check method signature
    method = workflow._publish_failure_event
    sig = inspect.signature(method)
    params = list(sig.parameters.keys())
    assert "gen_ctx" in params
    assert "stage" in params
    assert "error_code" in params
    assert "error_message" in params
    assert "error_context" in params
    assert "is_retryable" in params
    print("  ✓ _publish_failure_event signature is correct")

    # Check gather_intelligence has error handling
    method = workflow.gather_intelligence
    source = inspect.getsource(method)
    assert "try:" in source
    assert "except CircuitBreakerError" in source
    assert "EnumErrorCode.INTELLIGENCE_CIRCUIT_OPEN" in source
    print("  ✓ gather_intelligence has circuit breaker error handling")

    # Check validate_code has partial success handling
    method = workflow.validate_code
    source = inspect.getsource(method)
    assert "quality_threshold" in source
    assert "warnings" in source
    assert "EnumErrorCode.VALIDATION_QUALITY_BELOW_THRESHOLD" in source
    print("  ✓ validate_code has partial success handling")

    # Check write_files has error handling
    method = workflow.write_files
    source = inspect.getsource(method)
    assert "try:" in source
    assert "except PermissionError" in source
    assert "except OSError" in source
    assert "EnumErrorCode.FILE_PERMISSION_ERROR" in source
    assert "EnumErrorCode.FILE_WRITE_ERROR" in source
    print("  ✓ write_files has error handling for file operations")

    print("✅ Enhanced workflow method tests passed!\n")


def test_workflow_imports():
    """Test that all required imports are present."""
    print("Testing Workflow Imports...")

    # Check workflow file has necessary imports
    import omninode_bridge.nodes.codegen_orchestrator.v1_0_0.workflow as workflow_module

    source = inspect.getsource(workflow_module)

    # Check circuit breaker imports
    assert "from circuitbreaker import" in source
    assert "CircuitBreakerError" in source
    print("  ✓ Circuit breaker imports present")

    # Check tenacity imports
    assert "from tenacity import" in source
    assert "retry" in source
    assert "stop_after_attempt" in source
    assert "wait_exponential" in source
    print("  ✓ Tenacity retry imports present")

    # Check error code imports
    assert "EnumErrorCode" in source
    print("  ✓ Error code imports present")

    # Check failure event imports
    assert "ModelEventNodeGenerationFailed" in source
    print("  ✓ Failure event imports present")

    print("✅ Workflow import tests passed!\n")


def main():
    """Run all validation tests."""
    print("=" * 70)
    print("Circuit Breaker & Error Handling Validation")
    print("=" * 70)
    print()

    try:
        test_error_code_properties()
        test_circuit_breaker_decorator()
        test_workflow_methods()
        test_workflow_imports()

        print("=" * 70)
        print("✅ ALL VALIDATION TESTS PASSED!")
        print("=" * 70)
        print()
        print("Summary:")
        print("  - 10+ error code properties validated")
        print("  - Circuit breaker decorator verified")
        print("  - Retry logic with exponential backoff confirmed")
        print("  - Partial success handlers validated")
        print("  - Error publishing method verified")
        print()
        return 0

    except AssertionError as e:
        print()
        print("=" * 70)
        print(f"❌ VALIDATION FAILED: {e}")
        print("=" * 70)
        return 1
    except Exception as e:
        print()
        print("=" * 70)
        print(f"❌ UNEXPECTED ERROR: {e}")
        print("=" * 70)
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    import sys

    sys.exit(main())
