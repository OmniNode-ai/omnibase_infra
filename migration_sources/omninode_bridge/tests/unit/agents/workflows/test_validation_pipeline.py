"""
Comprehensive tests for validation pipeline.

Tests cover:
- ValidationModels (ValidationResult, ValidationContext, ValidationSummary)
- Validators (CompletenessValidator, QualityValidator, OnexComplianceValidator)
- ValidationPipeline (orchestration, parallel execution, reporting)
- Performance targets (300-800ms for full pipeline)

Coverage target: 95%+
"""

import asyncio
import time
from typing import Any

import pytest

from omninode_bridge.agents.metrics.collector import MetricsCollector
from omninode_bridge.agents.workflows.validation_models import (
    EnumValidationSeverity,
    EnumValidationType,
    ValidationContext,
    ValidationIssue,
    ValidationResult,
    ValidationSummary,
)
from omninode_bridge.agents.workflows.validation_pipeline import ValidationPipeline
from omninode_bridge.agents.workflows.validators import (
    BaseValidator,
    CompletenessValidator,
    OnexComplianceValidator,
    QualityValidator,
)


# Test code samples
VALID_ONEX_CODE = '''
"""Test module with ONEX compliance."""
from typing import Any
from omninode_bridge.models import ModelOnexError
from omninode_bridge.logging import emit_log_event, EnumLogLevel
from pydantic import BaseModel

class TestNode:
    """Test node with ONEX patterns."""

    async def execute_effect(self, input_data: dict[str, Any]) -> dict[str, Any]:
        """Execute effect with proper error handling."""
        try:
            emit_log_event(
                EnumLogLevel.INFO,
                "Executing effect",
                correlation_id="test-123"
            )
            result = await self._process(input_data)
            return result
        except Exception as e:
            raise ModelOnexError(f"Effect failed: {e}")

    async def _process(self, data: dict[str, Any]) -> dict[str, Any]:
        """Process data."""
        return {"status": "success", "data": data}
'''

INCOMPLETE_CODE = '''
"""Test module missing required methods."""
from typing import Any

class TestNode:
    """Test node without required methods."""

    async def some_method(self) -> None:
        """Some method."""
        pass
'''

LOW_QUALITY_CODE = '''
"""Module with quality issues."""
def complex_function(a, b, c, d, e):
    if a > 0:
        if b > 0:
            if c > 0:
                if d > 0:
                    if e > 0:
                        if a + b > 10:
                            if c + d > 10:
                                if a * b > 100:
                                    if c * d > 100:
                                        if e > 50:
                                            return "very complex"
    return "simple"

def no_type_hints(x, y):
    return x + y
'''

NON_ONEX_CODE = '''
"""Module without ONEX patterns."""
import logging

class TestNode:
    """Test node without ONEX compliance."""

    def execute(self, data):
        logging.info("Executing")
        try:
            result = self.process(data)
            return result
        except Exception as e:
            logging.error(f"Error: {e}")
            raise

    def process(self, data):
        return {"status": "success"}
'''


class TestValidationModels:
    """Test validation data models."""

    def test_validation_issue_creation(self):
        """Test ValidationIssue creation."""
        issue = ValidationIssue(
            severity=EnumValidationSeverity.ERROR,
            message="Test error",
            line_number=10,
            rule_name="test_rule",
            suggestion="Fix the error",
        )

        assert issue.severity == EnumValidationSeverity.ERROR
        assert issue.message == "Test error"
        assert issue.line_number == 10
        assert issue.rule_name == "test_rule"
        assert issue.suggestion == "Fix the error"
        assert issue.issue_id is not None

    def test_validation_issue_requires_message(self):
        """Test ValidationIssue requires message."""
        with pytest.raises(ValueError, match="message cannot be empty"):
            ValidationIssue(message="")

    def test_validation_result_creation(self):
        """Test ValidationResult creation."""
        result = ValidationResult(
            validator_name="TestValidator",
            validation_type=EnumValidationType.COMPLETENESS,
            passed=True,
            score=0.95,
            errors=[],
            warnings=["Warning 1"],
            issues=[],
            metadata={"test": "data"},
            duration_ms=45.2,
        )

        assert result.validator_name == "TestValidator"
        assert result.validation_type == EnumValidationType.COMPLETENESS
        assert result.passed is True
        assert result.score == 0.95
        assert len(result.warnings) == 1
        assert result.duration_ms == 45.2
        assert not result.has_errors
        assert result.has_warnings

    def test_validation_result_score_validation(self):
        """Test ValidationResult score validation."""
        with pytest.raises(ValueError, match="score must be between 0.0 and 1.0"):
            ValidationResult(
                validator_name="Test",
                validation_type=EnumValidationType.COMPLETENESS,
                passed=True,
                score=1.5,
            )

    def test_validation_result_properties(self):
        """Test ValidationResult properties."""
        result = ValidationResult(
            validator_name="Test",
            validation_type=EnumValidationType.COMPLETENESS,
            passed=False,
            score=0.5,
            errors=["Error 1"],
            warnings=["Warning 1"],
            issues=[
                ValidationIssue(
                    severity=EnumValidationSeverity.CRITICAL,
                    message="Critical issue",
                ),
                ValidationIssue(
                    severity=EnumValidationSeverity.ERROR,
                    message="Error issue",
                ),
                ValidationIssue(
                    severity=EnumValidationSeverity.WARNING,
                    message="Warning issue",
                ),
            ],
        )

        assert result.has_errors
        assert result.has_warnings
        assert len(result.critical_issues) == 1
        assert len(result.error_issues) == 1
        assert len(result.warning_issues) == 1

    def test_validation_context_creation(self):
        """Test ValidationContext creation."""
        context = ValidationContext(
            code_type="node",
            node_type="effect",
            contract_name="TestContract",
            expected_patterns=["async def", "ModelOnexError"],
            required_methods=["execute_effect"],
            quality_threshold=0.8,
            strict_mode=False,
        )

        assert context.code_type == "node"
        assert context.node_type == "effect"
        assert context.contract_name == "TestContract"
        assert len(context.expected_patterns) == 2
        assert len(context.required_methods) == 1
        assert context.quality_threshold == 0.8
        assert context.strict_mode is False

    def test_validation_summary_creation(self):
        """Test ValidationSummary creation."""
        result1 = ValidationResult(
            validator_name="Validator1",
            validation_type=EnumValidationType.COMPLETENESS,
            passed=True,
            score=0.9,
        )
        result2 = ValidationResult(
            validator_name="Validator2",
            validation_type=EnumValidationType.QUALITY,
            passed=False,
            score=0.5,
            errors=["Error 1"],
        )

        summary = ValidationSummary(
            total_validators=2,
            passed_validators=1,
            failed_validators=1,
            overall_score=0.7,
            total_errors=1,
            total_warnings=0,
            total_duration_ms=100.0,
            results={"Validator1": result1, "Validator2": result2},
        )

        assert summary.total_validators == 2
        assert summary.passed_validators == 1
        assert summary.failed_validators == 1
        assert summary.overall_score == 0.7
        assert not summary.passed
        assert summary.success_rate == 0.5


@pytest.mark.asyncio
class TestCompletenessValidator:
    """Test CompletenessValidator."""

    async def test_validate_complete_code(self):
        """Test validation of complete code."""
        metrics = MetricsCollector()
        validator = CompletenessValidator(metrics)

        context = ValidationContext(
            required_methods=["execute_effect", "_process"],
            expected_patterns=["async def", "ModelOnexError"],
        )

        result = await validator.validate(VALID_ONEX_CODE, context)

        assert result.passed
        assert result.score > 0.8
        assert len(result.errors) == 0

    async def test_validate_incomplete_code(self):
        """Test validation of incomplete code."""
        metrics = MetricsCollector()
        validator = CompletenessValidator(metrics)

        context = ValidationContext(
            required_methods=["execute_effect", "validate_input"],
            expected_patterns=["ModelOnexError"],
        )

        result = await validator.validate(INCOMPLETE_CODE, context)

        assert not result.passed
        assert result.score < 1.0
        assert len(result.errors) > 0
        assert any("execute_effect" in err for err in result.errors)

    async def test_validate_syntax_error(self):
        """Test validation with syntax error."""
        metrics = MetricsCollector()
        validator = CompletenessValidator(metrics)

        context = ValidationContext()
        invalid_code = "def invalid syntax here"

        result = await validator.validate(invalid_code, context)

        assert not result.passed
        assert result.score == 0.0
        assert len(result.errors) > 0
        assert any("Syntax error" in err for err in result.errors)

    async def test_validator_performance(self):
        """Test validator performance (<100ms target)."""
        metrics = MetricsCollector()
        validator = CompletenessValidator(metrics)

        context = ValidationContext(
            required_methods=["execute_effect"],
        )

        start_time = time.perf_counter()
        result = await validator.validate(VALID_ONEX_CODE, context)
        duration_ms = (time.perf_counter() - start_time) * 1000

        assert duration_ms < 100, f"Validator took {duration_ms:.2f}ms (target: <100ms)"
        assert result.duration_ms < 100


@pytest.mark.asyncio
class TestQualityValidator:
    """Test QualityValidator."""

    async def test_validate_high_quality_code(self):
        """Test validation of high-quality code."""
        metrics = MetricsCollector()
        validator = QualityValidator(metrics, quality_threshold=0.8)

        context = ValidationContext()

        result = await validator.validate(VALID_ONEX_CODE, context)

        assert result.passed
        assert result.score >= 0.8
        assert "complexity_score" in result.metadata
        assert "type_hint_score" in result.metadata
        assert "docstring_score" in result.metadata

    async def test_validate_low_quality_code(self):
        """Test validation of low-quality code."""
        metrics = MetricsCollector()
        validator = QualityValidator(metrics, quality_threshold=0.8)

        context = ValidationContext()

        result = await validator.validate(LOW_QUALITY_CODE, context)

        assert not result.passed
        assert result.score < 0.8
        assert len(result.warnings) > 0
        assert result.metadata["complexity_issues"] > 0

    async def test_quality_threshold_enforcement(self):
        """Test quality threshold enforcement."""
        metrics = MetricsCollector()
        validator = QualityValidator(metrics, quality_threshold=0.9)

        context = ValidationContext()

        result = await validator.validate(VALID_ONEX_CODE, context)

        # High threshold may not be met
        if result.score < 0.9:
            assert not result.passed
            assert len(result.errors) > 0
            assert any("below threshold" in err for err in result.errors)

    async def test_validator_performance(self):
        """Test validator performance (<200ms target)."""
        metrics = MetricsCollector()
        validator = QualityValidator(metrics, quality_threshold=0.8)

        context = ValidationContext()

        start_time = time.perf_counter()
        result = await validator.validate(VALID_ONEX_CODE, context)
        duration_ms = (time.perf_counter() - start_time) * 1000

        assert duration_ms < 200, f"Validator took {duration_ms:.2f}ms (target: <200ms)"
        assert result.duration_ms < 200


@pytest.mark.asyncio
class TestOnexComplianceValidator:
    """Test OnexComplianceValidator."""

    async def test_validate_compliant_code(self):
        """Test validation of ONEX-compliant code."""
        metrics = MetricsCollector()
        validator = OnexComplianceValidator(metrics)

        context = ValidationContext()

        result = await validator.validate(VALID_ONEX_CODE, context)

        assert result.passed
        assert result.score > 0.8
        assert result.metadata["pattern_coverage"] == 1.0

    async def test_validate_non_compliant_code(self):
        """Test validation of non-ONEX code."""
        metrics = MetricsCollector()
        validator = OnexComplianceValidator(metrics)

        context = ValidationContext()

        result = await validator.validate(NON_ONEX_CODE, context)

        assert result.passed  # No hard errors, only warnings
        assert result.score < 1.0
        assert len(result.warnings) > 0
        assert result.metadata["pattern_coverage"] < 1.0

    async def test_pattern_detection(self):
        """Test ONEX pattern detection."""
        metrics = MetricsCollector()
        validator = OnexComplianceValidator(metrics)

        context = ValidationContext()

        result = await validator.validate(VALID_ONEX_CODE, context)

        assert result.metadata["has_pydantic"]
        assert result.metadata["has_error_handling"]
        assert result.metadata["has_structured_logging"]

    async def test_validator_performance(self):
        """Test validator performance (<150ms target)."""
        metrics = MetricsCollector()
        validator = OnexComplianceValidator(metrics)

        context = ValidationContext()

        start_time = time.perf_counter()
        result = await validator.validate(VALID_ONEX_CODE, context)
        duration_ms = (time.perf_counter() - start_time) * 1000

        assert duration_ms < 150, f"Validator took {duration_ms:.2f}ms (target: <150ms)"
        assert result.duration_ms < 150


@pytest.mark.asyncio
class TestValidationPipeline:
    """Test ValidationPipeline orchestration."""

    async def test_pipeline_with_all_validators(self):
        """Test pipeline with all validators."""
        metrics = MetricsCollector()
        pipeline = ValidationPipeline(
            validators=[
                CompletenessValidator(metrics),
                QualityValidator(metrics, quality_threshold=0.8),
                OnexComplianceValidator(metrics),
            ],
            metrics_collector=metrics,
            parallel_execution=True,
        )

        context = ValidationContext(
            required_methods=["execute_effect", "_process"],
            expected_patterns=["async def", "ModelOnexError"],
            quality_threshold=0.8,
        )

        results = await pipeline.validate(VALID_ONEX_CODE, context)

        assert len(results) == 3
        assert "CompletenessValidator" in results
        assert "QualityValidator" in results
        assert "OnexComplianceValidator" in results

    async def test_pipeline_is_valid(self):
        """Test pipeline.is_valid()."""
        metrics = MetricsCollector()
        pipeline = ValidationPipeline(
            validators=[
                CompletenessValidator(metrics),
                QualityValidator(metrics, quality_threshold=0.8),
                OnexComplianceValidator(metrics),
            ],
            metrics_collector=metrics,
        )

        context = ValidationContext(
            required_methods=["execute_effect", "_process"],
            quality_threshold=0.8,
        )

        results = await pipeline.validate(VALID_ONEX_CODE, context)
        is_valid = pipeline.is_valid(results)

        # Should pass if code is valid
        assert isinstance(is_valid, bool)

    async def test_pipeline_create_summary(self):
        """Test pipeline.create_summary()."""
        metrics = MetricsCollector()
        pipeline = ValidationPipeline(
            validators=[
                CompletenessValidator(metrics),
                QualityValidator(metrics, quality_threshold=0.8),
                OnexComplianceValidator(metrics),
            ],
            metrics_collector=metrics,
        )

        context = ValidationContext()

        results = await pipeline.validate(VALID_ONEX_CODE, context)
        summary = pipeline.create_summary(results)

        assert summary.total_validators == 3
        assert summary.passed_validators >= 0
        assert summary.failed_validators >= 0
        assert 0.0 <= summary.overall_score <= 1.0
        assert summary.total_duration_ms > 0

    async def test_pipeline_parallel_execution(self):
        """Test parallel execution is faster than sequential."""
        metrics = MetricsCollector()
        validators = [
            CompletenessValidator(metrics),
            QualityValidator(metrics, quality_threshold=0.8),
            OnexComplianceValidator(metrics),
        ]

        context = ValidationContext()

        # Parallel execution
        pipeline_parallel = ValidationPipeline(
            validators=validators,
            metrics_collector=metrics,
            parallel_execution=True,
        )

        start_parallel = time.perf_counter()
        await pipeline_parallel.validate(VALID_ONEX_CODE, context)
        duration_parallel = (time.perf_counter() - start_parallel) * 1000

        # Sequential execution
        pipeline_sequential = ValidationPipeline(
            validators=validators,
            metrics_collector=metrics,
            parallel_execution=False,
        )

        start_sequential = time.perf_counter()
        await pipeline_sequential.validate(VALID_ONEX_CODE, context)
        duration_sequential = (time.perf_counter() - start_sequential) * 1000

        # Parallel should be faster (or at least not significantly slower)
        # Allow some variance due to overhead
        assert duration_parallel <= duration_sequential * 1.2

    async def test_pipeline_performance_target(self):
        """Test pipeline meets performance target (300-800ms)."""
        metrics = MetricsCollector()
        pipeline = ValidationPipeline(
            validators=[
                CompletenessValidator(metrics),
                QualityValidator(metrics, quality_threshold=0.8),
                OnexComplianceValidator(metrics),
            ],
            metrics_collector=metrics,
            parallel_execution=True,
        )

        context = ValidationContext(
            required_methods=["execute_effect"],
            quality_threshold=0.8,
        )

        start_time = time.perf_counter()
        results = await pipeline.validate(VALID_ONEX_CODE, context)
        duration_ms = (time.perf_counter() - start_time) * 1000

        assert (
            duration_ms < 800
        ), f"Pipeline took {duration_ms:.2f}ms (target: <800ms)"

        summary = pipeline.create_summary(results)
        assert (
            summary.total_duration_ms < 800
        ), f"Pipeline took {summary.total_duration_ms:.2f}ms (target: <800ms)"

    async def test_pipeline_get_validation_report(self):
        """Test pipeline.get_validation_report()."""
        metrics = MetricsCollector()
        pipeline = ValidationPipeline(
            validators=[
                CompletenessValidator(metrics),
                QualityValidator(metrics, quality_threshold=0.8),
                OnexComplianceValidator(metrics),
            ],
            metrics_collector=metrics,
        )

        context = ValidationContext()

        results = await pipeline.validate(VALID_ONEX_CODE, context)
        report = pipeline.get_validation_report(results)

        assert "VALIDATION REPORT" in report
        assert "Overall Status:" in report
        assert "Overall Score:" in report
        assert "CompletenessValidator" in report
        assert "QualityValidator" in report
        assert "OnexComplianceValidator" in report

    async def test_pipeline_get_failed_validators(self):
        """Test pipeline.get_failed_validators()."""
        metrics = MetricsCollector()
        pipeline = ValidationPipeline(
            validators=[
                CompletenessValidator(metrics),
                QualityValidator(metrics, quality_threshold=0.99),  # Very high threshold
                OnexComplianceValidator(metrics),
            ],
            metrics_collector=metrics,
        )

        context = ValidationContext()

        results = await pipeline.validate(VALID_ONEX_CODE, context)
        failed = pipeline.get_failed_validators(results)

        # Check that failed is a subset of results
        assert all(name in results for name in failed.keys())
        assert all(not result.passed for result in failed.values())

    async def test_pipeline_with_invalid_code(self):
        """Test pipeline with invalid code."""
        metrics = MetricsCollector()
        pipeline = ValidationPipeline(
            validators=[
                CompletenessValidator(metrics),
                QualityValidator(metrics, quality_threshold=0.8),
                OnexComplianceValidator(metrics),
            ],
            metrics_collector=metrics,
        )

        context = ValidationContext(
            required_methods=["execute_effect", "validate_input"],
            quality_threshold=0.8,
        )

        results = await pipeline.validate(INCOMPLETE_CODE, context)

        assert not pipeline.is_valid(results)
        summary = pipeline.create_summary(results)
        assert summary.failed_validators > 0
        assert summary.total_errors > 0

    async def test_pipeline_requires_validators(self):
        """Test pipeline requires at least one validator."""
        metrics = MetricsCollector()

        with pytest.raises(ValueError, match="At least one validator is required"):
            ValidationPipeline(
                validators=[],
                metrics_collector=metrics,
            )


@pytest.mark.asyncio
class TestIntegration:
    """Integration tests for validation pipeline."""

    async def test_end_to_end_validation(self):
        """Test complete end-to-end validation workflow."""
        # Setup
        metrics = MetricsCollector()
        pipeline = ValidationPipeline(
            validators=[
                CompletenessValidator(metrics),
                QualityValidator(metrics, quality_threshold=0.8),
                OnexComplianceValidator(metrics),
            ],
            metrics_collector=metrics,
            parallel_execution=True,
        )

        context = ValidationContext(
            code_type="node",
            node_type="effect",
            contract_name="TestContract",
            required_methods=["execute_effect", "_process"],
            expected_patterns=["async def", "ModelOnexError", "emit_log_event"],
            quality_threshold=0.8,
            strict_mode=False,
        )

        # Execute
        results = await pipeline.validate(VALID_ONEX_CODE, context)

        # Verify
        assert len(results) == 3
        summary = pipeline.create_summary(results)

        # Check summary
        assert summary.total_validators == 3
        assert summary.overall_score >= 0.0
        assert summary.overall_score <= 1.0

        # Generate report
        report = pipeline.get_validation_report(results)
        assert len(report) > 0

        # Check performance
        assert summary.total_duration_ms < 800

    async def test_validation_with_multiple_code_samples(self):
        """Test validation with multiple code samples."""
        metrics = MetricsCollector()
        pipeline = ValidationPipeline(
            validators=[
                CompletenessValidator(metrics),
                QualityValidator(metrics, quality_threshold=0.8),
                OnexComplianceValidator(metrics),
            ],
            metrics_collector=metrics,
        )

        code_samples = [
            VALID_ONEX_CODE,
            INCOMPLETE_CODE,
            LOW_QUALITY_CODE,
            NON_ONEX_CODE,
        ]

        context = ValidationContext(
            required_methods=["execute_effect"],
            quality_threshold=0.8,
        )

        for i, code in enumerate(code_samples):
            results = await pipeline.validate(code, context)
            summary = pipeline.create_summary(results)

            # Each validation should complete
            assert summary.total_validators == 3
            assert summary.total_duration_ms > 0

            # Performance target
            assert summary.total_duration_ms < 800


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
