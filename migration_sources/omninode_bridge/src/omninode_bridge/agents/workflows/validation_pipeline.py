"""
Validation pipeline for multi-stage code validation.

Orchestrates multiple validators in parallel with aggregation and reporting.

Performance targets:
- Full pipeline (3 validators): 300-800ms
- Parallel execution when possible
- Aggregated results with summary

Example:
    ```python
    from omninode_bridge.agents.metrics import MetricsCollector
    from omninode_bridge.agents.workflows import ValidationPipeline
    from omninode_bridge.agents.workflows.validators import (
        CompletenessValidator,
        QualityValidator,
        OnexComplianceValidator,
    )

    # Create pipeline
    collector = MetricsCollector()
    pipeline = ValidationPipeline(
        validators=[
            CompletenessValidator(collector),
            QualityValidator(collector, quality_threshold=0.8),
            OnexComplianceValidator(collector),
        ],
        metrics_collector=collector,
    )

    # Validate code
    context = ValidationContext(
        code_type="node",
        required_methods=["execute_effect"],
        expected_patterns=["async def", "ModelOnexError"],
        quality_threshold=0.8
    )

    results = await pipeline.validate(code, context)
    summary = pipeline.create_summary(results)

    if pipeline.is_valid(results):
        print("All validations passed!")
        print(f"Overall score: {summary.overall_score:.2f}")
    else:
        print("Validation failed!")
        for name, result in results.items():
            if not result.passed:
                print(f"{name}: {result.errors}")
    ```
"""

import asyncio
import logging
import time
from typing import Any

from omninode_bridge.agents.metrics.collector import MetricsCollector
from omninode_bridge.agents.workflows.validation_models import (
    ValidationContext,
    ValidationResult,
    ValidationSummary,
)
from omninode_bridge.agents.workflows.validators import BaseValidator

logger = logging.getLogger(__name__)


class ValidationPipeline:
    """
    Multi-stage validation pipeline.

    Orchestrates multiple validators in parallel, aggregates results,
    and provides summary reporting.

    Performance:
    - Full pipeline (3 validators): 300-800ms
    - Parallel execution reduces total time
    - Metrics collection for observability

    Attributes:
        validators: List of validators to execute
        metrics: MetricsCollector for tracking performance
        parallel_execution: Enable parallel validator execution (default: True)
    """

    def __init__(
        self,
        validators: list[BaseValidator],
        metrics_collector: MetricsCollector,
        parallel_execution: bool = True,
    ):
        """
        Initialize validation pipeline.

        Args:
            validators: List of validators to execute
            metrics_collector: Metrics collector for tracking
            parallel_execution: Enable parallel execution (default: True)
        """
        if not validators:
            raise ValueError("At least one validator is required")

        self.validators = validators
        self.metrics = metrics_collector
        self.parallel_execution = parallel_execution

        logger.info(
            f"ValidationPipeline initialized with {len(validators)} validators "
            f"(parallel={parallel_execution})"
        )

    async def validate(
        self, code: str, context: ValidationContext
    ) -> dict[str, ValidationResult]:
        """
        Run all validators and return results.

        Args:
            code: Python code to validate
            context: Validation context

        Returns:
            Dictionary mapping validator names to ValidationResult

        Performance:
        - Parallel: 300-800ms (max of all validators)
        - Sequential: 450-800ms (sum of all validators)
        """
        start_time = time.perf_counter()

        try:
            if self.parallel_execution:
                # Run validators in parallel
                tasks = [
                    validator.validate(code, context) for validator in self.validators
                ]
                results_list = await asyncio.gather(*tasks, return_exceptions=True)

                # Build results dict
                results = {}
                for validator, result in zip(self.validators, results_list):
                    if isinstance(result, Exception):
                        logger.error(
                            f"Validator {validator.validator_name} failed: {result}"
                        )
                        # Create error result
                        results[validator.validator_name] = ValidationResult(
                            validator_name=validator.validator_name,
                            validation_type=validator.__class__.__name__,
                            passed=False,
                            score=0.0,
                            errors=[f"Validation error: {str(result)}"],
                            warnings=[],
                            issues=[],
                            metadata={},
                            duration_ms=0.0,
                            correlation_id=context.correlation_id,
                        )
                    else:
                        results[validator.validator_name] = result

            else:
                # Run validators sequentially
                results = {}
                for validator in self.validators:
                    try:
                        result = await validator.validate(code, context)
                        results[validator.validator_name] = result
                    except Exception as e:
                        logger.error(
                            f"Validator {validator.validator_name} failed: {e}"
                        )
                        results[validator.validator_name] = ValidationResult(
                            validator_name=validator.validator_name,
                            validation_type=validator.__class__.__name__,
                            passed=False,
                            score=0.0,
                            errors=[f"Validation error: {str(e)}"],
                            warnings=[],
                            issues=[],
                            metadata={},
                            duration_ms=0.0,
                            correlation_id=context.correlation_id,
                        )

            # Record overall metrics
            duration_ms = (time.perf_counter() - start_time) * 1000
            await self._record_pipeline_metrics(results, duration_ms)

            logger.info(
                f"Validation pipeline completed in {duration_ms:.2f}ms "
                f"({len(results)} validators)"
            )

            return results

        except Exception as e:
            duration_ms = (time.perf_counter() - start_time) * 1000
            logger.error(f"Validation pipeline error: {e}")
            await self.metrics.record_timing(
                "validation_pipeline_duration_ms",
                duration_ms,
                tags={"status": "error"},
                correlation_id=context.correlation_id,
            )
            raise

    def is_valid(self, results: dict[str, ValidationResult]) -> bool:
        """
        Check if all validations passed.

        Args:
            results: Validation results from validate()

        Returns:
            True if all validations passed, False otherwise
        """
        return all(result.passed for result in results.values())

    def create_summary(
        self, results: dict[str, ValidationResult]
    ) -> ValidationSummary:
        """
        Create validation summary from results.

        Args:
            results: Validation results from validate()

        Returns:
            ValidationSummary with aggregated metrics
        """
        total_validators = len(results)
        passed_validators = sum(1 for result in results.values() if result.passed)
        failed_validators = total_validators - passed_validators

        # Calculate overall score (weighted average)
        if total_validators > 0:
            overall_score = (
                sum(result.score for result in results.values()) / total_validators
            )
        else:
            overall_score = 0.0

        # Count total errors and warnings
        total_errors = sum(len(result.errors) for result in results.values())
        total_warnings = sum(len(result.warnings) for result in results.values())

        # Calculate total duration
        total_duration_ms = sum(result.duration_ms for result in results.values())

        # Get correlation_id from first result
        correlation_id = None
        if results:
            correlation_id = next(iter(results.values())).correlation_id

        return ValidationSummary(
            total_validators=total_validators,
            passed_validators=passed_validators,
            failed_validators=failed_validators,
            overall_score=overall_score,
            total_errors=total_errors,
            total_warnings=total_warnings,
            total_duration_ms=total_duration_ms,
            results=results,
            correlation_id=correlation_id,
        )

    def get_failed_validators(
        self, results: dict[str, ValidationResult]
    ) -> dict[str, ValidationResult]:
        """
        Get only failed validators from results.

        Args:
            results: Validation results from validate()

        Returns:
            Dictionary of failed validators
        """
        return {
            name: result for name, result in results.items() if not result.passed
        }

    def get_validation_report(self, results: dict[str, ValidationResult]) -> str:
        """
        Generate human-readable validation report.

        Args:
            results: Validation results from validate()

        Returns:
            Formatted validation report string
        """
        summary = self.create_summary(results)

        lines = [
            "=" * 80,
            "VALIDATION REPORT",
            "=" * 80,
            "",
            f"Overall Status: {'PASSED' if summary.passed else 'FAILED'}",
            f"Overall Score: {summary.overall_score:.2f} / 1.00",
            f"Success Rate: {summary.success_rate:.1%}",
            "",
            f"Total Validators: {summary.total_validators}",
            f"Passed: {summary.passed_validators}",
            f"Failed: {summary.failed_validators}",
            "",
            f"Total Errors: {summary.total_errors}",
            f"Total Warnings: {summary.total_warnings}",
            f"Total Duration: {summary.total_duration_ms:.2f}ms",
            "",
            "-" * 80,
            "VALIDATOR RESULTS",
            "-" * 80,
        ]

        for name, result in results.items():
            status = " PASSED" if result.passed else " FAILED"
            lines.append("")
            lines.append(f"{name}: {status}")
            lines.append(f"  Score: {result.score:.2f}")
            lines.append(f"  Duration: {result.duration_ms:.2f}ms")

            if result.errors:
                lines.append(f"  Errors ({len(result.errors)}):")
                for error in result.errors:
                    lines.append(f"    - {error}")

            if result.warnings:
                lines.append(f"  Warnings ({len(result.warnings)}):")
                for warning in result.warnings[:5]:  # Show first 5
                    lines.append(f"    - {warning}")
                if len(result.warnings) > 5:
                    lines.append(f"    ... and {len(result.warnings) - 5} more")

        lines.append("")
        lines.append("=" * 80)

        return "\n".join(lines)

    async def _record_pipeline_metrics(
        self, results: dict[str, ValidationResult], duration_ms: float
    ) -> None:
        """Record pipeline-level metrics."""
        summary = self.create_summary(results)

        # Overall duration
        await self.metrics.record_timing(
            "validation_pipeline_duration_ms",
            duration_ms,
            tags={
                "status": "passed" if summary.passed else "failed",
                "validator_count": str(len(results)),
                "parallel": str(self.parallel_execution),
            },
            correlation_id=summary.correlation_id,
        )

        # Overall score
        await self.metrics.record_gauge(
            "validation_pipeline_overall_score",
            summary.overall_score,
            unit="score",
            tags={
                "status": "passed" if summary.passed else "failed",
            },
            correlation_id=summary.correlation_id,
        )

        # Success rate
        await self.metrics.record_rate(
            "validation_pipeline_success_rate",
            summary.success_rate * 100.0,
            tags={
                "status": "passed" if summary.passed else "failed",
            },
            correlation_id=summary.correlation_id,
        )

        # Error and warning counts
        await self.metrics.record_gauge(
            "validation_pipeline_error_count",
            float(summary.total_errors),
            unit="count",
            correlation_id=summary.correlation_id,
        )

        await self.metrics.record_gauge(
            "validation_pipeline_warning_count",
            float(summary.total_warnings),
            unit="count",
            correlation_id=summary.correlation_id,
        )

        # Pipeline execution count
        await self.metrics.record_counter(
            "validation_pipeline_execution_count",
            1,
            tags={
                "status": "passed" if summary.passed else "failed",
            },
            correlation_id=summary.correlation_id,
        )
