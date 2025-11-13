#!/usr/bin/env python3
"""
Test Failure Analyzer for Generated Code.

Analyzes test failures and provides actionable recommendations for fixing
generated code issues. Classifies failures by root cause and suggests
specific remediation strategies.

Failure Categories:
1. Import Errors - Missing dependencies or incorrect imports
2. Type Errors - Type annotation or validation issues
3. Runtime Errors - Logic errors in implementation
4. Assertion Errors - Incorrect test expectations or behavior
5. Configuration Errors - Missing or invalid configuration
6. Infrastructure Errors - External service unavailability

Provides:
- Root cause analysis
- Affected files identification
- Specific fix recommendations
- Severity assessment
- Auto-fix feasibility
"""

import logging
import re
from enum import Enum

from pydantic import BaseModel, Field

from .test_executor import ModelTestResults

logger = logging.getLogger(__name__)


class EnumFailureSeverity(str, Enum):
    """Severity levels for test failures."""

    CRITICAL = "critical"  # Blocks all functionality
    HIGH = "high"  # Major functionality broken
    MEDIUM = "medium"  # Partial functionality affected
    LOW = "low"  # Minor issues, edge cases
    INFO = "info"  # Warnings, suggestions


class EnumFailureCategory(str, Enum):
    """Categories of test failures."""

    IMPORT_ERROR = "import_error"  # Missing dependencies
    TYPE_ERROR = "type_error"  # Type annotation issues
    RUNTIME_ERROR = "runtime_error"  # Logic errors
    ASSERTION_ERROR = "assertion_error"  # Test expectations wrong
    CONFIGURATION_ERROR = "configuration_error"  # Config issues
    INFRASTRUCTURE_ERROR = "infrastructure_error"  # External services
    TIMEOUT_ERROR = "timeout_error"  # Performance issues
    UNKNOWN = "unknown"  # Unclassified


class ModelFailureCause(BaseModel):
    """Root cause analysis for a failure."""

    category: EnumFailureCategory = Field(..., description="Failure category")
    description: str = Field(..., description="Human-readable cause description")
    affected_files: list[str] = Field(
        default_factory=list, description="Files affected by this cause"
    )
    error_patterns: list[str] = Field(
        default_factory=list, description="Error patterns identified"
    )


class ModelFailureAnalysis(BaseModel):
    """
    Comprehensive analysis of test failures.

    Provides root cause analysis, affected components, recommendations,
    and auto-fix feasibility assessment.
    """

    # Root causes
    root_causes: list[ModelFailureCause] = Field(
        default_factory=list, description="Identified root causes"
    )

    # Impact
    affected_files: list[str] = Field(
        default_factory=list, description="All files affected by failures"
    )
    affected_tests: list[str] = Field(
        default_factory=list, description="Failed test names"
    )

    # Recommendations
    recommended_fixes: list[str] = Field(
        default_factory=list, description="Specific fix recommendations"
    )

    # Assessment
    severity: EnumFailureSeverity = Field(..., description="Overall severity")
    auto_fixable: bool = Field(
        default=False, description="Whether failures can be auto-fixed"
    )
    estimated_fix_time_minutes: int = Field(
        default=30, ge=0, description="Estimated time to fix manually"
    )

    # Metrics
    failure_count: int = Field(..., ge=0, description="Total number of failures")
    unique_error_types: int = Field(
        ..., ge=0, description="Number of unique error types"
    )

    # Summary
    summary: str = Field(..., description="Executive summary of issues")

    def get_report(self) -> str:
        """
        Generate human-readable failure analysis report.

        Returns:
            Formatted markdown report
        """
        report_lines = [
            "# Test Failure Analysis Report",
            "",
            f"**Severity**: {self.severity.value.upper()}",
            f"**Failures**: {self.failure_count}",
            f"**Unique Error Types**: {self.unique_error_types}",
            f"**Auto-Fixable**: {'Yes' if self.auto_fixable else 'No'}",
            f"**Est. Fix Time**: {self.estimated_fix_time_minutes} minutes",
            "",
            "## Summary",
            "",
            self.summary,
            "",
        ]

        # Root causes
        if self.root_causes:
            report_lines.extend(
                [
                    "## Root Causes",
                    "",
                ]
            )
            for i, cause in enumerate(self.root_causes, 1):
                report_lines.extend(
                    [
                        f"### {i}. {cause.category.value.replace('_', ' ').title()}",
                        "",
                        cause.description,
                        "",
                    ]
                )
                if cause.affected_files:
                    report_lines.append("**Affected Files**:")
                    for file in cause.affected_files:
                        report_lines.append(f"- `{file}`")
                    report_lines.append("")

        # Recommendations
        if self.recommended_fixes:
            report_lines.extend(
                [
                    "## Recommended Fixes",
                    "",
                ]
            )
            for i, fix in enumerate(self.recommended_fixes, 1):
                report_lines.append(f"{i}. {fix}")
            report_lines.append("")

        # Failed tests
        if self.affected_tests:
            report_lines.extend(
                [
                    "## Failed Tests",
                    "",
                ]
            )
            for test in self.affected_tests[:10]:  # Show first 10
                report_lines.append(f"- `{test}`")
            if len(self.affected_tests) > 10:
                report_lines.append(f"- ... and {len(self.affected_tests) - 10} more")
            report_lines.append("")

        return "\n".join(report_lines)


class FailureAnalyzer:
    """
    Analyze test failures and suggest fixes.

    Classifies failures by root cause, identifies affected components,
    and provides actionable remediation recommendations.

    Example:
        >>> analyzer = FailureAnalyzer()
        >>> analysis = analyzer.analyze(test_results)
        >>> print(analysis.get_report())
        >>> for fix in analysis.recommended_fixes:
        ...     print(f"Fix: {fix}")
    """

    def __init__(self):
        """Initialize failure analyzer."""
        # Error pattern matchers
        self.import_error_patterns = [
            r"ModuleNotFoundError: No module named '(\w+)'",
            r"ImportError: cannot import name '(\w+)'",
            r"ImportError: No module named '(\w+)'",
        ]

        self.type_error_patterns = [
            r"TypeError: .* expected .* got .*",
            r"ValidationError: .*",
            r"pydantic.*ValidationError",
        ]

        self.runtime_error_patterns = [
            r"AttributeError: .* has no attribute .*",
            r"KeyError: .*",
            r"ValueError: .*",
            r"RuntimeError: .*",
        ]

        self.assertion_error_patterns = [
            r"AssertionError: .*",
            r"assert .* == .*",
        ]

    def analyze(self, test_results: ModelTestResults) -> ModelFailureAnalysis:
        """
        Analyze test failures and provide recommendations.

        Args:
            test_results: Test execution results from TestExecutor

        Returns:
            ModelFailureAnalysis with root causes and recommendations

        Example:
            >>> analysis = analyzer.analyze(test_results)
            >>> if not analysis.auto_fixable:
            ...     print(f"Manual fixes required: {len(analysis.recommended_fixes)}")
            >>> print(analysis.summary)
        """
        if test_results.is_passing:
            return ModelFailureAnalysis(
                root_causes=[],
                affected_files=[],
                affected_tests=[],
                recommended_fixes=[],
                severity=EnumFailureSeverity.INFO,
                auto_fixable=True,
                estimated_fix_time_minutes=0,
                failure_count=0,
                unique_error_types=0,
                summary="All tests passed! No failures to analyze.",
            )

        # Analyze failures
        root_causes = self._classify_failures(test_results)
        affected_files = self._extract_affected_files(test_results)
        affected_tests = [test["name"] for test in test_results.failed_tests]

        # Generate recommendations
        recommended_fixes = self._generate_recommendations(root_causes, test_results)

        # Assess severity
        severity = self._assess_severity(test_results, root_causes)

        # Check auto-fix feasibility
        auto_fixable = self._is_auto_fixable(root_causes)

        # Estimate fix time
        fix_time = self._estimate_fix_time(root_causes)

        # Generate summary
        summary = self._generate_summary(test_results, root_causes)

        # Count unique error types
        unique_errors = len({cause.category for cause in root_causes})

        return ModelFailureAnalysis(
            root_causes=root_causes,
            affected_files=affected_files,
            affected_tests=affected_tests,
            recommended_fixes=recommended_fixes,
            severity=severity,
            auto_fixable=auto_fixable,
            estimated_fix_time_minutes=fix_time,
            failure_count=test_results.failed,
            unique_error_types=unique_errors,
            summary=summary,
        )

    def _classify_failures(
        self, test_results: ModelTestResults
    ) -> list[ModelFailureCause]:
        """
        Classify failures by root cause.

        Args:
            test_results: Test results to analyze

        Returns:
            List of identified root causes
        """
        causes = []
        categorized = set()  # Track which failures we've categorized

        # Check for import errors
        import_causes = self._check_import_errors(test_results)
        if import_causes:
            causes.extend(import_causes)
            categorized.update(
                {
                    test["name"]
                    for cause in import_causes
                    for test in test_results.failed_tests
                    if any(
                        pattern in str(test.get("error", ""))
                        for pattern in cause.error_patterns
                    )
                }
            )

        # Check for type errors
        type_causes = self._check_type_errors(test_results)
        if type_causes:
            causes.extend(type_causes)
            categorized.update(
                {
                    test["name"]
                    for cause in type_causes
                    for test in test_results.failed_tests
                    if any(
                        pattern in str(test.get("error", ""))
                        for pattern in cause.error_patterns
                    )
                }
            )

        # Check for assertion errors
        assertion_causes = self._check_assertion_errors(test_results)
        if assertion_causes:
            causes.extend(assertion_causes)
            categorized.update(
                {
                    test["name"]
                    for cause in assertion_causes
                    for test in test_results.failed_tests
                    if any(
                        pattern in str(test.get("error", ""))
                        for pattern in cause.error_patterns
                    )
                }
            )

        # Catch remaining failures as runtime errors
        uncategorized = [
            test
            for test in test_results.failed_tests
            if test["name"] not in categorized
        ]
        if uncategorized:
            runtime_cause = ModelFailureCause(
                category=EnumFailureCategory.RUNTIME_ERROR,
                description=f"Runtime errors in {len(uncategorized)} test(s)",
                affected_files=list(
                    {test.get("file", "unknown") for test in uncategorized}
                ),
                error_patterns=["RuntimeError", "Exception"],
            )
            causes.append(runtime_cause)

        return causes

    def _check_import_errors(
        self, test_results: ModelTestResults
    ) -> list[ModelFailureCause]:
        """Check for import/dependency errors."""
        causes = []
        missing_modules = set()

        for test in test_results.failed_tests:
            error_msg = str(test.get("error", ""))
            for pattern in self.import_error_patterns:
                if match := re.search(pattern, error_msg):
                    missing_modules.add(match.group(1))

        if missing_modules:
            description = (
                f"Missing Python dependencies: {', '.join(sorted(missing_modules))}"
            )
            causes.append(
                ModelFailureCause(
                    category=EnumFailureCategory.IMPORT_ERROR,
                    description=description,
                    affected_files=["pyproject.toml", "requirements.txt"],
                    error_patterns=list(missing_modules),
                )
            )

        return causes

    def _check_type_errors(
        self, test_results: ModelTestResults
    ) -> list[ModelFailureCause]:
        """Check for type annotation and validation errors."""
        causes = []
        type_issues = []

        for test in test_results.failed_tests:
            error_msg = str(test.get("error", ""))
            for pattern in self.type_error_patterns:
                if re.search(pattern, error_msg):
                    type_issues.append(test)
                    break

        if type_issues:
            affected_files = list({test.get("file", "unknown") for test in type_issues})
            description = (
                f"Type validation errors in {len(type_issues)} test(s). "
                "Generated code may have incorrect type annotations or "
                "Pydantic model validation issues."
            )
            causes.append(
                ModelFailureCause(
                    category=EnumFailureCategory.TYPE_ERROR,
                    description=description,
                    affected_files=affected_files,
                    error_patterns=["TypeError", "ValidationError"],
                )
            )

        return causes

    def _check_assertion_errors(
        self, test_results: ModelTestResults
    ) -> list[ModelFailureCause]:
        """Check for assertion failures (test expectations)."""
        causes = []
        assertion_failures = []

        for test in test_results.failed_tests:
            error_msg = str(test.get("error", ""))
            for pattern in self.assertion_error_patterns:
                if re.search(pattern, error_msg):
                    assertion_failures.append(test)
                    break

        if assertion_failures:
            affected_files = list(
                {test.get("file", "unknown") for test in assertion_failures}
            )
            description = (
                f"Assertion failures in {len(assertion_failures)} test(s). "
                "Generated code behavior doesn't match test expectations. "
                "This may indicate template logic errors or incomplete implementations."
            )
            causes.append(
                ModelFailureCause(
                    category=EnumFailureCategory.ASSERTION_ERROR,
                    description=description,
                    affected_files=affected_files,
                    error_patterns=["AssertionError"],
                )
            )

        return causes

    def _extract_affected_files(self, test_results: ModelTestResults) -> list[str]:
        """Extract unique list of affected files."""
        files = set()
        for test in test_results.failed_tests:
            if file := test.get("file"):
                files.add(file)
        return sorted(files)

    def _generate_recommendations(
        self, root_causes: list[ModelFailureCause], test_results: ModelTestResults
    ) -> list[str]:
        """Generate actionable fix recommendations."""
        recommendations = []

        for cause in root_causes:
            if cause.category == EnumFailureCategory.IMPORT_ERROR:
                recommendations.append(
                    f"Add missing dependencies to pyproject.toml: {', '.join(cause.error_patterns)}"
                )
                recommendations.append("Run `poetry install` to install dependencies")

            elif cause.category == EnumFailureCategory.TYPE_ERROR:
                recommendations.append(
                    "Review generated Pydantic models for correct type annotations"
                )
                recommendations.append(
                    "Check template logic in template_engine.py for type generation bugs"
                )

            elif cause.category == EnumFailureCategory.ASSERTION_ERROR:
                recommendations.append(
                    "Review generated node implementation for missing or incorrect logic"
                )
                recommendations.append(
                    "Update templates to generate complete implementations (not stubs)"
                )

            elif cause.category == EnumFailureCategory.RUNTIME_ERROR:
                recommendations.append(
                    "Review error messages and tracebacks for specific issues"
                )
                recommendations.append("Add defensive error handling in generated code")

        # Coverage recommendations
        if test_results.coverage_percent and test_results.coverage_percent < 80.0:
            recommendations.append(
                f"Increase test coverage from {test_results.coverage_percent:.1f}% to 80%+"
            )

        return recommendations

    def _assess_severity(
        self, test_results: ModelTestResults, root_causes: list[ModelFailureCause]
    ) -> EnumFailureSeverity:
        """Assess overall failure severity."""
        # All tests failed = CRITICAL
        if test_results.total > 0 and test_results.passed == 0:
            return EnumFailureSeverity.CRITICAL

        # >50% failure rate = HIGH
        if test_results.success_rate < 0.5:
            return EnumFailureSeverity.HIGH

        # Import errors are always HIGH (blocks functionality)
        if any(c.category == EnumFailureCategory.IMPORT_ERROR for c in root_causes):
            return EnumFailureSeverity.HIGH

        # <20% failure rate = MEDIUM
        if test_results.success_rate >= 0.8:
            return EnumFailureSeverity.MEDIUM

        # Default
        return EnumFailureSeverity.HIGH

    def _is_auto_fixable(self, root_causes: list[ModelFailureCause]) -> bool:
        """Check if failures can be automatically fixed."""
        # Import errors are auto-fixable (add to pyproject.toml)
        auto_fixable_categories = {EnumFailureCategory.IMPORT_ERROR}

        return all(cause.category in auto_fixable_categories for cause in root_causes)

    def _estimate_fix_time(self, root_causes: list[ModelFailureCause]) -> int:
        """Estimate manual fix time in minutes."""
        # Base time per category
        time_estimates = {
            EnumFailureCategory.IMPORT_ERROR: 5,  # Quick dependency add
            EnumFailureCategory.TYPE_ERROR: 15,  # Fix type annotations
            EnumFailureCategory.ASSERTION_ERROR: 30,  # Fix logic issues
            EnumFailureCategory.RUNTIME_ERROR: 20,  # Debug runtime issues
            EnumFailureCategory.CONFIGURATION_ERROR: 10,  # Config updates
            EnumFailureCategory.INFRASTRUCTURE_ERROR: 30,  # External deps
        }

        total_time = sum(
            time_estimates.get(cause.category, 15) for cause in root_causes
        )

        return min(total_time, 120)  # Cap at 2 hours

    def _generate_summary(
        self, test_results: ModelTestResults, root_causes: list[ModelFailureCause]
    ) -> str:
        """Generate executive summary of issues."""
        if not root_causes:
            return "No failures to report."

        # Group by category
        categories: dict[str, int] = {}
        for cause in root_causes:
            category_name = cause.category.value.replace("_", " ").title()
            categories[category_name] = categories.get(category_name, 0) + 1

        category_summary = ", ".join(
            f"{count} {cat}" for cat, count in categories.items()
        )

        return (
            f"Test suite failed with {test_results.failed}/{test_results.total} failures "
            f"({test_results.success_rate:.1%} success rate). "
            f"Root causes identified: {category_summary}. "
            f"Primary issues: {root_causes[0].description}"
        )


# Export
__all__ = [
    "FailureAnalyzer",
    "ModelFailureAnalysis",
    "ModelFailureCause",
    "EnumFailureSeverity",
    "EnumFailureCategory",
]
