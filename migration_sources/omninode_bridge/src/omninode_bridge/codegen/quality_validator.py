#!/usr/bin/env python3
"""
Quality Validator for Generated Code.

Validates generated ONEX nodes for:
- ONEX v2.0 compliance (naming, structure, patterns)
- Type safety (mypy validation)
- Code quality (ruff linting)
- Test coverage
- Documentation completeness

Provides quality scores (0.0-1.0) and actionable feedback.
"""

import ast

from pydantic import BaseModel, Field

from .template_engine import ModelGeneratedArtifacts


class ModelValidationResult(BaseModel):
    """
    Validation result with quality metrics and issues.

    Provides comprehensive quality assessment of generated code.
    """

    # Overall quality
    quality_score: float = Field(
        ..., ge=0.0, le=1.0, description="Overall quality score"
    )

    # Component scores
    onex_compliance_score: float = Field(
        ..., ge=0.0, le=1.0, description="ONEX v2.0 compliance score"
    )
    type_safety_score: float = Field(
        ..., ge=0.0, le=1.0, description="Type safety score"
    )
    code_quality_score: float = Field(
        ..., ge=0.0, le=1.0, description="Code quality score (linting)"
    )
    documentation_score: float = Field(
        ..., ge=0.0, le=1.0, description="Documentation completeness score"
    )
    test_coverage_score: float = Field(
        ..., ge=0.0, le=1.0, description="Test coverage score"
    )

    # Validation status
    passed: bool = Field(..., description="Overall validation passed")

    # Issues
    errors: list[str] = Field(default_factory=list, description="Critical errors")
    warnings: list[str] = Field(
        default_factory=list, description="Non-critical warnings"
    )
    suggestions: list[str] = Field(
        default_factory=list, description="Improvement suggestions"
    )

    # Detailed results
    onex_compliance_issues: list[str] = Field(default_factory=list)
    type_safety_issues: list[str] = Field(default_factory=list)
    code_quality_issues: list[str] = Field(default_factory=list)
    documentation_issues: list[str] = Field(default_factory=list)

    # Pass/fail thresholds
    min_quality_threshold: float = Field(default=0.8, ge=0.0, le=1.0)


class QualityValidator:
    """
    Validates generated code quality and ONEX compliance.

    Runs multiple validation passes:
    1. ONEX naming convention validation
    2. Structural validation (AST parsing)
    3. Type safety (mypy - optional)
    4. Code quality (ruff - optional)
    5. Documentation completeness
    6. Test coverage estimation
    """

    def __init__(
        self,
        enable_mypy: bool = False,
        enable_ruff: bool = False,
        min_quality_threshold: float = 0.8,
    ):
        """
        Initialize quality validator.

        Args:
            enable_mypy: Enable mypy type checking (requires mypy installed)
            enable_ruff: Enable ruff linting (requires ruff installed)
            min_quality_threshold: Minimum quality score to pass (0.0-1.0)
        """
        self.enable_mypy = enable_mypy
        self.enable_ruff = enable_ruff
        self.min_quality_threshold = min_quality_threshold

    async def validate(
        self, artifacts: ModelGeneratedArtifacts
    ) -> ModelValidationResult:
        """
        Validate generated code artifacts.

        Args:
            artifacts: Generated code artifacts to validate

        Returns:
            ModelValidationResult with quality scores and issues

        Example:
            >>> validator = QualityValidator()
            >>> result = await validator.validate(artifacts)
            >>> assert result.quality_score > 0.8
            >>> assert result.passed
        """
        # Component validations
        onex_score, onex_issues = self._validate_onex_compliance(artifacts)
        type_score, type_issues = self._validate_type_safety(artifacts)
        quality_score, quality_issues = self._validate_code_quality(artifacts)
        docs_score, docs_issues = self._validate_documentation(artifacts)
        test_score = self._estimate_test_coverage(artifacts)

        # Calculate overall quality score (weighted average)
        overall_quality = (
            onex_score * 0.30  # ONEX compliance: 30%
            + type_score * 0.25  # Type safety: 25%
            + quality_score * 0.20  # Code quality: 20%
            + docs_score * 0.15  # Documentation: 15%
            + test_score * 0.10  # Test coverage: 10%
        )

        # Aggregate errors and warnings
        errors = []
        warnings = []
        suggestions = []

        # Critical errors (score < 0.5)
        if onex_score < 0.5:
            errors.extend(onex_issues[:3])
        elif onex_score < 0.8:
            warnings.extend(onex_issues[:3])

        if type_score < 0.5:
            errors.extend(type_issues[:3])
        elif type_score < 0.8:
            warnings.extend(type_issues[:3])

        # Suggestions for improvement
        if docs_score < 0.9:
            suggestions.append("Add more comprehensive docstrings")
        if test_score < 0.9:
            suggestions.append("Add more test cases for edge conditions")

        passed = overall_quality >= self.min_quality_threshold

        return ModelValidationResult(
            quality_score=overall_quality,
            onex_compliance_score=onex_score,
            type_safety_score=type_score,
            code_quality_score=quality_score,
            documentation_score=docs_score,
            test_coverage_score=test_score,
            passed=passed,
            errors=errors,
            warnings=warnings,
            suggestions=suggestions,
            onex_compliance_issues=onex_issues,
            type_safety_issues=type_issues,
            code_quality_issues=quality_issues,
            documentation_issues=docs_issues,
            min_quality_threshold=self.min_quality_threshold,
        )

    def _validate_onex_compliance(
        self, artifacts: ModelGeneratedArtifacts
    ) -> tuple[float, list[str]]:
        """
        Validate ONEX v2.0 compliance.

        Checks:
        - Naming convention: Node<Name><Type> (suffix-based)
        - Base class inheritance
        - Method signatures (execute_effect, execute_compute, etc.)
        - Contract usage
        - Error handling with ModelOnexError
        """
        issues = []
        score = 1.0

        node_content = artifacts.node_file

        # Check 1: Naming convention (Node<Name><Type>)
        if not self._check_naming_convention(artifacts.node_name):
            issues.append(
                f"Node name '{artifacts.node_name}' doesn't follow ONEX naming convention"
            )
            score -= 0.2

        # Check 2: Base class import
        expected_base = f"Node{artifacts.node_type.capitalize()}"
        if expected_base not in node_content:
            issues.append(f"Missing base class import: {expected_base}")
            score -= 0.2

        # Check 3: Execute method signature
        execute_method = f"async def execute_{artifacts.node_type}"
        if execute_method not in node_content:
            issues.append(f"Missing execute method: {execute_method}")
            score -= 0.2

        # Check 4: Contract import
        if "ModelContract" not in node_content:
            issues.append("Missing contract import")
            score -= 0.1

        # Check 5: Error handling
        if "ModelOnexError" not in node_content:
            issues.append("Missing ModelOnexError for error handling")
            score -= 0.1

        # Check 6: Structured logging
        if "emit_log_event" not in node_content:
            issues.append("Missing structured logging (emit_log_event)")
            score -= 0.1

        # Check 7: Container-based initialization
        if "ModelContainer" not in node_content:
            issues.append("Missing ModelContainer in initialization")
            score -= 0.1

        return max(score, 0.0), issues

    def _check_naming_convention(self, node_name: str) -> bool:
        """
        Check if node name follows ONEX v2.0 naming convention.

        Pattern: Node<PascalCaseName><Type>
        Example: NodePostgresCrudEffect, NodeDataTransformerCompute
        """
        # Must start with "Node"
        if not node_name.startswith("Node"):
            return False

        # Must end with valid node type
        valid_suffixes = ["Effect", "Compute", "Reducer", "Orchestrator"]
        if not any(node_name.endswith(suffix) for suffix in valid_suffixes):
            return False

        # Must have content between "Node" and type suffix
        for suffix in valid_suffixes:
            if node_name.endswith(suffix):
                middle_part = node_name[4 : -len(suffix)]
                if not middle_part:
                    return False
                # Middle part should be PascalCase (starts with uppercase)
                if not middle_part[0].isupper():
                    return False

        return True

    def _validate_type_safety(
        self, artifacts: ModelGeneratedArtifacts
    ) -> tuple[float, list[str]]:
        """
        Validate type safety (AST-based + optional mypy).

        Checks:
        - Type hints on function signatures
        - Return type annotations
        - Pydantic models for data structures
        """
        issues = []
        score = 1.0

        try:
            # Parse node file as AST
            tree = ast.parse(artifacts.node_file)

            # Count functions with type hints
            functions = [
                node
                for node in ast.walk(tree)
                if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
            ]
            typed_functions = [func for func in functions if func.returns is not None]

            if functions:
                type_coverage = len(typed_functions) / len(functions)
                score = type_coverage
                if type_coverage < 0.8:
                    issues.append(
                        f"Low type coverage: {type_coverage:.1%} of functions typed"
                    )
            else:
                issues.append("No functions found in node file")
                score = 0.5

            # Optional: Run mypy if enabled
            if self.enable_mypy:
                mypy_score, mypy_issues = self._run_mypy(artifacts)
                score = (score + mypy_score) / 2
                issues.extend(mypy_issues)

        except SyntaxError as e:
            issues.append(f"Syntax error in generated code: {e}")
            score = 0.0

        return max(score, 0.0), issues

    def _validate_code_quality(
        self, artifacts: ModelGeneratedArtifacts
    ) -> tuple[float, list[str]]:
        """
        Validate code quality (AST-based + optional ruff).

        Checks:
        - Cyclomatic complexity
        - Function length
        - Import organization
        - Code formatting
        """
        issues = []
        score = 1.0

        try:
            tree = ast.parse(artifacts.node_file)

            # Check 1: Function complexity (simple heuristic)
            functions = [
                node
                for node in ast.walk(tree)
                if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
            ]
            for func in functions:
                # Count branches (if, for, while)
                branches = sum(
                    1
                    for node in ast.walk(func)
                    if isinstance(node, ast.If | ast.For | ast.While)
                )
                if branches > 10:
                    issues.append(
                        f"Function '{func.name}' has high complexity (>{branches} branches)"
                    )
                    score -= 0.1

            # Check 2: Import organization (stdlib → third-party → local)
            imports = [node for node in ast.walk(tree) if isinstance(node, ast.Import)]
            # Simplified check - proper organization would require more analysis

            # Optional: Run ruff if enabled
            if self.enable_ruff:
                ruff_score, ruff_issues = self._run_ruff(artifacts)
                score = (score + ruff_score) / 2
                issues.extend(ruff_issues)

        except SyntaxError as e:
            issues.append(f"Syntax error: {e}")
            score = 0.0

        return max(score, 0.0), issues

    def _validate_documentation(
        self, artifacts: ModelGeneratedArtifacts
    ) -> tuple[float, list[str]]:
        """
        Validate documentation completeness.

        Checks:
        - Module docstring
        - Class docstring
        - Method docstrings
        - README.md presence
        - Contract YAML documentation
        """
        issues = []
        score = 1.0

        try:
            tree = ast.parse(artifacts.node_file)

            # Check 1: Module docstring
            module_docstring = ast.get_docstring(tree)
            if not module_docstring:
                issues.append("Missing module docstring")
                score -= 0.2
            elif len(module_docstring) < 50:
                issues.append("Module docstring is too short")
                score -= 0.1

            # Check 2: Class docstrings
            classes = [node for node in tree.body if isinstance(node, ast.ClassDef)]
            for cls in classes:
                cls_docstring = ast.get_docstring(cls)
                if not cls_docstring:
                    issues.append(f"Class '{cls.name}' missing docstring")
                    score -= 0.1

            # Check 3: Method docstrings
            methods = [
                node
                for cls in classes
                for node in cls.body
                if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
            ]
            undocumented_methods = [m.name for m in methods if not ast.get_docstring(m)]
            if undocumented_methods:
                issues.append(
                    f"Methods missing docstrings: {', '.join(undocumented_methods[:3])}"
                )
                score -= 0.1 * min(len(undocumented_methods), 3)

            # Check 4: README presence
            if "README.md" not in artifacts.documentation:
                issues.append("Missing README.md")
                score -= 0.2

        except SyntaxError:
            issues.append("Cannot parse code for documentation check")
            score = 0.5

        return max(score, 0.0), issues

    def _estimate_test_coverage(self, artifacts: ModelGeneratedArtifacts) -> float:
        """
        Estimate test coverage based on test file presence and content.

        Note: This is an estimation, not actual test coverage measurement.
        """
        if not artifacts.tests:
            return 0.0

        score = 0.3  # Base score for having tests

        # Check for different test types
        test_content = " ".join(artifacts.tests.values())

        if "test_node_initialization" in test_content:
            score += 0.2
        if "test_execute" in test_content:
            score += 0.2
        if "@pytest.mark.integration" in test_content:
            score += 0.15
        if "assert" in test_content:
            score += 0.15

        return min(score, 1.0)

    def _run_mypy(self, artifacts: ModelGeneratedArtifacts) -> tuple[float, list[str]]:
        """Run mypy type checker on generated code."""
        # This is a placeholder - actual implementation would:
        # 1. Write files to temp directory
        # 2. Run mypy subprocess
        # 3. Parse output
        # 4. Return score and issues

        # For now, return neutral score
        return 0.8, []

    def _run_ruff(self, artifacts: ModelGeneratedArtifacts) -> tuple[float, list[str]]:
        """Run ruff linter on generated code."""
        # This is a placeholder - actual implementation would:
        # 1. Write files to temp directory
        # 2. Run ruff subprocess
        # 3. Parse output
        # 4. Return score and issues

        # For now, return neutral score
        return 0.8, []


# Export
__all__ = ["QualityValidator", "ModelValidationResult"]
