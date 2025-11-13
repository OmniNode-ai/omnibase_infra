#!/usr/bin/env python3
"""
Unit tests for QualityGatePipeline.

Tests multi-stage validation pipeline for generated code.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from omninode_bridge.codegen.business_logic.models import GenerationContext
from omninode_bridge.codegen.quality_gates import (
    QualityGatePipeline,
    StageResult,
    ValidationLevel,
    ValidationResult,
    ValidationStage,
)


class TestQualityGatePipeline:
    """Test QualityGatePipeline functionality."""

    @pytest.fixture
    def pipeline_strict(self):
        """Create pipeline with strict validation."""
        return QualityGatePipeline(
            validation_level="strict",
            enable_mypy=False,  # Disable mypy for unit tests (faster)
        )

    @pytest.fixture
    def pipeline_permissive(self):
        """Create pipeline with permissive validation."""
        return QualityGatePipeline(
            validation_level="permissive",
            enable_mypy=False,
        )

    @pytest.fixture
    def pipeline_development(self):
        """Create pipeline with development validation."""
        return QualityGatePipeline(
            validation_level="development",
            enable_mypy=False,
        )

    def test_pipeline_initialization_strict(self, pipeline_strict):
        """Test pipeline initializes with strict mode."""
        assert pipeline_strict.validation_level == ValidationLevel.STRICT
        assert pipeline_strict.code_validator is not None

    def test_pipeline_initialization_invalid_level_raises_error(self):
        """Test that invalid validation level raises error."""
        with pytest.raises(ValueError, match="Invalid validation_level"):
            QualityGatePipeline(validation_level="invalid_level")

    @pytest.mark.asyncio
    async def test_validate_syntax_valid_code(self, pipeline_strict, sample_valid_code):
        """Test syntax validation passes for valid code."""
        result = await pipeline_strict._validate_syntax(sample_valid_code)

        assert result.stage == ValidationStage.SYNTAX
        assert result.passed is True
        assert len(result.issues) == 0
        assert result.execution_time_ms >= 0

    @pytest.mark.asyncio
    async def test_validate_syntax_invalid_code(
        self, pipeline_strict, sample_code_with_syntax_error
    ):
        """Test syntax validation catches syntax errors."""
        result = await pipeline_strict._validate_syntax(sample_code_with_syntax_error)

        assert result.stage == ValidationStage.SYNTAX
        assert result.passed is False
        assert len(result.issues) > 0
        assert any("syntax" in issue.lower() for issue in result.issues)

    @pytest.mark.asyncio
    async def test_validate_security_clean_code(
        self, pipeline_strict, sample_valid_code
    ):
        """Test security validation passes for clean code."""
        # Mock the code_validator to avoid actual validation
        mock_result = MagicMock()
        mock_result.security_issues = []
        mock_result.security_clean = True

        with patch.object(
            pipeline_strict.code_validator,
            "validate",
            new_callable=AsyncMock,
            return_value=mock_result,
        ):
            result = await pipeline_strict._validate_security(sample_valid_code)

            assert result.stage == ValidationStage.SECURITY
            assert result.passed is True
            assert len(result.issues) == 0

    @pytest.mark.asyncio
    async def test_validate_security_with_issues(
        self, pipeline_strict, sample_code_with_security_issues
    ):
        """Test security validation catches security issues."""
        # Mock the code_validator to return security issues
        mock_result = MagicMock()
        mock_result.security_issues = [
            "Hardcoded secret detected",
            "Dangerous eval usage",
            "SQL injection vulnerability",
        ]
        mock_result.security_clean = False

        with patch.object(
            pipeline_strict.code_validator,
            "validate",
            new_callable=AsyncMock,
            return_value=mock_result,
        ):
            result = await pipeline_strict._validate_security(
                sample_code_with_security_issues
            )

            assert result.stage == ValidationStage.SECURITY
            assert result.passed is False
            assert len(result.issues) >= 3

    @pytest.mark.asyncio
    async def test_validate_code_injection_no_stubs(
        self, pipeline_strict, sample_valid_code
    ):
        """Test code injection validation passes when no stubs present."""
        result = await pipeline_strict._validate_code_injection(sample_valid_code)

        assert result.stage == ValidationStage.CODE_INJECTION
        assert result.passed is True
        assert len(result.issues) == 0

    @pytest.mark.asyncio
    async def test_validate_code_injection_with_stubs(
        self, pipeline_strict, sample_code_with_stubs
    ):
        """Test code injection validation catches stubs."""
        result = await pipeline_strict._validate_code_injection(sample_code_with_stubs)

        assert result.stage == ValidationStage.CODE_INJECTION
        assert result.passed is False
        assert len(result.issues) >= 2  # Should detect TODO and IMPLEMENTATION REQUIRED

    @pytest.mark.asyncio
    async def test_validate_onex_compliance(self, pipeline_strict, sample_valid_code):
        """Test ONEX compliance validation."""
        # Mock the code_validator
        mock_result = MagicMock()
        mock_result.onex_issues = []
        mock_result.type_hint_issues = []
        mock_result.quality_issues = []
        mock_result.onex_compliant = True

        context = GenerationContext(
            node_type="effect",
            service_name="test_service",
            business_description="Test business logic",
        )

        with patch.object(
            pipeline_strict.code_validator,
            "validate",
            new_callable=AsyncMock,
            return_value=mock_result,
        ):
            result = await pipeline_strict._validate_onex_compliance(
                sample_valid_code, context
            )

            assert result.stage == ValidationStage.ONEX_COMPLIANCE
            assert result.passed is True

    @pytest.mark.asyncio
    async def test_validate_full_pipeline_valid_code(
        self, pipeline_strict, sample_valid_code
    ):
        """Test full validation pipeline with valid code."""
        # Mock validators to avoid complex dependencies
        mock_validator_result = MagicMock()
        mock_validator_result.onex_issues = []
        mock_validator_result.type_hint_issues = []
        mock_validator_result.quality_issues = []
        mock_validator_result.security_issues = []
        mock_validator_result.onex_compliant = True
        mock_validator_result.security_clean = True

        with patch.object(
            pipeline_strict.code_validator,
            "validate",
            new_callable=AsyncMock,
            return_value=mock_validator_result,
        ):
            result = await pipeline_strict.validate(sample_valid_code)

            # Should pass all stages
            assert result.passed is True
            assert result.validation_level == ValidationLevel.STRICT
            assert result.quality_score >= 0.8  # High quality score

            # Check stages
            assert (
                len(result.stage_results) >= 4
            )  # Syntax, Type, ONEX, Injection, Security
            assert len(result.failed_stages) == 0
            assert result.total_issues_count == 0

    @pytest.mark.asyncio
    async def test_validate_full_pipeline_with_syntax_error(
        self, pipeline_strict, sample_code_with_syntax_error
    ):
        """Test that syntax errors stop pipeline early."""
        result = await pipeline_strict.validate(sample_code_with_syntax_error)

        # Should fail at syntax stage
        assert result.passed is False
        assert "syntax" in result.failed_stages

        # Other stages should be skipped or minimal
        assert result.quality_score < 0.5

    @pytest.mark.asyncio
    async def test_validate_development_mode_skips_stages(
        self, pipeline_development, sample_code_with_stubs
    ):
        """Test that development mode skips non-critical stages."""
        # Mock validators
        mock_validator_result = MagicMock()
        mock_validator_result.security_issues = []
        mock_validator_result.security_clean = True

        with patch.object(
            pipeline_development.code_validator,
            "validate",
            new_callable=AsyncMock,
            return_value=mock_validator_result,
        ):
            result = await pipeline_development.validate(sample_code_with_stubs)

            # Development mode skips ONEX compliance and code injection
            skipped = [r for r in result.stage_results if r.skipped]
            assert len(skipped) >= 2  # At least ONEX and Injection

    @pytest.mark.asyncio
    async def test_validate_permissive_mode_allows_warnings(
        self, pipeline_permissive, sample_valid_code
    ):
        """Test that permissive mode allows warnings."""
        # Mock validators with warnings but no critical issues
        mock_validator_result = MagicMock()
        mock_validator_result.onex_issues = []
        mock_validator_result.type_hint_issues = ["Missing type hint on parameter"]
        mock_validator_result.quality_issues = ["Consider adding docstring"]
        mock_validator_result.security_issues = []
        mock_validator_result.onex_compliant = True
        mock_validator_result.security_clean = True

        with patch.object(
            pipeline_permissive.code_validator,
            "validate",
            new_callable=AsyncMock,
            return_value=mock_validator_result,
        ):
            result = await pipeline_permissive.validate(sample_valid_code)

            # Should pass despite warnings in permissive mode
            assert result.passed is True
            assert result.total_warnings_count >= 2

    @pytest.mark.asyncio
    async def test_calculate_quality_score(self, pipeline_strict):
        """Test quality score calculation."""
        stage_results = [
            StageResult(
                stage=ValidationStage.SYNTAX, passed=True, execution_time_ms=10
            ),
            StageResult(
                stage=ValidationStage.TYPE_CHECKING, passed=True, execution_time_ms=50
            ),
            StageResult(
                stage=ValidationStage.ONEX_COMPLIANCE, passed=True, execution_time_ms=30
            ),
            StageResult(
                stage=ValidationStage.CODE_INJECTION, passed=True, execution_time_ms=20
            ),
            StageResult(
                stage=ValidationStage.SECURITY, passed=True, execution_time_ms=25
            ),
        ]

        score = pipeline_strict._calculate_quality_score(stage_results)

        # All stages passed, should be 1.0
        assert score == 1.0

    @pytest.mark.asyncio
    async def test_calculate_quality_score_partial_pass(self, pipeline_strict):
        """Test quality score with some failed stages."""
        stage_results = [
            StageResult(
                stage=ValidationStage.SYNTAX, passed=True, execution_time_ms=10
            ),
            StageResult(
                stage=ValidationStage.TYPE_CHECKING,
                passed=False,
                issues=["Type error"],
                execution_time_ms=50,
            ),
            StageResult(
                stage=ValidationStage.ONEX_COMPLIANCE, passed=True, execution_time_ms=30
            ),
            StageResult(
                stage=ValidationStage.CODE_INJECTION,
                passed=False,
                issues=["Stub found"],
                execution_time_ms=20,
            ),
            StageResult(
                stage=ValidationStage.SECURITY, passed=True, execution_time_ms=25
            ),
        ]

        score = pipeline_strict._calculate_quality_score(stage_results)

        # Some failures, should be less than 1.0
        assert 0.0 < score < 1.0

    @pytest.mark.asyncio
    async def test_build_validation_result_aggregates_issues(self, pipeline_strict):
        """Test that validation result aggregates issues from all stages."""
        stage_results = [
            StageResult(
                stage=ValidationStage.SYNTAX,
                passed=False,
                issues=["Syntax error line 10"],
                execution_time_ms=10,
            ),
            StageResult(
                stage=ValidationStage.SECURITY,
                passed=False,
                issues=["Hardcoded secret", "SQL injection"],
                execution_time_ms=25,
            ),
        ]

        result = pipeline_strict._build_validation_result(
            stage_results=stage_results, total_time_ms=100.0
        )

        # Should aggregate all issues
        assert len(result.all_issues) == 3
        assert len(result.critical_issues) >= 3  # Syntax and security are critical
        assert result.total_issues_count == 3
        assert len(result.failed_stages) == 2

    @pytest.mark.asyncio
    async def test_validation_tracks_execution_time(
        self, pipeline_strict, sample_valid_code
    ):
        """Test that validation tracks execution time."""
        mock_validator_result = MagicMock()
        mock_validator_result.onex_issues = []
        mock_validator_result.type_hint_issues = []
        mock_validator_result.quality_issues = []
        mock_validator_result.security_issues = []

        with patch.object(
            pipeline_strict.code_validator,
            "validate",
            new_callable=AsyncMock,
            return_value=mock_validator_result,
        ):
            result = await pipeline_strict.validate(sample_valid_code)

            # Should track total execution time
            assert result.total_execution_time_ms >= 0
            assert (
                result.total_execution_time_ms < 10000
            )  # Should be fast for unit test

            # Each stage should track time
            for stage_result in result.stage_results:
                if not stage_result.skipped:
                    assert stage_result.execution_time_ms >= 0


class TestValidationModels:
    """Test validation model structures."""

    def test_stage_result_model(self):
        """Test StageResult model."""
        result = StageResult(
            stage=ValidationStage.SYNTAX,
            passed=True,
            issues=[],
            warnings=["Minor warning"],
            execution_time_ms=50.5,
            skipped=False,
        )

        assert result.stage == ValidationStage.SYNTAX
        assert result.passed is True
        assert len(result.warnings) == 1
        assert result.execution_time_ms == 50.5

    def test_validation_result_model(self):
        """Test ValidationResult model."""
        result = ValidationResult(
            passed=True,
            validation_level=ValidationLevel.STRICT,
            quality_score=0.95,
            stage_results=[],
            failed_stages=[],
            passed_stages=["syntax", "security"],
            skipped_stages=[],
            all_issues=[],
            all_warnings=["Minor warning"],
            critical_issues=[],
            total_execution_time_ms=250.0,
            total_issues_count=0,
            total_warnings_count=1,
        )

        assert result.passed is True
        assert result.quality_score == 0.95
        assert result.total_warnings_count == 1
        assert len(result.passed_stages) == 2
