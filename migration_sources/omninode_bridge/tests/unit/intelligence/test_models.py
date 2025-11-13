"""
Unit tests for LLM metrics Pydantic models.

Tests validation, serialization, and business logic for intelligence models.
"""

from datetime import datetime
from uuid import uuid4

import pytest
from pydantic import ValidationError

from omninode_bridge.intelligence.models import (
    LLMGenerationHistory,
    LLMGenerationMetric,
    LLMPattern,
    MetricsSummary,
)


class TestLLMGenerationMetric:
    """Test LLMGenerationMetric model."""

    def test_valid_metric_creation(self):
        """Test creating a valid generation metric."""
        metric = LLMGenerationMetric(
            session_id="sess_test_123",
            node_type="effect",
            model_tier="tier_2",
            model_name="claude-sonnet-4",
            prompt_tokens=1500,
            completion_tokens=800,
            total_tokens=2300,
            latency_ms=3500.0,
            cost_usd=0.0345,
            success=True,
        )

        assert metric.session_id == "sess_test_123"
        assert metric.node_type == "effect"
        assert metric.total_tokens == 2300
        assert metric.success is True
        assert metric.metric_id is not None

    def test_total_tokens_validation(self):
        """Test that total_tokens is auto-corrected if mismatch."""
        metric = LLMGenerationMetric(
            session_id="sess_test_123",
            node_type="effect",
            model_tier="tier_2",
            model_name="claude-sonnet-4",
            prompt_tokens=1500,
            completion_tokens=800,
            total_tokens=1000,  # Wrong total
            latency_ms=3500.0,
            cost_usd=0.0345,
            success=True,
        )

        # Should be auto-corrected to 2300
        assert metric.total_tokens == 2300

    def test_negative_tokens_validation(self):
        """Test that negative tokens raise validation error."""
        with pytest.raises(ValidationError) as exc_info:
            LLMGenerationMetric(
                session_id="sess_test_123",
                node_type="effect",
                model_tier="tier_2",
                model_name="claude-sonnet-4",
                prompt_tokens=-100,  # Invalid
                completion_tokens=800,
                total_tokens=700,
                latency_ms=3500.0,
                cost_usd=0.0345,
                success=True,
            )

        assert "greater than or equal to 0" in str(exc_info.value)

    def test_negative_cost_validation(self):
        """Test that negative cost raises validation error."""
        with pytest.raises(ValidationError) as exc_info:
            LLMGenerationMetric(
                session_id="sess_test_123",
                node_type="effect",
                model_tier="tier_2",
                model_name="claude-sonnet-4",
                prompt_tokens=1500,
                completion_tokens=800,
                total_tokens=2300,
                latency_ms=3500.0,
                cost_usd=-0.01,  # Invalid
                success=True,
            )

        assert "greater than or equal to 0" in str(exc_info.value)

    def test_optional_fields(self):
        """Test that optional fields work correctly."""
        metric = LLMGenerationMetric(
            session_id="sess_test_123",
            node_type="effect",
            model_tier="tier_2",
            model_name="claude-sonnet-4",
            prompt_tokens=1500,
            completion_tokens=800,
            total_tokens=2300,
            latency_ms=3500.0,
            cost_usd=0.0345,
            success=False,
            error_message="Generation timeout",
            metadata={"retry_count": 3},
        )

        assert metric.error_message == "Generation timeout"
        assert metric.metadata["retry_count"] == 3

    def test_serialization(self):
        """Test model serialization to dict."""
        metric = LLMGenerationMetric(
            session_id="sess_test_123",
            node_type="effect",
            model_tier="tier_2",
            model_name="claude-sonnet-4",
            prompt_tokens=1500,
            completion_tokens=800,
            total_tokens=2300,
            latency_ms=3500.0,
            cost_usd=0.0345,
            success=True,
        )

        data = metric.model_dump()
        assert data["session_id"] == "sess_test_123"
        assert data["total_tokens"] == 2300
        assert isinstance(data["metric_id"], uuid4().__class__)


class TestLLMGenerationHistory:
    """Test LLMGenerationHistory model."""

    def test_valid_history_creation(self):
        """Test creating a valid generation history."""
        metric_id = uuid4()
        history = LLMGenerationHistory(
            metric_id=metric_id,
            prompt_text="Generate a Python Effect node...",
            generated_text="class NodeMyEffect:\n    async def execute_effect(self)...",
            quality_score=0.92,
            validation_passed=True,
        )

        assert history.metric_id == metric_id
        assert history.quality_score == 0.92
        assert history.validation_passed is True
        assert history.history_id is not None

    def test_quality_score_range_validation(self):
        """Test that quality_score must be between 0 and 1."""
        metric_id = uuid4()

        # Test score > 1
        with pytest.raises(ValidationError) as exc_info:
            LLMGenerationHistory(
                metric_id=metric_id,
                prompt_text="Test prompt",
                generated_text="Test output",
                quality_score=1.5,  # Invalid
                validation_passed=True,
            )

        assert "less than or equal to 1" in str(exc_info.value)

        # Test score < 0
        with pytest.raises(ValidationError) as exc_info:
            LLMGenerationHistory(
                metric_id=metric_id,
                prompt_text="Test prompt",
                generated_text="Test output",
                quality_score=-0.1,  # Invalid
                validation_passed=True,
            )

        assert "greater than or equal to 0" in str(exc_info.value)

    def test_optional_validation_errors(self):
        """Test that validation_errors field works correctly."""
        metric_id = uuid4()
        history = LLMGenerationHistory(
            metric_id=metric_id,
            prompt_text="Generate a Python Effect node...",
            generated_text="class NodeMyEffect:\n    async def execute_effect(self)...",
            quality_score=0.5,
            validation_passed=False,
            validation_errors={
                "syntax_errors": ["Missing import statement"],
                "type_errors": ["Invalid type annotation"],
            },
        )

        assert history.validation_passed is False
        assert "syntax_errors" in history.validation_errors
        assert len(history.validation_errors["syntax_errors"]) == 1


class TestLLMPattern:
    """Test LLMPattern model."""

    def test_valid_pattern_creation(self):
        """Test creating a valid learned pattern."""
        pattern = LLMPattern(
            pattern_type="prompt_template",
            node_type="effect",
            pattern_data={
                "template": "Generate a Python Effect node that...",
                "variables": ["node_name", "operation_type"],
            },
            usage_count=15,
            avg_quality_score=0.93,
            success_rate=0.95,
        )

        assert pattern.pattern_type == "prompt_template"
        assert pattern.usage_count == 15
        assert pattern.avg_quality_score == 0.93
        assert pattern.pattern_id is not None

    def test_negative_usage_count_validation(self):
        """Test that negative usage_count raises validation error."""
        with pytest.raises(ValidationError) as exc_info:
            LLMPattern(
                pattern_type="prompt_template",
                pattern_data={"template": "test"},
                usage_count=-1,  # Invalid
            )

        assert "greater than or equal to 0" in str(exc_info.value)

    def test_quality_score_range_validation(self):
        """Test that quality scores must be between 0 and 1."""
        # Test avg_quality_score > 1
        with pytest.raises(ValidationError) as exc_info:
            LLMPattern(
                pattern_type="prompt_template",
                pattern_data={"template": "test"},
                avg_quality_score=1.5,  # Invalid
            )

        assert "less than or equal to 1" in str(exc_info.value)

        # Test success_rate < 0
        with pytest.raises(ValidationError) as exc_info:
            LLMPattern(
                pattern_type="prompt_template",
                pattern_data={"template": "test"},
                success_rate=-0.1,  # Invalid
            )

        assert "greater than or equal to 0" in str(exc_info.value)

    def test_optional_node_type(self):
        """Test that node_type is optional."""
        pattern = LLMPattern(
            pattern_type="general_template",
            pattern_data={"template": "Generic template"},
        )

        assert pattern.node_type is None

    def test_timestamps(self):
        """Test that timestamps are auto-generated."""
        pattern = LLMPattern(
            pattern_type="prompt_template",
            pattern_data={"template": "test"},
        )

        assert pattern.created_at is not None
        assert pattern.updated_at is not None
        assert isinstance(pattern.created_at, datetime)
        assert isinstance(pattern.updated_at, datetime)


class TestMetricsSummary:
    """Test MetricsSummary model."""

    def test_valid_summary_creation(self):
        """Test creating a valid metrics summary."""
        now = datetime.utcnow()
        summary = MetricsSummary(
            total_generations=150,
            successful_generations=142,
            failed_generations=8,
            total_tokens=450000,
            total_cost_usd=12.45,
            avg_latency_ms=3200.0,
            success_rate=0.947,
            period_start=now,
            period_end=now,
        )

        assert summary.total_generations == 150
        assert summary.successful_generations == 142
        assert summary.success_rate == 0.947

    def test_success_rate_range_validation(self):
        """Test that success_rate must be between 0 and 1."""
        now = datetime.utcnow()

        with pytest.raises(ValidationError) as exc_info:
            MetricsSummary(
                total_generations=150,
                successful_generations=142,
                failed_generations=8,
                total_tokens=450000,
                total_cost_usd=12.45,
                avg_latency_ms=3200.0,
                success_rate=1.5,  # Invalid
                period_start=now,
                period_end=now,
            )

        assert "less than or equal to 1" in str(exc_info.value)

    def test_negative_values_validation(self):
        """Test that negative values raise validation errors."""
        now = datetime.utcnow()

        with pytest.raises(ValidationError) as exc_info:
            MetricsSummary(
                total_generations=-10,  # Invalid
                successful_generations=142,
                failed_generations=8,
                total_tokens=450000,
                total_cost_usd=12.45,
                avg_latency_ms=3200.0,
                success_rate=0.947,
                period_start=now,
                period_end=now,
            )

        assert "greater than or equal to 0" in str(exc_info.value)
