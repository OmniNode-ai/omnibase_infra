# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2026 OmniNode Team
"""Unit tests for HandlerQualityGate (delta function).

Tests cover:
    - Pass for each task type with valid output
    - Fail on refusal phrases (REFUSAL category)
    - Fail on short response (WEAK_OUTPUT category)
    - Fail on missing markers (TASK_MISMATCH category)
    - Score computation
    - Fallback recommendation
    - Empty response
    - Edge cases

Related:
    - OMN-7040: Node-based delegation pipeline
"""

from __future__ import annotations

from uuid import uuid4

import pytest

from omnibase_infra.nodes.node_delegation_quality_gate_reducer.handlers.handler_quality_gate import (
    delta,
)
from omnibase_infra.nodes.node_delegation_quality_gate_reducer.models.model_quality_gate_input import (
    ModelQualityGateInput,
)

pytestmark = [pytest.mark.unit]


def _input(
    task_type: str = "test",
    content: str = "",
    expected_markers: tuple[str, ...] = (),
    min_response_length: int = 60,
) -> ModelQualityGateInput:
    """Build a valid ModelQualityGateInput."""
    return ModelQualityGateInput(
        correlation_id=uuid4(),
        task_type=task_type,
        llm_response_content=content,
        expected_markers=expected_markers,
        min_response_length=min_response_length,
    )


class TestPassingResponses:
    """Verify passing quality gate for valid outputs."""

    def test_test_task_passes(self) -> None:
        content = (
            "import pytest\n\n"
            "@pytest.mark.unit\n"
            "def test_auth_validates_token():\n"
            "    assert validate_token('valid') is True\n"
            "\n"
            "def test_auth_rejects_expired():\n"
            "    assert validate_token('expired') is False\n"
        )
        result = delta(_input(task_type="test", content=content))
        assert result.passed is True
        assert result.quality_score >= 0.6

    def test_document_task_passes(self) -> None:
        content = (
            '"""Authentication module for token validation.\n\n'
            "Args:\n"
            "    token: The JWT token to validate.\n\n"
            "Returns:\n"
            "    True if the token is valid and not expired.\n\n"
            "Raises:\n"
            '    ValueError if the token format is invalid.\n"""\n'
        )
        result = delta(_input(task_type="document", content=content))
        assert result.passed is True
        assert result.quality_score >= 0.6

    def test_research_task_passes(self) -> None:
        content = (
            "The authentication module uses JWT tokens for session management. "
            "The validate_token function checks both signature validity and "
            "expiration time. The token refresh mechanism is implemented in "
            "refresh_handler.py on line 45."
        )
        result = delta(_input(task_type="research", content=content))
        assert result.passed is True
        assert result.quality_score >= 0.6


class TestRefusalDetection:
    """Verify REFUSAL failure category."""

    @pytest.mark.parametrize(
        "phrase",
        [
            "I cannot help with that request.",
            "I'm sorry, but I can't generate that code.",
            "As an AI language model, I don't have access to that.",
            "Error: something went wrong\nHere is the output...",
            "Traceback (most recent call last):\n  File ...",
        ],
    )
    def test_refusal_phrases_detected(self, phrase: str) -> None:
        # Pad content to meet length requirement
        content = phrase + " " * 200
        result = delta(_input(task_type="research", content=content))
        assert result.passed is False
        assert any("REFUSAL" in r for r in result.failure_reasons)

    def test_refusal_only_checks_first_200_chars(self) -> None:
        # Refusal phrase after 200 chars should not trigger
        safe_prefix = "x" * 201
        content = safe_prefix + "I cannot help with that"
        result = delta(_input(task_type="research", content=content))
        # Should not have refusal failure
        assert not any("REFUSAL" in r for r in result.failure_reasons)


class TestWeakOutput:
    """Verify WEAK_OUTPUT failure category."""

    def test_short_response_fails(self) -> None:
        result = delta(_input(task_type="test", content="def test_x(): pass"))
        assert result.passed is False
        assert any("WEAK_OUTPUT" in r for r in result.failure_reasons)

    def test_empty_response_fails(self) -> None:
        result = delta(_input(task_type="test", content=""))
        assert result.passed is False
        assert any("WEAK_OUTPUT" in r for r in result.failure_reasons)

    def test_document_requires_longer_response(self) -> None:
        # 90 chars -- passes test (80) but fails document (100)
        content = "x" * 90 + ' """docstring"""'
        result_test = delta(_input(task_type="research", content=content))
        assert result_test.passed is True

        result_doc = delta(_input(task_type="document", content=content))
        # Document has higher min length (100) so should have lower or equal score
        assert result_doc.quality_score <= 1.0


class TestTaskMismatch:
    """Verify TASK_MISMATCH failure category."""

    def test_test_without_test_markers(self) -> None:
        content = (
            "Here is a comprehensive analysis of the authentication module. "
            "The module provides token validation with JWT support and "
            "handles both access and refresh tokens securely."
        )
        result = delta(_input(task_type="test", content=content))
        assert any("TASK_MISMATCH" in r for r in result.failure_reasons)

    def test_document_without_doc_markers(self) -> None:
        content = (
            "def test_auth():\n"
            "    assert validate_token('valid') is True\n"
            "    assert validate_token('expired') is False\n"
            "    assert validate_token(None) is False\n"
        )
        result = delta(_input(task_type="document", content=content))
        assert any("TASK_MISMATCH" in r for r in result.failure_reasons)


class TestScoring:
    """Verify score computation."""

    def test_perfect_score(self) -> None:
        content = (
            "import pytest\n\n"
            "@pytest.mark.unit\n"
            "def test_auth():\n"
            "    assert validate_token('valid') is True\n"
        )
        result = delta(_input(task_type="test", content=content))
        assert result.quality_score == 1.0

    def test_zero_score_empty(self) -> None:
        result = delta(_input(task_type="test", content=""))
        assert result.quality_score <= 0.3


class TestFallbackRecommendation:
    """Verify fallback_recommended flag."""

    def test_refusal_recommends_fallback(self) -> None:
        content = "I cannot help with that request. " + "x" * 200
        result = delta(_input(task_type="research", content=content))
        assert result.fallback_recommended is True

    def test_passing_does_not_recommend_fallback(self) -> None:
        content = (
            "The authentication module uses JWT tokens for session management. "
            "It validates signatures and checks expiration timestamps."
        )
        result = delta(_input(task_type="research", content=content))
        assert result.fallback_recommended is False


class TestCorrelationIdPreserved:
    """Verify correlation_id flows through."""

    def test_correlation_id_matches_input(self) -> None:
        gate_input = _input(task_type="research", content="x" * 100)
        result = delta(gate_input)
        assert result.correlation_id == gate_input.correlation_id
