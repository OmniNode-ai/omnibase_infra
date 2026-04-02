# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Tests for LLM summary generator."""

import pytest

from omnibase_infra.services.agent_learning_extraction.summary_generator import (
    build_summary_prompt,
    parse_summary_response,
)


@pytest.mark.unit
class TestBuildSummaryPrompt:
    def test_includes_repo_and_errors(self) -> None:
        prompt = build_summary_prompt(
            repo="omnibase_infra",
            file_paths=["src/foo.py", "tests/test_foo.py"],
            error_signatures=["ImportError: cannot import 'bar'"],
            tool_names=["Edit", "Bash", "Bash"],
        )
        assert "omnibase_infra" in prompt
        assert "ImportError" in prompt
        assert "src/foo.py" in prompt

    def test_truncates_long_errors(self) -> None:
        long_error = "x" * 2000
        prompt = build_summary_prompt(
            repo="test",
            file_paths=[],
            error_signatures=[long_error],
            tool_names=[],
        )
        assert len(prompt) < 5000


@pytest.mark.unit
class TestParseSummaryResponse:
    def test_extracts_content(self) -> None:
        response = {
            "choices": [{"message": {"content": "Fixed by adding the missing import."}}]
        }
        assert parse_summary_response(response) == "Fixed by adding the missing import."

    def test_fallback_on_empty(self) -> None:
        response = {"choices": [{"message": {"content": ""}}]}
        result = parse_summary_response(response)
        assert result == "Session completed successfully (no summary generated)."

    def test_truncates_long_response(self) -> None:
        response = {"choices": [{"message": {"content": "x" * 3000}}]}
        result = parse_summary_response(response)
        assert len(result) <= 2000
