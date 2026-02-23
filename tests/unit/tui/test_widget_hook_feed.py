# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Unit tests for WidgetHookFeed helper functions.

Tests:
- _format_hook_line produces correct output for known payloads
- _outcome_text returns styled text for known/unknown outcomes
- Edge cases: missing fields, empty payload

Run with:
    uv run pytest tests/unit/tui/ -m unit -v

Related Tickets:
    - OMN-2657: Phase 3 — TUI ONEX Status Terminal View (omnibase_infra)
"""

from __future__ import annotations

import pytest

from omnibase_infra.tui.widgets.widget_hook_feed import (
    _format_hook_line,
    _outcome_text,
)


class TestOutcomeText:
    @pytest.mark.unit
    def test_pass_outcome(self) -> None:
        text = _outcome_text("pass")
        assert "PASS" in text.plain

    @pytest.mark.unit
    def test_fail_outcome(self) -> None:
        text = _outcome_text("fail")
        assert "FAIL" in text.plain

    @pytest.mark.unit
    def test_allowed_outcome(self) -> None:
        text = _outcome_text("allowed")
        assert "ALLOWED" in text.plain

    @pytest.mark.unit
    def test_blocked_outcome(self) -> None:
        text = _outcome_text("blocked")
        assert "BLOCKED" in text.plain

    @pytest.mark.unit
    def test_unknown_outcome_does_not_crash(self) -> None:
        text = _outcome_text("some_new_outcome")
        assert "SOME_NEW_OUTCOME" in text.plain

    @pytest.mark.unit
    def test_case_insensitive(self) -> None:
        """Outcome lookup is case-insensitive."""
        text_lower = _outcome_text("pass")
        text_upper = _outcome_text("PASS")
        assert text_lower.plain == text_upper.plain


class TestFormatHookLine:
    @pytest.mark.unit
    def test_full_payload_renders_all_fields(self) -> None:
        payload = {
            "event_type": "onex.evt.git.hook.v1",
            "hook": "pre-commit",
            "repo": "OmniNode-ai/omniclaude",
            "branch": "main",
            "author": "jsmith",
            "outcome": "pass",
            "gates": ["lint", "tests"],
            "emitted_at": "2026-02-23T10:00:00Z",
        }
        line = _format_hook_line(payload)
        assert "PRE-COMMIT" in line.plain
        assert "OmniNode-ai/omniclaude" in line.plain
        assert "main" in line.plain
        assert "jsmith" in line.plain
        assert "PASS" in line.plain

    @pytest.mark.unit
    def test_empty_payload_does_not_crash(self) -> None:
        """Missing fields fall back to defaults gracefully."""
        line = _format_hook_line({})
        assert line is not None
        # Should contain fallback values
        assert "UNKNOWN" in line.plain or len(line.plain) > 0

    @pytest.mark.unit
    def test_emitted_at_truncated_to_19_chars(self) -> None:
        """emitted_at display is truncated to 'YYYY-MM-DD HH:MM:SS'."""
        payload = {
            "hook": "post-receive",
            "repo": "OmniNode-ai/omnibase_core",
            "branch": "feature-x",
            "author": "dev",
            "outcome": "fail",
            "emitted_at": "2026-02-23T10:00:00.000000+00:00",
        }
        line = _format_hook_line(payload)
        # The timestamp should appear truncated (no microseconds/timezone)
        assert "2026-02-23 10:00:00" in line.plain

    @pytest.mark.unit
    def test_post_receive_hook(self) -> None:
        payload = {
            "hook": "post-receive",
            "repo": "OmniNode-ai/omnibase_infra",
            "branch": "jonah/feature",
            "author": "jonah",
            "outcome": "allowed",
            "emitted_at": "2026-02-23T12:34:56Z",
        }
        line = _format_hook_line(payload)
        assert "POST-RECEIVE" in line.plain
        assert "ALLOWED" in line.plain
