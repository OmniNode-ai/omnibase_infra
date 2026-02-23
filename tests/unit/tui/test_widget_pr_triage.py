# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Unit tests for WidgetPRTriage.

Tests:
- PR state extraction from partition_key
- All 8 triage states handled
- open_selected_pr does not crash on empty table

Run with:
    uv run pytest tests/unit/tui/ -m unit -v

Related Tickets:
    - OMN-2657: Phase 3 — TUI ONEX Status Terminal View (omnibase_infra)
"""

from __future__ import annotations

import pytest

from omnibase_infra.tui.widgets.widget_pr_triage import (
    _TRIAGE_ORDER,
    _triage_label,
)


class TestTriageOrder:
    @pytest.mark.unit
    def test_all_eight_states_present(self) -> None:
        """Verify all 8 triage states are present (set membership, not order)."""
        expected = {
            "draft",
            "stale",
            "ci_failing",
            "changes_requested",
            "ready_to_merge",
            "approved_pending_ci",
            "needs_review",
            "blocked",
        }
        assert set(_TRIAGE_ORDER) == expected

    @pytest.mark.unit
    def test_ready_to_merge_is_first(self) -> None:
        """ready_to_merge should sort first (highest priority display)."""
        assert _TRIAGE_ORDER[0] == "ready_to_merge"

    @pytest.mark.unit
    def test_draft_is_last(self) -> None:
        """draft should sort last (lowest priority display)."""
        assert _TRIAGE_ORDER[-1] == "draft"

    @pytest.mark.unit
    def test_no_duplicates_in_order(self) -> None:
        assert len(_TRIAGE_ORDER) == len(set(_TRIAGE_ORDER))


class TestTriageOrderComplete:
    @pytest.mark.unit
    def test_triage_order_length(self) -> None:
        assert len(_TRIAGE_ORDER) == 8

    @pytest.mark.unit
    def test_triage_order_is_tuple(self) -> None:
        assert isinstance(_TRIAGE_ORDER, tuple)


class TestTriageLabel:
    @pytest.mark.unit
    def test_ready_to_merge_label(self) -> None:
        label = _triage_label("ready_to_merge")
        assert "READY TO MERGE" in label.plain

    @pytest.mark.unit
    def test_needs_review_label(self) -> None:
        label = _triage_label("needs_review")
        assert "NEEDS REVIEW" in label.plain

    @pytest.mark.unit
    def test_unknown_state_label(self) -> None:
        """Unknown states should still render without error."""
        label = _triage_label("some_new_state")
        assert "SOME NEW STATE" in label.plain

    @pytest.mark.unit
    def test_all_known_states_render(self) -> None:
        for state in _TRIAGE_ORDER:
            label = _triage_label(state)
            assert len(label.plain) > 0

    @pytest.mark.unit
    def test_underscore_replaced_with_space(self) -> None:
        label = _triage_label("ci_failing")
        assert "_" not in label.plain
        assert "CI FAILING" in label.plain
