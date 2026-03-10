# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2025 OmniNode Team
"""Unit tests for fix-PR label parsing in NodeDeltaBundleEffect.

Tests the parse_stabilizes_label() pure function that extracts the
stabilized PR ref from the ``stabilizes:<pr_ref>`` label convention.

Related Tickets:
    - OMN-3142: NodeDeltaBundleEffect implementation
"""

from __future__ import annotations

import pytest

from omnibase_infra.nodes.node_delta_bundle_effect.handlers.handler_write_bundle import (
    parse_stabilizes_label,
)


@pytest.mark.unit
class TestParseStabilizesLabel:
    """Tests for parse_stabilizes_label()."""

    def test_returns_none_for_empty_labels(self) -> None:
        """Empty label list returns None."""
        assert parse_stabilizes_label([]) is None

    def test_returns_none_when_no_stabilizes_label(self) -> None:
        """Labels without stabilizes: prefix return None."""
        labels = ["bug", "enhancement", "priority:high"]
        assert parse_stabilizes_label(labels) is None

    def test_extracts_pr_ref_from_stabilizes_label(self) -> None:
        """Standard stabilizes:<pr_ref> label is correctly parsed."""
        labels = ["bug", "stabilizes:owner/repo#42"]
        assert parse_stabilizes_label(labels) == "owner/repo#42"

    def test_returns_first_match(self) -> None:
        """When multiple stabilizes labels exist, returns the first match."""
        labels = ["stabilizes:owner/repo#10", "stabilizes:owner/repo#20"]
        assert parse_stabilizes_label(labels) == "owner/repo#10"

    def test_strips_whitespace_from_value(self) -> None:
        """Whitespace around the PR ref value is stripped."""
        labels = ["stabilizes:  owner/repo#42  "]
        assert parse_stabilizes_label(labels) == "owner/repo#42"

    def test_ignores_empty_value_after_prefix(self) -> None:
        """Label with just 'stabilizes:' and no value returns None."""
        labels = ["stabilizes:"]
        assert parse_stabilizes_label(labels) is None

    def test_ignores_whitespace_only_value(self) -> None:
        """Label with just whitespace after prefix returns None."""
        labels = ["stabilizes:   "]
        assert parse_stabilizes_label(labels) is None

    def test_case_sensitive_prefix(self) -> None:
        """Prefix matching is case-sensitive."""
        labels = ["Stabilizes:owner/repo#42", "STABILIZES:owner/repo#42"]
        assert parse_stabilizes_label(labels) is None

    def test_complex_pr_ref(self) -> None:
        """Handles complex PR ref formats."""
        labels = ["stabilizes:OmniNode-ai/omnibase_infra#123"]
        assert parse_stabilizes_label(labels) == "OmniNode-ai/omnibase_infra#123"

    def test_mixed_labels_with_stabilizes(self) -> None:
        """Finds stabilizes label among many other labels."""
        labels = [
            "priority:critical",
            "type:fix",
            "stabilizes:org/repo#99",
            "reviewed",
            "auto-merge",
        ]
        assert parse_stabilizes_label(labels) == "org/repo#99"
