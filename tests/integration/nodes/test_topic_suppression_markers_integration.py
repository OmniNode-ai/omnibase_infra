# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Integration tests for OMN-9151: topic suppression marker coverage.

Verifies that hardcoded topic strings in handler docstrings and SQL constants
have been correctly annotated with `# onex-topic-allow` or `# onex-topic-sot`
markers, and that the handlers remain functionally correct after the annotation
sweep.
"""

from __future__ import annotations

import re

import pytest

SUPPRESSION_PATTERN = re.compile(r"#\s*onex-topic-(allow|sot)")
# Matches topic strings assigned to a variable or in SQL: = "onex...."
TOPIC_ASSIGNMENT_PATTERN = re.compile(
    r'=\s*["\']([^"\']*onex\.\w+\.\w+\.\w[\w-]*\.v\d+)'
)


def _find_unsuppressed_assignment_topics(source: str) -> list[str]:
    """Return lines with hardcoded topics in assignment context lacking a suppression marker."""
    violations = []
    for line in source.splitlines():
        if TOPIC_ASSIGNMENT_PATTERN.search(line) and not SUPPRESSION_PATTERN.search(
            line
        ):
            if not line.strip().startswith("#"):
                violations.append(line.strip())
    return violations


@pytest.mark.integration
class TestTopicSuppressionMarkersIntegration:
    def test_handler_postgres_topic_update_suppression(self) -> None:
        import inspect

        import omnibase_infra.nodes.node_contract_persistence_effect.handlers.handler_postgres_topic_update as mod

        source = inspect.getsource(mod)
        violations = _find_unsuppressed_assignment_topics(source)
        assert violations == [], (
            f"handler_postgres_topic_update has unsuppressed topic strings: {violations}"
        )

    def test_handler_gmail_archive_cleanup_suppression(self) -> None:
        import inspect

        import omnibase_infra.nodes.node_gmail_archive_cleanup_effect.handlers.handler_gmail_archive_cleanup as mod

        source = inspect.getsource(mod)
        violations = _find_unsuppressed_assignment_topics(source)
        assert violations == [], (
            f"handler_gmail_archive_cleanup has unsuppressed topic strings: {violations}"
        )

    def test_handler_catalog_request_suppression(self) -> None:
        import inspect

        import omnibase_infra.nodes.node_registration_orchestrator.handlers.handler_catalog_request as mod

        source = inspect.getsource(mod)
        violations = _find_unsuppressed_assignment_topics(source)
        assert violations == [], (
            f"handler_catalog_request has unsuppressed topic strings: {violations}"
        )

    def test_handler_validation_ledger_projection_suppression(self) -> None:
        import inspect

        import omnibase_infra.nodes.node_validation_ledger_projection_compute.handlers.handler_validation_ledger_projection as mod

        source = inspect.getsource(mod)
        violations = _find_unsuppressed_assignment_topics(source)
        assert violations == [], (
            f"handler_validation_ledger_projection has unsuppressed topic strings: {violations}"
        )

    def test_suppressed_handlers_are_importable(self) -> None:
        """Verify all 4 annotated handlers import without errors after the sweep."""
        import omnibase_infra.nodes.node_contract_persistence_effect.handlers.handler_postgres_topic_update
        import omnibase_infra.nodes.node_gmail_archive_cleanup_effect.handlers.handler_gmail_archive_cleanup
        import omnibase_infra.nodes.node_registration_orchestrator.handlers.handler_catalog_request
        import omnibase_infra.nodes.node_validation_ledger_projection_compute.handlers.handler_validation_ledger_projection
