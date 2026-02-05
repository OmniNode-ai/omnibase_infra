# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Unit tests for notification event models.

Tests the Pydantic models for notification events:
- ModelNotificationBlocked: Agent blocked waiting for human input
- ModelNotificationCompleted: Ticket work completed

Related Tickets:
    - OMN-1831: Implement event-driven Slack notifications via runtime
"""

from __future__ import annotations

from uuid import UUID, uuid4

import pytest
from pydantic import ValidationError

from omnibase_infra.runtime.emit_daemon.models import (
    ModelNotificationBlocked,
    ModelNotificationCompleted,
)


class TestModelNotificationBlocked:
    """Tests for ModelNotificationBlocked Pydantic model."""

    def test_minimal_required_fields(self) -> None:
        """Should create model with only required fields."""
        blocked = ModelNotificationBlocked(
            ticket_identifier="OMN-1234",
            reason="Waiting for approval",
            repo="omniclaude",
            session_id=uuid4(),
        )
        assert blocked.ticket_identifier == "OMN-1234"
        assert blocked.reason == "Waiting for approval"
        assert blocked.repo == "omniclaude"
        assert isinstance(blocked.session_id, UUID)
        assert blocked.details == []
        assert isinstance(blocked.correlation_id, UUID)

    def test_full_fields(self) -> None:
        """Should create model with all fields."""
        correlation = UUID("12345678-1234-5678-1234-567812345678")
        blocked = ModelNotificationBlocked(
            ticket_identifier="OMN-1234",
            reason="Waiting for approval",
            details=["Phase: spec", "Gate: approve spec"],
            repo="omniclaude",
            session_id=uuid4(),
            correlation_id=correlation,
        )
        assert blocked.details == ["Phase: spec", "Gate: approve spec"]
        assert blocked.correlation_id == correlation

    def test_frozen_model(self) -> None:
        """Should raise when attempting to modify frozen model."""
        blocked = ModelNotificationBlocked(
            ticket_identifier="OMN-1234",
            reason="Waiting for approval",
            repo="omniclaude",
            session_id=uuid4(),
        )
        with pytest.raises(ValidationError):
            blocked.ticket_identifier = "OMN-5678"  # type: ignore[misc]

    def test_forbids_extra_fields(self) -> None:
        """Should raise when extra fields are provided."""
        with pytest.raises(ValidationError):
            ModelNotificationBlocked(
                ticket_identifier="OMN-1234",
                reason="Waiting for approval",
                repo="omniclaude",
                session_id=uuid4(),
                extra_field="not allowed",  # type: ignore[call-arg]
            )

    def test_validates_required_ticket_identifier(self) -> None:
        """Should raise when ticket_identifier is missing."""
        with pytest.raises(ValidationError):
            ModelNotificationBlocked(
                reason="Waiting for approval",
                repo="omniclaude",
                session_id=uuid4(),
            )  # type: ignore[call-arg]

    def test_validates_non_empty_ticket_identifier(self) -> None:
        """Should raise when ticket_identifier is empty."""
        with pytest.raises(ValidationError):
            ModelNotificationBlocked(
                ticket_identifier="",
                reason="Waiting for approval",
                repo="omniclaude",
                session_id=uuid4(),
            )

    def test_validates_non_empty_reason(self) -> None:
        """Should raise when reason is empty."""
        with pytest.raises(ValidationError):
            ModelNotificationBlocked(
                ticket_identifier="OMN-1234",
                reason="",
                repo="omniclaude",
                session_id=uuid4(),
            )


class TestModelNotificationCompleted:
    """Tests for ModelNotificationCompleted Pydantic model."""

    def test_minimal_required_fields(self) -> None:
        """Should create model with only required fields."""
        completed = ModelNotificationCompleted(
            ticket_identifier="OMN-1234",
            summary="Feature implemented",
            repo="omniclaude",
            session_id=uuid4(),
        )
        assert completed.ticket_identifier == "OMN-1234"
        assert completed.summary == "Feature implemented"
        assert completed.repo == "omniclaude"
        assert isinstance(completed.session_id, UUID)
        assert completed.pr_url is None
        assert isinstance(completed.correlation_id, UUID)

    def test_full_fields(self) -> None:
        """Should create model with all fields."""
        correlation = UUID("12345678-1234-5678-1234-567812345678")
        completed = ModelNotificationCompleted(
            ticket_identifier="OMN-1234",
            summary="Feature implemented",
            repo="omniclaude",
            pr_url="https://github.com/org/repo/pull/123",
            session_id=uuid4(),
            correlation_id=correlation,
        )
        assert completed.pr_url == "https://github.com/org/repo/pull/123"
        assert completed.correlation_id == correlation

    def test_frozen_model(self) -> None:
        """Should raise when attempting to modify frozen model."""
        completed = ModelNotificationCompleted(
            ticket_identifier="OMN-1234",
            summary="Feature implemented",
            repo="omniclaude",
            session_id=uuid4(),
        )
        with pytest.raises(ValidationError):
            completed.ticket_identifier = "OMN-5678"  # type: ignore[misc]

    def test_forbids_extra_fields(self) -> None:
        """Should raise when extra fields are provided."""
        with pytest.raises(ValidationError):
            ModelNotificationCompleted(
                ticket_identifier="OMN-1234",
                summary="Feature implemented",
                repo="omniclaude",
                session_id=uuid4(),
                extra_field="not allowed",  # type: ignore[call-arg]
            )

    def test_validates_required_summary(self) -> None:
        """Should raise when summary is missing."""
        with pytest.raises(ValidationError):
            ModelNotificationCompleted(
                ticket_identifier="OMN-1234",
                repo="omniclaude",
                session_id=uuid4(),
            )  # type: ignore[call-arg]

    def test_validates_non_empty_summary(self) -> None:
        """Should raise when summary is empty."""
        with pytest.raises(ValidationError):
            ModelNotificationCompleted(
                ticket_identifier="OMN-1234",
                summary="",
                repo="omniclaude",
                session_id=uuid4(),
            )

    def test_pr_url_can_be_none(self) -> None:
        """Should allow pr_url to be None."""
        completed = ModelNotificationCompleted(
            ticket_identifier="OMN-1234",
            summary="Feature implemented",
            repo="omniclaude",
            session_id=uuid4(),
            pr_url=None,
        )
        assert completed.pr_url is None
