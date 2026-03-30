# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Tests for session registry models (OMN-6853).

Validates Pydantic model construction, frozen immutability,
enum validation, and phase ordering for Doctrine D3 compliance.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from omnibase_infra.services.session_registry.enum_session_phase import EnumSessionPhase
from omnibase_infra.services.session_registry.enum_session_registry_status import (
    EnumSessionRegistryStatus,
)
from omnibase_infra.services.session_registry.models import (
    ModelSessionRegistryEntry,
    phase_is_forward,
)


@pytest.mark.unit
class TestModelSessionRegistryEntry:
    """ModelSessionRegistryEntry construction and validation."""

    def test_minimal_creation(self) -> None:
        """Entry can be created with just task_id."""
        entry = ModelSessionRegistryEntry(task_id="OMN-1234")
        assert entry.task_id == "OMN-1234"
        assert entry.status == EnumSessionRegistryStatus.ACTIVE
        assert entry.current_phase is None
        assert entry.files_touched == []
        assert entry.session_ids == []

    def test_full_creation(self) -> None:
        """Entry accepts all fields."""
        now = datetime(2026, 3, 28, 12, 0, 0, tzinfo=UTC)
        entry = ModelSessionRegistryEntry(
            task_id="OMN-1234",
            status=EnumSessionRegistryStatus.ACTIVE,
            current_phase=EnumSessionPhase.IMPLEMENTING,
            worktree_path="/omni_worktrees/OMN-1234/omnibase_core",
            files_touched=["src/models/foo.py", "tests/test_foo.py"],
            depends_on=["OMN-1230"],
            session_ids=["session-abc-123"],
            correlation_ids=["550e8400-e29b-41d4-a716-446655440000"],
            decisions=["Chose asyncpg over SQLAlchemy for perf"],
            last_activity=now,
            created_at=now,
        )
        assert entry.task_id == "OMN-1234"
        assert entry.current_phase == EnumSessionPhase.IMPLEMENTING
        assert len(entry.files_touched) == 2
        assert entry.last_activity == now

    def test_frozen_immutability(self) -> None:
        """Model is frozen -- field reassignment raises."""
        entry = ModelSessionRegistryEntry(
            task_id="OMN-1234",
            status=EnumSessionRegistryStatus.ACTIVE,
        )
        with pytest.raises(Exception):
            entry.status = EnumSessionRegistryStatus.COMPLETED  # type: ignore[misc]

    def test_extra_fields_forbidden(self) -> None:
        """Extra fields are rejected (extra='forbid')."""
        with pytest.raises(Exception):
            ModelSessionRegistryEntry(
                task_id="OMN-1234",
                unknown_field="should_fail",  # type: ignore[call-arg]
            )

    def test_status_rejects_bare_string(self) -> None:
        """Status must use EnumSessionRegistryStatus, not arbitrary strings."""
        with pytest.raises(Exception):
            ModelSessionRegistryEntry(
                task_id="OMN-1234",
                status="unknown_status",  # type: ignore[arg-type]
            )

    def test_phase_rejects_bare_string(self) -> None:
        """current_phase must use EnumSessionPhase, not arbitrary strings."""
        with pytest.raises(Exception):
            ModelSessionRegistryEntry(
                task_id="OMN-1234",
                current_phase="unknown_phase",  # type: ignore[arg-type]
            )

    def test_task_id_min_length(self) -> None:
        """task_id must be at least 1 character."""
        with pytest.raises(Exception):
            ModelSessionRegistryEntry(task_id="")

    def test_task_id_max_length(self) -> None:
        """task_id enforces max_length=64."""
        with pytest.raises(Exception):
            ModelSessionRegistryEntry(task_id="A" * 65)


@pytest.mark.unit
class TestEnumSessionPhase:
    """EnumSessionPhase values and ordering."""

    def test_all_phases_exist(self) -> None:
        phases = [
            "planning",
            "implementing",
            "reviewing",
            "merging",
            "deploying",
            "completed",
            "stalled",
        ]
        for phase in phases:
            assert EnumSessionPhase(phase) is not None

    def test_phase_string_values(self) -> None:
        assert str(EnumSessionPhase.IMPLEMENTING) == "implementing"
        assert str(EnumSessionPhase.COMPLETED) == "completed"


@pytest.mark.unit
class TestPhaseIsForward:
    """Doctrine D3: phase ordering for replay-safe advancement."""

    def test_implementing_after_planning(self) -> None:
        assert (
            phase_is_forward(EnumSessionPhase.PLANNING, EnumSessionPhase.IMPLEMENTING)
            is True
        )

    def test_planning_after_implementing_is_backward(self) -> None:
        assert (
            phase_is_forward(EnumSessionPhase.IMPLEMENTING, EnumSessionPhase.PLANNING)
            is False
        )

    def test_same_phase_is_not_forward(self) -> None:
        assert (
            phase_is_forward(EnumSessionPhase.REVIEWING, EnumSessionPhase.REVIEWING)
            is False
        )

    def test_completed_after_merging(self) -> None:
        assert (
            phase_is_forward(EnumSessionPhase.MERGING, EnumSessionPhase.COMPLETED)
            is True
        )

    def test_full_forward_sequence(self) -> None:
        """Walk the full lifecycle forward."""
        phases = [
            EnumSessionPhase.PLANNING,
            EnumSessionPhase.IMPLEMENTING,
            EnumSessionPhase.REVIEWING,
            EnumSessionPhase.MERGING,
            EnumSessionPhase.DEPLOYING,
            EnumSessionPhase.COMPLETED,
        ]
        for i in range(len(phases) - 1):
            assert phase_is_forward(phases[i], phases[i + 1]) is True


@pytest.mark.unit
class TestEnumSessionRegistryStatus:
    """EnumSessionRegistryStatus values."""

    def test_all_statuses_exist(self) -> None:
        statuses = ["active", "completed", "stalled"]
        for status in statuses:
            assert EnumSessionRegistryStatus(status) is not None

    def test_active_is_default(self) -> None:
        entry = ModelSessionRegistryEntry(task_id="OMN-1234")
        assert entry.status == EnumSessionRegistryStatus.ACTIVE
