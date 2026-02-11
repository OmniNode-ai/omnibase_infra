# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Tests for session state effect models."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from omnibase_infra.enums import EnumSessionLifecycleState
from omnibase_infra.nodes.node_session_state_effect.models import (
    ModelRunContext,
    ModelSessionIndex,
    ModelSessionStateResult,
)

# ============================================================
# ModelSessionIndex tests
# ============================================================


class TestModelSessionIndex:
    """Tests for ModelSessionIndex."""

    def test_default_construction(self) -> None:
        """Default index has no active run and empty recent_run_ids."""
        idx = ModelSessionIndex()
        assert idx.active_run_id is None
        assert idx.recent_run_ids == ()
        assert idx.updated_at.tzinfo is not None

    def test_with_run_added(self) -> None:
        """Adding a run prepends it to recent_run_ids."""
        idx = ModelSessionIndex()
        idx2 = idx.with_run_added("run-1")
        assert idx2.recent_run_ids == ("run-1",)
        assert idx2.active_run_id is None  # Not set unless requested

    def test_with_run_added_set_active(self) -> None:
        """Adding a run with set_active=True sets active_run_id."""
        idx = ModelSessionIndex()
        idx2 = idx.with_run_added("run-1", set_active=True)
        assert idx2.active_run_id == "run-1"
        assert idx2.recent_run_ids == ("run-1",)

    def test_with_run_added_deduplicates(self) -> None:
        """Adding an existing run_id moves it to the front."""
        idx = ModelSessionIndex(recent_run_ids=("run-1", "run-2", "run-3"))
        idx2 = idx.with_run_added("run-2")
        assert idx2.recent_run_ids == ("run-2", "run-1", "run-3")

    def test_with_run_removed(self) -> None:
        """Removing a run filters it from recent_run_ids."""
        idx = ModelSessionIndex(
            recent_run_ids=("run-1", "run-2"),
            active_run_id="run-1",
        )
        idx2 = idx.with_run_removed("run-1")
        assert idx2.recent_run_ids == ("run-2",)
        assert idx2.active_run_id is None  # Cleared when active is removed

    def test_with_run_removed_non_active(self) -> None:
        """Removing a non-active run preserves active_run_id."""
        idx = ModelSessionIndex(
            recent_run_ids=("run-1", "run-2"),
            active_run_id="run-1",
        )
        idx2 = idx.with_run_removed("run-2")
        assert idx2.recent_run_ids == ("run-1",)
        assert idx2.active_run_id == "run-1"

    def test_with_active_run(self) -> None:
        """Setting active_run_id succeeds for existing runs."""
        idx = ModelSessionIndex(recent_run_ids=("run-1", "run-2"))
        idx2 = idx.with_active_run("run-2")
        assert idx2.active_run_id == "run-2"

    def test_with_active_run_invalid(self) -> None:
        """Setting active_run_id to non-existent run raises ValueError."""
        idx = ModelSessionIndex(recent_run_ids=("run-1",))
        with pytest.raises(ValueError, match="not in recent_run_ids"):
            idx.with_active_run("run-unknown")

    def test_immutability(self) -> None:
        """Model is frozen — field assignment raises TypeError."""
        idx = ModelSessionIndex()
        with pytest.raises(Exception):
            idx.active_run_id = "run-1"  # type: ignore[misc]

    def test_timezone_validation(self) -> None:
        """updated_at must be timezone-aware."""
        with pytest.raises(ValueError, match="timezone-aware"):
            ModelSessionIndex(updated_at=datetime(2026, 1, 1))

    @pytest.mark.parametrize(
        "bad_id",
        ["../etc/passwd", "foo/bar", "foo\\bar", "foo\0bar", ".."],
    )
    def test_active_run_id_path_traversal_rejected(self, bad_id: str) -> None:
        """active_run_id with unsafe characters is rejected by allowlist."""
        with pytest.raises(ValueError, match="run_id"):
            ModelSessionIndex(active_run_id=bad_id, recent_run_ids=())

    def test_active_run_id_valid_accepted(self) -> None:
        """active_run_id with safe characters is accepted."""
        idx = ModelSessionIndex(
            active_run_id="run-abc.123_test",
            recent_run_ids=("run-abc.123_test",),
        )
        assert idx.active_run_id == "run-abc.123_test"

    def test_active_run_id_none_accepted(self) -> None:
        """active_run_id=None bypasses validation."""
        idx = ModelSessionIndex(active_run_id=None)
        assert idx.active_run_id is None

    def test_with_run_added_caps_at_max(self) -> None:
        """recent_run_ids is trimmed to MAX_RECENT_RUNS."""
        max_runs = ModelSessionIndex.MAX_RECENT_RUNS
        ids = tuple(f"run-{i}" for i in range(max_runs))
        idx = ModelSessionIndex(recent_run_ids=ids)
        idx2 = idx.with_run_added("run-new")
        assert len(idx2.recent_run_ids) == max_runs
        assert idx2.recent_run_ids[0] == "run-new"
        # The oldest entry was trimmed
        assert f"run-{max_runs - 1}" not in idx2.recent_run_ids

    def test_with_run_added_clears_trimmed_active(self) -> None:
        """active_run_id is cleared when it gets trimmed by MAX_RECENT_RUNS."""
        max_runs = ModelSessionIndex.MAX_RECENT_RUNS
        # The oldest run is the active one
        oldest = f"run-{max_runs - 1}"
        ids = tuple(f"run-{i}" for i in range(max_runs))
        idx = ModelSessionIndex(recent_run_ids=ids, active_run_id=oldest)
        # Adding a new run trims the oldest (which is the active one)
        idx2 = idx.with_run_added("run-new")
        assert oldest not in idx2.recent_run_ids
        assert idx2.active_run_id is None  # Cleared because it was trimmed

    def test_with_run_added_preserves_active_when_not_trimmed(self) -> None:
        """active_run_id is preserved when not affected by trim."""
        ids = tuple(f"run-{i}" for i in range(5))
        idx = ModelSessionIndex(recent_run_ids=ids, active_run_id="run-0")
        idx2 = idx.with_run_added("run-new")
        assert idx2.active_run_id == "run-0"  # Still present, not trimmed


# ============================================================
# ModelRunContext tests
# ============================================================


class TestModelRunContext:
    """Tests for ModelRunContext."""

    def test_construction(self) -> None:
        """Run context has required run_id and defaults."""
        ctx = ModelRunContext(run_id="run-abc")
        assert ctx.run_id == "run-abc"
        assert ctx.status == EnumSessionLifecycleState.RUN_CREATED
        assert ctx.metadata == {}
        assert ctx.created_at.tzinfo is not None

    def test_with_status(self) -> None:
        """Status transition returns a new instance."""
        ctx = ModelRunContext(run_id="run-abc")
        ctx2 = ctx.with_status(EnumSessionLifecycleState.RUN_ACTIVE)
        assert ctx2.status == EnumSessionLifecycleState.RUN_ACTIVE
        assert ctx.status == EnumSessionLifecycleState.RUN_CREATED  # Original unchanged

    def test_with_metadata(self) -> None:
        """Adding metadata returns a new instance with the entry."""
        ctx = ModelRunContext(run_id="run-abc")
        ctx2 = ctx.with_metadata("ticket", "OMN-2117")
        assert ctx2.metadata == {"ticket": "OMN-2117"}
        assert ctx.metadata == {}  # Original unchanged

    def test_is_stale_fresh(self) -> None:
        """A just-created context is not stale."""
        ctx = ModelRunContext(run_id="run-abc")
        assert not ctx.is_stale()

    def test_is_stale_old(self) -> None:
        """A context with old updated_at is stale."""
        old = datetime(2020, 1, 1, tzinfo=UTC)
        ctx = ModelRunContext(
            run_id="run-abc",
            created_at=old,
            updated_at=old,
        )
        assert ctx.is_stale()

    def test_empty_run_id_rejected(self) -> None:
        """Empty run_id is rejected by min_length=1."""
        with pytest.raises(ValueError):
            ModelRunContext(run_id="")

    def test_extra_fields_forbidden(self) -> None:
        """Extra fields are rejected by extra='forbid'."""
        with pytest.raises(ValueError):
            ModelRunContext(run_id="run-abc", unknown_field="oops")  # type: ignore[call-arg]

    def test_metadata_defensive_copy(self) -> None:
        """Mutating the original dict does not affect the model's metadata."""
        original = {"key": "value"}
        ctx = ModelRunContext(run_id="run-abc", metadata=original)
        # Mutate the original dict after construction
        original["injected"] = "oops"
        assert "injected" not in ctx.metadata
        assert ctx.metadata == {"key": "value"}

    @pytest.mark.parametrize(
        "bad_id",
        ["../etc/passwd", "foo/bar", "foo\\bar", "foo\0bar", ".."],
    )
    def test_path_traversal_run_id_rejected(self, bad_id: str) -> None:
        """run_id containing unsafe characters is rejected by allowlist."""
        with pytest.raises(ValueError, match="run_id"):
            ModelRunContext(run_id=bad_id)


# ============================================================
# ModelSessionStateResult tests
# ============================================================


class TestModelSessionStateResult:
    """Tests for ModelSessionStateResult."""

    def test_success_result(self) -> None:
        """A successful result evaluates as True."""
        from uuid import uuid4

        result = ModelSessionStateResult(
            success=True,
            operation="test",
            correlation_id=uuid4(),
        )
        assert bool(result) is True

    def test_failure_result(self) -> None:
        """A failed result evaluates as False."""
        from uuid import uuid4

        result = ModelSessionStateResult(
            success=False,
            operation="test",
            correlation_id=uuid4(),
            error="oops",
            error_code="TEST_ERROR",
        )
        assert bool(result) is False
