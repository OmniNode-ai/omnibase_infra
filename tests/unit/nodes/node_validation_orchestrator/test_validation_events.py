# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for cross-repo validation event models.

Verifies construction, frozen immutability, extra-field rejection,
and JSON serialization for all 4 validation event models.

Field contracts are pinned to omnidash Zod schemas.

Reference: OMN-5184 Batch B Task 6
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from omnibase_infra.nodes.node_validation_orchestrator.models.model_cross_repo_run_completed_event import (
    ModelCrossRepoRunCompletedEvent,
)
from omnibase_infra.nodes.node_validation_orchestrator.models.model_cross_repo_run_started_event import (
    ModelCrossRepoRunStartedEvent,
)
from omnibase_infra.nodes.node_validation_orchestrator.models.model_cross_repo_violations_batch_event import (
    ModelCrossRepoViolationsBatchEvent,
)
from omnibase_infra.nodes.node_validation_orchestrator.models.model_lifecycle_candidate_upserted_event import (
    ModelLifecycleCandidateUpsertedEvent,
)
from omnibase_infra.nodes.node_validation_orchestrator.models.model_violation import (
    ModelViolation,
)

pytestmark = pytest.mark.unit

NOW = datetime(2026, 3, 17, 12, 0, 0, tzinfo=UTC)


class TestModelCrossRepoRunStartedEvent:
    def test_construct_valid(self) -> None:
        event = ModelCrossRepoRunStartedEvent(
            run_id="run-123",
            repos=["omniclaude", "omnibase_core"],
            validators=["dep-check", "naming-lint"],
            triggered_by="ci",
            timestamp=NOW,
        )
        assert event.run_id == "run-123"
        assert event.repos == ["omniclaude", "omnibase_core"]
        assert event.event_type == "ValidationRunStarted"

    def test_frozen(self) -> None:
        event = ModelCrossRepoRunStartedEvent(
            run_id="x", repos=[], validators=[], timestamp=NOW
        )
        with pytest.raises(Exception):
            event.run_id = "y"  # type: ignore[misc]

    def test_extra_forbidden(self) -> None:
        with pytest.raises(ValueError):
            ModelCrossRepoRunStartedEvent(
                run_id="x",
                repos=[],
                validators=[],
                timestamp=NOW,
                extra="bad",  # type: ignore[call-arg]
            )

    def test_triggered_by_defaults_to_empty_string(self) -> None:
        event = ModelCrossRepoRunStartedEvent(
            run_id="x", repos=[], validators=[], timestamp=NOW
        )
        assert event.triggered_by == ""

    def test_json_keys(self) -> None:
        event = ModelCrossRepoRunStartedEvent(
            run_id="x", repos=[], validators=[], timestamp=NOW
        )
        data = event.model_dump(mode="json")
        assert set(data.keys()) == {
            "event_type",
            "run_id",
            "repos",
            "validators",
            "triggered_by",
            "timestamp",
        }


class TestModelCrossRepoViolationsBatchEvent:
    def test_construct_with_violations(self) -> None:
        v = ModelViolation(
            rule_id="R001",
            severity="error",
            message="bad",
            repo="omniclaude",
            file_path="src/foo.py",
            line=42,
            validator="naming-lint",
        )
        event = ModelCrossRepoViolationsBatchEvent(
            run_id="run-123",
            violations=[v],
            batch_index=0,
            timestamp=NOW,
        )
        assert len(event.violations) == 1
        assert event.violations[0].rule_id == "R001"

    def test_json_keys(self) -> None:
        event = ModelCrossRepoViolationsBatchEvent(
            run_id="x",
            violations=[],
            batch_index=0,
            timestamp=NOW,
        )
        data = event.model_dump(mode="json")
        assert set(data.keys()) == {
            "event_type",
            "run_id",
            "violations",
            "batch_index",
            "timestamp",
        }


class TestModelCrossRepoRunCompletedEvent:
    def test_construct_valid(self) -> None:
        event = ModelCrossRepoRunCompletedEvent(
            run_id="run-123",
            status="passed",
            total_violations=0,
            duration_ms=4500,
            timestamp=NOW,
        )
        assert event.status == "passed"
        assert event.total_violations == 0

    def test_violations_by_severity(self) -> None:
        event = ModelCrossRepoRunCompletedEvent(
            run_id="run-123",
            status="failed",
            total_violations=5,
            violations_by_severity={"error": 3, "warning": 2},
            duration_ms=8000,
            timestamp=NOW,
        )
        assert event.violations_by_severity == {"error": 3, "warning": 2}

    def test_violations_by_severity_rejects_negative(self) -> None:
        with pytest.raises(ValueError, match="must be >= 0"):
            ModelCrossRepoRunCompletedEvent(
                run_id="run-123",
                status="failed",
                total_violations=5,
                violations_by_severity={"error": -1},
                duration_ms=8000,
                timestamp=NOW,
            )

    def test_json_keys(self) -> None:
        event = ModelCrossRepoRunCompletedEvent(
            run_id="x",
            status="passed",
            total_violations=0,
            duration_ms=100,
            timestamp=NOW,
        )
        data = event.model_dump(mode="json")
        assert set(data.keys()) == {
            "event_type",
            "run_id",
            "status",
            "total_violations",
            "violations_by_severity",
            "duration_ms",
            "timestamp",
        }


class TestModelLifecycleCandidateUpsertedEvent:
    def test_construct_valid(self) -> None:
        event = ModelLifecycleCandidateUpsertedEvent(
            candidate_id="cand-1",
            rule_name="no-hardcoded-urls",
            rule_id="R042",
            tier="suggested",
            status="pass",
            source_repo="omniclaude",
            entered_tier_at=NOW,
            last_validated_at=NOW,
            pass_streak=5,
            fail_streak=0,
            total_runs=10,
            timestamp=NOW,
        )
        assert event.tier == "suggested"
        assert event.pass_streak == 5

    def test_tz_validator_names_field(self) -> None:
        """Timezone validator error message should reference the actual field name."""
        with pytest.raises(ValueError, match="entered_tier_at must be timezone-aware"):
            ModelLifecycleCandidateUpsertedEvent(
                candidate_id="c1",
                rule_name="r",
                rule_id="R1",
                tier="observed",
                status="pending",
                source_repo="repo",
                entered_tier_at=datetime(2026, 1, 1),  # naive - no tz
                last_validated_at=NOW,
                pass_streak=0,
                fail_streak=0,
                total_runs=0,
                timestamp=NOW,
            )

    def test_roundtrip(self) -> None:
        event = ModelLifecycleCandidateUpsertedEvent(
            candidate_id="c1",
            rule_name="r",
            rule_id="R1",
            tier="observed",
            status="pending",
            source_repo="repo",
            entered_tier_at=NOW,
            last_validated_at=NOW,
            pass_streak=0,
            fail_streak=0,
            total_runs=0,
            timestamp=NOW,
        )
        data = event.model_dump(mode="json")
        reconstructed = ModelLifecycleCandidateUpsertedEvent.model_validate(data)
        assert reconstructed == event
