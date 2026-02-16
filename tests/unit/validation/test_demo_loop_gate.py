# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Unit tests for the Demo Loop Assertion Gate (OMN-2297).

Tests cover all six assertion checks:
    1. Canonical pipeline exclusivity
    2. Required event types
    3. Schema version compatibility
    4. Projector health
    5. Dashboard config
    6. No duplicate events

Also tests the aggregate result model, CLI formatter, and edge cases.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from omnibase_infra.validation.demo_loop_gate import (
    CANONICAL_EVENT_TOPICS,
    LEGACY_TOPIC_MAPPINGS,
    DemoLoopGate,
    EnumAssertionStatus,
    ModelAssertionResult,
    ModelDemoLoopResult,
    format_result,
    main,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def gate_ci_mode() -> DemoLoopGate:
    """Gate in CI mode (skips projector and dashboard checks)."""
    return DemoLoopGate(
        projector_check_enabled=False,
        dashboard_check_enabled=False,
    )


@pytest.fixture
def gate_full() -> DemoLoopGate:
    """Gate with all checks enabled."""
    return DemoLoopGate(
        projector_check_enabled=True,
        dashboard_check_enabled=True,
    )


# =============================================================================
# Test: ModelAssertionResult
# =============================================================================


class TestModelAssertionResult:
    """Tests for the individual assertion result model."""

    def test_create_passed(self) -> None:
        result = ModelAssertionResult(
            name="test",
            status=EnumAssertionStatus.PASSED,
            message="All good",
        )
        assert result.name == "test"
        assert result.status == EnumAssertionStatus.PASSED
        assert result.message == "All good"
        assert result.details == ()

    def test_create_failed_with_details(self) -> None:
        result = ModelAssertionResult(
            name="test",
            status=EnumAssertionStatus.FAILED,
            message="Issues found",
            details=("detail 1", "detail 2"),
        )
        assert result.status == EnumAssertionStatus.FAILED
        assert len(result.details) == 2

    def test_frozen(self) -> None:
        result = ModelAssertionResult(
            name="test",
            status=EnumAssertionStatus.PASSED,
            message="ok",
        )
        with pytest.raises(Exception):
            result.name = "changed"  # type: ignore[misc]


# =============================================================================
# Test: ModelDemoLoopResult
# =============================================================================


class TestModelDemoLoopResult:
    """Tests for the aggregate demo loop result model."""

    def test_bool_true_when_ready(self) -> None:
        result = ModelDemoLoopResult(
            assertions=(),
            passed=3,
            failed=0,
            skipped=0,
            is_ready=True,
        )
        assert bool(result) is True

    def test_bool_false_when_not_ready(self) -> None:
        result = ModelDemoLoopResult(
            assertions=(),
            passed=2,
            failed=1,
            skipped=0,
            is_ready=False,
        )
        assert bool(result) is False

    def test_frozen(self) -> None:
        result = ModelDemoLoopResult(is_ready=True)
        with pytest.raises(Exception):
            result.is_ready = False  # type: ignore[misc]


# =============================================================================
# Test: Assertion 1 -- Canonical Pipeline Exclusivity
# =============================================================================


class TestCanonicalPipelineExclusivity:
    """Tests for the canonical pipeline exclusivity assertion."""

    def test_passes_with_no_legacy_registrations(
        self, gate_ci_mode: DemoLoopGate
    ) -> None:
        result = gate_ci_mode.assert_canonical_pipeline_exclusivity()
        assert result.status == EnumAssertionStatus.PASSED
        assert result.name == "canonical_pipeline"

    def test_passes_with_empty_legacy_mappings(self) -> None:
        gate = DemoLoopGate(
            legacy_mappings={},
            projector_check_enabled=False,
            dashboard_check_enabled=False,
        )
        result = gate.assert_canonical_pipeline_exclusivity()
        assert result.status == EnumAssertionStatus.PASSED

    def test_fails_when_legacy_topic_in_registrations(self) -> None:
        """Simulate a case where a legacy topic is found in registrations.

        We patch ALL_EVENT_REGISTRATIONS to include a registration that uses
        a legacy topic template.
        """
        from omnibase_infra.runtime.emit_daemon.event_registry import (
            ModelEventRegistration,
        )

        legacy_topic = "onex.cmd.omniintelligence.session-outcome.v1"
        canonical_topic = "onex.evt.omniintelligence.session-outcome.v1"

        fake_registrations = (
            ModelEventRegistration(
                event_type="session.outcome",
                topic_template=legacy_topic,
                schema_version="1.0.0",
            ),
        )

        gate = DemoLoopGate(
            legacy_mappings={legacy_topic: canonical_topic},
            projector_check_enabled=False,
            dashboard_check_enabled=False,
        )

        with patch(
            "omnibase_infra.validation.demo_loop_gate.ALL_EVENT_REGISTRATIONS",
            fake_registrations,
        ):
            result = gate.assert_canonical_pipeline_exclusivity()

        assert result.status == EnumAssertionStatus.FAILED
        assert "Legacy pipeline detected" in result.message
        assert len(result.details) == 1


# =============================================================================
# Test: Assertion 2 -- Required Event Types
# =============================================================================


class TestRequiredEventTypes:
    """Tests for the required event types assertion."""

    def test_passes_with_default_canonical_topics(
        self, gate_ci_mode: DemoLoopGate
    ) -> None:
        result = gate_ci_mode.assert_required_event_types()
        assert result.status == EnumAssertionStatus.PASSED
        total = len(CANONICAL_EVENT_TOPICS)
        assert f"{total}/{total}" in result.message

    def test_fails_with_invalid_topic(self) -> None:
        gate = DemoLoopGate(
            canonical_topics=(
                "onex.evt.platform.node-introspection.v1",
                "invalid-topic-format",
            ),
            projector_check_enabled=False,
            dashboard_check_enabled=False,
        )
        result = gate.assert_required_event_types()
        assert result.status == EnumAssertionStatus.FAILED
        assert "1 of 2" in result.message
        assert any("invalid-topic-format" in d for d in result.details)

    def test_fails_with_all_invalid(self) -> None:
        gate = DemoLoopGate(
            canonical_topics=("bad1", "bad2", "bad3"),
            projector_check_enabled=False,
            dashboard_check_enabled=False,
        )
        result = gate.assert_required_event_types()
        assert result.status == EnumAssertionStatus.FAILED
        assert "3 of 3" in result.message

    def test_passes_with_empty_topics(self) -> None:
        gate = DemoLoopGate(
            canonical_topics=(),
            projector_check_enabled=False,
            dashboard_check_enabled=False,
        )
        result = gate.assert_required_event_types()
        assert result.status == EnumAssertionStatus.PASSED
        assert "0/0" in result.message


# =============================================================================
# Test: Assertion 3 -- Schema Version Compatibility
# =============================================================================


class TestSchemaVersionCompatibility:
    """Tests for the schema version compatibility assertion."""

    def test_passes_with_matching_versions(self, gate_ci_mode: DemoLoopGate) -> None:
        result = gate_ci_mode.assert_schema_version_compatibility()
        assert result.status == EnumAssertionStatus.PASSED
        assert "1.0.0" in result.message

    def test_fails_with_mismatched_version(self) -> None:
        from omnibase_infra.runtime.emit_daemon.event_registry import (
            ModelEventRegistration,
        )

        fake_registrations = (
            ModelEventRegistration(
                event_type="test.event",
                topic_template="onex.evt.test.event.v1",
                schema_version="2.0.0",
            ),
        )

        gate = DemoLoopGate(
            expected_schema_version="1.0.0",
            projector_check_enabled=False,
            dashboard_check_enabled=False,
        )

        with patch(
            "omnibase_infra.validation.demo_loop_gate.ALL_EVENT_REGISTRATIONS",
            fake_registrations,
        ):
            result = gate.assert_schema_version_compatibility()

        assert result.status == EnumAssertionStatus.FAILED
        assert "mismatch" in result.message.lower()
        assert any("2.0.0" in d for d in result.details)


# =============================================================================
# Test: Assertion 4 -- Projector Health
# =============================================================================


class TestProjectorHealth:
    """Tests for the projector health assertion."""

    def test_skipped_in_ci_mode(self, gate_ci_mode: DemoLoopGate) -> None:
        result = gate_ci_mode.assert_projector_health()
        assert result.status == EnumAssertionStatus.SKIPPED
        assert "skipped" in result.message.lower()

    def test_skipped_on_import_error(self) -> None:
        gate = DemoLoopGate(
            projector_check_enabled=True,
            dashboard_check_enabled=False,
        )
        with patch(
            "omnibase_infra.validation.demo_loop_gate.DemoLoopGate.assert_projector_health"
        ) as mock_check:
            mock_check.return_value = ModelAssertionResult(
                name="projector_health",
                status=EnumAssertionStatus.SKIPPED,
                message="Projector health: skipped (ImportError)",
            )
            result = gate.assert_projector_health()
        assert result.status == EnumAssertionStatus.SKIPPED


# =============================================================================
# Test: Assertion 5 -- Dashboard Config
# =============================================================================


class TestDashboardConfig:
    """Tests for the dashboard config assertion."""

    def test_skipped_in_ci_mode(self, gate_ci_mode: DemoLoopGate) -> None:
        result = gate_ci_mode.assert_dashboard_config()
        assert result.status == EnumAssertionStatus.SKIPPED
        assert "skipped" in result.message.lower()

    def test_passes_with_kafka_servers_set(self) -> None:
        gate = DemoLoopGate(
            projector_check_enabled=False,
            dashboard_check_enabled=True,
        )
        with patch.dict(
            "os.environ",
            {"KAFKA_BOOTSTRAP_SERVERS": "192.168.86.200:29092"},
        ):
            result = gate.assert_dashboard_config()
        assert result.status == EnumAssertionStatus.PASSED
        assert "192.168.86.200:29092" in result.message

    def test_fails_with_no_kafka_servers(self) -> None:
        gate = DemoLoopGate(
            projector_check_enabled=False,
            dashboard_check_enabled=True,
        )
        with patch.dict("os.environ", {}, clear=True):
            result = gate.assert_dashboard_config()
        assert result.status == EnumAssertionStatus.FAILED
        assert "KAFKA_BOOTSTRAP_SERVERS" in result.message


# =============================================================================
# Test: Assertion 6 -- No Duplicate Events
# =============================================================================


class TestNoDuplicateEvents:
    """Tests for the no duplicate events assertion."""

    def test_passes_with_no_overlap(self, gate_ci_mode: DemoLoopGate) -> None:
        result = gate_ci_mode.assert_no_duplicate_events()
        assert result.status == EnumAssertionStatus.PASSED

    def test_fails_with_dual_emission(self) -> None:
        from omnibase_infra.runtime.emit_daemon.event_registry import (
            ModelEventRegistration,
        )

        legacy_topic = "onex.cmd.omniintelligence.session-outcome.v1"
        canonical_topic = "onex.evt.omniintelligence.session-outcome.v1"

        fake_registrations = (
            ModelEventRegistration(
                event_type="session.outcome.legacy",
                topic_template=legacy_topic,
                schema_version="1.0.0",
            ),
            ModelEventRegistration(
                event_type="session.outcome.canonical",
                topic_template=canonical_topic,
                schema_version="1.0.0",
            ),
        )

        gate = DemoLoopGate(
            legacy_mappings={legacy_topic: canonical_topic},
            projector_check_enabled=False,
            dashboard_check_enabled=False,
        )

        with patch(
            "omnibase_infra.validation.demo_loop_gate.ALL_EVENT_REGISTRATIONS",
            fake_registrations,
        ):
            result = gate.assert_no_duplicate_events()

        assert result.status == EnumAssertionStatus.FAILED
        assert "Duplicate events detected" in result.message
        assert len(result.details) == 1

    def test_passes_with_empty_legacy_mappings(self) -> None:
        gate = DemoLoopGate(
            legacy_mappings={},
            projector_check_enabled=False,
            dashboard_check_enabled=False,
        )
        result = gate.assert_no_duplicate_events()
        assert result.status == EnumAssertionStatus.PASSED


# =============================================================================
# Test: run_all() Aggregate
# =============================================================================


class TestRunAll:
    """Tests for the aggregate run_all() method."""

    def test_ci_mode_all_pass(self, gate_ci_mode: DemoLoopGate) -> None:
        result = gate_ci_mode.run_all()
        assert result.is_ready is True
        assert result.failed == 0
        # In CI mode, projector and dashboard are skipped
        assert result.skipped == 2
        assert result.passed == 4

    def test_returns_false_on_failure(self) -> None:
        gate = DemoLoopGate(
            canonical_topics=("invalid-topic",),
            projector_check_enabled=False,
            dashboard_check_enabled=False,
        )
        result = gate.run_all()
        assert result.is_ready is False
        assert result.failed >= 1
        assert bool(result) is False

    def test_all_six_assertions_present(self, gate_ci_mode: DemoLoopGate) -> None:
        result = gate_ci_mode.run_all()
        assert len(result.assertions) == 6
        names = {a.name for a in result.assertions}
        assert names == {
            "canonical_pipeline",
            "required_event_types",
            "schema_versions",
            "projector_health",
            "dashboard_config",
            "no_duplicate_events",
        }


# =============================================================================
# Test: format_result()
# =============================================================================


class TestFormatResult:
    """Tests for the CLI output formatter."""

    def test_format_passing_result(self, gate_ci_mode: DemoLoopGate) -> None:
        result = gate_ci_mode.run_all()
        output = format_result(result)
        assert "PASS: Demo loop ready" in output
        assert "[PASS]" in output
        assert "[SKIP]" in output

    def test_format_failing_result(self) -> None:
        gate = DemoLoopGate(
            canonical_topics=("bad-topic",),
            projector_check_enabled=False,
            dashboard_check_enabled=False,
        )
        result = gate.run_all()
        output = format_result(result)
        assert "FAIL: Demo loop not ready" in output
        assert "[FAIL]" in output

    def test_format_includes_details(self) -> None:
        gate = DemoLoopGate(
            canonical_topics=("bad-topic",),
            projector_check_enabled=False,
            dashboard_check_enabled=False,
        )
        result = gate.run_all()
        output = format_result(result)
        assert "bad-topic" in output


# =============================================================================
# Test: CLI main()
# =============================================================================


class TestCLIMain:
    """Tests for the CLI entry point."""

    def test_ci_mode_exits_zero(self) -> None:
        exit_code = main(["--ci"])
        assert exit_code == 0

    def test_ci_mode_verbose(self) -> None:
        exit_code = main(["--ci", "--verbose"])
        assert exit_code == 0

    def test_invalid_topics_exits_nonzero(self) -> None:
        gate = DemoLoopGate(
            canonical_topics=("bad-topic",),
            projector_check_enabled=False,
            dashboard_check_enabled=False,
        )
        result = gate.run_all()
        assert not result.is_ready

    def test_help_flag(self) -> None:
        with pytest.raises(SystemExit) as exc_info:
            main(["--help"])
        assert exc_info.value.code == 0


# =============================================================================
# Test: Constants
# =============================================================================


class TestConstants:
    """Tests for module-level constants."""

    def test_canonical_topics_not_empty(self) -> None:
        assert len(CANONICAL_EVENT_TOPICS) > 0

    def test_all_canonical_topics_are_onex_format(self) -> None:
        for topic in CANONICAL_EVENT_TOPICS:
            assert topic.startswith("onex."), (
                f"Topic {topic} doesn't start with 'onex.'"
            )
            parts = topic.split(".")
            assert len(parts) == 5, f"Topic {topic} doesn't have 5 segments"

    def test_legacy_mappings_has_entries(self) -> None:
        assert len(LEGACY_TOPIC_MAPPINGS) > 0

    def test_legacy_mappings_values_are_canonical(self) -> None:
        for legacy, canonical in LEGACY_TOPIC_MAPPINGS.items():
            assert "cmd" in legacy or "legacy" in legacy.lower(), (
                f"Legacy topic '{legacy}' doesn't look legacy"
            )
            assert "evt" in canonical, (
                f"Canonical topic '{canonical}' doesn't use 'evt' kind"
            )
