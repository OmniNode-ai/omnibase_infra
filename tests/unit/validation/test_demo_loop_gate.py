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
        canonical_topic = "onex.evt.omniclaude.session-outcome.v1"

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

    def test_fails_with_empty_topics(self) -> None:
        gate = DemoLoopGate(
            canonical_topics=(),
            projector_check_enabled=False,
            dashboard_check_enabled=False,
        )
        result = gate.assert_required_event_types()
        assert result.status == EnumAssertionStatus.FAILED
        assert "no coverage" in result.message.lower()


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
        with patch.dict(
            "sys.modules",
            {"omnibase_infra.runtime.projector_plugin_loader": None},
        ):
            result = gate.assert_projector_health()
        assert result.status == EnumAssertionStatus.SKIPPED
        assert "ModuleNotFoundError" in result.message

    def test_passed_with_contract_files(self, tmp_path: object) -> None:
        """Projector health passes when contracts directory has matching files.

        The real ``__file__`` lives at ``<pkg>/validation/demo_loop_gate.py``,
        so ``Path(__file__).parent.parent`` resolves to ``<pkg>/``, and the
        contracts dir is ``<pkg>/projectors/contracts/``.  We replicate that
        directory layout under ``tmp_path``.
        """
        from pathlib import Path

        pkg_root = Path(str(tmp_path)) / "fake_pkg"
        validation_dir = pkg_root / "validation"
        validation_dir.mkdir(parents=True)
        contracts_dir = pkg_root / "projectors" / "contracts"
        contracts_dir.mkdir(parents=True)
        (contracts_dir / "session_projector.yaml").write_text("name: test")

        gate = DemoLoopGate(
            projector_check_enabled=True,
            dashboard_check_enabled=False,
        )

        fake_file = str(validation_dir / "demo_loop_gate.py")
        with patch(
            "omnibase_infra.validation.demo_loop_gate.__file__",
            fake_file,
        ):
            result = gate.assert_projector_health()

        assert result.status == EnumAssertionStatus.PASSED
        assert "1 contract(s)" in result.message

    def test_failed_no_contracts_in_directory(self, tmp_path: object) -> None:
        """Projector health fails when contracts directory exists but is empty."""
        from pathlib import Path

        pkg_root = Path(str(tmp_path)) / "fake_pkg"
        validation_dir = pkg_root / "validation"
        validation_dir.mkdir(parents=True)
        contracts_dir = pkg_root / "projectors" / "contracts"
        contracts_dir.mkdir(parents=True)
        # Put a non-matching file so the dir exists but has no contracts
        (contracts_dir / "unrelated.txt").write_text("not a contract")

        gate = DemoLoopGate(
            projector_check_enabled=True,
            dashboard_check_enabled=False,
        )

        fake_file = str(validation_dir / "demo_loop_gate.py")
        with patch(
            "omnibase_infra.validation.demo_loop_gate.__file__",
            fake_file,
        ):
            result = gate.assert_projector_health()

        assert result.status == EnumAssertionStatus.FAILED
        assert "no projector contracts discovered" in result.message
        assert any("contains no matching contracts" in d for d in result.details)

    def test_failed_contracts_directory_missing(self, tmp_path: object) -> None:
        """Projector health fails with descriptive message when directory is absent."""
        from pathlib import Path

        pkg_root = Path(str(tmp_path)) / "fake_pkg"
        validation_dir = pkg_root / "validation"
        validation_dir.mkdir(parents=True)
        # Do NOT create projectors/contracts -- it should not exist

        gate = DemoLoopGate(
            projector_check_enabled=True,
            dashboard_check_enabled=False,
        )

        fake_file = str(validation_dir / "demo_loop_gate.py")
        with patch(
            "omnibase_infra.validation.demo_loop_gate.__file__",
            fake_file,
        ):
            result = gate.assert_projector_health()

        assert result.status == EnumAssertionStatus.FAILED
        assert "no projector contracts discovered" in result.message
        assert any("does not exist" in d for d in result.details)


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
        assert "Kafka bootstrap configured" in result.message

    def test_fails_with_no_kafka_servers(self) -> None:
        gate = DemoLoopGate(
            projector_check_enabled=False,
            dashboard_check_enabled=True,
        )
        with patch.dict("os.environ", {}, clear=True):
            result = gate.assert_dashboard_config()
        assert result.status == EnumAssertionStatus.FAILED
        assert "KAFKA_BOOTSTRAP_SERVERS" in result.message

    def test_fails_with_missing_port(self) -> None:
        """KAFKA_BOOTSTRAP_SERVERS without a port suffix should fail validation."""
        gate = DemoLoopGate(
            projector_check_enabled=False,
            dashboard_check_enabled=True,
        )
        with patch.dict(
            "os.environ",
            {"KAFKA_BOOTSTRAP_SERVERS": "just-a-hostname"},
        ):
            result = gate.assert_dashboard_config()
        assert result.status == EnumAssertionStatus.FAILED
        assert "host:port" in result.message
        assert any("missing" in d for d in result.details)

    def test_fails_with_empty_host(self) -> None:
        """A value like ':9092' (empty host) should fail validation."""
        gate = DemoLoopGate(
            projector_check_enabled=False,
            dashboard_check_enabled=True,
        )
        with patch.dict(
            "os.environ",
            {"KAFKA_BOOTSTRAP_SERVERS": ":9092"},
        ):
            result = gate.assert_dashboard_config()
        assert result.status == EnumAssertionStatus.FAILED
        assert "host:port" in result.message

    def test_fails_with_empty_port(self) -> None:
        """A value like 'localhost:' (empty port) should fail validation."""
        gate = DemoLoopGate(
            projector_check_enabled=False,
            dashboard_check_enabled=True,
        )
        with patch.dict(
            "os.environ",
            {"KAFKA_BOOTSTRAP_SERVERS": "localhost:"},
        ):
            result = gate.assert_dashboard_config()
        assert result.status == EnumAssertionStatus.FAILED
        assert "host:port" in result.message

    def test_passes_with_multiple_brokers(self) -> None:
        """Comma-separated broker list should pass when all are valid."""
        gate = DemoLoopGate(
            projector_check_enabled=False,
            dashboard_check_enabled=True,
        )
        with patch.dict(
            "os.environ",
            {"KAFKA_BOOTSTRAP_SERVERS": "broker1:9092,broker2:9093"},
        ):
            result = gate.assert_dashboard_config()
        assert result.status == EnumAssertionStatus.PASSED


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
        canonical_topic = "onex.evt.omniclaude.session-outcome.v1"

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

    def test_invalid_topics_returns_failed_result(self) -> None:
        """Call main() with a gate that has invalid topics and verify exit 1.

        We wrap the real DemoLoopGate class so that any construction
        transparently injects invalid topics, without monkey-patching
        ``__init__`` with a non-standard signature.
        """

        class _GateWithBadTopics(DemoLoopGate):
            def __init__(self, **kwargs: object) -> None:
                kwargs["canonical_topics"] = ("not-a-valid-topic",)
                super().__init__(**kwargs)  # type: ignore[arg-type]

        with patch(
            "omnibase_infra.validation.demo_loop_gate.DemoLoopGate",
            _GateWithBadTopics,
        ):
            exit_code = main(["--ci"])
        assert exit_code == 1

    def test_help_flag(self) -> None:
        with pytest.raises(SystemExit) as exc_info:
            main(["--help"])
        assert exc_info.value.code == 0


# =============================================================================
# Test: _load_env_file
# =============================================================================


class TestLoadEnvFile:
    """Tests for the _load_env_file helper."""

    def test_handles_export_prefix(self, tmp_path: object) -> None:
        """_load_env_file strips the 'export' prefix from keys."""
        from pathlib import Path

        from omnibase_infra.validation.demo_loop_gate import _load_env_file

        env_file = Path(str(tmp_path)) / ".env"
        env_file.write_text(
            "export MY_DEMO_VAR=hello\nexport ANOTHER_VAR=world\nPLAIN_VAR=plain\n"
        )

        with patch.dict("os.environ", {}, clear=True):
            _load_env_file(str(env_file))

            import os

            assert os.environ.get("MY_DEMO_VAR") == "hello"
            assert os.environ.get("ANOTHER_VAR") == "world"
            assert os.environ.get("PLAIN_VAR") == "plain"
            # Ensure the raw "export MY_DEMO_VAR" key was NOT set
            assert "export MY_DEMO_VAR" not in os.environ

    def test_strips_double_quotes_from_values(self, tmp_path: object) -> None:
        """Double-quoted values have their quotes removed."""
        from pathlib import Path

        from omnibase_infra.validation.demo_loop_gate import _load_env_file

        env_file = Path(str(tmp_path)) / ".env"
        env_file.write_text('DOUBLE_QUOTED="some value"\n')

        with patch.dict("os.environ", {}, clear=True):
            _load_env_file(str(env_file))

            import os

            assert os.environ.get("DOUBLE_QUOTED") == "some value"

    def test_strips_single_quotes_from_values(self, tmp_path: object) -> None:
        """Single-quoted values have their quotes removed."""
        from pathlib import Path

        from omnibase_infra.validation.demo_loop_gate import _load_env_file

        env_file = Path(str(tmp_path)) / ".env"
        env_file.write_text("SINGLE_QUOTED='another value'\n")

        with patch.dict("os.environ", {}, clear=True):
            _load_env_file(str(env_file))

            import os

            assert os.environ.get("SINGLE_QUOTED") == "another value"

    def test_skips_comment_lines(self, tmp_path: object) -> None:
        """Lines starting with # are ignored."""
        from pathlib import Path

        from omnibase_infra.validation.demo_loop_gate import _load_env_file

        env_file = Path(str(tmp_path)) / ".env"
        env_file.write_text(
            "# This is a comment\nREAL_KEY=real_value\n# Another comment\n"
        )

        with patch.dict("os.environ", {}, clear=True):
            _load_env_file(str(env_file))

            import os

            assert os.environ.get("REAL_KEY") == "real_value"
            # No key should exist from comment lines
            assert "# This is a comment" not in os.environ

    def test_skips_lines_without_equals(self, tmp_path: object) -> None:
        """Lines without an = sign are silently skipped."""
        from pathlib import Path

        from omnibase_infra.validation.demo_loop_gate import _load_env_file

        env_file = Path(str(tmp_path)) / ".env"
        env_file.write_text("no_equals_here\nVALID_KEY=valid_value\njust_text\n")

        with patch.dict("os.environ", {}, clear=True):
            _load_env_file(str(env_file))

            import os

            assert os.environ.get("VALID_KEY") == "valid_value"

    def test_nonexistent_file_does_not_crash(self) -> None:
        """Passing a path to a file that does not exist logs a warning but does not raise."""
        from omnibase_infra.validation.demo_loop_gate import _load_env_file

        # Should not raise any exception
        _load_env_file("/nonexistent/path/to/.env")

    def test_does_not_override_existing_env_vars(self, tmp_path: object) -> None:
        """If a variable is already set in os.environ, _load_env_file does NOT override it."""
        from pathlib import Path

        from omnibase_infra.validation.demo_loop_gate import _load_env_file

        env_file = Path(str(tmp_path)) / ".env"
        env_file.write_text("EXISTING_VAR=from_file\nNEW_VAR=from_file\n")

        with patch.dict("os.environ", {"EXISTING_VAR": "already_set"}, clear=True):
            _load_env_file(str(env_file))

            import os

            # Existing var should retain its original value
            assert os.environ.get("EXISTING_VAR") == "already_set"
            # New var should be loaded from file
            assert os.environ.get("NEW_VAR") == "from_file"

    def test_strips_inline_comments_from_unquoted_values(
        self, tmp_path: object
    ) -> None:
        """Inline comments (space + #) are stripped from unquoted values."""
        from pathlib import Path

        from omnibase_infra.validation.demo_loop_gate import _load_env_file

        env_file = Path(str(tmp_path)) / ".env"
        env_file.write_text("HOST=localhost # the host\nPORT=5432 # postgres port\n")

        with patch.dict("os.environ", {}, clear=True):
            _load_env_file(str(env_file))

            import os

            assert os.environ.get("HOST") == "localhost"
            assert os.environ.get("PORT") == "5432"

    def test_preserves_hash_in_quoted_values(self, tmp_path: object) -> None:
        """A '#' inside quotes is literal and must NOT be stripped."""
        from pathlib import Path

        from omnibase_infra.validation.demo_loop_gate import _load_env_file

        env_file = Path(str(tmp_path)) / ".env"
        env_file.write_text(
            "DOUBLE=\"value # with hash\"\nSINGLE='value # with hash'\n"
        )

        with patch.dict("os.environ", {}, clear=True):
            _load_env_file(str(env_file))

            import os

            assert os.environ.get("DOUBLE") == "value # with hash"
            assert os.environ.get("SINGLE") == "value # with hash"

    def test_hash_without_leading_space_is_not_a_comment(
        self, tmp_path: object
    ) -> None:
        """A '#' not preceded by a space is part of the value, not a comment."""
        from pathlib import Path

        from omnibase_infra.validation.demo_loop_gate import _load_env_file

        env_file = Path(str(tmp_path)) / ".env"
        env_file.write_text("COLOR=#ff0000\n")

        with patch.dict("os.environ", {}, clear=True):
            _load_env_file(str(env_file))

            import os

            assert os.environ.get("COLOR") == "#ff0000"


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
