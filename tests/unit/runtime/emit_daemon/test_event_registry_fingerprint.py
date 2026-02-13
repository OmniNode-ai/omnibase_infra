# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Unit tests for event registry fingerprint computation and validation (OMN-2088).

Tests cover the fingerprint computation, assertion, diff generation, model
serialization, and CLI entry points.  All tests are pure unit tests with
no live infrastructure dependencies.

Test coverage includes:
- Deterministic fingerprint ordering (same registrations different order -> same hash)
- Empty registry fingerprint
- Single and multiple registration fingerprints
- assert_fingerprint: matching -> no error
- assert_fingerprint: additions, removals, modifications -> mismatch with diff
- Bounded diff (max 10 lines)
- Field-level change detection in diff
- JSON round-trip (to_json -> from_json_path)
- CLI stamp writes valid JSON
- CLI stamp --dry-run does not write
- CLI verify succeeds with matching artifact
- CLI verify fails with mismatched artifact
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from omnibase_infra.errors.error_event_registry_fingerprint import (
    EventRegistryFingerprintMismatchError,
    EventRegistryFingerprintMissingError,
)
from omnibase_infra.runtime.emit_daemon.event_registry import (
    _ARTIFACT_DEFAULT_PATH,
    EventRegistry,
    ModelEventRegistration,
    _cli_stamp,
    _cli_verify,
    _compute_registry_diff,
    _main,
    _sha256_json,
)
from omnibase_infra.runtime.emit_daemon.model_event_registry_fingerprint import (
    ModelEventRegistryFingerprint,
    ModelEventRegistryFingerprintElement,
)

pytestmark = pytest.mark.unit

# ---------------------------------------------------------------------------
# Fixed registrations for deterministic tests
# ---------------------------------------------------------------------------

REG_ALPHA = ModelEventRegistration(
    event_type="alpha.event",
    topic_template="onex.evt.alpha.event.v1",
    partition_key_field="alpha_id",
    required_fields=["alpha_id", "payload"],
    schema_version="1.0.0",
)

REG_BETA = ModelEventRegistration(
    event_type="beta.event",
    topic_template="onex.evt.beta.event.v1",
    partition_key_field=None,
    required_fields=["data"],
    schema_version="2.0.0",
)

REG_GAMMA = ModelEventRegistration(
    event_type="gamma.event",
    topic_template="onex.evt.gamma.event.v1",
    partition_key_field="gid",
    required_fields=[],
    schema_version="1.0.0",
)


# ---------------------------------------------------------------------------
# TestSha256Json
# ---------------------------------------------------------------------------


class TestSha256Json:
    """Tests for the _sha256_json helper function."""

    def test_deterministic_for_same_input(self) -> None:
        """Same input produces same hash."""
        obj = {"a": 1, "b": [2, 3]}
        assert _sha256_json(obj) == _sha256_json(obj)

    def test_different_input_different_hash(self) -> None:
        """Different inputs produce different hashes."""
        assert _sha256_json({"a": 1}) != _sha256_json({"a": 2})

    def test_key_order_does_not_affect_hash(self) -> None:
        """JSON serialization sorts keys, so dict order is irrelevant."""
        a = {"z": 1, "a": 2}
        b = {"a": 2, "z": 1}
        assert _sha256_json(a) == _sha256_json(b)

    def test_returns_64_char_hex(self) -> None:
        """SHA-256 produces 64 hex characters."""
        result = _sha256_json("test")
        assert len(result) == 64
        assert all(c in "0123456789abcdef" for c in result)

    def test_empty_list(self) -> None:
        """Empty list produces valid hash."""
        result = _sha256_json([])
        assert len(result) == 64

    def test_tuple_serialized_as_list(self) -> None:
        """Tuples and lists should produce the same hash (JSON has no tuple)."""
        assert _sha256_json((1, 2, 3)) == _sha256_json([1, 2, 3])


# ---------------------------------------------------------------------------
# TestComputeFingerprint
# ---------------------------------------------------------------------------


class TestComputeFingerprint:
    """Tests for EventRegistry.compute_fingerprint()."""

    def test_empty_registry(self) -> None:
        """Empty registry produces a valid fingerprint with zero elements."""
        registry = EventRegistry()
        fp = registry.compute_fingerprint()
        assert fp.version == 1
        assert len(fp.fingerprint_sha256) == 64
        assert fp.elements == ()

    def test_single_registration(self) -> None:
        """Single registration produces fingerprint with one element."""
        registry = EventRegistry()
        registry.register(REG_ALPHA)
        fp = registry.compute_fingerprint()
        assert len(fp.elements) == 1
        assert fp.elements[0].event_type == "alpha.event"
        assert fp.elements[0].topic_template == "onex.evt.alpha.event.v1"
        assert fp.elements[0].schema_version == "1.0.0"
        assert fp.elements[0].partition_key_field == "alpha_id"
        assert fp.elements[0].required_fields == ("alpha_id", "payload")
        assert len(fp.elements[0].element_sha256) == 64

    def test_multiple_registrations(self) -> None:
        """Multiple registrations produce sorted elements."""
        registry = EventRegistry()
        registry.register_batch([REG_BETA, REG_ALPHA])
        fp = registry.compute_fingerprint()
        assert len(fp.elements) == 2
        # Sorted by event_type
        assert fp.elements[0].event_type == "alpha.event"
        assert fp.elements[1].event_type == "beta.event"

    def test_deterministic_ordering(self) -> None:
        """Same registrations in different order produce same fingerprint."""
        reg1 = EventRegistry()
        reg1.register_batch([REG_ALPHA, REG_BETA, REG_GAMMA])

        reg2 = EventRegistry()
        reg2.register_batch([REG_GAMMA, REG_ALPHA, REG_BETA])

        fp1 = reg1.compute_fingerprint()
        fp2 = reg2.compute_fingerprint()
        assert fp1.fingerprint_sha256 == fp2.fingerprint_sha256

    def test_required_fields_sorted(self) -> None:
        """Required fields are sorted alphabetically in the element."""
        reg = ModelEventRegistration(
            event_type="test.sort",
            topic_template="onex.evt.test.sort.v1",
            required_fields=["zebra", "apple", "middle"],
        )
        registry = EventRegistry()
        registry.register(reg)
        fp = registry.compute_fingerprint()
        assert fp.elements[0].required_fields == ("apple", "middle", "zebra")

    def test_none_partition_key_becomes_empty_string(self) -> None:
        """None partition_key_field is represented as empty string."""
        registry = EventRegistry()
        registry.register(REG_BETA)
        fp = registry.compute_fingerprint()
        assert fp.elements[0].partition_key_field == ""

    def test_different_registrations_different_fingerprint(self) -> None:
        """Different registrations produce different overall fingerprints."""
        r1 = EventRegistry()
        r1.register(REG_ALPHA)

        r2 = EventRegistry()
        r2.register(REG_BETA)

        fp1 = r1.compute_fingerprint()
        fp2 = r2.compute_fingerprint()
        assert fp1.fingerprint_sha256 != fp2.fingerprint_sha256

    def test_added_registration_changes_fingerprint(self) -> None:
        """Adding a registration changes the overall fingerprint."""
        r1 = EventRegistry()
        r1.register(REG_ALPHA)
        fp1 = r1.compute_fingerprint()

        r2 = EventRegistry()
        r2.register_batch([REG_ALPHA, REG_BETA])
        fp2 = r2.compute_fingerprint()

        assert fp1.fingerprint_sha256 != fp2.fingerprint_sha256


# ---------------------------------------------------------------------------
# TestAssertFingerprint
# ---------------------------------------------------------------------------


class TestAssertFingerprint:
    """Tests for EventRegistry.assert_fingerprint()."""

    def test_matching_fingerprint_no_error(self) -> None:
        """Matching fingerprint does not raise."""
        registry = EventRegistry()
        registry.register_batch([REG_ALPHA, REG_BETA])
        expected = registry.compute_fingerprint()
        # Should not raise
        registry.assert_fingerprint(expected)

    def test_added_registration_raises_mismatch(self) -> None:
        """Adding a registration triggers mismatch error with diff."""
        registry_original = EventRegistry()
        registry_original.register(REG_ALPHA)
        expected = registry_original.compute_fingerprint()

        registry_modified = EventRegistry()
        registry_modified.register_batch([REG_ALPHA, REG_BETA])

        with pytest.raises(EventRegistryFingerprintMismatchError) as exc_info:
            registry_modified.assert_fingerprint(expected)

        assert "+ added: beta.event" in exc_info.value.diff_summary

    def test_removed_registration_raises_mismatch(self) -> None:
        """Removing a registration triggers mismatch error with diff."""
        registry_original = EventRegistry()
        registry_original.register_batch([REG_ALPHA, REG_BETA])
        expected = registry_original.compute_fingerprint()

        registry_modified = EventRegistry()
        registry_modified.register(REG_ALPHA)

        with pytest.raises(EventRegistryFingerprintMismatchError) as exc_info:
            registry_modified.assert_fingerprint(expected)

        assert "- removed: beta.event" in exc_info.value.diff_summary

    def test_modified_registration_raises_mismatch(self) -> None:
        """Modifying a registration triggers mismatch with field details."""
        registry_original = EventRegistry()
        registry_original.register(REG_ALPHA)
        expected = registry_original.compute_fingerprint()

        modified_alpha = ModelEventRegistration(
            event_type="alpha.event",
            topic_template="onex.evt.alpha.event.v2",  # changed
            partition_key_field="alpha_id",
            required_fields=["alpha_id", "payload", "extra"],  # changed
            schema_version="1.0.0",
        )
        registry_modified = EventRegistry()
        registry_modified.register(modified_alpha)

        with pytest.raises(EventRegistryFingerprintMismatchError) as exc_info:
            registry_modified.assert_fingerprint(expected)

        diff = exc_info.value.diff_summary
        assert "~ changed: alpha.event" in diff
        assert "topic_template" in diff
        assert "required_fields" in diff

    def test_mismatch_error_has_expected_and_actual(self) -> None:
        """Mismatch error includes both expected and actual fingerprints."""
        registry = EventRegistry()
        registry.register(REG_ALPHA)
        expected = registry.compute_fingerprint()

        registry2 = EventRegistry()
        registry2.register(REG_BETA)

        with pytest.raises(EventRegistryFingerprintMismatchError) as exc_info:
            registry2.assert_fingerprint(expected)

        assert exc_info.value.expected_fingerprint == expected.fingerprint_sha256
        assert len(exc_info.value.actual_fingerprint) == 64

    def test_empty_expected_vs_populated_raises(self) -> None:
        """Empty expected vs populated registry raises mismatch."""
        empty_registry = EventRegistry()
        expected = empty_registry.compute_fingerprint()

        populated = EventRegistry()
        populated.register(REG_ALPHA)

        with pytest.raises(EventRegistryFingerprintMismatchError):
            populated.assert_fingerprint(expected)


# ---------------------------------------------------------------------------
# TestComputeRegistryDiff
# ---------------------------------------------------------------------------


class TestComputeRegistryDiff:
    """Tests for _compute_registry_diff()."""

    def _make_fingerprint(
        self, registrations: list[ModelEventRegistration]
    ) -> ModelEventRegistryFingerprint:
        """Helper to build a fingerprint from registrations."""
        registry = EventRegistry()
        registry.register_batch(registrations)
        return registry.compute_fingerprint()

    def test_no_diff_when_identical(self) -> None:
        """Identical fingerprints produce empty diff."""
        fp = self._make_fingerprint([REG_ALPHA])
        diff = _compute_registry_diff(fp, fp)
        assert diff == ""

    def test_addition_shown(self) -> None:
        """Added event type is shown with + prefix."""
        expected = self._make_fingerprint([REG_ALPHA])
        actual = self._make_fingerprint([REG_ALPHA, REG_BETA])
        diff = _compute_registry_diff(expected, actual)
        assert "+ added: beta.event" in diff

    def test_removal_shown(self) -> None:
        """Removed event type is shown with - prefix."""
        expected = self._make_fingerprint([REG_ALPHA, REG_BETA])
        actual = self._make_fingerprint([REG_ALPHA])
        diff = _compute_registry_diff(expected, actual)
        assert "- removed: beta.event" in diff

    def test_modification_shows_changed_fields(self) -> None:
        """Modified event type shows which fields changed."""
        expected = self._make_fingerprint([REG_ALPHA])
        modified = ModelEventRegistration(
            event_type="alpha.event",
            topic_template="onex.evt.alpha.event.v2",  # changed
            partition_key_field="alpha_id",
            required_fields=["alpha_id", "payload"],
            schema_version="1.0.0",
        )
        actual = self._make_fingerprint([modified])
        diff = _compute_registry_diff(expected, actual)
        assert "~ changed: alpha.event (topic_template)" in diff

    def test_multiple_field_changes(self) -> None:
        """Multiple changed fields are listed together."""
        expected = self._make_fingerprint([REG_ALPHA])
        modified = ModelEventRegistration(
            event_type="alpha.event",
            topic_template="onex.evt.alpha.event.v2",
            partition_key_field="new_key",
            required_fields=["new_field"],
            schema_version="3.0.0",
        )
        actual = self._make_fingerprint([modified])
        diff = _compute_registry_diff(expected, actual)
        assert "topic_template" in diff
        assert "schema_version" in diff
        assert "partition_key_field" in diff
        assert "required_fields" in diff

    def test_bounded_to_10_lines(self) -> None:
        """Diff is truncated to 10 lines with overflow indicator."""
        # Create 12 different event types that will all show as added
        expected_fp = self._make_fingerprint([])
        regs = [
            ModelEventRegistration(
                event_type=f"event.type.{i:02d}",
                topic_template=f"onex.evt.type.{i:02d}.v1",
            )
            for i in range(12)
        ]
        actual_fp = self._make_fingerprint(regs)
        diff = _compute_registry_diff(expected_fp, actual_fp)
        lines = diff.split("\n")
        assert len(lines) == 10
        assert "... and" in lines[-1]

    def test_exactly_10_lines_no_truncation(self) -> None:
        """Exactly 10 diff lines are shown without truncation."""
        expected_fp = self._make_fingerprint([])
        regs = [
            ModelEventRegistration(
                event_type=f"event.type.{i:02d}",
                topic_template=f"onex.evt.type.{i:02d}.v1",
            )
            for i in range(10)
        ]
        actual_fp = self._make_fingerprint(regs)
        diff = _compute_registry_diff(expected_fp, actual_fp)
        lines = diff.split("\n")
        assert len(lines) == 10
        assert "... and" not in lines[-1]


# ---------------------------------------------------------------------------
# TestModelEventRegistryFingerprint
# ---------------------------------------------------------------------------


class TestModelEventRegistryFingerprint:
    """Tests for ModelEventRegistryFingerprint Pydantic model."""

    def test_json_round_trip(self, tmp_path: Path) -> None:
        """Write to JSON and read back produces identical model."""
        registry = EventRegistry()
        registry.register_batch([REG_ALPHA, REG_BETA])
        original = registry.compute_fingerprint()

        artifact_path = tmp_path / "fingerprint.json"
        original.to_json(artifact_path)

        loaded = ModelEventRegistryFingerprint.from_json_path(artifact_path)
        assert loaded.fingerprint_sha256 == original.fingerprint_sha256
        assert loaded.version == original.version
        assert len(loaded.elements) == len(original.elements)
        for orig, load in zip(original.elements, loaded.elements, strict=True):
            assert orig.event_type == load.event_type
            assert orig.element_sha256 == load.element_sha256
            assert orig.required_fields == load.required_fields

    def test_to_json_writes_valid_json(self, tmp_path: Path) -> None:
        """to_json writes valid parseable JSON."""
        registry = EventRegistry()
        registry.register(REG_ALPHA)
        fp = registry.compute_fingerprint()

        path = tmp_path / "fp.json"
        fp.to_json(path)

        raw = path.read_text(encoding="utf-8")
        data = json.loads(raw)
        assert "fingerprint_sha256" in data
        assert "elements" in data
        assert isinstance(data["elements"], list)

    def test_from_json_path_file_not_found(self, tmp_path: Path) -> None:
        """from_json_path raises FileNotFoundError for missing files."""
        with pytest.raises(FileNotFoundError):
            ModelEventRegistryFingerprint.from_json_path(tmp_path / "nonexistent.json")

    def test_frozen_model(self) -> None:
        """Model is frozen (immutable)."""
        registry = EventRegistry()
        registry.register(REG_ALPHA)
        fp = registry.compute_fingerprint()

        with pytest.raises(Exception):
            fp.fingerprint_sha256 = "modified"  # type: ignore[misc]

    def test_extra_fields_forbidden(self) -> None:
        """Extra fields are rejected."""
        with pytest.raises(Exception):
            ModelEventRegistryFingerprintElement(
                event_type="test",
                topic_template="onex.evt.test.v1",
                schema_version="1.0.0",
                partition_key_field="",
                required_fields=(),
                element_sha256="a" * 64,
                extra_field="bad",  # type: ignore[call-arg]
            )

    def test_default_version(self) -> None:
        """Default version is 1."""
        registry = EventRegistry()
        fp = registry.compute_fingerprint()
        assert fp.version == 1


# ---------------------------------------------------------------------------
# TestCli
# ---------------------------------------------------------------------------


class TestCli:
    """Tests for CLI stamp/verify subcommands."""

    def test_stamp_writes_valid_json(self, tmp_path: Path) -> None:
        """stamp writes a valid JSON artifact."""
        artifact = tmp_path / "fingerprint.json"
        _cli_stamp(str(artifact))

        assert artifact.exists()
        data = json.loads(artifact.read_text(encoding="utf-8"))
        assert "fingerprint_sha256" in data
        assert "elements" in data
        assert data["version"] == 1

    def test_stamp_dry_run_does_not_write(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """stamp --dry-run does not write the artifact file."""
        artifact = tmp_path / "fingerprint.json"
        _cli_stamp(str(artifact), dry_run=True)

        assert not artifact.exists()
        captured = capsys.readouterr()
        assert "--dry-run" in captured.out

    def test_stamp_prints_fingerprint(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """stamp prints the fingerprint hash."""
        artifact = tmp_path / "fingerprint.json"
        _cli_stamp(str(artifact))

        captured = capsys.readouterr()
        assert "fingerprint:" in captured.out
        assert "registrations:" in captured.out

    def test_verify_succeeds_with_matching_artifact(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """verify succeeds when artifact matches live registrations."""
        artifact = tmp_path / "fingerprint.json"
        # First stamp
        _cli_stamp(str(artifact))
        # Then verify
        _cli_verify(str(artifact))

        captured = capsys.readouterr()
        assert "OK" in captured.out

    def test_verify_fails_with_missing_artifact(self, tmp_path: Path) -> None:
        """verify raises when artifact file does not exist."""
        artifact = tmp_path / "nonexistent.json"
        with pytest.raises(EventRegistryFingerprintMissingError):
            _cli_verify(str(artifact))

    def test_verify_fails_with_mismatched_artifact(self, tmp_path: Path) -> None:
        """verify raises when artifact does not match live registry."""
        artifact = tmp_path / "fingerprint.json"

        # Write a fingerprint with different registrations
        fake_reg = ModelEventRegistration(
            event_type="fake.event",
            topic_template="onex.evt.fake.event.v1",
        )
        fake_registry = EventRegistry()
        fake_registry.register(fake_reg)
        fake_fp = fake_registry.compute_fingerprint()
        fake_fp.to_json(artifact)

        # Verify against live registry (which has phase.metrics, not fake.event)
        with pytest.raises(EventRegistryFingerprintMismatchError):
            _cli_verify(str(artifact))


class TestMainCli:
    """Tests for _main() argument parsing and dispatch."""

    def test_no_args_exits_1(self) -> None:
        """No subcommand prints help and exits 1."""
        with (
            patch("sys.argv", ["prog"]),
            pytest.raises(SystemExit) as exc_info,
        ):
            _main()
        assert exc_info.value.code == 1

    def test_stamp_dispatches_to_cli_stamp(self, tmp_path: Path) -> None:
        """'stamp' subcommand dispatches to _cli_stamp."""
        artifact = str(tmp_path / "fp.json")
        with (
            patch("sys.argv", ["prog", "stamp", "--artifact", artifact]),
            patch(
                "omnibase_infra.runtime.emit_daemon.event_registry._cli_stamp",
            ) as mock_stamp,
        ):
            _main()
        mock_stamp.assert_called_once_with(artifact, dry_run=False)

    def test_stamp_dry_run_flag(self, tmp_path: Path) -> None:
        """'stamp --dry-run' passes dry_run=True."""
        artifact = str(tmp_path / "fp.json")
        with (
            patch("sys.argv", ["prog", "stamp", "--dry-run", "--artifact", artifact]),
            patch(
                "omnibase_infra.runtime.emit_daemon.event_registry._cli_stamp",
            ) as mock_stamp,
        ):
            _main()
        mock_stamp.assert_called_once_with(artifact, dry_run=True)

    def test_verify_dispatches_to_cli_verify(self, tmp_path: Path) -> None:
        """'verify' subcommand dispatches to _cli_verify."""
        artifact = str(tmp_path / "fp.json")
        with (
            patch("sys.argv", ["prog", "verify", "--artifact", artifact]),
            patch(
                "omnibase_infra.runtime.emit_daemon.event_registry._cli_verify",
            ) as mock_verify,
        ):
            _main()
        mock_verify.assert_called_once_with(artifact)

    def test_mismatch_error_exits_2(self, tmp_path: Path) -> None:
        """EventRegistryFingerprintMismatchError causes exit code 2."""
        artifact = str(tmp_path / "fp.json")
        with (
            patch("sys.argv", ["prog", "verify", "--artifact", artifact]),
            patch(
                "omnibase_infra.runtime.emit_daemon.event_registry._cli_verify",
                side_effect=EventRegistryFingerprintMismatchError(
                    "mismatch",
                    expected_fingerprint="aaa",
                    actual_fingerprint="bbb",
                ),
            ),
            pytest.raises(SystemExit) as exc_info,
        ):
            _main()
        assert exc_info.value.code == 2

    def test_missing_error_exits_2(self, tmp_path: Path) -> None:
        """EventRegistryFingerprintMissingError causes exit code 2."""
        artifact = str(tmp_path / "fp.json")
        with (
            patch("sys.argv", ["prog", "verify", "--artifact", artifact]),
            patch(
                "omnibase_infra.runtime.emit_daemon.event_registry._cli_verify",
                side_effect=EventRegistryFingerprintMissingError(
                    "missing",
                    artifact_path="/fake/path",
                ),
            ),
            pytest.raises(SystemExit) as exc_info,
        ):
            _main()
        assert exc_info.value.code == 2

    def test_generic_exception_exits_1(self, tmp_path: Path) -> None:
        """Unexpected exceptions cause exit code 1."""
        artifact = str(tmp_path / "fp.json")
        with (
            patch("sys.argv", ["prog", "stamp", "--artifact", artifact]),
            patch(
                "omnibase_infra.runtime.emit_daemon.event_registry._cli_stamp",
                side_effect=RuntimeError("boom"),
            ),
            pytest.raises(SystemExit) as exc_info,
        ):
            _main()
        assert exc_info.value.code == 1

    def test_default_artifact_path(self) -> None:
        """Default artifact path constant points to co-located JSON file."""
        assert _ARTIFACT_DEFAULT_PATH.endswith("event_registry_fingerprint.json")
        assert Path(_ARTIFACT_DEFAULT_PATH).is_absolute()
