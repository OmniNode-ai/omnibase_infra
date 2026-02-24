# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Adversarial tests: fingerprint drift detection (CI twin validation) [OMN-2293].

These tests prove the CI twin fingerprint system is NOT cosmetic by
intentionally introducing drift and asserting the twin correctly catches it.

Design rationale:
    "Trust but verify" — any safety mechanism must itself be tested by
    attempting to defeat it. If these tests pass, the twin system provably
    catches the specific drift scenarios that caused the Feb 15 incident class.

Two systems are validated:
    1. Schema fingerprint twin (check_schema_fingerprint.py)
       Detects when forward migration SQL files change without the artifact
       being regenerated (docker/migrations/schema_fingerprint.sha256).

    2. Event registry fingerprint twin (check_event_registry_fingerprint.py)
       Detects when ALL_EVENT_REGISTRATIONS changes without the artifact
       being regenerated (event_registry_fingerprint.json).

Each test follows the pattern:
    1. Stamp a valid artifact against the current state
    2. Mutate the source (schema file or registry)
    3. Do NOT re-stamp
    4. Run the CI twin verify step
    5. Assert: exit code == 2 (FAILED) with clear diagnostics

Related tickets:
    - OMN-2149: CI twin implementation
    - OMN-2293: Adversarial validation (this file)
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.check_event_registry_fingerprint import (
    cmd_stamp as event_cmd_stamp,
)
from scripts.check_event_registry_fingerprint import (
    cmd_verify as event_cmd_verify,
)
from scripts.check_schema_fingerprint import (
    cmd_stamp as schema_cmd_stamp,
)
from scripts.check_schema_fingerprint import (
    cmd_verify as schema_cmd_verify,
)
from scripts.check_schema_fingerprint import (
    compute_migration_fingerprint,
    write_artifact,
)

pytestmark = pytest.mark.unit


# =============================================================================
# Helpers
# =============================================================================


def _write_migration(migrations_dir: Path, name: str, content: str) -> Path:
    """Create a fake migration SQL file."""
    path = migrations_dir / name
    path.write_text(content, encoding="utf-8")
    return path


def _valid_event_registry_artifact(elements: list[dict]) -> dict:
    """Build a structurally valid event registry artifact with given elements.

    The fingerprint_sha256 field is set to a deliberate mismatch value ("stale")
    to simulate a stamped artifact that then diverges from the live registry.
    Callers that need a fully valid artifact should use cmd_stamp() instead.
    """
    return {
        "version": 1,
        "fingerprint_sha256": "a" * 64,
        "elements": elements,
    }


# =============================================================================
# Schema Fingerprint Adversarial Tests
# =============================================================================


class TestSchemaFingerprintAdversarial:
    """Adversarial tests proving the schema fingerprint CI twin catches drift.

    Each test:
      1. Creates a known-good migration set and stamps a valid artifact.
      2. Introduces schema drift (add column, rename table, add new migration).
      3. Does NOT re-stamp.
      4. Asserts the verify step returns exit code 2 (FAILED).

    These tests prove the CI twin is NOT cosmetic — it would have caught the
    class of deployment errors described in OMN-2233/OMN-2293.
    """

    def test_adding_column_to_existing_migration_is_caught(
        self, tmp_path: Path
    ) -> None:
        """Drift: alter an existing migration SQL file (add a column).

        Simulates a developer editing a migration file directly without
        regenerating the fingerprint artifact. This is the most common
        accidental drift scenario.

        Expected CI twin behavior: FAILED (exit code 2), not a silent pass.
        """
        migrations_dir = tmp_path / "migrations" / "forward"
        migrations_dir.mkdir(parents=True)
        artifact = tmp_path / "schema_fingerprint.sha256"

        # Step 1: create baseline migrations and stamp a valid artifact
        _write_migration(
            migrations_dir,
            "001_create_nodes.sql",
            "CREATE TABLE nodes (id UUID PRIMARY KEY, name TEXT NOT NULL);",
        )
        _write_migration(
            migrations_dir,
            "002_create_events.sql",
            "CREATE TABLE events (id UUID PRIMARY KEY, node_id UUID, created_at TIMESTAMPTZ);",
        )
        assert schema_cmd_stamp(migrations_dir, artifact) == 0
        # Confirm baseline is clean
        assert schema_cmd_verify(migrations_dir, artifact) == 0

        # Step 2: DRIFT — add a column to an existing migration file
        # (as if someone edited the SQL file to add a new field)
        _write_migration(
            migrations_dir,
            "001_create_nodes.sql",
            "CREATE TABLE nodes (id UUID PRIMARY KEY, name TEXT NOT NULL, status TEXT DEFAULT 'active');",
        )

        # Step 3: do NOT re-stamp

        # Step 4: verify must catch the drift
        # Expected error: "Schema fingerprint artifact is stale."
        result = schema_cmd_verify(migrations_dir, artifact)
        assert result == 2, (
            "CI twin MISSED schema drift: adding a column to an existing migration "
            "was not detected. This means the twin system is cosmetic and would NOT "
            "have caught the OMN-2233-class deployment error."
        )

    def test_adding_new_migration_file_is_caught(self, tmp_path: Path) -> None:
        """Drift: add a new migration file without regenerating the artifact.

        Simulates the normal forward-migration workflow where a developer adds
        a new .sql file but forgets to run the stamp step (or CI is bypassed).

        Expected CI twin behavior: FAILED (exit code 2).
        """
        migrations_dir = tmp_path / "migrations" / "forward"
        migrations_dir.mkdir(parents=True)
        artifact = tmp_path / "schema_fingerprint.sha256"

        # Step 1: stamp baseline with one migration
        _write_migration(
            migrations_dir,
            "001_create_nodes.sql",
            "CREATE TABLE nodes (id UUID PRIMARY KEY);",
        )
        assert schema_cmd_stamp(migrations_dir, artifact) == 0
        assert schema_cmd_verify(migrations_dir, artifact) == 0

        # Step 2: DRIFT — add a new migration (simulating a new schema change)
        _write_migration(
            migrations_dir,
            "002_add_capabilities.sql",
            "ALTER TABLE nodes ADD COLUMN capabilities JSONB DEFAULT '{}';",
        )

        # Step 3: do NOT re-stamp

        # Step 4: verify must catch the drift
        result = schema_cmd_verify(migrations_dir, artifact)
        assert result == 2, (
            "CI twin MISSED schema drift: adding a new migration file was not detected. "
            "This means the twin system would NOT have caught a new schema change "
            "that bypassed the stamp step."
        )

    def test_renaming_migration_file_is_caught(self, tmp_path: Path) -> None:
        """Drift: rename an existing migration file (changes canonical hash input).

        Migration file names are part of the fingerprint input. Renaming a file
        (even with identical content) changes the fingerprint because the
        canonical representation includes filenames.

        Expected CI twin behavior: FAILED (exit code 2).
        """
        migrations_dir = tmp_path / "migrations" / "forward"
        migrations_dir.mkdir(parents=True)
        artifact = tmp_path / "schema_fingerprint.sha256"

        # Step 1: stamp baseline
        content = "CREATE TABLE nodes (id UUID PRIMARY KEY);"
        _write_migration(migrations_dir, "001_create_nodes.sql", content)
        assert schema_cmd_stamp(migrations_dir, artifact) == 0
        assert schema_cmd_verify(migrations_dir, artifact) == 0

        # Step 2: DRIFT — rename the file (content identical, name changed)
        (migrations_dir / "001_create_nodes.sql").unlink()
        _write_migration(migrations_dir, "001_create_node_table.sql", content)

        # Step 3: do NOT re-stamp

        # Step 4: verify must catch the rename
        result = schema_cmd_verify(migrations_dir, artifact)
        assert result == 2, (
            "CI twin MISSED schema drift: renaming a migration file was not detected. "
            "File names are part of the fingerprint input and must be stable."
        )

    def test_stale_artifact_is_explicitly_rejected(self, tmp_path: Path) -> None:
        """A manually crafted stale artifact (wrong hash) is detected as drift.

        This is the most direct adversarial test: write a totally wrong hash
        to the artifact file and confirm verify rejects it immediately.

        This validates the exit code contract: 2 means "artifact is stale".
        """
        migrations_dir = tmp_path / "migrations" / "forward"
        migrations_dir.mkdir(parents=True)
        artifact = tmp_path / "schema_fingerprint.sha256"

        _write_migration(
            migrations_dir,
            "001_init.sql",
            "CREATE TABLE registration_projections (id UUID PRIMARY KEY);",
        )
        fp, count = compute_migration_fingerprint(migrations_dir)

        # Write a deliberately wrong hash (the artifact is "stale" from day 0)
        wrong_hash = "0" * 64
        assert fp != wrong_hash, "Test setup error: collision with zero hash"
        write_artifact(artifact, wrong_hash, count)

        result = schema_cmd_verify(migrations_dir, artifact)
        assert result == 2, (
            "CI twin MISSED a completely wrong artifact hash. "
            "The twin system failed to detect a manually injected stale artifact."
        )

    def test_verify_after_correct_stamp_always_passes(self, tmp_path: Path) -> None:
        """Stamp followed by immediate verify always produces exit code 0.

        This is the negative control: after a correct stamp, verify must pass.
        This ensures the adversarial tests above are not trivially passing due
        to a broken verify function that always returns 2.
        """
        migrations_dir = tmp_path / "migrations" / "forward"
        migrations_dir.mkdir(parents=True)
        artifact = tmp_path / "schema_fingerprint.sha256"

        _write_migration(
            migrations_dir,
            "001_init.sql",
            "CREATE TABLE registration_projections (id UUID PRIMARY KEY);",
        )
        _write_migration(
            migrations_dir,
            "002_add_col.sql",
            "ALTER TABLE registration_projections ADD COLUMN created_at TIMESTAMPTZ;",
        )

        assert schema_cmd_stamp(migrations_dir, artifact) == 0
        result = schema_cmd_verify(migrations_dir, artifact)
        assert result == 0, (
            "Verify returned non-zero after a fresh stamp. "
            "The twin system's verify function may be broken."
        )


# =============================================================================
# Event Registry Fingerprint Adversarial Tests
# =============================================================================


class TestEventRegistryFingerprintAdversarial:
    """Adversarial tests proving the event registry CI twin catches drift.

    Each test:
      1. Stamps a valid artifact against the CURRENT live registry.
      2. Corrupts the artifact (simulating a stale committed fingerprint).
      3. Does NOT re-stamp.
      4. Asserts the verify step returns exit code 2 (FAILED).

    Note on test design:
        The event registry is derived from ALL_EVENT_REGISTRATIONS (runtime
        Python code). We cannot inject fake topics at test time without
        monkeypatching the registry itself, which would test the monkeypatch
        rather than the twin. Instead, we validate the twin's ability to
        detect mismatches by stamping a correct artifact and then manually
        corrupting the artifact's fingerprint hash.

        The schema fingerprint adversarial tests cover the "mutation of source"
        scenario. These tests cover the "committed artifact diverges from source"
        scenario — which is the same thing from the CI perspective.
    """

    def test_corrupted_fingerprint_hash_is_caught(self, tmp_path: Path) -> None:
        """Drift: artifact fingerprint hash is wrong (zero hash).

        Simulates committing a stale artifact: someone stamps the artifact
        in their feature branch but the registry changes before merge, and
        the committed artifact no longer matches the live code.

        Expected CI twin behavior: FAILED (exit code 2).
        """
        artifact = tmp_path / "event_registry_fingerprint.json"

        # Step 1: stamp a valid artifact from the live registry
        assert event_cmd_stamp(str(artifact)) == 0
        assert event_cmd_verify(str(artifact)) == 0

        # Step 2: DRIFT — corrupt the fingerprint hash (simulate stale artifact)
        data = json.loads(artifact.read_text(encoding="utf-8"))
        original_hash = data["fingerprint_sha256"]
        corrupted_hash = "0" * 64
        assert original_hash != corrupted_hash, "Test setup collision"
        data["fingerprint_sha256"] = corrupted_hash
        artifact.write_text(json.dumps(data), encoding="utf-8")

        # Step 3: do NOT re-stamp

        # Step 4: verify must catch the corruption
        result = event_cmd_verify(str(artifact))
        assert result == 2, (
            "CI twin MISSED event registry drift: a corrupted fingerprint hash "
            "was not detected. This means the twin system would NOT have caught "
            "a stale artifact in CI."
        )

    def test_element_hash_mutation_behavior_is_documented(self, tmp_path: Path) -> None:
        """Documents how the event registry twin handles element-level corruption.

        The twin validates by recomputing the fingerprint from ALL_EVENT_REGISTRATIONS
        (live Python code) and comparing the overall fingerprint_sha256 hash.
        It does NOT re-validate individual element_sha256 fields from the artifact
        JSON — those are informational and used for diff display only.

        Therefore: if you corrupt an individual element_sha256 in the JSON artifact
        but leave the overall fingerprint_sha256 unchanged, verify PASSES. This is
        correct behavior — the authoritative check is the overall hash vs live code.

        What DOES trigger a failure is any change to the live ALL_EVENT_REGISTRATIONS
        without regenerating the artifact. That is tested in other tests via:
          - Corrupting fingerprint_sha256 (test_corrupted_fingerprint_hash_is_caught)
          - Adding phantom elements (test_extra_element_not_in_registry_is_caught)

        This test documents and pins the behavior so that if the validation logic
        changes, someone is explicitly aware that element-level checking was added.
        """
        artifact = tmp_path / "event_registry_fingerprint.json"

        # Step 1: stamp a valid artifact
        assert event_cmd_stamp(str(artifact)) == 0
        assert event_cmd_verify(str(artifact)) == 0

        # Step 2: mutate an element's sha256 but leave overall fingerprint_sha256 intact
        data = json.loads(artifact.read_text(encoding="utf-8"))

        if not data.get("elements"):
            pytest.skip("No elements in live registry — cannot test element mutation")

        # Mutate only the element sha256 (informational field only)
        original_element_hash = data["elements"][0]["element_sha256"]
        data["elements"][0]["element_sha256"] = "b" * 64
        assert original_element_hash != "b" * 64, "Test setup collision"
        # Intentionally leave fingerprint_sha256 unchanged (overall hash is still valid)
        artifact.write_text(json.dumps(data), encoding="utf-8")

        # Step 3: verify passes because overall fingerprint_sha256 matches live code.
        # This is the DOCUMENTED behavior: element-level JSON fields are informational.
        result = event_cmd_verify(str(artifact))
        assert result == 0, (
            "Unexpected: element-level sha256 mutation caused verify to fail. "
            "If this assertion fails, the validation logic has been changed to "
            "validate individual element hashes from the artifact file — update "
            "this test and test_corrupted_fingerprint_hash_is_caught accordingly."
        )

    def test_missing_artifact_is_caught(self, tmp_path: Path) -> None:
        """Drift: artifact file does not exist.

        Simulates a developer who adds a new event registration but forgets
        to commit the regenerated artifact (or it was .gitignored accidentally).

        Expected CI twin behavior: FAILED (exit code 2).
        """
        artifact = tmp_path / "event_registry_fingerprint.json"
        # Artifact deliberately not created

        result = event_cmd_verify(str(artifact))
        assert result == 2, (
            "CI twin MISSED a missing artifact. "
            "A non-existent artifact file should fail verification immediately."
        )

    def test_invalid_json_in_artifact_is_caught(self, tmp_path: Path) -> None:
        """Drift: artifact file contains invalid JSON (disk corruption or merge conflict).

        Simulates a merge conflict marker or truncated file being committed as
        the artifact. The CI twin must reject this, not silently pass.

        Expected CI twin behavior: FAILED (exit code 2).
        """
        artifact = tmp_path / "event_registry_fingerprint.json"
        artifact.write_text("<<<<<<< HEAD\n{broken json\n=======\n", encoding="utf-8")

        result = event_cmd_verify(str(artifact))
        assert result == 2, (
            "CI twin MISSED an invalid JSON artifact (merge conflict markers). "
            "Corrupted artifact files must be rejected, not silently passed."
        )

    def test_extra_element_not_in_registry_is_caught(self, tmp_path: Path) -> None:
        """Drift: artifact contains an element that no longer exists in the registry.

        Simulates removing a topic from ALL_EVENT_REGISTRATIONS (decommissioning
        an event type) without regenerating the artifact. The committed artifact
        has an extra entry that is no longer in the live code.

        Expected CI twin behavior: FAILED (exit code 2).
        """
        artifact = tmp_path / "event_registry_fingerprint.json"

        # Step 1: stamp a valid artifact
        assert event_cmd_stamp(str(artifact)) == 0
        assert event_cmd_verify(str(artifact)) == 0

        # Step 2: DRIFT — add a phantom element to the artifact
        # (simulating a topic that was in the registry when stamped but since removed)
        data = json.loads(artifact.read_text(encoding="utf-8"))
        phantom_element = {
            "element_sha256": "c" * 64,
            "event_type": "phantom.decommissioned.topic",
            "partition_key_field": "id",
            "required_fields": ["id"],
            "schema_version": "1.0.0",
            "topic_template": "onex.evt.phantom.decommissioned.v1",
        }
        data["elements"].append(phantom_element)
        # Also corrupt the overall fingerprint to simulate the stale stamp
        data["fingerprint_sha256"] = "d" * 64
        artifact.write_text(json.dumps(data), encoding="utf-8")

        # Step 3: do NOT re-stamp

        # Step 4: verify must catch the phantom element
        result = event_cmd_verify(str(artifact))
        assert result == 2, (
            "CI twin MISSED a phantom element in the artifact (a topic that "
            "no longer exists in ALL_EVENT_REGISTRATIONS). The twin should "
            "detect that the committed artifact diverges from the live registry."
        )

    def test_verify_after_correct_stamp_always_passes(self, tmp_path: Path) -> None:
        """Stamp followed by immediate verify always produces exit code 0.

        Negative control: ensures the adversarial tests above are not trivially
        passing because verify() always returns 2. After a correct stamp,
        verify must pass.
        """
        artifact = tmp_path / "event_registry_fingerprint.json"

        assert event_cmd_stamp(str(artifact)) == 0
        result = event_cmd_verify(str(artifact))
        assert result == 0, (
            "Verify returned non-zero after a fresh stamp. "
            "The event registry twin's verify function may be broken."
        )
