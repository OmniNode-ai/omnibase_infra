# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Unit tests for onex-linear-relay CLI.

Tests cover:
    - Valid snapshot file emit
    - Kafka-unavailable spool path (mocked)
    - Invalid snapshot file handling
    - Event payload structure

Related Tickets:
    - OMN-2656: Phase 2 — Effect Nodes & CLIs (omnibase_infra)
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from click.testing import CliRunner

from omnibase_infra.cli.linear_relay import (
    _build_event,
    cli,
)


class TestBuildEvent:
    @pytest.mark.unit
    def test_event_structure(self) -> None:
        snapshot = {"workstreams": ["ws-a", "ws-b"], "issues": []}
        event = _build_event(snapshot, "test-uuid-1234")
        assert event["event_type"] == "onex.evt.linear.snapshot.v1"
        assert event["snapshot_id"] == "test-uuid-1234"
        assert event["workstreams"] == ["ws-a", "ws-b"]
        assert event["snapshot"] == snapshot
        assert "emitted_at" in event

    @pytest.mark.unit
    def test_event_empty_workstreams(self) -> None:
        snapshot: dict[str, object] = {"issues": []}
        event = _build_event(snapshot, "uuid-abc")
        assert event["workstreams"] == []

    @pytest.mark.unit
    def test_event_non_list_workstreams_normalized(self) -> None:
        """If workstreams is not a list, it is normalized to []."""
        snapshot: dict[str, object] = {"workstreams": "single"}
        event = _build_event(snapshot, "uuid-xyz")
        assert event["workstreams"] == []


class TestEmitCommandSuccess:
    @pytest.mark.unit
    def test_emit_valid_snapshot(self, tmp_path: Path) -> None:
        snapshot_file = tmp_path / "snapshot.json"
        snapshot_data = {"workstreams": ["ws-1"], "issues": [{"id": "OMN-1"}]}
        snapshot_file.write_text(json.dumps(snapshot_data), encoding="utf-8")

        runner = CliRunner()
        with patch(
            "omnibase_infra.cli.linear_relay._publish_event",
            new=AsyncMock(return_value=True),
        ):
            result = runner.invoke(
                cli,
                ["emit", "--snapshot-file", str(snapshot_file)],
                catch_exceptions=False,
            )
        assert result.exit_code == 0
        assert "Published Linear snapshot" in result.output

    @pytest.mark.unit
    def test_emit_with_explicit_snapshot_id(self, tmp_path: Path) -> None:
        snapshot_file = tmp_path / "snapshot.json"
        snapshot_file.write_text('{"workstreams": []}', encoding="utf-8")

        runner = CliRunner()
        with patch(
            "omnibase_infra.cli.linear_relay._publish_event",
            new=AsyncMock(return_value=True),
        ):
            result = runner.invoke(
                cli,
                [
                    "emit",
                    "--snapshot-file",
                    str(snapshot_file),
                    "--snapshot-id",
                    "my-custom-uuid",
                ],
                catch_exceptions=False,
            )
        assert result.exit_code == 0
        assert "my-custom-uuid" in result.output


class TestEmitKafkaUnavailableSpool:
    @pytest.mark.unit
    def test_spools_when_kafka_unavailable(self, tmp_path: Path) -> None:
        """When Kafka publish fails, event is spooled and CLI exits 0."""
        snapshot_file = tmp_path / "snapshot.json"
        snapshot_file.write_text('{"workstreams": ["ws-a"]}', encoding="utf-8")
        spool_file = tmp_path / "spool" / "linear-snapshots.jsonl"

        runner = CliRunner()
        with (
            patch(
                "omnibase_infra.cli.linear_relay._publish_event",
                new=AsyncMock(return_value=False),
            ),
            patch(
                "omnibase_infra.cli.linear_relay._SPOOL_FILE",
                spool_file,
            ),
            patch(
                "omnibase_infra.cli.linear_relay._SPOOL_DIR",
                tmp_path / "spool",
            ),
        ):
            result = runner.invoke(
                cli,
                ["emit", "--snapshot-file", str(snapshot_file)],
                catch_exceptions=False,
            )

        # Always exits 0 — non-blocking
        assert result.exit_code == 0
        # Spool file should contain the event
        assert spool_file.exists()
        lines = spool_file.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 1
        event = json.loads(lines[0])
        assert event["event_type"] == "onex.evt.linear.snapshot.v1"
        assert event["workstreams"] == ["ws-a"]


class TestEmitInvalidInput:
    @pytest.mark.unit
    def test_invalid_json_file(self, tmp_path: Path) -> None:
        bad_file = tmp_path / "bad.json"
        bad_file.write_text("not-json{{", encoding="utf-8")

        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["emit", "--snapshot-file", str(bad_file)],
        )
        assert result.exit_code == 1
        assert "Failed to read" in (result.output + str(result.stderr or ""))

    @pytest.mark.unit
    def test_non_object_json_file(self, tmp_path: Path) -> None:
        array_file = tmp_path / "array.json"
        array_file.write_text('["not", "an", "object"]', encoding="utf-8")

        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["emit", "--snapshot-file", str(array_file)],
        )
        assert result.exit_code == 1
        assert "must contain a JSON object" in (
            result.output + str(result.stderr or "")
        )
