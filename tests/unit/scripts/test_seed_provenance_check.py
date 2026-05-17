# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Unit tests for scripts/check_seed_provenance.py [OMN-11208]."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT_PATH = (
    Path(__file__).resolve().parents[3] / "scripts" / "check_seed_provenance.py"
)

# Import the check functions directly for unit testing
sys.path.insert(0, str(SCRIPT_PATH.parent))
from check_seed_provenance import (
    _has_provenance,
    _is_seed_or_demo,
    _publishes_events,
    check_scripts,
)


@pytest.mark.unit
class TestHelperFunctions:
    def test_is_seed_or_demo_matches_seed(self) -> None:
        assert _is_seed_or_demo(Path("seed-infisical.py"))

    def test_is_seed_or_demo_matches_demo(self) -> None:
        assert _is_seed_or_demo(Path("demo_runtime_verification.py"))

    def test_is_seed_or_demo_rejects_other(self) -> None:
        assert not _is_seed_or_demo(Path("check_topic_drift.py"))

    def test_publishes_events_detects_kafka_producer(self) -> None:
        assert _publishes_events("producer = AIOKafkaProducer(bootstrap_servers=...)")

    def test_publishes_events_detects_send_and_wait(self) -> None:
        assert _publishes_events("await producer.send_and_wait(topic, body)")

    def test_publishes_events_detects_emit(self) -> None:
        assert _publishes_events("event_bus.emit(envelope)")

    def test_publishes_events_false_for_docstring_only(self) -> None:
        # "produces" (with s) does not match \bproduce\b
        assert not _publishes_events("re-running against a correct realm produces all")

    def test_has_provenance_true(self) -> None:
        assert _has_provenance('payload["data_provenance"] = "demo_seeded"')

    def test_has_provenance_false(self) -> None:
        assert not _has_provenance("event_type = 'baselines.computed'")


@pytest.mark.unit
class TestCheckScripts:
    def test_no_warnings_when_provenance_present(self, tmp_path: Path) -> None:
        script = tmp_path / "seed_example.py"
        script.write_text(
            "async def run():\n"
            '    payload = {"data_provenance": "demo_seeded"}\n'
            '    await producer.send_and_wait("topic", json.dumps(payload).encode())\n'
        )
        warnings = check_scripts(tmp_path)
        assert warnings == []

    def test_warning_when_provenance_missing(self, tmp_path: Path) -> None:
        script = tmp_path / "seed_example.py"
        script.write_text(
            "async def run():\n"
            '    payload = {"event_type": "foo"}\n'
            '    await producer.send_and_wait("topic", json.dumps(payload).encode())\n'
        )
        warnings = check_scripts(tmp_path)
        assert len(warnings) == 1
        assert "data_provenance" in warnings[0]
        assert "seed_example.py" in warnings[0]

    def test_non_seed_demo_scripts_not_checked(self, tmp_path: Path) -> None:
        script = tmp_path / "publish_pr_merged_event.py"
        script.write_text(
            'async def run():\n    await producer.send_and_wait("topic", b"data")\n'
        )
        warnings = check_scripts(tmp_path)
        assert warnings == []

    def test_seed_script_without_event_publish_not_warned(self, tmp_path: Path) -> None:
        script = tmp_path / "seed_config.py"
        script.write_text(
            'def run():\n    client.create_secret(key="FOO", value="bar")\n'
        )
        warnings = check_scripts(tmp_path)
        assert warnings == []

    def test_multiple_flagged_scripts(self, tmp_path: Path) -> None:
        for name in ("seed_a.py", "demo_b.py"):
            (tmp_path / name).write_text(
                'await producer.send_and_wait("topic", b"payload")\n'
            )
        warnings = check_scripts(tmp_path)
        assert len(warnings) == 2


@pytest.mark.unit
class TestCLI:
    def test_exits_zero_always(self) -> None:
        result = subprocess.run(
            [sys.executable, str(SCRIPT_PATH)],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        assert result.returncode == 0

    def test_output_contains_advisory_label(self) -> None:
        result = subprocess.run(
            [sys.executable, str(SCRIPT_PATH)],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        assert "Seed Provenance" in result.stdout
