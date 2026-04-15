# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Unit tests for scripts/trigger_rebuild_on_merge.py [OMN-8917].

Tests assert path-based and label-based trigger logic with mocked Kafka publish.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

SCRIPT_PATH = (
    Path(__file__).resolve().parents[3] / "scripts" / "trigger_rebuild_on_merge.py"
)


def _import_trigger_module():
    """Import the trigger module for unit-testing logic functions directly."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "trigger_rebuild_on_merge", SCRIPT_PATH
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


@pytest.mark.unit
class TestRebuildTriggerLogic:
    """Unit tests for should_trigger() path/label detection logic."""

    def setup_method(self) -> None:
        self.mod = _import_trigger_module()

    def test_runtime_change_label_triggers(self) -> None:
        """runtime_change label alone should trigger rebuild."""
        assert self.mod.should_trigger(
            changed_files=[],
            labels=["runtime_change"],
        )

    def test_omnimarket_src_path_triggers(self) -> None:
        """Changed file under src/omnimarket/ should trigger rebuild."""
        assert self.mod.should_trigger(
            changed_files=["src/omnimarket/nodes/foo/handler.py"],
            labels=[],
        )

    def test_omnibase_infra_nodes_path_triggers(self) -> None:
        """Changed file under src/omnibase_infra/nodes/ should trigger rebuild."""
        assert self.mod.should_trigger(
            changed_files=["src/omnibase_infra/nodes/node_foo/contract.yaml"],
            labels=[],
        )

    def test_non_runtime_path_does_not_trigger(self) -> None:
        """Changed file outside runtime paths should not trigger rebuild."""
        assert not self.mod.should_trigger(
            changed_files=["docs/plans/some-plan.md", "tests/unit/test_foo.py"],
            labels=[],
        )

    def test_mixed_paths_one_match_triggers(self) -> None:
        """Any single matching file among many should trigger rebuild."""
        assert self.mod.should_trigger(
            changed_files=[
                "README.md",
                "src/omnimarket/nodes/bar/node.py",
                "pyproject.toml",
            ],
            labels=[],
        )

    def test_empty_inputs_no_trigger(self) -> None:
        """No files and no labels should not trigger."""
        assert not self.mod.should_trigger(changed_files=[], labels=[])

    def test_unrelated_label_does_not_trigger(self) -> None:
        """Labels other than runtime_change should not trigger."""
        assert not self.mod.should_trigger(
            changed_files=[],
            labels=["bug", "documentation"],
        )

    def test_multiple_labels_with_runtime_change_triggers(self) -> None:
        """runtime_change among other labels should trigger."""
        assert self.mod.should_trigger(
            changed_files=[],
            labels=["bug", "runtime_change", "enhancement"],
        )


@pytest.mark.unit
class TestRebuildTriggerPublish:
    """Unit tests for publish_rebuild_event() Kafka call shape."""

    def setup_method(self) -> None:
        self.mod = _import_trigger_module()

    def test_publish_calls_producer_with_correct_topic(self) -> None:
        """publish_rebuild_event should produce to onex.cmd.deploy.rebuild-requested.v1."""
        mock_producer = MagicMock()
        mock_producer.flush.return_value = None

        with patch.dict(
            "os.environ",
            {
                "KAFKA_BOOTSTRAP_SERVERS": "broker:9092",
                "KAFKA_SASL_USERNAME": "user",
                "KAFKA_SASL_PASSWORD": "pass",
                "DEPLOY_AGENT_HMAC_SECRET": "testsecret",
            },
        ):
            with patch("confluent_kafka.Producer", return_value=mock_producer):
                self.mod.publish_rebuild_event(
                    bootstrap_servers="broker:9092",
                    username="user",
                    password="pass",
                    hmac_secret="testsecret",
                    git_ref="origin/main",
                    correlation_id="test-corr-id",
                    requested_by="gha-trigger",
                )

        mock_producer.produce.assert_called_once()
        call_kwargs = mock_producer.produce.call_args
        assert call_kwargs.kwargs["topic"] == "onex.cmd.deploy.rebuild-requested.v1"

    def test_publish_event_payload_shape(self) -> None:
        """Published payload must include scope=runtime and git_ref."""
        import json

        mock_producer = MagicMock()
        captured_value: list[bytes] = []

        def fake_produce(topic, key, value, on_delivery):
            captured_value.append(value)

        mock_producer.produce.side_effect = fake_produce
        mock_producer.flush.return_value = None

        with patch("confluent_kafka.Producer", return_value=mock_producer):
            self.mod.publish_rebuild_event(
                bootstrap_servers="broker:9092",
                username="user",
                password="pass",
                hmac_secret="testsecret",
                git_ref="origin/main",
                correlation_id="test-corr-id",
                requested_by="gha-trigger",
            )

        assert captured_value, "produce was not called"
        payload = json.loads(captured_value[0])
        assert payload["scope"] == "runtime"
        assert payload["git_ref"] == "origin/main"
        assert "correlation_id" in payload
        assert "_signature" in payload


@pytest.mark.unit
class TestRebuildTriggerCLI:
    """CLI integration tests using --dry-run flag."""

    def test_dry_run_no_trigger_exits_zero(self) -> None:
        """--dry-run with no matching files or labels should exit 0 without publishing."""
        result = subprocess.run(
            [
                sys.executable,
                str(SCRIPT_PATH),
                "--changed-files",
                "README.md,docs/plans/foo.md",
                "--labels",
                "",
                "--git-ref",
                "origin/main",
                "--dry-run",
            ],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"
        assert "no rebuild trigger" in result.stdout.lower()

    def test_dry_run_with_runtime_change_label(self) -> None:
        """--dry-run with runtime_change label should report trigger without publishing."""
        result = subprocess.run(
            [
                sys.executable,
                str(SCRIPT_PATH),
                "--changed-files",
                "",
                "--labels",
                "runtime_change",
                "--git-ref",
                "origin/main",
                "--dry-run",
            ],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"
        assert (
            "rebuild triggered" in result.stdout.lower()
            or "dry-run" in result.stdout.lower()
        )

    def test_dry_run_with_omnimarket_path(self) -> None:
        """--dry-run with omnimarket src path should report trigger."""
        result = subprocess.run(
            [
                sys.executable,
                str(SCRIPT_PATH),
                "--changed-files",
                "src/omnimarket/nodes/foo/handler.py",
                "--labels",
                "",
                "--git-ref",
                "origin/main",
                "--dry-run",
            ],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"
        assert (
            "rebuild triggered" in result.stdout.lower()
            or "dry-run" in result.stdout.lower()
        )
