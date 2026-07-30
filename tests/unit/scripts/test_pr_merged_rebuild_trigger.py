# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Unit tests for scripts/trigger_rebuild_on_merge.py [OMN-8917, OMN-12573].

Tests assert path-based and label-based trigger logic with mocked Kafka publish.
OMN-12573 re-points the script to publish the node_redeploy start command
(onex.cmd.omnimarket.redeploy-start.v1) carrying the triggering lane + ref,
instead of the deploy-agent rebuild command with a hardcoded origin/main.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_PATH = REPO_ROOT / "scripts" / "trigger_rebuild_on_merge.py"
WORKFLOW_PATH = REPO_ROOT / ".github" / "workflows" / "runtime-rebuild-trigger.yml"


def _script_env() -> dict[str, str]:
    """Subprocess env with hostile ambient ``PYTHONPATH`` replaced (OMN-14744).

    The ``TestRedeployStartCLI`` cases spawn ``trigger_rebuild_on_merge.py`` via
    ``sys.executable``, which must resolve ``omnibase_infra`` from THIS worktree
    (its editable install) so dev-only modules like
    ``omnibase_infra.utils.util_producer_effect_assertion`` are importable. But
    ``scripts/monitor_logs.py`` (imported by the ``test_monitor_*`` suites earlier
    in the session) runs ``_load_omnibase_env()`` at import, copying the
    ``PYTHONPATH`` line from ``~/.omnibase/.env`` -- which points at the CANONICAL
    ``$OMNI_HOME/omnibase_infra/src`` clone (frequently behind ``dev``) -- into the
    global ``os.environ`` when it is not already set. That entry lands ahead of the
    editable ``.pth`` and shadows the worktree, so the child import resolves the
    wrong checkout. Replacing ``PYTHONPATH`` with this worktree's ``src`` directory
    makes the child resolve the checked-out package deterministically, independent
    of collection order.
    """
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    env["PYTHONPATH"] = str(REPO_ROOT / "src")
    return env


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
def test_workflow_uses_authoritative_overlay_not_raw_kafka_secrets() -> None:
    """The post-merge producer must resolve its target from checked-in truth."""
    workflow = WORKFLOW_PATH.read_text()

    assert "repository: OmniNode-ai/omnimarket" in workflow
    assert "config/ci_bus_lanes.yaml" in workflow
    assert "model_redeploy_start_command.py" in workflow
    assert '--bus-lane "dev"' in workflow
    assert "--bus-overlay" in workflow
    assert "--consumer-model" in workflow
    assert "secrets.KAFKA_BOOTSTRAP_SERVERS" not in workflow
    assert "secrets.KAFKA_SASL_USERNAME" not in workflow
    assert "secrets.KAFKA_SASL_PASSWORD" not in workflow
    assert "secrets.DEPLOY_AGENT_HMAC_SECRET" not in workflow


@pytest.mark.unit
class TestRedeployStartPublish:
    """Unit tests for publish_redeploy_start_event() Kafka call shape (OMN-12573).

    CI publishes the node_redeploy start command, not the deploy-agent rebuild
    command directly.
    """

    def setup_method(self) -> None:
        self.mod = _import_trigger_module()

    @staticmethod
    def _write_consumer_model(tmp_path: Path) -> Path:
        model_path = tmp_path / "model_redeploy_start_command.py"
        model_path.write_text(
            "class ModelRedeployStartCommand(BaseModel):\n"
            "    model_config = ConfigDict(frozen=True, extra='forbid')\n"
            "    correlation_id: UUID = Field(...)\n"
            "    scope: str = Field(default='full')\n"
            "    git_ref: str = Field(default='origin/main')\n"
            "    runtime_lane: str = Field(default='dev')\n"
            "    build_source: str = Field(default='release')\n"
            "    requested_by: str = Field(default='node_redeploy_orchestrator')\n"
            "    dry_run: bool = Field(default=False)\n"
        )
        return model_path

    def test_overlay_declared_local_broker_builds_plaintext_transport(
        self, tmp_path: Path
    ) -> None:
        """The dev overlay reaches local Redpanda without fake SASL credentials."""
        overlay = tmp_path / "ci_bus_lanes.yaml"
        overlay.write_text(
            "default: inmemory\n"
            "lanes:\n"
            "  dev:\n"
            "    broker: omninode-pc.tail75df5e.ts.net:19092\n"
        )

        model = self.mod.load_ci_bus_overlay(overlay)
        broker = self.mod.resolve_ci_bus_broker(
            overlay=model,
            lane="dev",
            injected_broker="",
        )
        config = self.mod.build_kafka_producer_config(
            broker,
            username="",
            password="",
        )

        assert broker == "omninode-pc.tail75df5e.ts.net:19092"
        assert config == {"bootstrap.servers": "omninode-pc.tail75df5e.ts.net:19092"}

    def test_overlay_rejects_injected_broker_drift(self, tmp_path: Path) -> None:
        """An opaque broker injection may not override checked-in lane truth."""
        overlay = tmp_path / "ci_bus_lanes.yaml"
        overlay.write_text(
            "default: inmemory\nlanes:\n  dev:\n    broker: declared:19092\n"
        )

        model = self.mod.load_ci_bus_overlay(overlay)

        with pytest.raises(ValueError, match="LANE BUS DRIFT"):
            self.mod.resolve_ci_bus_broker(
                overlay=model,
                lane="dev",
                injected_broker="wrong:9092",
            )

    @pytest.mark.parametrize(
        ("username", "password"),
        [("user-only", ""), ("", "password-only")],
    )
    def test_partial_sasl_credentials_fail_closed(
        self, username: str, password: str
    ) -> None:
        """A half-configured SASL transport must not fall back to plaintext."""
        with pytest.raises(ValueError, match="both be set or both be empty"):
            self.mod.build_kafka_producer_config(
                "broker:9092",
                username=username,
                password=password,
            )

    def test_malformed_overlay_fails_validation(self, tmp_path: Path) -> None:
        """Unknown config fields are rejected rather than silently ignored."""
        overlay = tmp_path / "ci_bus_lanes.yaml"
        overlay.write_text(
            "default: inmemory\n"
            "lanes:\n"
            "  dev:\n"
            "    broker: declared:19092\n"
            "    typo_broker: wrong:9092\n"
        )

        with pytest.raises(ValueError, match="Invalid CI bus overlay"):
            self.mod.load_ci_bus_overlay(overlay)

    def test_cli_uses_overlay_broker_without_kafka_secrets(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """OMN-15009: a runtime merge publishes without any Kafka/HMAC secret."""
        overlay = tmp_path / "ci_bus_lanes.yaml"
        overlay.write_text(
            "default: inmemory\nlanes:\n  dev:\n    broker: declared:19092\n"
        )
        monkeypatch.delenv("KAFKA_BOOTSTRAP_SERVERS", raising=False)
        monkeypatch.delenv("KAFKA_SASL_USERNAME", raising=False)
        monkeypatch.delenv("KAFKA_SASL_PASSWORD", raising=False)
        monkeypatch.delenv("DEPLOY_AGENT_HMAC_SECRET", raising=False)
        consumer_model = self._write_consumer_model(tmp_path)
        captured: dict[str, Any] = {}

        def _fake_publish(**kwargs: Any) -> int:
            captured.update(kwargs)
            return 1

        monkeypatch.setattr(self.mod, "publish_redeploy_start_event", _fake_publish)

        result = CliRunner().invoke(
            self.mod.main,
            [
                "--changed-files",
                "src/omnibase_infra/nodes/node_runtime_sweep/handler.py",
                "--base-branch",
                "dev",
                "--source-sha",
                "deadbeef",
                "--bus-lane",
                "dev",
                "--bus-overlay",
                str(overlay),
                "--consumer-model",
                str(consumer_model),
            ],
        )

        assert result.exit_code == 0, result.output
        assert captured["bootstrap_servers"] == "declared:19092"
        assert captured["username"] == ""
        assert captured["password"] == ""

    def test_publish_calls_producer_with_redeploy_start_topic(self) -> None:
        """publish_redeploy_start_event publishes onex.cmd.omnimarket.redeploy-start.v1."""
        mock_producer = MagicMock()
        mock_producer.flush.return_value = None

        with patch("confluent_kafka.Producer", return_value=mock_producer):
            self.mod.publish_redeploy_start_event(
                bootstrap_servers="broker:9092",
                username="user",
                password="pass",
                runtime_lane="dev",
                build_source="workspace",
                source_sha="abc1234",
                correlation_id="d35d0dd8-e1a5-4fa7-a323-b1704ee44406",
                requested_by="gha-trigger",
            )

        mock_producer.produce.assert_called_once()
        call_kwargs = mock_producer.produce.call_args
        assert call_kwargs.kwargs["topic"] == "onex.cmd.omnimarket.redeploy-start.v1"

    def test_publish_event_payload_shape(self) -> None:
        """Payload carries the triggering lane + ref, never a hardcoded origin/main."""
        import json

        mock_producer = MagicMock()
        captured_value: list[bytes] = []

        def fake_produce(topic, key, value, on_delivery):
            captured_value.append(value)

        mock_producer.produce.side_effect = fake_produce
        mock_producer.flush.return_value = None

        with patch("confluent_kafka.Producer", return_value=mock_producer):
            self.mod.publish_redeploy_start_event(
                bootstrap_servers="broker:9092",
                username="user",
                password="pass",
                runtime_lane="stability-test",
                build_source="release",
                source_sha="deadbeef",
                correlation_id="d35d0dd8-e1a5-4fa7-a323-b1704ee44406",
                requested_by="gha-trigger",
            )

        assert captured_value, "produce was not called"
        payload = json.loads(captured_value[0])
        assert payload == {
            "correlation_id": "d35d0dd8-e1a5-4fa7-a323-b1704ee44406",
            "scope": "full",
            "git_ref": "deadbeef",
            "runtime_lane": "stability-test",
            "build_source": "release",
            "requested_by": "gha-trigger",
        }
        assert (
            not {
                "source_branch",
                "source_sha",
                "requires_occ",
                "requires_readiness_gate",
                "requested_at",
                "_signature",
            }
            & payload.keys()
        )

    def test_consumer_model_seam_rejects_legacy_extra_fields(
        self, tmp_path: Path
    ) -> None:
        """The producer checks its exact keys against the strict consumer model."""
        consumer_model = self._write_consumer_model(tmp_path)
        legacy_payload = {
            "correlation_id": "d35d0dd8-e1a5-4fa7-a323-b1704ee44406",
            "runtime_lane": "dev",
            "source_sha": "deadbeef",
            "_signature": "obsolete",
        }

        with pytest.raises(ValueError, match="consumer rejects extra fields"):
            self.mod.assert_consumer_model_accepts_payload(
                payload=legacy_payload,
                model_path=consumer_model,
            )


@pytest.mark.unit
class TestRedeployStartCLI:
    """CLI integration tests using --dry-run flag (OMN-12573)."""

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
                "--base-branch",
                "dev",
                "--source-sha",
                "abc123",
                "--dry-run",
            ],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
            env=_script_env(),
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"
        assert "no rebuild trigger" in result.stdout.lower()

    def test_dry_run_with_runtime_change_label_reports_dev_lane(self) -> None:
        """--dry-run with runtime_change label reports the dev lane and ref."""
        result = subprocess.run(
            [
                sys.executable,
                str(SCRIPT_PATH),
                "--changed-files",
                "",
                "--labels",
                "runtime_change",
                "--base-branch",
                "dev",
                "--source-sha",
                "abc123",
                "--dry-run",
            ],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
            env=_script_env(),
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"
        assert "runtime_lane=dev" in result.stdout
        assert "source_sha=abc123" in result.stdout

    def test_dry_run_with_main_base_reports_stability_lane(self) -> None:
        """--dry-run with omnimarket src path and main base reports the stability lane."""
        result = subprocess.run(
            [
                sys.executable,
                str(SCRIPT_PATH),
                "--changed-files",
                "src/omnimarket/nodes/foo/handler.py",
                "--labels",
                "",
                "--base-branch",
                "main",
                "--source-sha",
                "deadbeef",
                "--dry-run",
            ],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
            env=_script_env(),
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"
        assert "runtime_lane=stability-test" in result.stdout
        assert "source_sha=deadbeef" in result.stdout

    def test_unknown_base_branch_fails(self) -> None:
        """An unmapped base branch must fail closed (no silent default lane)."""
        result = subprocess.run(
            [
                sys.executable,
                str(SCRIPT_PATH),
                "--changed-files",
                "src/omnimarket/nodes/foo/handler.py",
                "--labels",
                "",
                "--base-branch",
                "release",
                "--source-sha",
                "abc123",
                "--dry-run",
            ],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
            env=_script_env(),
        )
        assert result.returncode != 0
        assert "release" in (result.stdout + result.stderr)
