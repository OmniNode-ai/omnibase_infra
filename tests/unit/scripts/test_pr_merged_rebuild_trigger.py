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
RUNTIME_PATH_VALIDATOR = REPO_ROOT / "tests" / "fixtures" / "runtime_path_classifier.py"


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
class _AckMessage:
    """Minimal confluent-kafka Message stand-in carrying delivery coordinates."""

    def __init__(self, partition: int, offset: int) -> None:
        self._partition = partition
        self._offset = offset

    def partition(self) -> int:
        return self._partition

    def offset(self) -> int:
        return self._offset


def _ack_produce(topic, key, value, on_delivery):  # type: ignore[no-untyped-def]
    """Acknowledge the produced message the way a live broker would."""
    on_delivery(None, _AckMessage(partition=0, offset=17))


class TestRebuildTriggerLogic:
    """Unit tests for canonical runtime-path and label trigger logic."""

    def setup_method(self) -> None:
        self.mod = _import_trigger_module()
        self.classifier = self.mod.load_runtime_path_classifier(RUNTIME_PATH_VALIDATOR)

    def test_runtime_change_label_triggers(self) -> None:
        """runtime_change label alone should trigger rebuild."""
        assert self.mod.should_trigger(
            runtime_paths=[],
            labels=["runtime_change"],
        )

    def test_omnimarket_src_path_triggers(self) -> None:
        """Changed file under src/omnimarket/ should trigger rebuild."""
        assert self.mod.should_trigger(
            runtime_paths=self.classifier(["src/omnimarket/nodes/foo/handler.py"]),
            labels=[],
        )

    def test_omnibase_infra_nodes_path_triggers(self) -> None:
        """Changed file under src/omnibase_infra/nodes/ should trigger rebuild."""
        assert self.mod.should_trigger(
            runtime_paths=self.classifier(
                ["src/omnibase_infra/nodes/node_foo/contract.yaml"]
            ),
            labels=[],
        )

    def test_docker_compose_path_triggers(self) -> None:
        """The exact OMN-15009 false-green Docker path must trigger rebuild."""
        runtime_paths = self.mod.classify_runtime_paths(
            ["docker/docker-compose.infra.yml"], self.classifier
        )

        assert runtime_paths == ["docker/docker-compose.infra.yml"]
        assert self.mod.should_trigger(runtime_paths=runtime_paths, labels=[])

    def test_non_runtime_path_does_not_trigger(self) -> None:
        """Changed file outside runtime paths should not trigger rebuild."""
        assert not self.mod.should_trigger(
            runtime_paths=self.classifier(
                ["docs/plans/some-plan.md", "tests/unit/test_foo.py"]
            ),
            labels=[],
        )

    def test_mixed_paths_one_match_triggers(self) -> None:
        """Any single matching file among many should trigger rebuild."""
        assert self.mod.should_trigger(
            runtime_paths=self.classifier(
                [
                    "README.md",
                    "src/omnimarket/nodes/bar/node.py",
                    "pyproject.toml",
                ]
            ),
            labels=[],
        )

    def test_empty_inputs_no_trigger(self) -> None:
        """No files and no labels should not trigger."""
        assert not self.mod.should_trigger(runtime_paths=[], labels=[])

    def test_unrelated_label_does_not_trigger(self) -> None:
        """Labels other than runtime_change should not trigger."""
        assert not self.mod.should_trigger(
            runtime_paths=[],
            labels=["bug", "documentation"],
        )

    def test_multiple_labels_with_runtime_change_triggers(self) -> None:
        """runtime_change among other labels should trigger."""
        assert self.mod.should_trigger(
            runtime_paths=[],
            labels=["bug", "runtime_change", "enhancement"],
        )

    def test_classifier_without_canonical_callable_fails_closed(
        self, tmp_path: Path
    ) -> None:
        """A fetched validator missing the canonical seam must stop the job."""
        validator = tmp_path / "validator.py"
        validator.write_text("RUNTIME_PATH_PATTERNS = []\n", encoding="utf-8")

        with pytest.raises(ValueError, match="does not define find_runtime_paths"):
            self.mod.load_runtime_path_classifier(validator)

    def test_invalid_classifier_result_fails_closed(self) -> None:
        """The hosted validator may not silently change its return contract."""

        def invalid_classifier(_changed_files: list[str]) -> tuple[str, ...]:
            return ("docker/docker-compose.infra.yml",)

        with pytest.raises(ValueError, match="returned an invalid path list"):
            self.mod.classify_runtime_paths(
                ["docker/docker-compose.infra.yml"], invalid_classifier
            )


@pytest.mark.unit
def test_workflow_uses_authoritative_overlay_not_raw_kafka_secrets() -> None:
    """The post-merge producer must resolve its target from checked-in truth.

    CONFIG vs CREDENTIAL, and why only one of them is a secret here (OMN-18012).
    The broker address and the transport are CONFIG: both are declared in
    omnimarket's ``config/ci_bus_lanes.yaml``, which the job checks out, so
    neither may arrive as an opaque secret — a ``KAFKA_BOOTSTRAP_SERVERS``
    injection is exactly what let a silent dev->stability repoint run green
    (OMN-14800). The SCRAM principal is a CREDENTIAL and can only arrive as a
    secret. The lane now declares ``SASL_PLAINTEXT`` / ``SCRAM-SHA-256``, and
    the publisher refuses to downgrade a declared SASL transport, so these two
    secrets MUST be injected — their absence is the failure this asserts against.
    """
    workflow = WORKFLOW_PATH.read_text()

    assert "repository: OmniNode-ai/omnimarket" in workflow
    assert "repository: OmniNode-ai/omniclaude" in workflow
    assert "config/ci_bus_lanes.yaml" in workflow
    assert "model_redeploy_start_command.py" in workflow
    assert "validate_pr_deploy_required.py" in workflow
    assert '--bus-lane "dev"' in workflow
    assert "--bus-overlay" in workflow
    assert "--consumer-model" in workflow
    assert "--runtime-path-validator" in workflow
    assert "secrets.KAFKA_BOOTSTRAP_SERVERS" not in workflow
    assert "secrets.KAFKA_SASL_USERNAME" in workflow
    assert "secrets.KAFKA_SASL_PASSWORD" in workflow
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
            "    security_protocol: PLAINTEXT\n"
        )

        model = self.mod.load_ci_bus_overlay(overlay)
        broker = self.mod.resolve_ci_bus_broker(
            overlay=model,
            lane="dev",
            injected_broker="",
        )
        protocol, mechanism = self.mod.resolve_ci_bus_security(
            overlay=model,
            lane="dev",
        )
        config = self.mod.build_kafka_producer_config(
            broker,
            "",
            "",
            protocol,
            mechanism,
        )

        assert broker == "omninode-pc.tail75df5e.ts.net:19092"
        assert config == {
            "bootstrap.servers": "omninode-pc.tail75df5e.ts.net:19092",
            "security.protocol": "PLAINTEXT",
        }

    def test_overlay_rejects_injected_broker_drift(self, tmp_path: Path) -> None:
        """An opaque broker injection may not override checked-in lane truth."""
        overlay = tmp_path / "ci_bus_lanes.yaml"
        overlay.write_text(
            "default: inmemory\nlanes:\n  dev:\n"
            "    broker: declared:19092\n"
            "    security_protocol: PLAINTEXT\n"
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
                username,
                password,
                "SASL_PLAINTEXT",
                "SCRAM-SHA-256",
            )

    def test_malformed_overlay_fails_validation(self, tmp_path: Path) -> None:
        """Unknown config fields are rejected rather than silently ignored."""
        overlay = tmp_path / "ci_bus_lanes.yaml"
        overlay.write_text(
            "default: inmemory\n"
            "lanes:\n"
            "  dev:\n"
            "    broker: declared:19092\n"
            "    security_protocol: PLAINTEXT\n"
            "    typo_broker: wrong:9092\n"
        )

        with pytest.raises(ValueError, match="Invalid CI bus overlay"):
            self.mod.load_ci_bus_overlay(overlay)

    def test_overlay_accepts_declared_projection_readback(self, tmp_path: Path) -> None:
        """OMN-18060: the lane's declared link-2 DSN NAME is modelled, not rejected.

        This is the live shape of omnimarket ``config/ci_bus_lanes.yaml`` after
        omnimarket#2420. Before the field existed here, ``extra="forbid"``
        rejected the real overlay outright -- "lanes.dev.projection_readback
        Extra inputs are not permitted" -- and every agent-path dev-lane rebuild
        trigger failed on it, which is the whole unblock chain for the chain
        canary, the OCC mint outage and the OMN-18072 verification.
        """
        overlay = tmp_path / "ci_bus_lanes.yaml"
        overlay.write_text(
            "default: inmemory\n"
            "lanes:\n"
            "  dev:\n"
            "    broker: omninode-pc.tail75df5e.ts.net:19092\n"
            "    security_protocol: SASL_PLAINTEXT\n"
            "    sasl_mechanism: SCRAM-SHA-256\n"
            "    projection_readback:\n"
            "      dsn_env: CHAIN_CANARY_PROJECTION_DSN\n"
            "  stability:\n"
            "    broker: inmemory\n"
            "  prod:\n"
            "    broker: inmemory\n"
        )

        model = self.mod.load_ci_bus_overlay(overlay)

        declaration = model.lanes["dev"].projection_readback
        assert declaration is not None
        assert declaration.dsn_env == "CHAIN_CANARY_PROJECTION_DSN"
        # The block must not disturb what this publisher actually reads.
        assert (
            self.mod.resolve_ci_bus_broker(
                overlay=model, lane="dev", injected_broker=""
            )
            == "omninode-pc.tail75df5e.ts.net:19092"
        )
        assert self.mod.resolve_ci_bus_security(overlay=model, lane="dev") == (
            "SASL_PLAINTEXT",
            "SCRAM-SHA-256",
        )

    def test_lane_without_projection_readback_is_none(self, tmp_path: Path) -> None:
        """Absent means absent -- the field is optional, with no invented default."""
        overlay = tmp_path / "ci_bus_lanes.yaml"
        overlay.write_text(
            "default: inmemory\n"
            "lanes:\n"
            "  dev:\n"
            "    broker: declared:19092\n"
            "    security_protocol: PLAINTEXT\n"
        )

        model = self.mod.load_ci_bus_overlay(overlay)

        assert model.lanes["dev"].projection_readback is None

    def test_projection_readback_rejects_unknown_key(self, tmp_path: Path) -> None:
        """The nested block is strict too; the fix widened the model, not the door."""
        overlay = tmp_path / "ci_bus_lanes.yaml"
        overlay.write_text(
            "default: inmemory\n"
            "lanes:\n"
            "  dev:\n"
            "    broker: declared:19092\n"
            "    security_protocol: PLAINTEXT\n"
            "    projection_readback:\n"
            "      dsn_env: CHAIN_CANARY_PROJECTION_DSN\n"
            "      dsn: postgresql://someone@host:5432/db\n"
        )

        with pytest.raises(ValueError, match="Invalid CI bus overlay"):
            self.mod.load_ci_bus_overlay(overlay)

    @pytest.mark.parametrize(
        "declared",
        [
            "postgresql://reader:pw@db.example.invalid:5432/onex",
            "host=db.example.invalid dbname=onex password=pw",
            "",
        ],
    )
    def test_projection_readback_refuses_a_value_where_a_name_belongs(
        self, tmp_path: Path, declared: str
    ) -> None:
        """A DSN pasted where a NAME belongs is a red gate, not committed config.

        The declaration is a variable NAME, so the check is a NAME check --
        which excludes every connection string by shape, since a DSN carries
        ``:``, ``/``, ``@`` or ``=`` and an environment-variable name carries
        none of them. Empty is refused on the same terms rather than treated as
        "not declared": ``projection_readback: {dsn_env: ""}`` is a half-written
        declaration, and silently reading it as ABSENT would hand the chain
        canary a SKIPPED verdict for a lane that meant to declare one.

        HONEST LIMIT, stated rather than implied: the validator's own message
        names the field and never the value, but ``load_ci_bus_overlay`` wraps
        pydantic's ``ValidationError``, whose rendering includes
        ``input_value``. So a DSN committed here still reaches the run log. The
        gate this test pins is that the overlay REFUSES the value, not that the
        log hides it -- a credential in a committed, CODEOWNERS-reviewed file is
        already disclosed by the commit, and the fix is to never land it, never
        to quiet the error.
        """
        overlay = tmp_path / "ci_bus_lanes.yaml"
        overlay.write_text(
            "default: inmemory\n"
            "lanes:\n"
            "  dev:\n"
            "    broker: declared:19092\n"
            "    security_protocol: PLAINTEXT\n"
            "    projection_readback:\n"
            f"      dsn_env: {declared!r}\n"
        )

        with pytest.raises(ValueError, match="Invalid CI bus overlay") as excinfo:
            self.mod.load_ci_bus_overlay(overlay)

        assert "dsn_env" in str(excinfo.value)

    def test_cli_uses_overlay_broker_without_kafka_secrets(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """OMN-15009: a runtime merge publishes without any Kafka/HMAC secret."""
        overlay = tmp_path / "ci_bus_lanes.yaml"
        overlay.write_text(
            "default: inmemory\nlanes:\n  dev:\n"
            "    broker: declared:19092\n"
            "    security_protocol: PLAINTEXT\n"
        )
        monkeypatch.delenv("KAFKA_BOOTSTRAP_SERVERS", raising=False)
        monkeypatch.delenv("KAFKA_SASL_USERNAME", raising=False)
        monkeypatch.delenv("KAFKA_SASL_PASSWORD", raising=False)
        monkeypatch.delenv("DEPLOY_AGENT_HMAC_SECRET", raising=False)
        consumer_model = self._write_consumer_model(tmp_path)
        captured: dict[str, Any] = {}

        def _fake_publish(**kwargs: Any) -> tuple[int, str]:
            captured.update(kwargs)
            # (delivered, broker-assigned coordinates) — OMN-17888: publication
            # is proven by the offset the broker assigned, not by intent.
            return 1, "partition=0 offset=17"

        monkeypatch.setattr(self.mod, "publish_redeploy_start_event", _fake_publish)

        result = CliRunner().invoke(
            self.mod.main,
            [
                "--changed-files",
                "src/omnibase_infra/nodes/node_runtime_sweep/handler.py",
                "--runtime-path-validator",
                str(RUNTIME_PATH_VALIDATOR),
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
        # OMN-17888: a drained flush with no delivery callback is unproven and
        # now raises, so the fake must acknowledge the message the way a broker
        # does — with a partition and an offset.
        mock_producer.produce.side_effect = _ack_produce

        with patch("confluent_kafka.Producer", return_value=mock_producer):
            self.mod.publish_redeploy_start_event(
                bootstrap_servers="broker:9092",
                username="user",
                password="pass",
                security_protocol="SASL_PLAINTEXT",
                sasl_mechanism="SCRAM-SHA-256",
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
            on_delivery(None, _AckMessage(partition=0, offset=17))

        mock_producer.produce.side_effect = fake_produce
        mock_producer.flush.return_value = None

        with patch("confluent_kafka.Producer", return_value=mock_producer):
            self.mod.publish_redeploy_start_event(
                bootstrap_servers="broker:9092",
                username="user",
                password="pass",
                security_protocol="SASL_PLAINTEXT",
                sasl_mechanism="SCRAM-SHA-256",
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
                "--runtime-path-validator",
                str(RUNTIME_PATH_VALIDATOR),
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
                "--runtime-path-validator",
                str(RUNTIME_PATH_VALIDATOR),
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
                "--runtime-path-validator",
                str(RUNTIME_PATH_VALIDATOR),
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
                "--runtime-path-validator",
                str(RUNTIME_PATH_VALIDATOR),
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


@pytest.mark.unit
class TestLaneDeclaredTransport:
    """OMN-18012 — the bus transport is read from the lane, never inferred.

    omnimarket#2387 added ``security_protocol`` / ``sasl_mechanism`` to the
    ``dev`` lane of ``config/ci_bus_lanes.yaml`` because the .201 dev-lane
    Redpanda external listener began requiring SASL/SCRAM-SHA-256 over PLAINTEXT.
    ``runtime-rebuild-trigger.yml`` sparse-checks that exact file out of
    ``omnimarket@dev`` and validates it here with ``extra="forbid"``, so a model
    that knew only ``broker`` rejected the live overlay and took the dev-lane
    redeploy trigger red on every runtime-touching PR (run 34160709151, job
    101861818486: ``Extra inputs are not permitted ... lanes.dev.sasl_mechanism``).

    ``_LIVE_DEV_LANE`` below is the live declaration verbatim. On ``origin/dev``
    the first test fails with that same validation error.
    """

    _LIVE_DEV_LANE = (
        "default: inmemory\n"
        "lanes:\n"
        "  dev:\n"
        '    broker: "omninode-pc.tail75df5e.ts.net:19092"\n'
        "    security_protocol: SASL_PLAINTEXT\n"
        "    sasl_mechanism: SCRAM-SHA-256\n"
        "  stability:\n"
        "    broker: inmemory\n"
        "  prod:\n"
        "    broker: inmemory\n"
    )

    def setup_method(self) -> None:
        self.mod = _import_trigger_module()

    def _overlay(self, tmp_path: Path, body: str) -> Path:
        overlay = tmp_path / "ci_bus_lanes.yaml"
        overlay.write_text(body)
        return overlay

    def test_live_omnimarket_overlay_validates(self, tmp_path: Path) -> None:
        """The overlay CI actually checks out loads and yields its transport."""
        model = self.mod.load_ci_bus_overlay(
            self._overlay(tmp_path, self._LIVE_DEV_LANE)
        )

        assert model.lanes["dev"].security_protocol == "SASL_PLAINTEXT"
        assert model.lanes["dev"].sasl_mechanism == "SCRAM-SHA-256"
        assert self.mod.resolve_ci_bus_security(overlay=model, lane="dev") == (
            "SASL_PLAINTEXT",
            "SCRAM-SHA-256",
        )

    def test_producer_config_honours_the_declared_sasl_plaintext(
        self, tmp_path: Path
    ) -> None:
        """SASL over PLAINTEXT is used verbatim — never upgraded to SASL_SSL."""
        model = self.mod.load_ci_bus_overlay(
            self._overlay(tmp_path, self._LIVE_DEV_LANE)
        )
        protocol, mechanism = self.mod.resolve_ci_bus_security(
            overlay=model, lane="dev"
        )

        config = self.mod.build_kafka_producer_config(
            "omninode-pc.tail75df5e.ts.net:19092",
            "ci-principal",
            "ci-secret",
            protocol,
            mechanism,
        )

        assert config["security.protocol"] == "SASL_PLAINTEXT"
        assert config["sasl.mechanisms"] == "SCRAM-SHA-256"
        assert config["sasl.username"] == "ci-principal"

    def test_sasl_lane_without_credentials_fails_closed(self, tmp_path: Path) -> None:
        """A SASL lane with no credentials must red, not downgrade to plaintext."""
        model = self.mod.load_ci_bus_overlay(
            self._overlay(tmp_path, self._LIVE_DEV_LANE)
        )
        protocol, mechanism = self.mod.resolve_ci_bus_security(
            overlay=model, lane="dev"
        )

        with pytest.raises(ValueError, match="are not set in this job's environment"):
            self.mod.build_kafka_producer_config(
                "broker:19092", "", "", protocol, mechanism
            )

    def test_publishing_lane_must_declare_a_security_protocol(
        self, tmp_path: Path
    ) -> None:
        """A concrete broker with no declared transport is a wiring gap."""
        with pytest.raises(ValueError, match="no security_protocol"):
            self.mod.load_ci_bus_overlay(
                self._overlay(
                    tmp_path,
                    "default: inmemory\nlanes:\n  dev:\n    broker: declared:19092\n",
                )
            )

    def test_mechanism_beside_non_sasl_protocol_is_contradictory(
        self, tmp_path: Path
    ) -> None:
        """Half a transport declaration is rejected rather than half-applied."""
        with pytest.raises(ValueError, match="carries no"):
            self.mod.load_ci_bus_overlay(
                self._overlay(
                    tmp_path,
                    "default: inmemory\n"
                    "lanes:\n"
                    "  dev:\n"
                    "    broker: declared:19092\n"
                    "    security_protocol: PLAINTEXT\n"
                    "    sasl_mechanism: SCRAM-SHA-256\n",
                )
            )

    def test_sasl_protocol_without_mechanism_is_rejected(self, tmp_path: Path) -> None:
        """SASL without a mechanism would leave librdkafka to pick one."""
        with pytest.raises(ValueError, match="requires a"):
            self.mod.load_ci_bus_overlay(
                self._overlay(
                    tmp_path,
                    "default: inmemory\n"
                    "lanes:\n"
                    "  dev:\n"
                    "    broker: declared:19092\n"
                    "    security_protocol: SASL_PLAINTEXT\n",
                )
            )

    def test_inmemory_lane_needs_no_transport(self, tmp_path: Path) -> None:
        """An in-memory lane publishes nothing cross-process, so declares none."""
        model = self.mod.load_ci_bus_overlay(
            self._overlay(tmp_path, self._LIVE_DEV_LANE)
        )

        assert model.lanes["stability"].security_protocol is None
        assert model.lanes["prod"].sasl_mechanism is None

    def test_unknown_security_protocol_is_rejected(self, tmp_path: Path) -> None:
        """Only librdkafka's four protocol names are accepted."""
        with pytest.raises(ValueError, match="not a librdkafka security"):
            self.mod.load_ci_bus_overlay(
                self._overlay(
                    tmp_path,
                    "default: inmemory\n"
                    "lanes:\n"
                    "  dev:\n"
                    "    broker: declared:19092\n"
                    "    security_protocol: SASL_TLS\n",
                )
            )


@pytest.mark.unit
class TestNoPublishRunStillValidatesTheOverlay:
    """OMN-18060 — a run that publishes nothing must still be probative.

    THE FAIL-OPEN THIS CLOSES. ``main()`` classified the merge first and
    returned before it had looked at the overlay at all: a merge with no
    runtime path and no ``runtime_change`` label printed "No rebuild trigger"
    and exited 0 without parsing ``config/ci_bus_lanes.yaml``. Every such run
    was GREEN and proved nothing about the overlay contract, so a producer-side
    key added in omnimarket sat undetected until the first RUNTIME merge, which
    then failed on a skew introduced by an unrelated repository hours earlier.
    That is exactly how 2026-09-09 played out: the ``projection_readback`` key
    landed at 02:15Z and the ten trigger runs between then and the first
    runtime merge all reported success.

    The publisher validates the overlay it was HANDED, on every path that can
    reach an exit. The no-publish semantics are unchanged -- no broker is
    contacted, no command is produced, ``published=false`` is still emitted --
    the run simply now fails when the checked-in contract it was given does not
    load.

    HONEST LIMIT, pinned rather than implied: with no ``--bus-overlay`` there
    is nothing to validate and the early exit stays green. That is not a hole
    in CI, because the workflow always passes the flag -- asserted by
    ``test_workflow_uses_authoritative_overlay_not_raw_kafka_secrets`` above --
    but it does mean a hand-run invocation without the flag is not a skew
    check, and the local test below says so.
    """

    _SKEWED_OVERLAY = (
        "default: inmemory\n"
        "lanes:\n"
        "  dev:\n"
        "    broker: declared:19092\n"
        "    security_protocol: PLAINTEXT\n"
        "    a_key_this_publisher_has_never_learned: whatever\n"
    )

    _VALID_OVERLAY = (
        "default: inmemory\n"
        "lanes:\n"
        "  dev:\n"
        "    broker: declared:19092\n"
        "    security_protocol: PLAINTEXT\n"
        "    projection_readback:\n"
        "      dsn_env: CHAIN_CANARY_PROJECTION_DSN\n"
    )

    @staticmethod
    def _run(
        overlay: Path | None,
        *,
        labels: str = "",
        dry_run: bool = False,
    ):
        argv = [
            sys.executable,
            str(SCRIPT_PATH),
            "--changed-files",
            "README.md,docs/plans/foo.md",
            "--runtime-path-validator",
            str(RUNTIME_PATH_VALIDATOR),
            "--labels",
            labels,
            "--base-branch",
            "dev",
            "--source-sha",
            "abc123",
        ]
        if overlay is not None:
            argv += ["--bus-lane", "dev", "--bus-overlay", str(overlay)]
        if dry_run:
            argv.append("--dry-run")
        return subprocess.run(
            argv,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
            env=_script_env(),
        )

    def _overlay(self, tmp_path: Path, body: str) -> Path:
        overlay = tmp_path / "ci_bus_lanes.yaml"
        overlay.write_text(body)
        return overlay

    def test_no_runtime_change_with_skewed_overlay_fails_closed(
        self, tmp_path: Path
    ) -> None:
        """A docs-only merge reds on overlay skew instead of exiting 0 green."""
        result = self._run(self._overlay(tmp_path, self._SKEWED_OVERLAY))

        assert result.returncode == 1, f"stdout: {result.stdout}"
        combined = result.stdout + result.stderr
        assert "Invalid CI bus overlay" in combined
        assert "a_key_this_publisher_has_never_learned" in combined

    def test_no_runtime_change_with_valid_overlay_still_exits_zero(
        self, tmp_path: Path
    ) -> None:
        """The no-publish semantics are unchanged when the contract is intact."""
        result = self._run(self._overlay(tmp_path, self._VALID_OVERLAY))

        assert result.returncode == 0, f"stderr: {result.stderr}"
        assert "no rebuild trigger" in result.stdout.lower()

    def test_no_runtime_change_never_contacts_a_broker(self, tmp_path: Path) -> None:
        """Validation is a parse of a checked-in file, not a connection.

        The declared broker is a name that does not resolve. A run that tried
        to reach it would hang for the 30-second flush and time out here; the
        early exit must return immediately.
        """
        result = self._run(self._overlay(tmp_path, self._VALID_OVERLAY))

        assert result.returncode == 0
        assert "Published redeploy-start" not in result.stdout

    def test_missing_overlay_file_fails_closed(self, tmp_path: Path) -> None:
        """A checkout that silently produced no overlay is a wiring gap, not a skip."""
        result = self._run(tmp_path / "does_not_exist.yaml")

        assert result.returncode == 1
        assert "CI bus overlay does not exist" in (result.stdout + result.stderr)

    def test_without_the_flag_the_early_exit_is_unchanged(self) -> None:
        """The stated limit: no overlay handed in means no skew check."""
        result = self._run(None)

        assert result.returncode == 0, f"stderr: {result.stderr}"
        assert "no rebuild trigger" in result.stdout.lower()

    def test_dry_run_with_a_runtime_change_also_validates(self, tmp_path: Path) -> None:
        """--dry-run is the local skew check, so it reds on a skewed overlay too."""
        result = self._run(
            self._overlay(tmp_path, self._SKEWED_OVERLAY),
            labels="runtime_change",
            dry_run=True,
        )

        assert result.returncode == 1, f"stdout: {result.stdout}"
        assert "Invalid CI bus overlay" in (result.stdout + result.stderr)
