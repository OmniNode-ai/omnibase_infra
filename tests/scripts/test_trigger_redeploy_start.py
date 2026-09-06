# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Unit tests for scripts/trigger_rebuild_on_merge.py re-point to node_redeploy.

OMN-12573 (blocker B1): CI must publish ``onex.cmd.omnimarket.redeploy-start.v1``
(consumed by ``node_redeploy``) carrying the triggering lane + ref — not
``onex.cmd.deploy.rebuild-requested.v1`` directly, and not a hardcoded
``origin/main``. ``node_redeploy`` remains the sole emitter of
``onex.cmd.deploy.rebuild-requested.v1`` to the deploy agent.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest
from click.testing import CliRunner

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts" / "trigger_rebuild_on_merge.py"
RUNTIME_PATH_VALIDATOR = REPO_ROOT / "tests" / "fixtures" / "runtime_path_classifier.py"

REDEPLOY_START_TOPIC = "onex.cmd.omnimarket.redeploy-start.v1"
REBUILD_REQUESTED_TOPIC = "onex.cmd.deploy.rebuild-requested.v1"


def _write_bus_contracts(tmp_path: Path) -> tuple[Path, Path]:
    overlay = tmp_path / "ci_bus_lanes.yaml"
    overlay.write_text(
        "default: inmemory\nlanes:\n  dev:\n    broker: declared:19092\n"
    )
    consumer_model = tmp_path / "model_redeploy_start_command.py"
    consumer_model.write_text(
        "class ModelRedeployStartCommand(BaseModel):\n"
        "    model_config = ConfigDict(frozen=True, extra='forbid')\n"
        "    correlation_id: UUID = Field(...)\n"
        "    scope: str = Field(default='full')\n"
        "    git_ref: str = Field(default='origin/main')\n"
        "    runtime_lane: str = Field(default='dev')\n"
        "    build_source: str = Field(default='release')\n"
        "    requested_by: str = Field(default='node_redeploy_orchestrator')\n"
    )
    return overlay, consumer_model


def _load_trigger_module() -> Any:
    spec = importlib.util.spec_from_file_location(
        "trigger_rebuild_on_merge", SCRIPT_PATH
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["trigger_rebuild_on_merge"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def trigger_module() -> Any:
    return _load_trigger_module()


class _DeliveredMessage:
    """Minimal confluent-kafka Message stand-in carrying delivery coordinates."""

    def __init__(self, partition: int, offset: int) -> None:
        self._partition = partition
        self._offset = offset

    def partition(self) -> int:
        return self._partition

    def offset(self) -> int:
        return self._offset


class _CapturingProducer:
    """Fake confluent-kafka Producer capturing the published topic + payload."""

    last: _CapturingProducer | None = None

    def __init__(self, config: dict[str, object]) -> None:
        self.config = config
        self.produced: list[dict[str, Any]] = []
        _CapturingProducer.last = self

    def produce(
        self,
        topic: str,
        key: bytes,
        value: bytes,
        on_delivery: Any,
    ) -> None:
        self.produced.append(
            {
                "topic": topic,
                "key": key,
                "payload": json.loads(value.decode("utf-8")),
            }
        )
        on_delivery(None, _DeliveredMessage(partition=0, offset=17))

    def flush(self, timeout: float) -> int:
        return 0


class _SilentProducer(_CapturingProducer):
    """Producer whose flush drains but whose delivery callback never fires.

    This is the OMN-17378 green-but-silent shape: no broker-assigned offset,
    therefore no proof of publication, therefore not a success.
    """

    def produce(
        self,
        topic: str,
        key: bytes,
        value: bytes,
        on_delivery: Any,
    ) -> None:
        self.produced.append(
            {
                "topic": topic,
                "key": key,
                "payload": json.loads(value.decode("utf-8")),
            }
        )


def _code_lines(source: str) -> str:
    """Return source with comment lines stripped (operative code only).

    Comments legitimately reference the deploy-agent topic to explain *why* CI
    must not publish it; the guarantee under test is that no code path actually
    targets it.
    """
    return "\n".join(
        line for line in source.splitlines() if not line.lstrip().startswith("#")
    )


@pytest.mark.unit
def test_topic_constant_is_redeploy_start_not_rebuild_requested(
    trigger_module: Any,
) -> None:
    """The script must publish to node_redeploy, not the deploy agent directly."""
    assert trigger_module.TOPIC == REDEPLOY_START_TOPIC
    # node_redeploy is the SOLE emitter of rebuild-requested to the deploy agent;
    # this script must not publish that topic from any code path.
    code = _code_lines(SCRIPT_PATH.read_text())
    assert REBUILD_REQUESTED_TOPIC not in code


@pytest.mark.unit
def test_lane_for_base_branch_maps_dev_and_main(trigger_module: Any) -> None:
    """dev merges deploy the dev lane; main (promotion) merges the stability lane."""
    assert trigger_module.lane_for_base_branch("dev") == "dev"
    assert trigger_module.lane_for_base_branch("main") == "stability-test"


@pytest.mark.unit
def test_lane_for_base_branch_rejects_unknown(trigger_module: Any) -> None:
    """Unknown base branches must fail closed (no silent default lane)."""
    with pytest.raises(ValueError, match="release"):
        trigger_module.lane_for_base_branch("release")


@pytest.mark.unit
def test_publish_redeploy_start_carries_triggering_lane_and_ref(
    trigger_module: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Payload carries the actual lane/ref/sha, never a hardcoded origin/main."""
    import types

    fake_confluent = types.SimpleNamespace(Producer=_CapturingProducer)
    monkeypatch.setitem(sys.modules, "confluent_kafka", fake_confluent)

    trigger_module.publish_redeploy_start_event(
        bootstrap_servers="broker:9092",
        username="user",
        password="secret",
        runtime_lane="dev",
        build_source="workspace",
        source_sha="abc1234",
        correlation_id="d35d0dd8-e1a5-4fa7-a323-b1704ee44406",
        requested_by="gha/omnibase_infra/pr-42",
    )

    producer = _CapturingProducer.last
    assert producer is not None
    assert len(producer.produced) == 1
    msg = producer.produced[0]

    assert msg["topic"] == REDEPLOY_START_TOPIC
    payload = msg["payload"]
    assert payload["runtime_lane"] == "dev"
    assert payload["git_ref"] == "abc1234"
    assert payload["build_source"] == "workspace"
    # No hardcoded origin/main anywhere in the payload.
    assert "origin/main" not in json.dumps(payload)
    assert "_signature" not in payload
    assert "source_branch" not in payload
    assert "source_sha" not in payload


@pytest.mark.unit
def test_cli_publishes_redeploy_start_with_main_lane(
    trigger_module: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """End-to-end CLI: a main-base trigger publishes the stability-test lane."""
    monkeypatch.delenv("KAFKA_BOOTSTRAP_SERVERS", raising=False)
    monkeypatch.delenv("KAFKA_SASL_USERNAME", raising=False)
    monkeypatch.delenv("KAFKA_SASL_PASSWORD", raising=False)
    monkeypatch.delenv("DEPLOY_AGENT_HMAC_SECRET", raising=False)
    overlay, consumer_model = _write_bus_contracts(tmp_path)

    captured: dict[str, Any] = {}

    def _fake_publish(**kwargs: Any) -> tuple[int, str]:
        captured.update(kwargs)
        # publish_redeploy_start_event returns (delivered, coordinates); the
        # caller asserts delivered >=1 (RT-5 fail-closed on zero output) and
        # prints the broker-assigned coordinates as the publication proof.
        return 1, "partition=0 offset=17"

    monkeypatch.setattr(trigger_module, "publish_redeploy_start_event", _fake_publish)

    result = CliRunner().invoke(
        trigger_module.main,
        [
            "--changed-files",
            "src/omnibase_infra/nodes/node_runtime_sweep/handler.py",
            "--runtime-path-validator",
            str(RUNTIME_PATH_VALIDATOR),
            "--base-branch",
            "main",
            "--source-sha",
            "deadbeef",
            "--correlation-id",
            "d35d0dd8-e1a5-4fa7-a323-b1704ee44406",
            "--requested-by",
            "gha/omnibase_infra/pr-7",
            "--bus-lane",
            "dev",
            "--bus-overlay",
            str(overlay),
            "--consumer-model",
            str(consumer_model),
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured["runtime_lane"] == "stability-test"
    assert captured["build_source"] == "release"
    assert captured["source_sha"] == "deadbeef"


@pytest.mark.unit
def test_cli_dry_run_reports_lane_and_ref(
    trigger_module: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Dry-run prints the resolved lane/ref and does not publish."""

    def _explode(**_kwargs: Any) -> None:
        raise AssertionError("dry-run must not publish")

    monkeypatch.setattr(trigger_module, "publish_redeploy_start_event", _explode)

    result = CliRunner().invoke(
        trigger_module.main,
        [
            "--changed-files",
            "src/omnimarket/nodes/foo/handler.py",
            "--runtime-path-validator",
            str(RUNTIME_PATH_VALIDATOR),
            "--base-branch",
            "dev",
            "--source-sha",
            "cafe",
            "--dry-run",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "runtime_lane=dev" in result.output
    assert "source_sha=cafe" in result.output


@pytest.mark.unit
def test_workflow_passes_triggering_lane_and_ref_not_origin_main() -> None:
    """The GHA workflow must hand the script the real base branch + merge SHA."""
    workflow = (
        REPO_ROOT / ".github" / "workflows" / "runtime-rebuild-trigger.yml"
    ).read_text()
    code = _code_lines(workflow)
    # The hardcoded git ref is gone from the operative invocation.
    assert "--git-ref" not in code
    assert "origin/main" not in code
    # The script receives the merged PR's base branch and merge SHA.
    assert "--base-branch" in code
    assert "github.event.pull_request.base.ref" in code
    assert "--source-sha" in code
    assert "github.event.pull_request.merge_commit_sha" in code


# ---------------------------------------------------------------------------
# OMN-17888: the publisher is LAN-bound and must say so
#
# The dev control-bus lane declares a tailnet broker. OMN-16682 retargeted the
# shared OMNI_TRUSTED_CI_RUNS_ON_JSON seam to ["ubuntu-latest"] at org AND repo
# scope on 2026-08-26, which relocated this job onto GitHub-hosted compute that
# cannot resolve that name. Every run since that had a runtime change to publish
# failed on a DNS error after a 30-second flush; the runs that read green were
# no-ops with nothing to send. These tests pin BOTH halves of the remedy: the
# dedicated runner knob, and a refusal that names the routing defect instead of
# timing out against an unresolvable host.
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_workflow_pins_the_publisher_to_the_lan_fleet() -> None:
    """The trusted arm must NOT read the shared trusted-CI runner seam."""
    workflow = (
        REPO_ROOT / ".github" / "workflows" / "runtime-rebuild-trigger.yml"
    ).read_text()
    code = _code_lines(workflow)

    # A dedicated knob, deliberately left unset so the self-hosted literal wins.
    assert "vars.OMNI_RUNTIME_REBUILD_RUNS_ON_JSON" in code
    assert '\'["self-hosted","omnibase-ci"]\'' in code
    # The shared seam is what moved this LAN-bound publisher onto hosted compute.
    assert "OMNI_TRUSTED_CI_RUNS_ON_JSON" not in code
    # Fork PRs still route to hosted compute — untrusted code never reaches the
    # fleet — and that arm is still gated on the fork predicate.
    assert "vars.OMNI_PUBLIC_PR_RUNS_ON_JSON" in code
    assert "head.repo.full_name != github.repository" in code
    # The runner class reaches the script so it can refuse by name.
    assert "runner.environment" in code
    assert "--runner-environment" in code


@pytest.mark.unit
def test_cli_refuses_a_live_publish_from_a_github_hosted_runner(
    trigger_module: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A hosted runner cannot reach the lane broker; refuse before producing."""
    monkeypatch.delenv("KAFKA_BOOTSTRAP_SERVERS", raising=False)
    overlay, consumer_model = _write_bus_contracts(tmp_path)

    def _explode(**_kwargs: Any) -> None:
        raise AssertionError("must not construct a producer on a hosted runner")

    monkeypatch.setattr(trigger_module, "publish_redeploy_start_event", _explode)

    result = CliRunner().invoke(
        trigger_module.main,
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
            "--runner-environment",
            "github-hosted",
        ],
    )

    assert result.exit_code == 1, result.output
    assert "refusing to publish from a github-hosted runner" in result.output
    assert "OMNI_RUNTIME_REBUILD_RUNS_ON_JSON" in result.output


@pytest.mark.unit
def test_cli_publishes_normally_from_the_self_hosted_fleet(
    trigger_module: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The refusal is scoped to hosted compute; the fleet path is unchanged."""
    monkeypatch.delenv("KAFKA_BOOTSTRAP_SERVERS", raising=False)
    overlay, consumer_model = _write_bus_contracts(tmp_path)

    def _fake_publish(**_kwargs: Any) -> tuple[int, str]:
        return 1, "partition=0 offset=64"

    monkeypatch.setattr(trigger_module, "publish_redeploy_start_event", _fake_publish)

    result = CliRunner().invoke(
        trigger_module.main,
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
            "--runner-environment",
            "self-hosted",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "partition=0 offset=64" in result.output


@pytest.mark.unit
def test_decision_line_does_not_claim_delivery(
    trigger_module: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The pre-flush line records the DECISION, never the publication.

    "Redeploy triggered: ..." was emitted before a 30-second flush that then
    timed out with the command undelivered, so anyone reading the step summary
    saw a receipt for a message that never left the runner. A surrogate for the
    behaviour is not the behaviour.
    """

    def _explode(**_kwargs: Any) -> None:
        raise AssertionError("dry-run must not publish")

    monkeypatch.setattr(trigger_module, "publish_redeploy_start_event", _explode)

    result = CliRunner().invoke(
        trigger_module.main,
        [
            "--changed-files",
            "src/omnimarket/nodes/foo/handler.py",
            "--runtime-path-validator",
            str(RUNTIME_PATH_VALIDATOR),
            "--base-branch",
            "dev",
            "--source-sha",
            "cafe",
            "--dry-run",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Redeploy triggered" not in result.output
    assert "delivery NOT yet confirmed" in result.output
    assert "runtime_lane=dev" in result.output


@pytest.mark.unit
def test_publish_reports_broker_assigned_coordinates(
    trigger_module: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A publish is proven by the offset the broker assigned it, not by intent."""
    import types

    monkeypatch.setitem(
        sys.modules,
        "confluent_kafka",
        types.SimpleNamespace(Producer=_CapturingProducer),
    )

    delivered, coordinates = trigger_module.publish_redeploy_start_event(
        bootstrap_servers="broker:9092",
        username="",
        password="",
        runtime_lane="dev",
        build_source="workspace",
        source_sha="abc1234",
        correlation_id="d35d0dd8-e1a5-4fa7-a323-b1704ee44406",
        requested_by="gha/omnibase_infra/pr-42",
    )

    assert delivered == 1
    assert coordinates == "partition=0 offset=17"


@pytest.mark.unit
def test_publish_fails_closed_when_no_delivery_callback_fires(
    trigger_module: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A drained queue with no acknowledgement is unproven, not successful."""
    import types

    monkeypatch.setitem(
        sys.modules, "confluent_kafka", types.SimpleNamespace(Producer=_SilentProducer)
    )

    with pytest.raises(
        RuntimeError, match=r"no broker-assigned offset, so publication is unproven"
    ):
        trigger_module.publish_redeploy_start_event(
            bootstrap_servers="broker:9092",
            username="",
            password="",
            runtime_lane="dev",
            build_source="workspace",
            source_sha="abc1234",
            correlation_id="d35d0dd8-e1a5-4fa7-a323-b1704ee44406",
            requested_by="gha/omnibase_infra/pr-42",
        )
