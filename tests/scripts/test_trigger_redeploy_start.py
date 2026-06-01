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

REDEPLOY_START_TOPIC = "onex.cmd.omnimarket.redeploy-start.v1"
REBUILD_REQUESTED_TOPIC = "onex.cmd.deploy.rebuild-requested.v1"


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
        on_delivery(None, None)

    def flush(self, timeout: float) -> int:
        return 0


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
        hmac_secret="hmac-secret",
        runtime_lane="dev",
        source_branch="dev",
        source_sha="abc123",
        correlation_id="corr-123",
        requested_by="gha/omnibase_infra/pr-42",
    )

    producer = _CapturingProducer.last
    assert producer is not None
    assert len(producer.produced) == 1
    msg = producer.produced[0]

    assert msg["topic"] == REDEPLOY_START_TOPIC
    payload = msg["payload"]
    assert payload["runtime_lane"] == "dev"
    assert payload["source_branch"] == "dev"
    assert payload["source_sha"] == "abc123"
    # No hardcoded origin/main anywhere in the payload.
    assert "origin/main" not in json.dumps(payload)
    # HMAC signature is still applied.
    assert "_signature" in payload


@pytest.mark.unit
def test_cli_publishes_redeploy_start_with_main_lane(
    trigger_module: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """End-to-end CLI: a main-base trigger publishes the stability-test lane."""
    monkeypatch.setenv("KAFKA_BOOTSTRAP_SERVERS", "broker:9092")
    monkeypatch.setenv("KAFKA_SASL_USERNAME", "user")
    monkeypatch.setenv("KAFKA_SASL_PASSWORD", "secret")
    monkeypatch.setenv("DEPLOY_AGENT_HMAC_SECRET", "hmac-secret")

    captured: dict[str, Any] = {}

    def _fake_publish(**kwargs: Any) -> None:
        captured.update(kwargs)

    monkeypatch.setattr(trigger_module, "publish_redeploy_start_event", _fake_publish)

    result = CliRunner().invoke(
        trigger_module.main,
        [
            "--changed-files",
            "src/omnibase_infra/nodes/node_runtime_sweep/handler.py",
            "--base-branch",
            "main",
            "--source-sha",
            "deadbeef",
            "--correlation-id",
            "corr-xyz",
            "--requested-by",
            "gha/omnibase_infra/pr-7",
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured["runtime_lane"] == "stability-test"
    assert captured["source_branch"] == "main"
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
