# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-19152: a skill's returned terminal result reaches the declared lane.

``onex skill dod_verify`` dispatches on the in-memory bus, so the verdict its
node returns used to die with the process and the durable verdict table
(OMN-18900) stayed empty on the path production runs. The dispatch stays
in-memory; after it returns, the returned result is published to the
contract's ``terminal_event`` topic on the lane the skill mapping declares.

The publish is fail-soft by construction (AC-2): a missing lane, an
unreachable broker or a timeout is reported and never changes the verdict,
the receipt or the exit code.
"""

from __future__ import annotations

import json
import socket
import time
from pathlib import Path
from uuid import UUID, uuid4

import pytest
from click.testing import CliRunner
from pydantic import JsonValue

from omnibase_core.enums.enum_skill_result_status import EnumSkillResultStatus
from omnibase_core.models.dispatch.model_skill_result import ModelSkillResult
from omnibase_infra.cli import cli_skill
from omnibase_infra.cli.cli_skill import load_skill_registry, run_skill_by_name
from omnibase_infra.cli.enum_skill_terminal_publish_outcome import (
    EnumSkillTerminalPublishOutcome,
)
from omnibase_infra.cli.model_receipt_runtime_summary import ModelReceiptRuntimeSummary
from omnibase_infra.cli.model_skill_terminal_publish_target import (
    ModelSkillTerminalPublishTarget,
)
from omnibase_infra.cli.skill_terminal_publish import publish_skill_terminal_event
from omnibase_infra.runtime_identity import collect_runtime_identity

pytestmark = pytest.mark.unit

TERMINAL_TOPIC = "onex.evt.omnimarket.dod-verify-completed.v1"
RESULT_MODEL = (
    "omnimarket.nodes.node_dod_verify.models.model_dod_verify_state.ModelDodVerifyState"
)


@pytest.fixture(autouse=True)
def _clear_registry_cache() -> None:
    load_skill_registry.cache_clear()


def _contract(tmp_path: Path) -> Path:
    path = tmp_path / "contract.yaml"
    path.write_text(
        "name: node_dod_verify\n"
        f"terminal_event: {TERMINAL_TOPIC}\n"
        "event_bus:\n"
        f"  publish_topics: [{TERMINAL_TOPIC}]\n",
        encoding="utf-8",
    )
    return path


def _verdict(status: str) -> dict[str, JsonValue]:
    return {
        "ticket_id": "OMN-19152",
        "correlation_id": str(uuid4()),
        "status": status,
        "total_checks": 2,
        "passed_checks": 2 if status == "verified" else 1,
        "failed_checks": 0 if status == "verified" else 1,
        "started_at": "2026-09-25T01:00:00+00:00",
        "completed_at": "2026-09-25T01:00:05+00:00",
        "dry_run": False,
    }


def _success_receipt(result: dict[str, JsonValue]) -> object:
    return ModelSkillResult[JsonValue](
        skill_name="node_dod_verify",
        node_name="node_dod_verify",
        status=EnumSkillResultStatus.SUCCESS,
        correlation_id=uuid4(),
        run_id=uuid4(),
        exit_code=0,
        duration_ms=10,
        result=result,
        result_model=RESULT_MODEL,
        runtime_identity=collect_runtime_identity(config_source="contract.yaml"),
    )


def _failure_receipt(handler_result: dict[str, JsonValue] | None) -> object:
    summary = ModelReceiptRuntimeSummary(
        workflow_result="failed",
        exit_code=1,
        workflow="contract.yaml",
        handler_result=handler_result,
    )
    return ModelSkillResult[ModelReceiptRuntimeSummary](
        skill_name="node_dod_verify",
        node_name="node_dod_verify",
        status=EnumSkillResultStatus.FAILED,
        correlation_id=uuid4(),
        run_id=uuid4(),
        exit_code=1,
        duration_ms=10,
        result=summary,
        result_model="omnibase_infra.cli.receipt_mode.ModelReceiptRuntimeSummary",
        runtime_identity=collect_runtime_identity(config_source="contract.yaml"),
    )


class _RecordingPublisher:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, bytes | None, bytes]] = []

    async def __call__(
        self,
        target: ModelSkillTerminalPublishTarget,
        topic: str,
        key: bytes | None,
        value: bytes,
        timeout_seconds: float,
    ) -> None:
        self.calls.append((target.bootstrap_servers, topic, key, value))


def _target(bootstrap: str = "broker.test:19092") -> ModelSkillTerminalPublishTarget:
    return ModelSkillTerminalPublishTarget(
        lane="dev", bootstrap_servers=bootstrap, transport=None
    )


def test_dod_verify_mapping_declares_the_dev_lane_and_no_other_skill_does() -> None:
    registry = load_skill_registry()
    dod = registry.get("dod_verify")
    assert dod is not None
    assert dod.publish_terminal_to_lane == "dev"
    assert dod.event_bus == "inmemory", "the dispatch itself stays in-memory"
    others = [
        s.skill_name
        for s in registry.skills
        if s.skill_name != "dod_verify" and s.publish_terminal_to_lane is not None
    ]
    assert others == []


def test_a_verified_verdict_is_published_as_the_terminal_envelope(
    tmp_path: Path,
) -> None:
    verdict = _verdict("verified")
    publisher = _RecordingPublisher()
    report = publish_skill_terminal_event(
        receipt=_success_receipt(verdict),
        result_model=RESULT_MODEL,
        contract_path=_contract(tmp_path),
        lane="dev",
        resolve_target=lambda _lane: _target(),
        publisher=publisher,
    )
    assert report.outcome is EnumSkillTerminalPublishOutcome.PUBLISHED
    assert report.topic == TERMINAL_TOPIC
    assert report.correlation_id == UUID(str(verdict["correlation_id"]))
    [(bootstrap, topic, key, value)] = publisher.calls
    assert (bootstrap, topic) == ("broker.test:19092", TERMINAL_TOPIC)
    assert key == str(verdict["correlation_id"]).encode()
    envelope = json.loads(value)
    assert envelope["payload"] == verdict
    assert envelope["correlation_id"] == verdict["correlation_id"]


def test_a_failed_verdict_is_published_too(tmp_path: Path) -> None:
    """A failure is as durable as a pass (OMN-18900 AC2).

    A failed verdict makes the run exit non-zero, and receipt mode then
    carries the handler result inside its failure summary rather than as the
    receipt's result. It must still be published.
    """
    verdict = _verdict("failed")
    publisher = _RecordingPublisher()
    report = publish_skill_terminal_event(
        receipt=_failure_receipt(verdict),
        result_model=RESULT_MODEL,
        contract_path=_contract(tmp_path),
        lane="dev",
        resolve_target=lambda _lane: _target(),
        publisher=publisher,
    )
    assert report.outcome is EnumSkillTerminalPublishOutcome.PUBLISHED
    [(_, _, _, value)] = publisher.calls
    assert json.loads(value)["payload"] == verdict


def test_a_run_that_returned_no_result_publishes_nothing(tmp_path: Path) -> None:
    publisher = _RecordingPublisher()
    report = publish_skill_terminal_event(
        receipt=_failure_receipt(None),
        result_model=RESULT_MODEL,
        contract_path=_contract(tmp_path),
        lane="dev",
        resolve_target=lambda _lane: _target(),
        publisher=publisher,
    )
    assert report.outcome is EnumSkillTerminalPublishOutcome.SKIPPED_NO_RESULT
    assert publisher.calls == []


def test_an_unresolvable_lane_is_reported_not_raised(tmp_path: Path) -> None:
    def _refuse(_lane: str) -> ModelSkillTerminalPublishTarget:
        raise RuntimeError("no lane declaration on this host")

    publisher = _RecordingPublisher()
    report = publish_skill_terminal_event(
        receipt=_success_receipt(_verdict("verified")),
        result_model=RESULT_MODEL,
        contract_path=_contract(tmp_path),
        lane="dev",
        resolve_target=_refuse,
        publisher=publisher,
    )
    assert report.outcome is EnumSkillTerminalPublishOutcome.FAILED
    assert "no lane declaration" in report.detail
    assert publisher.calls == []


def _closed_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def test_an_unreachable_broker_fails_soft_within_the_timeout(tmp_path: Path) -> None:
    """AC-2, against the REAL Kafka publisher pointed at a closed port."""
    started = time.monotonic()
    report = publish_skill_terminal_event(
        receipt=_success_receipt(_verdict("verified")),
        result_model=RESULT_MODEL,
        contract_path=_contract(tmp_path),
        lane="dev",
        resolve_target=lambda _lane: _target(f"127.0.0.1:{_closed_port()}"),
        timeout_seconds=2.0,
    )
    elapsed = time.monotonic() - started
    assert report.outcome is EnumSkillTerminalPublishOutcome.FAILED
    assert elapsed < 15.0, (
        f"the publish must not hold the verification ({elapsed:.1f}s)"
    )


@pytest.mark.parametrize("dispatch_exit", [0, 1])
def test_the_skill_exit_code_is_the_dispatch_exit_code_when_the_publish_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, dispatch_exit: int
) -> None:
    """AC-2 at the command: a failed publish never changes the exit code."""
    monkeypatch.delenv("OMNI_HOME", raising=False)
    monkeypatch.setattr(cli_skill, "check_omnimarket_drift", lambda **_: None)
    contract = _contract(tmp_path)
    monkeypatch.setattr(cli_skill, "_resolve_packaged_contract", lambda _n: contract)
    verdict = _verdict("verified" if dispatch_exit == 0 else "failed")

    def _fake_receipt_mode(**kwargs: object) -> int:
        callback = kwargs["receipt_callback"]
        assert callable(callback)
        callback(
            _success_receipt(verdict)
            if dispatch_exit == 0
            else _failure_receipt(verdict)
        )
        assert kwargs["backend_overrides"] == {"event_bus": "inmemory"}
        return dispatch_exit

    def _refuse(_lane: str, **_: object) -> ModelSkillTerminalPublishTarget:
        raise RuntimeError("broker unreachable")

    monkeypatch.setattr(cli_skill, "run_receipt_mode", _fake_receipt_mode)
    monkeypatch.setattr(cli_skill, "resolve_skill_terminal_target", _refuse)

    result = CliRunner().invoke(
        run_skill_by_name,
        ["dod_verify", "OMN-19152", "--state-root", str(tmp_path / "state")],
    )
    assert result.exit_code == dispatch_exit, result.output
    assert "terminal publish failed" in result.output


def test_a_skill_with_no_declared_lane_never_publishes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("OMNI_HOME", raising=False)
    monkeypatch.setattr(cli_skill, "check_omnimarket_drift", lambda **_: None)
    monkeypatch.setattr(
        cli_skill, "_resolve_packaged_contract", lambda _n: _contract(tmp_path)
    )

    def _fake_receipt_mode(**kwargs: object) -> int:
        assert kwargs.get("receipt_callback") is None
        return 0

    def _must_not_resolve(_lane: str, **_: object) -> ModelSkillTerminalPublishTarget:
        raise AssertionError("a skill with no declared lane resolved one")

    monkeypatch.setattr(cli_skill, "run_receipt_mode", _fake_receipt_mode)
    monkeypatch.setattr(cli_skill, "resolve_skill_terminal_target", _must_not_resolve)
    result = CliRunner().invoke(
        run_skill_by_name,
        ["compliance_sweep", "--state-root", str(tmp_path / "state")],
    )
    assert result.exit_code == 0, result.output
