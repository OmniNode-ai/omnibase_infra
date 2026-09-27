# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""``onex delegate`` rides out a lane rebind end to end through ``run_delegate`` (OMN-18843).

The unit tests pin the gate. This pins the path a caller actually takes: the
CLI entry resolves the locus, the gate waits while nothing is bound, and the
decision that reaches receipt mode is a dispatched run that records how long
it waited. The counter-case is the refusal at the bound: nothing is
dispatched and nothing runs here.

The broker is the only stub. The live listing on the .201 dev lane was
exercised against a real broker in omnibase_infra#4028's lab readback; this
file keeps the wiring from the CLI entry to receipt mode honest in CI.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import click
import pytest
import yaml

from omnibase_infra.cli import cli_delegate, delegate_locus
from omnibase_infra.cli.cli_delegate import run_delegate
from omnibase_infra.enums.enum_delegate_locus import EnumDelegateLocus

pytestmark = pytest.mark.integration

_COMMAND_TOPIC = "onex.cmd.omnimarket.delegate-skill.v1"
_TERMINAL_TOPIC = "onex.evt.omnimarket.delegate-skill-completed.v1"
# The committed stand-in task-class contract: this repo does not depend on
# omnimarket, so its packaged contract is absent in the test venv (OMN-18305).
_STAND_IN_TASK_CLASS_CONTRACT = (
    Path(__file__).resolve().parents[2]
    / "fixtures"
    / "delegation"
    / "omn18305"
    / "task_class_contracts_vocabulary.yaml"
)
_GROUP = (
    "local.omnimarket.node_delegate_skill_orchestrator.consume.1.3.0"
    ".__i.runtime-effects.__t." + _COMMAND_TOPIC
)


@pytest.fixture
def harness(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    contract = tmp_path / "contract.yaml"
    contract.write_text(
        yaml.safe_dump(
            {
                "name": "node_delegate_skill_orchestrator",
                "terminal_event": _TERMINAL_TOPIC,
                "event_bus": {
                    "publish_topics": [_TERMINAL_TOPIC],
                    "subscribe_topics": [_COMMAND_TOPIC],
                },
            }
        ),
        encoding="utf-8",
    )
    dispatched: list[dict[str, object]] = []

    def _fake_receipt_mode(**kwargs: object) -> int:
        dispatched.append(kwargs)
        return 0

    monkeypatch.delenv("OMNI_HOME", raising=False)
    monkeypatch.setattr(cli_delegate, "check_omnimarket_drift", lambda **_: None)
    monkeypatch.setattr(cli_delegate, "_resolve_packaged_contract", lambda _n: contract)
    monkeypatch.setattr(cli_delegate, "run_receipt_mode", _fake_receipt_mode)
    monkeypatch.setattr(
        cli_delegate,
        "resolve_task_class_contract_path",
        lambda: _STAND_IN_TASK_CLASS_CONTRACT,
    )

    clock = {"now": 0.0, "slept": 0.0}

    def _sleep(seconds: float) -> None:
        clock["now"] += seconds
        clock["slept"] += seconds

    monkeypatch.setattr(delegate_locus, "_monotonic", lambda: clock["now"])
    monkeypatch.setattr(delegate_locus, "_sleep", _sleep)
    return {"dispatched": dispatched, "clock": clock, "state": tmp_path / "state"}


def _answer(monkeypatch: pytest.MonkeyPatch, answers: list[tuple[str, ...]]) -> None:
    asked: list[int] = []

    def _fake(
        *, topic: str, bootstrap_servers: str | None, timeout: float
    ) -> tuple[str, ...]:
        assert topic == _COMMAND_TOPIC
        asked.append(1)
        return answers[min(len(asked), len(answers)) - 1]

    monkeypatch.setattr(delegate_locus, "live_consumer_groups", _fake)


def _delegate(state_root: Path, tmp_path: Path) -> int:
    return run_delegate(
        prompt="probe",
        task_type="research",
        max_tokens=None,
        bus="kafka",
        locus=EnumDelegateLocus.DEPLOYED_LANE,
        kafka_bootstrap="broker.invalid:19092",
        state_root=state_root,
        timeout=5,
        verbose=False,
        emit_socket=tmp_path / "emit.sock",
    )


def test_a_rebinding_lane_is_dispatched_with_the_wait_recorded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, harness: dict[str, Any]
) -> None:
    _answer(monkeypatch, [(), (), (), (_GROUP,)])

    assert _delegate(harness["state"], tmp_path) == 0

    assert len(harness["dispatched"]) == 1
    call = harness["dispatched"][0]
    assert call["host_handlers"] is False
    decision = call["locus_decision"]
    assert decision.locus is EnumDelegateLocus.DEPLOYED_LANE
    assert decision.lane_consumer_groups == (_GROUP,)
    assert decision.consumer_bind_wait_seconds == pytest.approx(
        3 * delegate_locus._REBIND_POLL_SECONDS
    )


def test_a_lane_still_unbound_at_the_bound_dispatches_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, harness: dict[str, Any]
) -> None:
    _answer(monkeypatch, [()])

    with pytest.raises(click.ClickException, match="delegate-consumer-rebind-window"):
        _delegate(harness["state"], tmp_path)

    assert harness["dispatched"] == []
    assert harness["clock"]["slept"] <= delegate_locus._REBIND_WAIT_SECONDS
