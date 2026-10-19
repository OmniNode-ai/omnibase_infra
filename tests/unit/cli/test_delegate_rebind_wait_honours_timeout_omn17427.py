# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The rebind probe and terminal wait share the caller's timeout (OMN-17427)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from omnibase_infra.backends.model_consumer_group_owner import ModelConsumerGroupOwner
from omnibase_infra.cli import cli_delegate, delegate_locus
from omnibase_infra.cli.delegate_locus import (
    DelegateLocusRefusedError,
    rebind_wait_for_timeout,
    resolve_delegate_locus,
)
from omnibase_infra.cli.model_delegate_locus_decision import ModelDelegateLocusDecision
from omnibase_infra.enums.enum_delegate_locus import EnumDelegateLocus

pytestmark = pytest.mark.unit

_COMMAND_TOPIC = "onex.cmd.omnimarket.delegate-skill.v1"
_TERMINAL_TOPIC = "onex.evt.omnimarket.delegate-skill-completed.v1"
_BROKER = "broker.invalid:19092"
_GROUP = (
    "local.omnimarket.node_delegate_skill_orchestrator.consume.1.3.0"
    ".__i.runtime-effects.__t." + _COMMAND_TOPIC
)


class _FakeClock:
    """Time moves only when the code under test sleeps."""

    def __init__(self) -> None:
        self.now = 1000.0
        self.sleeps: list[float] = []

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> _FakeClock:
    fake = _FakeClock()
    monkeypatch.setattr(delegate_locus, "_monotonic", fake.monotonic)
    monkeypatch.setattr(delegate_locus, "_sleep", fake.sleep)
    return fake


def _contract(tmp_path: Path) -> Path:
    contract: dict[str, Any] = {
        "name": "node_delegate_skill_orchestrator",
        "terminal_event": _TERMINAL_TOPIC,
        "event_bus": {
            "publish_topics": [_TERMINAL_TOPIC],
            "subscribe_topics": [_COMMAND_TOPIC],
        },
    }
    path = tmp_path / "contract.yaml"
    path.write_text(yaml.safe_dump(contract), encoding="utf-8")
    return path


def _answers(
    monkeypatch: pytest.MonkeyPatch, answers: list[tuple[str, ...]]
) -> list[dict[str, object]]:
    """Answer successive probes from *answers*, repeating the last one."""
    asked: list[dict[str, object]] = []

    def _fake(
        *,
        topic: str,
        bootstrap_servers: str | None,
        timeout: float,
        owner: ModelConsumerGroupOwner | None = None,
    ) -> tuple[str, ...]:
        asked.append({"topic": topic, "owner": owner})
        return answers[min(len(asked), len(answers)) - 1]

    monkeypatch.setattr(delegate_locus, "live_consumer_groups", _fake)
    return asked


def _resolve(
    tmp_path: Path, rebind_wait_seconds: float | None
) -> ModelDelegateLocusDecision:
    return resolve_delegate_locus(
        requested=EnumDelegateLocus.DEPLOYED_LANE,
        bus="kafka",
        kafka_bootstrap=_BROKER,
        contract_path=_contract(tmp_path),
        shared_bus_value="kafka",
        rebind_wait_seconds=rebind_wait_seconds,
    )


def test_rebind_wait_for_timeout() -> None:
    assert rebind_wait_for_timeout(None) is None
    assert rebind_wait_for_timeout(120) == 60.0
    assert rebind_wait_for_timeout(900) == delegate_locus._REBIND_WAIT_SECONDS


def test_rebind_wait_reads_globals_at_call_time(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(delegate_locus, "_REBIND_WAIT_SECONDS", 45.0)
    assert rebind_wait_for_timeout(120) == 45.0
    monkeypatch.setattr(delegate_locus, "_REBIND_SHARE_OF_TIMEOUT", 0.25)
    assert rebind_wait_for_timeout(120) == 30.0


def test_unbound_lane_refuses_within_half_the_callers_timeout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, clock: _FakeClock
) -> None:
    _answers(monkeypatch, [()])
    with pytest.raises(DelegateLocusRefusedError, match="half the caller's --timeout"):
        _resolve(tmp_path, rebind_wait_seconds=60)
    assert sum(clock.sleeps) <= 60


def test_lane_binding_on_third_probe_returns_the_wait(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, clock: _FakeClock
) -> None:
    asked = _answers(monkeypatch, [(), (), (_GROUP,)])
    decision = _resolve(tmp_path, rebind_wait_seconds=60)
    assert len(asked) == 3
    assert decision.lane_consumer_groups == (_GROUP,)
    assert (
        decision.consumer_bind_wait_seconds == 2 * delegate_locus._REBIND_POLL_SECONDS
    )
    assert sum(clock.sleeps) == decision.consumer_bind_wait_seconds


@pytest.mark.parametrize("rebind_wait_seconds", [None, 900.0])
def test_default_window_is_retained_and_larger_bounds_are_clamped(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    clock: _FakeClock,
    rebind_wait_seconds: float | None,
) -> None:
    _answers(monkeypatch, [()])
    with pytest.raises(DelegateLocusRefusedError) as excinfo:
        _resolve(tmp_path, rebind_wait_seconds=rebind_wait_seconds)
    assert 60 < sum(clock.sleeps) <= delegate_locus._REBIND_WAIT_SECONDS
    assert sum(clock.sleeps) == pytest.approx(delegate_locus._REBIND_WAIT_SECONDS)
    assert "half the caller's --timeout" not in str(excinfo.value)


@pytest.mark.parametrize(
    ("timeout", "probe_seconds", "expected"),
    [(None, 50.0, None), (120, 50.2, 69), (120, 500.0, 1)],
)
def test_timeout_left_after_probe(
    timeout: int | None, probe_seconds: float, expected: int | None
) -> None:
    assert cli_delegate._timeout_left_after_probe(timeout, probe_seconds) == expected
