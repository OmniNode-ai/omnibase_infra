# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The dispatch gate rides out a lane rebuild instead of refusing it (OMN-18843).

Failure class ``delegate-consumer-rebind-window``: every runtime-affecting merge
to ``dev`` force-recreates the containers that bind the delegate command topic,
so for a bounded window the broker answers and no consumer group is bound.
Measured on the .201 dev lane on 2026-09-23 after omnibase_infra#3939: effects
container created 17:58:02.183Z, delegate-skill orchestrator group joined
17:59:40.214Z, 98.0 s. The gate used to refuse on the first empty answer, so
every rebuild became refused delegations.

The positive assertions (a rebinding lane is waited for) are paired with the
counter-assertions that keep the gate fail-closed: the wait is bounded, a lane
still unbound at the bound is refused, and an unanswerable broker is never
retried.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from omnibase_infra.backends.backend_probe import ConsumerGroupLivenessUnknownError
from omnibase_infra.cli import delegate_locus
from omnibase_infra.cli.delegate_locus import (
    REBIND_WINDOW_FAILURE_CLASS,
    DelegateLocusRefusedError,
    resolve_delegate_locus,
)
from omnibase_infra.enums.enum_delegate_locus import EnumDelegateLocus

pytestmark = pytest.mark.unit

_COMMAND_TOPIC = "onex.cmd.omnimarket.delegate-skill.v1"
_TERMINAL_TOPIC = "onex.evt.omnimarket.delegate-skill-completed.v1"
_BROKER = "broker.invalid:19092"
_GROUP = (
    "local.omnimarket.node_delegate_skill_orchestrator.consume.1.3.0"
    ".__i.runtime-effects.__t." + _COMMAND_TOPIC
)
# The post-#3939 window measured on the dev lane, created -> joined.
_MEASURED_WINDOW_SECONDS = 98.0


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
) -> list[str]:
    """Answer successive probes from *answers*, repeating the last one."""
    asked: list[str] = []

    def _fake(
        *, topic: str, bootstrap_servers: str | None, timeout: float
    ) -> tuple[str, ...]:
        asked.append(topic)
        return answers[min(len(asked), len(answers)) - 1]

    monkeypatch.setattr(delegate_locus, "live_consumer_groups", _fake)
    return asked


def _resolve(tmp_path: Path) -> Any:
    return resolve_delegate_locus(
        requested=EnumDelegateLocus.DEPLOYED_LANE,
        bus="kafka",
        kafka_bootstrap=_BROKER,
        contract_path=_contract(tmp_path),
        shared_bus_value="kafka",
    )


class TestRebindWindowIsWaitedFor:
    def test_bound_on_first_probe_does_not_wait(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, clock: _FakeClock
    ) -> None:
        asked = _answers(monkeypatch, [(_GROUP,)])
        decision = _resolve(tmp_path)
        assert asked == [_COMMAND_TOPIC]
        assert clock.sleeps == []
        assert decision.lane_consumer_groups == (_GROUP,)
        assert decision.consumer_bind_wait_seconds == 0.0

    def test_a_rebinding_lane_is_waited_for_and_the_wait_is_recorded(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, clock: _FakeClock
    ) -> None:
        asked = _answers(monkeypatch, [(), (), (_GROUP,)])
        decision = _resolve(tmp_path)
        assert len(asked) == 3
        assert clock.sleeps == [
            delegate_locus._REBIND_POLL_SECONDS,
            delegate_locus._REBIND_POLL_SECONDS,
        ]
        assert decision.lane_consumer_groups == (_GROUP,)
        assert decision.consumer_bind_wait_seconds == pytest.approx(
            2 * delegate_locus._REBIND_POLL_SECONDS
        )

    def test_the_measured_window_fits_inside_the_bound(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, clock: _FakeClock
    ) -> None:
        """A lane unbound for the whole measured 98 s window is still served."""
        unbound_probes = (
            int(_MEASURED_WINDOW_SECONDS // delegate_locus._REBIND_POLL_SECONDS) + 1
        )
        asked = _answers(monkeypatch, [()] * unbound_probes + [(_GROUP,)])
        decision = _resolve(tmp_path)
        assert len(asked) == unbound_probes + 1
        assert decision.consumer_bind_wait_seconds >= _MEASURED_WINDOW_SECONDS
        assert (
            decision.consumer_bind_wait_seconds <= delegate_locus._REBIND_WAIT_SECONDS
        )


class TestTheWaitStaysFailClosed:
    def test_a_lane_still_unbound_at_the_bound_is_refused(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, clock: _FakeClock
    ) -> None:
        asked = _answers(monkeypatch, [()])
        with pytest.raises(DelegateLocusRefusedError) as excinfo:
            _resolve(tmp_path)
        message = str(excinfo.value)
        assert "no live consumer group" in message
        assert REBIND_WINDOW_FAILURE_CLASS in message
        assert sum(clock.sleeps) <= delegate_locus._REBIND_WAIT_SECONDS
        expected_probes = (
            int(
                delegate_locus._REBIND_WAIT_SECONDS
                // delegate_locus._REBIND_POLL_SECONDS
            )
            + 1
        )
        assert len(asked) == expected_probes

    def test_unknown_liveness_is_refused_at_once_and_never_retried(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, clock: _FakeClock
    ) -> None:
        calls: list[int] = []

        def _raise(**_: object) -> tuple[str, ...]:
            calls.append(1)
            raise ConsumerGroupLivenessUnknownError("broker transport failure")

        monkeypatch.setattr(delegate_locus, "live_consumer_groups", _raise)
        with pytest.raises(DelegateLocusRefusedError, match="cannot confirm"):
            _resolve(tmp_path)
        assert calls == [1]
        assert clock.sleeps == []

    def test_the_bound_exceeds_the_measured_window(self) -> None:
        assert delegate_locus._REBIND_WAIT_SECONDS > _MEASURED_WINDOW_SECONDS

    def test_in_process_runs_never_probe_or_wait(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, clock: _FakeClock
    ) -> None:
        asked = _answers(monkeypatch, [()])
        decision = resolve_delegate_locus(
            requested=EnumDelegateLocus.IN_PROCESS,
            bus="inmemory",
            kafka_bootstrap=None,
            contract_path=_contract(tmp_path),
            shared_bus_value="kafka",
        )
        assert asked == []
        assert clock.sleeps == []
        assert decision.consumer_bind_wait_seconds == 0.0
