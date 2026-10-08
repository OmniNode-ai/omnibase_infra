# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-20646: bounded transport retries preserve the fail-closed dispatch gate."""

from __future__ import annotations

import os
import socket
from pathlib import Path
from typing import Any

import pytest
import yaml

from omnibase_infra.backends.backend_probe import (
    ConsumerGroupDescribeDeniedError,
    ConsumerGroupLivenessTransientError,
    ConsumerGroupLivenessUnknownError,
    ConsumerGroupSaslRefusedError,
    live_consumer_groups,
)
from omnibase_infra.cli import delegate_locus
from omnibase_infra.cli.delegate_locus import (
    DelegateDownstreamChainRefusedError,
    DelegateLocusAclRefusedError,
    DelegateLocusRefusedError,
    DelegateLocusSaslRefusedError,
    resolve_delegate_locus,
)
from omnibase_infra.cli.model_delegate_locus_decision import ModelDelegateLocusDecision
from omnibase_infra.enums.enum_delegate_locus import EnumDelegateLocus

pytestmark = pytest.mark.unit

_COMMAND_TOPIC = "onex.cmd.omnimarket.delegate-skill.v1"
_TERMINAL_TOPIC = "onex.evt.omnimarket.delegate-skill-completed.v1"
_CHAIN_TOPIC = "onex.cmd.synthetic.delegation-request.v1"
_CHAIN_COMPLETED = "onex.evt.synthetic.delegation-completed.v1"
_CHAIN_FAILED = "onex.evt.synthetic.delegation-failed.v1"
_BROKER = "broker.invalid:19092"
_GROUP = (
    "local.omnimarket.node_delegate_skill_orchestrator.consume.1.3.0"
    ".__i.runtime-effects.__t." + _COMMAND_TOPIC
)
_CHAIN_GROUP = f"local.runtime_config.delegation-orchestrator.__t.{_CHAIN_TOPIC}"


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


def _contract(tmp_path: Path, *, chain: bool = False) -> Path:
    contract: dict[str, Any] = {
        "name": "node_delegate_skill_orchestrator",
        "terminal_event": _TERMINAL_TOPIC,
        "event_bus": {
            "publish_topics": [_TERMINAL_TOPIC],
            "subscribe_topics": [_COMMAND_TOPIC],
        },
    }
    path = tmp_path / "nodes" / "node_delegate_skill_orchestrator" / "contract.yaml"
    path.parent.mkdir(parents=True)
    if chain:
        contract["delegation_runtime_dispatch"] = {
            "topics": {
                "command": _CHAIN_TOPIC,
                "completed": _CHAIN_COMPLETED,
                "failed": _CHAIN_FAILED,
            }
        }
        sibling = path.parent.parent / "node_delegation_orchestrator" / "contract.yaml"
        sibling.parent.mkdir()
        sibling.write_text(
            yaml.safe_dump(
                {
                    "name": "node_delegation_orchestrator",
                    "event_bus": {
                        "subscribe_topics": [
                            _CHAIN_TOPIC,
                            "onex.evt.synthetic.routing.v1",
                        ],
                        "publish_topics": [_CHAIN_COMPLETED, _CHAIN_FAILED],
                    },
                }
            ),
            encoding="utf-8",
        )
    path.write_text(yaml.safe_dump(contract), encoding="utf-8")
    return path


def _resolve(contract: Path) -> ModelDelegateLocusDecision:
    return resolve_delegate_locus(
        requested=EnumDelegateLocus.DEPLOYED_LANE,
        bus="kafka",
        kafka_bootstrap=_BROKER,
        contract_path=contract,
        shared_bus_value="kafka",
    )


def _answers(
    monkeypatch: pytest.MonkeyPatch,
    answers: list[tuple[str, ...] | Exception],
    *,
    chain: bool = False,
) -> list[dict[str, Any]]:
    asked: list[dict[str, Any]] = []

    def ask(**kwargs: Any) -> tuple[str, ...]:
        asked.append(kwargs)
        answer = answers[min(len(asked), len(answers)) - 1]
        if isinstance(answer, Exception):
            raise answer
        return answer

    name = "live_chain_consumer_groups" if chain else "live_consumer_groups"
    monkeypatch.setattr(delegate_locus, name, ask)
    return asked


def test_slow_but_alive_first_hop(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    clock: _FakeClock,
    caplog: pytest.LogCaptureFixture,
) -> None:
    asked = _answers(
        monkeypatch, [ConsumerGroupLivenessTransientError("slow")] * 2 + [(_GROUP,)]
    )
    decision = _resolve(_contract(tmp_path))
    assert decision.lane_consumer_groups == (_GROUP,)
    assert len(asked) == 3
    assert clock.sleeps == [2.0, 4.0]
    assert "first-hop liveness" in caplog.text
    assert "attempt 1 of 3" in caplog.text
    assert "attempt 2 of 3" in caplog.text
    assert "slow; asking again in 4.0 s" in caplog.text


def test_dead_first_hop_refuses_after_three_attempts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    clock: _FakeClock,
) -> None:
    asked = _answers(monkeypatch, [ConsumerGroupLivenessTransientError("down")])
    with pytest.raises(
        DelegateLocusRefusedError, match=r"3 attempts over 6\.0 s"
    ) as caught:
        _resolve(_contract(tmp_path))
    assert isinstance(caught.value.__cause__, ConsumerGroupLivenessTransientError)
    assert len(asked) == 3
    assert clock.sleeps == [2.0, 4.0]


@pytest.mark.parametrize(
    ("failure", "refusal"),
    [
        (ConsumerGroupLivenessUnknownError("decode"), DelegateLocusRefusedError),
        (
            ConsumerGroupSaslRefusedError(
                bootstrap_servers=_BROKER, principal="synthetic"
            ),
            DelegateLocusSaslRefusedError,
        ),
        (
            ConsumerGroupDescribeDeniedError(
                group_ids=(_GROUP,), bootstrap_servers=_BROKER, principal="synthetic"
            ),
            DelegateLocusAclRefusedError,
        ),
    ],
)
def test_nontransport_failures_refuse_after_one_ask(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    clock: _FakeClock,
    failure: Exception,
    refusal: type[Exception],
) -> None:
    asked = _answers(monkeypatch, [failure])
    with pytest.raises(refusal):
        _resolve(_contract(tmp_path))
    assert len(asked) == 1
    assert clock.sleeps == []


def test_slow_but_alive_downstream_chain(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    clock: _FakeClock,
) -> None:
    first_asked = _answers(monkeypatch, [(_GROUP,)])
    chain_asked = _answers(
        monkeypatch,
        [ConsumerGroupLivenessTransientError("slow"), (_CHAIN_GROUP,)],
        chain=True,
    )
    decision = _resolve(_contract(tmp_path, chain=True))
    assert decision.downstream_consumer_groups == (_CHAIN_GROUP,)
    assert len(first_asked) == 1
    assert len(chain_asked) == 2
    assert clock.sleeps == [2.0]


def test_dead_downstream_chain_refuses_after_three_attempts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    clock: _FakeClock,
) -> None:
    first_asked = _answers(monkeypatch, [(_GROUP,)])
    chain_asked = _answers(
        monkeypatch, [ConsumerGroupLivenessTransientError("down")], chain=True
    )
    with pytest.raises(DelegateDownstreamChainRefusedError, match="3 attempts"):
        _resolve(_contract(tmp_path, chain=True))
    assert len(first_asked) == 1
    assert len(chain_asked) == 3
    assert clock.sleeps == [2.0, 4.0]


def test_real_dead_broker_is_transient(monkeypatch: pytest.MonkeyPatch) -> None:
    # A real aiokafka bootstrap failure, without replacing any client methods.
    for name in list(os.environ):
        if name.startswith("KAFKA_SASL_") or name == "KAFKA_SECURITY_PROTOCOL":
            monkeypatch.delenv(name, raising=False)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as reservation:
        reservation.bind(("127.0.0.1", 0))
        port = reservation.getsockname()[1]
    with pytest.raises(ConsumerGroupLivenessTransientError):
        live_consumer_groups(
            topic=_COMMAND_TOPIC, bootstrap_servers=f"127.0.0.1:{port}", timeout=1.0
        )
