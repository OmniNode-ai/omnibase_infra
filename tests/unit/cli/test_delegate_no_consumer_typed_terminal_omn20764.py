# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""A delegation with no live consumer ends in a terminal that names it (OMN-20764).

Fingerprint ``fr-onex-delegate-no-listener-for-delegation-orchestrator``: on
2026-10-04, 2026-10-08 and 2026-10-11 a delegate call whose command topic had
no bound consumer group ended as a bare "timed out after 180 s" in the hourly
tick's log, or as a refusal whose only carrier of the topic and the absent
consumer was prose.

The refusal text already named both. What was missing is the typed terminal: the
written ``transport_refusal`` left ``command_topic`` empty and had no field for
the consumer that was absent, so a reader (the hourly tick's classifier) had
nothing to read but a sentence. These tests pin that the topic and the absent
consumer are typed fields of the terminal, written within the rebind bound.

Positive controls: a bound consumer does not refuse, and a refusal that is not
a missing consumer (an unanswerable broker) leaves the new fields empty rather
than naming a consumer it never established was absent.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml
from click.testing import CliRunner

from omnibase_infra.backends.backend_probe import ConsumerGroupLivenessUnknownError
from omnibase_infra.backends.model_consumer_group_owner import ModelConsumerGroupOwner
from omnibase_infra.cli import cli_delegate, delegate_locus
from omnibase_infra.cli.cli_delegate import delegate_command
from omnibase_infra.cli.delegate_locus import (
    DelegateLocusRefusedError,
    DelegateNoConsumerRefusedError,
    resolve_delegate_locus,
)
from omnibase_infra.cli.model_delegate_locus_decision import ModelDelegateLocusDecision
from omnibase_infra.enums.enum_delegate_locus import EnumDelegateLocus

pytestmark = pytest.mark.unit

_COMMAND_TOPIC = "onex.cmd.omnimarket.delegate-skill.v1"
_TERMINAL_TOPIC = "onex.evt.omnimarket.delegate-skill-completed.v1"
_BROKER = "broker.invalid:19092"
_ABSENT_CONSUMER = "omnimarket.node_delegate_skill_orchestrator"
_LIVE_GROUP = (
    "local.omnimarket.node_delegate_skill_orchestrator.consume.1.3.0"
    ".__i.runtime-effects.__t." + _COMMAND_TOPIC
)


class _FakeClock:
    """Time moves only when the code under test sleeps."""

    def __init__(self) -> None:
        self.now = 1000.0

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds


@pytest.fixture(autouse=True)
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
        "handler": {
            "module": "tests.fixtures.handler_correlated_noop",
            "class": "HandlerCorrelatedNoop",
            "input_model": (
                "tests.fixtures.handler_correlated_noop.ModelCorrelatedNoopRequest"
            ),
        },
        "handler_routing": {
            "default_handler": (
                "tests.fixtures.handler_correlated_noop:HandlerCorrelatedNoop"
            )
        },
    }
    path = tmp_path / "contract.yaml"
    path.write_text(yaml.safe_dump(contract), encoding="utf-8")
    return path


def _groups(monkeypatch: pytest.MonkeyPatch, groups: tuple[str, ...]) -> None:
    def _fake(
        *,
        topic: str,
        bootstrap_servers: str | None,
        timeout: float,
        owner: ModelConsumerGroupOwner | None = None,
    ) -> tuple[str, ...]:
        return groups

    monkeypatch.setattr(delegate_locus, "live_consumer_groups", _fake)


def _resolve(contract: Path) -> ModelDelegateLocusDecision:
    return resolve_delegate_locus(
        requested=EnumDelegateLocus.DEPLOYED_LANE,
        bus="kafka",
        kafka_bootstrap=_BROKER,
        contract_path=contract,
        shared_bus_value="kafka",
    )


def test_delegate_no_consumer_typed_terminal_error_carries_topic_and_consumer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, clock: _FakeClock
) -> None:
    _groups(monkeypatch, ())
    with pytest.raises(DelegateNoConsumerRefusedError) as caught:
        _resolve(_contract(tmp_path))
    assert isinstance(caught.value, DelegateLocusRefusedError)
    assert caught.value.command_topic == _COMMAND_TOPIC
    assert caught.value.consumer == _ABSENT_CONSUMER
    # Within its budget: the wait is the bound for the rebind class, not a hang.
    assert 0 < caught.value.waited_seconds <= delegate_locus._REBIND_WAIT_SECONDS
    assert clock.now - 1000.0 <= delegate_locus._REBIND_WAIT_SECONDS


def test_delegate_no_consumer_typed_terminal_a_bound_consumer_does_not_refuse(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _groups(monkeypatch, (_LIVE_GROUP,))
    decision = _resolve(_contract(tmp_path))
    assert decision.lane_consumer_groups == (_LIVE_GROUP,)


def test_delegate_no_consumer_typed_terminal_unanswerable_broker_names_no_consumer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Counter-assertion: an unknown answer is not the claim "no consumer"."""

    def _unknown(**_: object) -> tuple[str, ...]:
        raise ConsumerGroupLivenessUnknownError("broker did not answer")

    monkeypatch.setattr(delegate_locus, "live_consumer_groups", _unknown)
    with pytest.raises(DelegateLocusRefusedError) as caught:
        _resolve(_contract(tmp_path))
    assert not isinstance(caught.value, DelegateNoConsumerRefusedError)


def test_delegate_no_consumer_typed_terminal_receipt_names_topic_and_consumer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The written receipt.json, the surface every lane reads, types both."""
    monkeypatch.delenv("OMNI_HOME", raising=False)
    monkeypatch.setattr(cli_delegate, "check_omnimarket_drift", lambda **_: None)
    contract = _contract(tmp_path)
    monkeypatch.setattr(cli_delegate, "_resolve_packaged_contract", lambda _n: contract)
    monkeypatch.setenv("ONEX_ARTIFACT_STORE_ROOT", str(tmp_path / "artifacts"))
    _groups(monkeypatch, ())
    state_root = tmp_path / "state"
    result = CliRunner().invoke(
        delegate_command,
        [
            "Reply with exactly the word READY",
            "--json",
            "--bus",
            "kafka",
            "--kafka-bootstrap",
            _BROKER,
            "--locus",
            "deployed-lane",
            "--state-root",
            str(state_root),
            "--emit-socket",
            str(tmp_path / "no-daemon.sock"),
        ],
        catch_exceptions=False,
    )
    assert result.exit_code != 0
    written = sorted(state_root.glob("runs/*/receipt.json"))
    assert len(written) == 1
    receipt = json.loads(written[0].read_text(encoding="utf-8"))
    refusal = receipt["transport_refusal"]
    assert refusal["reason"] == "locus_probe_refused"
    assert refusal["transport_error_type"] == "DelegateNoConsumerRefusedError"
    assert refusal["command_topic"] == _COMMAND_TOPIC
    assert refusal["absent_consumer"] == _ABSENT_CONSUMER
    assert refusal["elapsed_seconds"] >= 0.0
    assert receipt["terminal_class"] == "transport"
    assert _COMMAND_TOPIC in receipt["failure_reason"]
