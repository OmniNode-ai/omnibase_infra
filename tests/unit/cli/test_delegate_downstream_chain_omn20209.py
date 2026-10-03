# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-20209: the first hop was proven live while its downstream chain was dead.

The deployed delegate-skill orchestrator accepted the command, then waited
300 s for the downstream consumer and returned a timeout with attempts=[].
Callers exhausted their budgets before receiving that terminal.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml
from click.testing import CliRunner

from omnibase_infra.backends.backend_probe import (
    ConsumerGroupDescribeDeniedError,
    ConsumerGroupLivenessUnknownError,
)
from omnibase_infra.cli import cli_delegate, delegate_locus
from omnibase_infra.cli.model_delegate_locus_decision import ModelDelegateLocusDecision
from omnibase_infra.enums.enum_delegate_locus import EnumDelegateLocus

pytestmark = pytest.mark.unit

_COMMAND = "onex.cmd.synthetic.delegate-skill.v1"
_TERMINAL = "onex.evt.synthetic.delegate-skill-completed.v1"
_REQUEST = "onex.cmd.synthetic.delegation-request.v1"
_COMPLETED = "onex.evt.synthetic.delegation-completed.v1"
_FAILED = "onex.evt.synthetic.delegation-failed.v1"
_SUBSCRIBE = (
    _REQUEST,
    "onex.evt.synthetic.routing-decision.v1",
    "onex.evt.synthetic.inference-response.v1",
)
_BROKER = "broker.invalid:19092"
_FIRST_GROUP = f"local.omnimarket.node_delegate_skill_orchestrator.__t.{_COMMAND}"
_CHAIN_GROUP = f"local.runtime_config.delegation-orchestrator.__t.{_REQUEST}"


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


def _write_contract(nodes: Path, name: str, data: dict[str, Any]) -> Path:
    path = nodes / name / "contract.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump({"name": name, **data}), encoding="utf-8")
    return path


@pytest.fixture
def contract(tmp_path: Path) -> Path:
    nodes = tmp_path / "nodes"
    path = _write_contract(
        nodes,
        "node_delegate_skill_orchestrator",
        {
            "terminal_event": _TERMINAL,
            "event_bus": {
                "subscribe_topics": [_COMMAND],
                "publish_topics": [_TERMINAL],
            },
            "delegation_runtime_dispatch": {
                "topics": {
                    "command": _REQUEST,
                    "completed": _COMPLETED,
                    "failed": _FAILED,
                }
            },
        },
    )
    _write_contract(
        nodes,
        "node_delegation_orchestrator",
        {
            "event_bus": {
                "subscribe_topics": list(_SUBSCRIBE),
                "publish_topics": [_COMPLETED, _FAILED],
            }
        },
    )
    _write_contract(
        nodes,
        "node_ledger_projection",
        {"event_bus": {"subscribe_topics": [_REQUEST], "publish_topics": []}},
    )
    return path


def _probes(
    monkeypatch: pytest.MonkeyPatch,
    chain_answers: list[tuple[str, ...]],
    first: tuple[str, ...] = (_FIRST_GROUP,),
) -> list[dict[str, Any]]:
    asked: list[dict[str, Any]] = []
    monkeypatch.setattr(delegate_locus, "live_consumer_groups", lambda **_: first)

    def chain(**kwargs: Any) -> tuple[str, ...]:
        asked.append(kwargs)
        return chain_answers[min(len(asked), len(chain_answers)) - 1]

    # Allows the original first-hop-only implementation to demonstrate RED.
    monkeypatch.setattr(
        delegate_locus, "live_chain_consumer_groups", chain, raising=False
    )
    return asked


def _resolve(contract: Path) -> ModelDelegateLocusDecision:
    return delegate_locus.resolve_delegate_locus(
        requested=EnumDelegateLocus.DEPLOYED_LANE,
        bus="kafka",
        kafka_bootstrap=_BROKER,
        contract_path=contract,
        shared_bus_value="kafka",
    )


def test_first_hop_live_downstream_never_live(
    contract: Path, monkeypatch: pytest.MonkeyPatch, clock: _FakeClock
) -> None:
    asked = _probes(monkeypatch, [()])
    with pytest.raises(delegate_locus.DelegateLocusRefusedError) as caught:
        _resolve(contract)
    assert isinstance(caught.value, delegate_locus.DelegateDownstreamChainRefusedError)
    message = str(caught.value)
    assert delegate_locus.DOWNSTREAM_CHAIN_STAGE in message
    assert _REQUEST in message
    assert "node_delegation_orchestrator" in message
    assert _FIRST_GROUP in message
    assert delegate_locus.REBIND_WINDOW_FAILURE_CLASS in message
    assert "execution budget" in message
    assert "--bus inmemory --locus in-process" in message
    assert sum(clock.sleeps) <= delegate_locus._REBIND_WAIT_SECONDS
    assert (
        len(asked)
        == int(
            delegate_locus._REBIND_WAIT_SECONDS / delegate_locus._REBIND_POLL_SECONDS
        )
        + 1
    )


def test_both_hops_live_records_full_chain_evidence(
    contract: Path, monkeypatch: pytest.MonkeyPatch, clock: _FakeClock
) -> None:
    asked = _probes(monkeypatch, [(_CHAIN_GROUP,)])
    decision = _resolve(contract)
    assert decision.downstream_consumer_groups == (_CHAIN_GROUP,)
    assert decision.downstream_command_topic == _REQUEST
    assert asked == [
        {
            "command_topic": _REQUEST,
            "subscribe_topics": _SUBSCRIBE,
            "bootstrap_servers": _BROKER,
            "timeout": delegate_locus._LIVENESS_TIMEOUT_SECONDS,
        }
    ]
    assert clock.sleeps == []


def test_downstream_rebind_waits_in_same_window(
    contract: Path,
    monkeypatch: pytest.MonkeyPatch,
    clock: _FakeClock,
    caplog: pytest.LogCaptureFixture,
) -> None:
    _probes(monkeypatch, [(), (), (_CHAIN_GROUP,)])
    decision = _resolve(contract)
    assert decision.downstream_consumer_groups == (_CHAIN_GROUP,)
    assert clock.sleeps == [delegate_locus._REBIND_POLL_SECONDS] * 2
    assert decision.consumer_bind_wait_seconds == sum(clock.sleeps)
    assert delegate_locus.DOWNSTREAM_CHAIN_STAGE in caplog.text
    assert "rebinding, not down" in caplog.text


def test_no_declared_chain_never_probes_downstream(
    contract: Path, monkeypatch: pytest.MonkeyPatch, clock: _FakeClock
) -> None:
    raw = yaml.safe_load(contract.read_text())
    del raw["delegation_runtime_dispatch"]
    contract.write_text(yaml.safe_dump(raw))
    asked = _probes(monkeypatch, [()])
    decision = _resolve(contract)
    assert decision.downstream_consumer_groups == ()
    assert decision.downstream_command_topic == ""
    assert asked == []


def test_ambiguous_contract_refuses_before_any_probe(
    contract: Path, monkeypatch: pytest.MonkeyPatch, clock: _FakeClock
) -> None:
    _write_contract(
        contract.parent.parent,
        "node_duplicate",
        {"event_bus": {"subscribe_topics": [_REQUEST], "publish_topics": [_COMPLETED]}},
    )

    def unexpected(**_: Any) -> tuple[str, ...]:
        pytest.fail("ambiguous chain must refuse before probing")

    monkeypatch.setattr(delegate_locus, "live_consumer_groups", unexpected)
    with pytest.raises(delegate_locus.DelegateLocusRefusedError) as caught:
        _resolve(contract)
    assert _REQUEST in str(caught.value)
    assert "2" in str(caught.value)
    assert "node_duplicate" in str(caught.value)
    assert "node_delegation_orchestrator" in str(caught.value)
    assert clock.sleeps == []


@pytest.mark.parametrize("denied", [False, True])
def test_downstream_unanswerable_refuses_at_once(
    contract: Path, monkeypatch: pytest.MonkeyPatch, clock: _FakeClock, denied: bool
) -> None:
    _probes(monkeypatch, [()])

    def unknown(**_: Any) -> tuple[str, ...]:
        if denied:
            raise ConsumerGroupDescribeDeniedError(
                group_ids=("g",), bootstrap_servers=_BROKER, principal="p"
            )
        raise ConsumerGroupLivenessUnknownError("cannot ask broker")

    monkeypatch.setattr(delegate_locus, "live_chain_consumer_groups", unknown)
    expected = (
        delegate_locus.DelegateLocusAclRefusedError
        if denied
        else delegate_locus.DelegateDownstreamChainRefusedError
    )
    with pytest.raises(expected) as caught:
        _resolve(contract)
    assert delegate_locus.DOWNSTREAM_CHAIN_STAGE in str(caught.value)
    assert _REQUEST in str(caught.value)
    assert clock.sleeps == []


def test_first_hop_unbound_preserves_existing_refusal(
    contract: Path, monkeypatch: pytest.MonkeyPatch, clock: _FakeClock
) -> None:
    asked = _probes(monkeypatch, [(_CHAIN_GROUP,)], first=())
    with pytest.raises(delegate_locus.DelegateLocusRefusedError) as caught:
        _resolve(contract)
    assert type(caught.value) is delegate_locus.DelegateLocusRefusedError
    assert "no live consumer group" in str(caught.value)
    assert _COMMAND in str(caught.value)
    assert asked == []


@pytest.mark.parametrize("refused", [False, True])
def test_cli_prints_dispatch_or_refusal_ids(
    contract: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, refused: bool
) -> None:
    calls: list[dict[str, Any]] = []

    def receipt(**kwargs: Any) -> int:
        calls.append(kwargs)
        return 0

    def resolve(**_: Any) -> ModelDelegateLocusDecision:
        if refused:
            raise delegate_locus.DelegateDownstreamChainRefusedError(
                f"{delegate_locus.DOWNSTREAM_CHAIN_STAGE}: {_REQUEST} unavailable"
            )
        return ModelDelegateLocusDecision(
            locus=EnumDelegateLocus.DEPLOYED_LANE,
            resolved_from="test",
            orchestrator_contract=str(contract),
            orchestrator_distribution="synthetic",
            command_topic=_COMMAND,
            broker=_BROKER,
            lane_consumer_groups=(_FIRST_GROUP,),
        )

    monkeypatch.delenv("OMNI_HOME", raising=False)
    monkeypatch.setattr(cli_delegate, "check_omnimarket_drift", lambda **_: None)
    monkeypatch.setattr(cli_delegate, "_resolve_packaged_contract", lambda _: contract)
    monkeypatch.setattr(cli_delegate, "run_receipt_mode", receipt)
    monkeypatch.setattr(cli_delegate, "resolve_delegate_locus", resolve)
    result = CliRunner().invoke(
        cli_delegate.delegate_command,
        [
            "probe",
            "--task-type",
            "research",
            "--bus",
            "kafka",
            "--locus",
            "deployed-lane",
            "--kafka-bootstrap",
            _BROKER,
            "--state-root",
            str(tmp_path / "state"),
            "--timeout",
            "5",
        ],
    )
    if refused:
        assert result.exit_code != 0
        assert "run " in result.output
        assert "correlation " in result.output
        assert delegate_locus.DOWNSTREAM_CHAIN_STAGE in result.output
        assert calls == []
        # The ids printed must identify the refusal artifacts this invocation wrote.
        refusal_path = next((tmp_path / "state" / "runs").glob("*/receipt.json"))
        refusal = json.loads(refusal_path.read_text())
        assert refusal_path.parent.name in result.output
        assert str(refusal["transport_refusal"]["correlation_id"]) in result.output
    else:
        assert result.exit_code == 0, result.output
        assert "dispatching: run " in result.stderr
        assert str(calls[0]["expected_correlation_id"]) in result.stderr
        assert _COMMAND in result.stderr
        assert _BROKER in result.stderr
