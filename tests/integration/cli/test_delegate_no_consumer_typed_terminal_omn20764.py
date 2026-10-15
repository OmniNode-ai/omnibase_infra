# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""End-to-end CLI coverage for the no-consumer typed terminal (OMN-20764).

The unit module ``tests/unit/cli/test_delegate_no_consumer_typed_terminal_omn20764.py``
pins the raised error. This one goes through the real ``onex delegate`` command
-- option parsing, the locus probe, the refusal path and the receipt writer -- and
reads back the ``receipt.json`` a lane or the hourly tick's held classifier reads.
It also loads that receipt's ``transport_refusal`` through the typed model, so a
field the writer and the model disagree on fails here.

Only the broker's consumer-group answer is replaced. Everything downstream of it
runs for real.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml
from click.testing import CliRunner, Result

from omnibase_infra.backends.backend_probe import ConsumerGroupLivenessUnknownError
from omnibase_infra.backends.model_consumer_group_owner import ModelConsumerGroupOwner
from omnibase_infra.cli import cli_delegate, delegate_locus
from omnibase_infra.cli.cli_delegate import delegate_command
from omnibase_infra.cli.model_delegate_transport_refusal import (
    ModelDelegateTransportRefusal,
)

pytestmark = pytest.mark.integration

_COMMAND_TOPIC = "onex.cmd.omnimarket.delegate-skill.v1"
_TERMINAL_TOPIC = "onex.evt.omnimarket.delegate-skill-completed.v1"
_BROKER = "broker.invalid:19092"
_ABSENT_CONSUMER = "omnimarket.node_delegate_skill_orchestrator"


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


@pytest.fixture(autouse=True)
def _delegate_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
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
    monkeypatch.delenv("OMNI_HOME", raising=False)
    monkeypatch.setattr(cli_delegate, "check_omnimarket_drift", lambda **_: None)
    monkeypatch.setattr(cli_delegate, "_resolve_packaged_contract", lambda _n: path)
    monkeypatch.setenv("ONEX_ARTIFACT_STORE_ROOT", str(tmp_path / "artifacts"))


def _no_group_bound(monkeypatch: pytest.MonkeyPatch) -> None:
    def _fake(
        *,
        topic: str,
        bootstrap_servers: str | None,
        timeout: float,
        owner: ModelConsumerGroupOwner | None = None,
    ) -> tuple[str, ...]:
        return ()

    monkeypatch.setattr(delegate_locus, "live_consumer_groups", _fake)


def _broker_does_not_answer(monkeypatch: pytest.MonkeyPatch) -> None:
    def _unknown(**_: object) -> tuple[str, ...]:
        raise ConsumerGroupLivenessUnknownError("broker did not answer")

    monkeypatch.setattr(delegate_locus, "live_consumer_groups", _unknown)


def _delegate(state_root: Path, tmp_path: Path) -> Result:
    return CliRunner().invoke(
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


def _only_receipt(state_root: Path) -> dict[str, Any]:
    written = sorted(state_root.glob("runs/*/receipt.json"))
    assert len(written) == 1
    receipt: dict[str, Any] = json.loads(written[0].read_text(encoding="utf-8"))
    return receipt


class TestNoConsumerEndsInATypedTerminalFromTheCommandLine:
    def test_the_receipt_types_the_topic_and_the_absent_consumer(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, clock: _FakeClock
    ) -> None:
        _no_group_bound(monkeypatch)
        state_root = tmp_path / "state"
        result = _delegate(state_root, tmp_path)
        assert result.exit_code != 0
        receipt = _only_receipt(state_root)
        refusal = ModelDelegateTransportRefusal.model_validate(
            receipt["transport_refusal"]
        )
        assert refusal.reason == "locus_probe_refused"
        assert refusal.transport_error_type == "DelegateNoConsumerRefusedError"
        assert refusal.command_topic == _COMMAND_TOPIC
        assert refusal.absent_consumer == _ABSENT_CONSUMER
        assert receipt["terminal_class"] == "transport"
        assert _COMMAND_TOPIC in receipt["failure_reason"]
        # Ends within the rebind bound: a terminal, not a hang.
        assert clock.now - 1000.0 <= delegate_locus._REBIND_WAIT_SECONDS

    def test_an_unanswerable_broker_names_no_absent_consumer(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Positive control: no answer is not the claim "no consumer is bound"."""
        _broker_does_not_answer(monkeypatch)
        state_root = tmp_path / "state"
        result = _delegate(state_root, tmp_path)
        assert result.exit_code != 0
        receipt = _only_receipt(state_root)
        refusal = ModelDelegateTransportRefusal.model_validate(
            receipt["transport_refusal"]
        )
        assert refusal.transport_error_type != "DelegateNoConsumerRefusedError"
        assert refusal.absent_consumer == ""
