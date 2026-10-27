# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""The no-consumer refusal reads as no_consumer to the hourly tick (OMN-20764 AC2).

The hourly tick's held classifier lives in another repository. It classifies a
held delegate window from the failure text the delegate left, and since
OMN-20764 it has a ``no_consumer`` class, checked before ``timeout``, that
matches ``NO_CONSUMER_FAILURE`` in its sweep script. That class only fires if
the text this repository's ``onex delegate`` writes for a missing consumer
carries one of the phrases the pattern looks for.

This test pins that seam from this side: it drives the real command with no
consumer group bound and requires the receipt's ``failure_reason`` and the
operator's output to match the classifier's pattern, copied verbatim below.
The control drives an unanswerable broker, which is not a missing consumer, and
requires that it does not match.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pytest
import yaml
from click.testing import CliRunner, Result

from omnibase_infra.backends.backend_probe import ConsumerGroupLivenessUnknownError
from omnibase_infra.cli import cli_delegate, delegate_locus
from omnibase_infra.cli.cli_delegate import delegate_command

pytestmark = pytest.mark.integration

# The hourly tick's NO_CONSUMER_FAILURE (an internal repository, sweep script
# linear-comments-hourly.js, added under OMN-20764), copied verbatim.
_HOURLY_TICK_NO_CONSUMER = re.compile(
    r"\bno (?:live )?consumer groups? (?:is|was|were|are) (?:bound|live)\b"
    r"|\bno service was listening\b"
    r"|\bDelegateNoConsumerRefusedError\b"
    r"|\babsent_consumer\b",
    re.IGNORECASE,
)


def _invoke_delegate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, broker_answers: bool
) -> tuple[Result, dict[str, Any]]:
    """Run the command with only its contract, clock and broker answer replaced."""
    command_topic = "onex.cmd.omnimarket.delegate-skill.v1"
    terminal_topic = "onex.evt.omnimarket.delegate-skill-completed.v1"
    contract = {
        "name": "node_delegate_skill_orchestrator",
        "terminal_event": terminal_topic,
        "event_bus": {
            "publish_topics": [terminal_topic],
            "subscribe_topics": [command_topic],
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
    contract_path = tmp_path / "contract.yaml"
    contract_path.write_text(yaml.safe_dump(contract), encoding="utf-8")
    monkeypatch.delenv("OMNI_HOME", raising=False)
    monkeypatch.setattr(cli_delegate, "check_omnimarket_drift", lambda **_: None)
    monkeypatch.setattr(
        cli_delegate, "_resolve_packaged_contract", lambda _n: contract_path
    )
    monkeypatch.setenv("ONEX_ARTIFACT_STORE_ROOT", str(tmp_path / "artifacts"))

    now = 1000.0

    def sleep(seconds: float) -> None:
        nonlocal now
        now += seconds

    def live_consumer_groups(**_: object) -> tuple[str, ...]:
        if not broker_answers:
            raise ConsumerGroupLivenessUnknownError("broker did not answer")
        return ()

    monkeypatch.setattr(delegate_locus, "_monotonic", lambda: now)
    monkeypatch.setattr(delegate_locus, "_sleep", sleep)
    monkeypatch.setattr(delegate_locus, "live_consumer_groups", live_consumer_groups)
    state_root = tmp_path / "state"
    result = CliRunner().invoke(
        delegate_command,
        [
            "Reply with exactly the word READY",
            "--json",
            "--bus",
            "kafka",
            "--kafka-bootstrap",
            "broker.invalid:19092",
            "--locus",
            "deployed-lane",
            "--state-root",
            str(state_root),
            "--emit-socket",
            str(tmp_path / "no-daemon.sock"),
        ],
        catch_exceptions=False,
    )
    written = sorted(state_root.glob("runs/*/receipt.json"))
    assert len(written) == 1
    receipt: dict[str, Any] = json.loads(written[0].read_text(encoding="utf-8"))
    return result, receipt


def test_hourly_tick_delegate_no_consumer_reason_matches_the_held_classifier(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """AC2, this side: the missing-consumer terminal carries the classifier's phrase."""
    result, receipt = _invoke_delegate(tmp_path, monkeypatch, broker_answers=True)

    assert result.exit_code != 0
    assert _HOURLY_TICK_NO_CONSUMER.search(receipt["failure_reason"]), receipt[
        "failure_reason"
    ]
    assert _HOURLY_TICK_NO_CONSUMER.search(result.output), result.output


def test_hourly_tick_delegate_no_consumer_reason_control_unanswerable_broker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Control: a broker that did not answer is not reported as a missing consumer."""
    result, receipt = _invoke_delegate(tmp_path, monkeypatch, broker_answers=False)

    assert result.exit_code != 0
    assert not _HOURLY_TICK_NO_CONSUMER.search(receipt["failure_reason"]), receipt[
        "failure_reason"
    ]
    assert not _HOURLY_TICK_NO_CONSUMER.search(result.output), result.output


@pytest.mark.parametrize(
    ("failure_reason", "matches"),
    [
        pytest.param("no live consumer groups are bound", True, id="consumer-group"),
        pytest.param("no service was listening", True, id="listener"),
        pytest.param("DelegateNoConsumerRefusedError", True, id="typed-error"),
        pytest.param("absent_consumer", True, id="typed-field"),
        pytest.param("broker did not answer", False, id="unknown-broker"),
        pytest.param("timed out after 180 s", False, id="generic-timeout"),
    ],
)
def test_hourly_tick_no_consumer_pattern_alternatives(
    failure_reason: str, matches: bool
) -> None:
    """Pin each copied classifier alternative and unrelated transport controls."""
    assert bool(_HOURLY_TICK_NO_CONSUMER.search(failure_reason)) is matches
