# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Execution-locus resolution and the fail-closed dispatch gate (OMN-17295).

Subject: whether ``onex delegate`` can still produce a receipt that reads as a
statement about a deployed lane it never reached.

The defect: ``--bus kafka`` selected the TRANSPORT and was read as relocating
execution. It never did. The orchestrator ran in-process from the caller's own
venv, and — worse, because the CLI also subscribed the command topic it was
publishing to — it executed a stale backlog command while the lane executed
the real one. Live on the .201 dev lane 2026-08-31, one probe advanced the
CLI's own committed offset from 53 to 54 against a log end of 152.

The positive assertions here are paired with counter-assertions on the
in-process path. Without them, "always refuse" and "never dispatch" would
satisfy the fail-closed tests while destroying the offline flow.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from omnibase_infra.backends.backend_probe import ConsumerGroupLivenessUnknownError
from omnibase_infra.cli import delegate_locus
from omnibase_infra.cli.delegate_locus import (
    DelegateLocusRefusedError,
    contract_command_topic,
    resolve_delegate_locus,
)
from omnibase_infra.enums.enum_delegate_locus import EnumDelegateLocus

_COMMAND_TOPIC = "onex.cmd.omnimarket.delegate-skill.v1"
_TERMINAL_TOPIC = "onex.evt.omnimarket.delegate-skill-completed.v1"
_BUS_KAFKA = "kafka"
_BUS_INMEMORY = "inmemory"
# Fixture broker address. Deliberately not a real lab IP: this file asserts on
# the value round-tripping into the decision, and a hardcoded private address
# in a test is both a portability trap and a gate violation.
_TEST_BROKER = "broker.invalid:19092"

_LANE_GROUP = (
    "local.omnimarket.node_delegate_skill_orchestrator.consume.1.1.0"
    ".__i.runtime-effects.__t." + _COMMAND_TOPIC
)


def _write_contract(tmp_path: Path, *, subscribe_topics: list[str] | None) -> Path:
    contract: dict[str, Any] = {
        "name": "node_delegate_skill_orchestrator",
        "terminal_event": _TERMINAL_TOPIC,
        "event_bus": {"publish_topics": [_TERMINAL_TOPIC]},
    }
    if subscribe_topics is not None:
        contract["event_bus"]["subscribe_topics"] = subscribe_topics
    path = tmp_path / "contract.yaml"
    path.write_text(yaml.safe_dump(contract), encoding="utf-8")
    return path


def _stub_groups(
    monkeypatch: pytest.MonkeyPatch, groups: tuple[str, ...]
) -> list[dict[str, object]]:
    """Record every liveness question and answer it with *groups*."""
    asked: list[dict[str, object]] = []

    def _fake(
        *, topic: str, bootstrap_servers: str | None, timeout: float
    ) -> tuple[str, ...]:
        asked.append({"topic": topic, "bootstrap": bootstrap_servers})
        return groups

    monkeypatch.setattr(delegate_locus, "live_consumer_groups", _fake)
    return asked


class TestContractCommandTopic:
    """The command topic is read from the contract, never named in code."""

    def test_reads_the_first_subscribe_topic(self, tmp_path: Path) -> None:
        contract = _write_contract(tmp_path, subscribe_topics=[_COMMAND_TOPIC])
        assert contract_command_topic(contract) == _COMMAND_TOPIC

    def test_refuses_a_contract_with_no_command_topic(self, tmp_path: Path) -> None:
        contract = _write_contract(tmp_path, subscribe_topics=None)
        with pytest.raises(DelegateLocusRefusedError, match="subscribe_topics"):
            contract_command_topic(contract)


class TestLocusResolution:
    """Locus follows the transport by default, and says so either way."""

    def test_auto_on_a_shared_bus_dispatches(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        contract = _write_contract(tmp_path, subscribe_topics=[_COMMAND_TOPIC])
        _stub_groups(monkeypatch, (_LANE_GROUP,))
        decision = resolve_delegate_locus(
            requested=EnumDelegateLocus.AUTO,
            bus=_BUS_KAFKA,
            kafka_bootstrap=_TEST_BROKER,
            contract_path=contract,
            shared_bus_value=_BUS_KAFKA,
        )
        assert decision.locus is EnumDelegateLocus.DEPLOYED_LANE
        assert "resolved from transport=kafka" in decision.resolved_from
        assert decision.lane_consumer_groups == (_LANE_GROUP,)

    def test_auto_on_an_in_process_bus_runs_here(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Counter-assertion: the offline path still resolves and still runs."""
        contract = _write_contract(tmp_path, subscribe_topics=[_COMMAND_TOPIC])
        asked = _stub_groups(monkeypatch, ())
        decision = resolve_delegate_locus(
            requested=EnumDelegateLocus.AUTO,
            bus=_BUS_INMEMORY,
            kafka_bootstrap=None,
            contract_path=contract,
            shared_bus_value=_BUS_KAFKA,
        )
        assert decision.locus is EnumDelegateLocus.IN_PROCESS
        assert decision.lane_consumer_groups == ()
        assert decision.broker == ""
        assert asked == [], (
            "an in-process run asked the broker about consumers — the offline "
            "flow must not acquire a broker dependency"
        )

    def test_explicit_locus_labels_itself_an_override(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A receipt must distinguish "the authority chose" from "a human typed"."""
        contract = _write_contract(tmp_path, subscribe_topics=[_COMMAND_TOPIC])
        _stub_groups(monkeypatch, ())
        decision = resolve_delegate_locus(
            requested=EnumDelegateLocus.IN_PROCESS,
            bus=_BUS_KAFKA,
            kafka_bootstrap=None,
            contract_path=contract,
            shared_bus_value=_BUS_KAFKA,
        )
        assert decision.locus is EnumDelegateLocus.IN_PROCESS
        assert "OVERRIDES" in decision.resolved_from

    def test_deployed_lane_over_an_in_process_bus_is_refused(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        contract = _write_contract(tmp_path, subscribe_topics=[_COMMAND_TOPIC])
        _stub_groups(monkeypatch, (_LANE_GROUP,))
        with pytest.raises(DelegateLocusRefusedError, match="incoherent"):
            resolve_delegate_locus(
                requested=EnumDelegateLocus.DEPLOYED_LANE,
                bus=_BUS_INMEMORY,
                kafka_bootstrap=None,
                contract_path=contract,
                shared_bus_value=_BUS_KAFKA,
            )


class TestFailClosedDispatchGate:
    """A dispatched run refuses unless something is provably consuming NOW."""

    def test_no_live_consumer_refuses(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        contract = _write_contract(tmp_path, subscribe_topics=[_COMMAND_TOPIC])
        _stub_groups(monkeypatch, ())
        with pytest.raises(DelegateLocusRefusedError, match="no live consumer group"):
            resolve_delegate_locus(
                requested=EnumDelegateLocus.DEPLOYED_LANE,
                bus=_BUS_KAFKA,
                kafka_bootstrap="broker:9092",
                contract_path=contract,
                shared_bus_value=_BUS_KAFKA,
            )

    def test_unknown_liveness_refuses_and_is_not_reported_as_absence(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """UNKNOWN is not permission, and it is not "nobody is bound" either.

        The two need different messages: one says fix the broker, the other
        says start the runtime. Collapsing them is the same conflation that let
        an unanswerable probe read as a negative verdict.
        """
        contract = _write_contract(tmp_path, subscribe_topics=[_COMMAND_TOPIC])

        def _raise(**_: object) -> tuple[str, ...]:
            raise ConsumerGroupLivenessUnknownError("broker transport failure")

        monkeypatch.setattr(delegate_locus, "live_consumer_groups", _raise)
        with pytest.raises(DelegateLocusRefusedError, match="cannot confirm"):
            resolve_delegate_locus(
                requested=EnumDelegateLocus.DEPLOYED_LANE,
                bus=_BUS_KAFKA,
                kafka_bootstrap="broker:9092",
                contract_path=contract,
                shared_bus_value=_BUS_KAFKA,
            )

    def test_liveness_is_checked_on_the_contract_topic_not_a_constant(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The gate must ask about the topic the command actually lands on.

        ``resolve_default_bus`` asks ``probe_kafka`` for AUTHORITATIVE liveness
        on ``SUFFIX_DELEGATION_REQUEST``
        (``onex.cmd.omnibase-infra.delegation-request.v1``) while this path
        publishes to the contract's own command topic. Both carry live groups
        on the dev lane, so that check passes for reasons unrelated to whether
        this command will be served.
        """
        other_topic = "onex.cmd.omnimarket.some-other-command.v1"
        contract = _write_contract(tmp_path, subscribe_topics=[other_topic])
        asked = _stub_groups(monkeypatch, (f"g.__t.{other_topic}",))
        decision = resolve_delegate_locus(
            requested=EnumDelegateLocus.DEPLOYED_LANE,
            bus=_BUS_KAFKA,
            kafka_bootstrap="broker:9092",
            contract_path=contract,
            shared_bus_value=_BUS_KAFKA,
        )
        assert [entry["topic"] for entry in asked] == [other_topic]
        assert decision.command_topic == other_topic


class TestDecisionCarriesItsEvidence:
    """The decision is only useful if it names what it was decided on."""

    def test_dispatched_decision_names_topic_broker_and_consumers(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        contract = _write_contract(tmp_path, subscribe_topics=[_COMMAND_TOPIC])
        _stub_groups(monkeypatch, (_LANE_GROUP,))
        decision = resolve_delegate_locus(
            requested=EnumDelegateLocus.DEPLOYED_LANE,
            bus=_BUS_KAFKA,
            kafka_bootstrap=_TEST_BROKER,
            contract_path=contract,
            shared_bus_value=_BUS_KAFKA,
        )
        assert decision.command_topic == _COMMAND_TOPIC
        assert decision.broker == _TEST_BROKER
        assert decision.lane_consumer_groups == (_LANE_GROUP,)
        assert decision.orchestrator_contract == str(contract)
        assert decision.orchestrator_distribution.startswith("omnimarket ")


class TestRunDelegateThreadsTheDecision:
    """The decision has to reach the runtime, or it is decoration."""

    @staticmethod
    def _capture_receipt_mode(
        monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> list[dict[str, object]]:
        from omnibase_infra.cli import cli_delegate

        calls: list[dict[str, object]] = []

        def _fake_receipt_mode(**kwargs: object) -> int:
            calls.append(kwargs)
            return 0

        contract = _write_contract(tmp_path, subscribe_topics=[_COMMAND_TOPIC])
        monkeypatch.delenv("OMNI_HOME", raising=False)
        monkeypatch.setattr(cli_delegate, "check_omnimarket_drift", lambda **_: None)
        monkeypatch.setattr(
            cli_delegate, "_resolve_packaged_contract", lambda _name: contract
        )
        monkeypatch.setattr(cli_delegate, "run_receipt_mode", _fake_receipt_mode)
        return calls

    def test_deployed_lane_hosts_nothing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """AC4: with a shared bus the CLI publishes and does NOT subscribe.

        ``host_handlers=True`` here is precisely the OMN-17295 defect — the
        caller becomes a second consumer group of the command topic it
        publishes to, executes a backlog command out of its own venv, and
        returns that as the run's receipt.
        """
        from omnibase_infra.cli.cli_delegate import run_delegate

        calls = self._capture_receipt_mode(monkeypatch, tmp_path)
        _stub_groups(monkeypatch, (_LANE_GROUP,))

        assert (
            run_delegate(
                prompt="probe",
                task_type="research",
                max_tokens=None,
                bus=_BUS_KAFKA,
                locus=EnumDelegateLocus.DEPLOYED_LANE,
                kafka_bootstrap="broker:9092",
                state_root=tmp_path / "state",
                timeout=5,
                verbose=False,
                emit_socket=tmp_path / "emit.sock",
            )
            == 0
        )
        assert len(calls) == 1
        assert calls[0]["host_handlers"] is False
        decision = calls[0]["locus_decision"]
        assert decision.locus is EnumDelegateLocus.DEPLOYED_LANE

    def test_in_process_still_hosts(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Counter-assertion: the offline path keeps hosting its handlers."""
        from omnibase_infra.cli.cli_delegate import run_delegate

        calls = self._capture_receipt_mode(monkeypatch, tmp_path)
        _stub_groups(monkeypatch, ())

        assert (
            run_delegate(
                prompt="probe",
                task_type="research",
                max_tokens=None,
                bus=_BUS_INMEMORY,
                locus=EnumDelegateLocus.AUTO,
                kafka_bootstrap=None,
                state_root=tmp_path / "state",
                timeout=5,
                verbose=False,
                emit_socket=tmp_path / "emit.sock",
            )
            == 0
        )
        assert len(calls) == 1
        assert calls[0]["host_handlers"] is True

    def test_a_refused_lane_never_falls_back_to_running_here(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The whole point: no consumer means no run, not a local run.

        A fallback here would reproduce the original defect exactly — a
        receipt produced locally while the operator believes they probed a
        lane — with the aggravation of having been asked for the lane
        explicitly.
        """
        import click

        from omnibase_infra.cli.cli_delegate import run_delegate

        calls = self._capture_receipt_mode(monkeypatch, tmp_path)
        _stub_groups(monkeypatch, ())

        with pytest.raises(click.ClickException, match="no live consumer group"):
            run_delegate(
                prompt="probe",
                task_type="research",
                max_tokens=None,
                bus=_BUS_KAFKA,
                locus=EnumDelegateLocus.DEPLOYED_LANE,
                kafka_bootstrap="broker:9092",
                state_root=tmp_path / "state",
                timeout=5,
                verbose=False,
                emit_socket=tmp_path / "emit.sock",
            )
        assert calls == [], "a refused lane probe still dispatched a run"
