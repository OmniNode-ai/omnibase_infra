# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""In-process delegations must not double dispatch on a shared bus (OMN-20236)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from omnibase_infra.cli import delegate_locus
from omnibase_infra.cli.delegate_locus import (
    DelegateLocusRefusedError,
    resolve_delegate_locus,
)
from omnibase_infra.enums.enum_delegate_locus import EnumDelegateLocus

pytestmark = pytest.mark.unit

_COMMAND_TOPIC = "onex.cmd.omnimarket.delegate-skill.v1"
_TERMINAL_TOPIC = "onex.evt.omnimarket.delegate-skill-completed.v1"
_BUS_KAFKA = "kafka"
_BUS_INMEMORY = "inmemory"
_TEST_BROKER = "broker.invalid:19092"


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


def _stub_groups(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, object]]:
    """Record any broker question, even when the deployed consumer is down."""
    asked: list[dict[str, object]] = []

    def _fake(**kwargs: object) -> tuple[str, ...]:
        # Keyword-tolerant: the probe's signature is owned by OMN-20235.
        asked.append(dict(kwargs))
        return ()

    monkeypatch.setattr(delegate_locus, "live_consumer_groups", _fake)
    return asked


def test_double_dispatch_shared_bus_in_process_refuses_without_probing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    contract = _write_contract(tmp_path, subscribe_topics=[_COMMAND_TOPIC])
    asked = _stub_groups(monkeypatch)

    with pytest.raises(DelegateLocusRefusedError) as exc_info:
        resolve_delegate_locus(
            requested=EnumDelegateLocus.IN_PROCESS,
            bus=_BUS_KAFKA,
            kafka_bootstrap=_TEST_BROKER,
            contract_path=contract,
            shared_bus_value=_BUS_KAFKA,
        )

    message = str(exc_info.value)
    for phrase in (
        "double dispatch",
        _COMMAND_TOPIC,
        "committed offset when it next starts",
        "second copy overwrites the first",
        "projection",
        "--locus deployed-lane",
        "--bus inmemory",
        "OMN-20236",
    ):
        assert phrase in message
    assert asked == []


@pytest.mark.parametrize(
    "requested", [EnumDelegateLocus.AUTO, EnumDelegateLocus.IN_PROCESS]
)
def test_double_dispatch_inmemory_still_resolves_without_probing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, requested: EnumDelegateLocus
) -> None:
    contract = _write_contract(tmp_path, subscribe_topics=[_COMMAND_TOPIC])
    asked = _stub_groups(monkeypatch)

    decision = resolve_delegate_locus(
        requested=requested,
        bus=_BUS_INMEMORY,
        kafka_bootstrap=None,
        contract_path=contract,
        shared_bus_value=_BUS_KAFKA,
    )

    assert decision.locus is EnumDelegateLocus.IN_PROCESS
    assert decision.command_topic == _COMMAND_TOPIC
    assert decision.broker == ""
    assert decision.lane_consumer_groups == ()
    assert asked == []


@pytest.mark.parametrize("subscribe_topics", [None, [], [""]])
def test_double_dispatch_shared_bus_without_command_topic_still_resolves(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    subscribe_topics: list[str] | None,
) -> None:
    contract = _write_contract(tmp_path, subscribe_topics=subscribe_topics)
    asked = _stub_groups(monkeypatch)

    decision = resolve_delegate_locus(
        requested=EnumDelegateLocus.IN_PROCESS,
        bus=_BUS_KAFKA,
        kafka_bootstrap=_TEST_BROKER,
        contract_path=contract,
        shared_bus_value=_BUS_KAFKA,
    )

    assert decision.locus is EnumDelegateLocus.IN_PROCESS
    assert decision.command_topic == ""
    assert asked == []


def test_double_dispatch_refusal_escapes_name_the_inmemory_bus() -> None:
    source = Path(delegate_locus.__file__).read_text(encoding="utf-8")
    assert "or pass --locus in-process" not in source
