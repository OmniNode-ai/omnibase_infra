# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""``onex delegate --locus in-process`` on a shared bus is refused end to end (OMN-20236).

The unit tests pin the locus gate. This pins the path a caller takes: the CLI
entry resolves the locus, an in-process run over a shared broker is refused
before anything is published or run here, and the true in-process run on an
in-memory bus still reaches receipt mode.

The broker is the only stub: the refusal fires before any broker question, so
the stub records that none was asked.
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
    asked: list[dict[str, object]] = []

    def _fake_receipt_mode(**kwargs: object) -> int:
        dispatched.append(kwargs)
        return 0

    def _fake_groups(**kwargs: object) -> tuple[str, ...]:
        asked.append(dict(kwargs))
        return ()

    monkeypatch.delenv("OMNI_HOME", raising=False)
    monkeypatch.setattr(cli_delegate, "check_omnimarket_drift", lambda **_: None)
    monkeypatch.setattr(cli_delegate, "_resolve_packaged_contract", lambda _n: contract)
    monkeypatch.setattr(cli_delegate, "run_receipt_mode", _fake_receipt_mode)
    monkeypatch.setattr(delegate_locus, "live_consumer_groups", _fake_groups)
    return {"dispatched": dispatched, "asked": asked, "state": tmp_path / "state"}


def _delegate(
    harness: dict[str, Any], tmp_path: Path, *, bus: str, kafka_bootstrap: str | None
) -> int:
    return run_delegate(
        prompt="probe",
        task_type="research",
        max_tokens=None,
        bus=bus,
        locus=EnumDelegateLocus.IN_PROCESS,
        kafka_bootstrap=kafka_bootstrap,
        state_root=harness["state"],
        timeout=5,
        verbose=False,
        emit_socket=tmp_path / "emit.sock",
    )


def test_in_process_on_a_shared_bus_dispatches_nothing(
    tmp_path: Path, harness: dict[str, Any]
) -> None:
    with pytest.raises(click.ClickException, match="OMN-20236"):
        _delegate(
            harness, tmp_path, bus="kafka", kafka_bootstrap="broker.invalid:19092"
        )

    assert harness["dispatched"] == []
    assert harness["asked"] == []


def test_in_process_on_an_inmemory_bus_still_reaches_receipt_mode(
    tmp_path: Path, harness: dict[str, Any]
) -> None:
    assert _delegate(harness, tmp_path, bus="inmemory", kafka_bootstrap=None) == 0

    assert len(harness["dispatched"]) == 1
    decision = harness["dispatched"][0]["locus_decision"]
    assert decision.locus is EnumDelegateLocus.IN_PROCESS
    assert decision.broker == ""
    assert harness["asked"] == []
