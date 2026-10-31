# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""``onex delegate --session-model`` carries the caller's session model (OMN-20557).

The savings baseline is the developer's own Claude Code session model when it is
known (ruling D3). The /onex:delegate plugin knows it and passes it here; this
CLI writes it into the request ``metadata`` map under ``session_model``, the key
omnimarket's delegate-skill handler resolves the baseline from. ``metadata`` is
the map every released request consumer already accepts, so a newer CLI never
breaks an older omnimarket.

What these tests pin:

* the option exists and writes exactly that key, only when named;
* a malformed value is a usage error naming the flag, never dropped;
* the model rides in the request payload the run is built from, in process and
  on the deployed lane.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest
from click.testing import CliRunner

from omnibase_infra.cli import cli_delegate
from omnibase_infra.cli.cli_delegate import (
    _request_payload,
    _write_payload,
    delegate_command,
)
from omnibase_infra.cli.model_delegate_locus_decision import ModelDelegateLocusDecision
from omnibase_infra.enums.enum_delegate_locus import EnumDelegateLocus

pytestmark = pytest.mark.unit

SESSION_MODEL = "claude-opus-5-5"


def test_the_command_declares_the_option() -> None:
    declared = {
        opt for param in delegate_command.params for opt in getattr(param, "opts", ())
    }
    assert "--session-model" in declared


def test_the_session_model_rides_in_the_request_metadata() -> None:
    payload = _request_payload(
        prompt="Summarise this.",
        task_type="document",
        source="claude-code",
        max_tokens=None,
        correlation_id=uuid4(),
        ticket_id="OMN-20557",
        session_model=SESSION_MODEL,
    )
    assert payload["metadata"] == {
        "ticket_id": "OMN-20557",
        "session_model": SESSION_MODEL,
    }
    assert "session_model" not in payload, "a metadata key, never a new wire field"


def test_the_payload_file_carries_it(tmp_path: Path) -> None:
    path = _write_payload(
        prompt="Summarise this.",
        task_type="document",
        source="claude-code",
        max_tokens=None,
        state_root=tmp_path / "state",
        run_id=uuid4(),
        correlation_id=uuid4(),
        session_model=SESSION_MODEL,
    )
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded["metadata"]["session_model"] == SESSION_MODEL


@pytest.mark.parametrize("absent", [None, ""])
def test_no_session_model_writes_no_key(absent: str | None) -> None:
    payload = _request_payload(
        prompt="Summarise this.",
        task_type="document",
        source="claude-code",
        max_tokens=None,
        correlation_id=uuid4(),
        session_model=absent,
    )
    assert "metadata" not in payload


@pytest.mark.parametrize(
    "value", ["   ", "claude opus", "x" * 129, "-claude", "claude\nopus"]
)
def test_a_malformed_session_model_is_a_usage_error(value: str) -> None:
    result = CliRunner().invoke(
        delegate_command,
        ["Summarise this.", "--task-type", "document", "--session-model", value],
    )
    assert result.exit_code == 2, result.output
    assert "--session-model" in result.output


_COMMAND = "onex.cmd.synthetic.delegate-skill.v1"
_BROKER = "broker.invalid:19092"


def _chain(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, locus: EnumDelegateLocus
) -> list[dict[str, Any]]:
    """Run the real command with a stand-in for the runtime; return what it was handed."""
    calls: list[dict[str, Any]] = []
    contract = tmp_path / "contract.yaml"
    contract.write_text("name: node_delegate_skill_orchestrator\n", encoding="utf-8")

    def receipt(**kwargs: Any) -> int:
        calls.append(
            {**kwargs, "payload": json.loads(kwargs["input_path"].read_text())}
        )
        return 0

    def resolve(**_: Any) -> ModelDelegateLocusDecision:
        return ModelDelegateLocusDecision(
            locus=locus,
            resolved_from="test",
            orchestrator_contract=str(contract),
            orchestrator_distribution="synthetic",
            command_topic=_COMMAND,
            broker=_BROKER if locus is EnumDelegateLocus.DEPLOYED_LANE else "",
            lane_consumer_groups=(),
        )

    monkeypatch.delenv("OMNI_HOME", raising=False)
    monkeypatch.setattr(cli_delegate, "check_omnimarket_drift", lambda **_: None)
    monkeypatch.setattr(cli_delegate, "_resolve_packaged_contract", lambda _: contract)
    monkeypatch.setattr(cli_delegate, "run_receipt_mode", receipt)
    monkeypatch.setattr(cli_delegate, "resolve_delegate_locus", resolve)
    return calls


@pytest.mark.parametrize(
    ("locus", "flags"),
    [
        (EnumDelegateLocus.IN_PROCESS, ["--bus", "inmemory", "--locus", "in-process"]),
        (
            EnumDelegateLocus.DEPLOYED_LANE,
            [
                "--bus",
                "kafka",
                "--kafka-bootstrap",
                _BROKER,
                "--locus",
                "deployed-lane",
            ],
        ),
    ],
)
def test_golden_chain_the_run_is_handed_the_session_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    locus: EnumDelegateLocus,
    flags: list[str],
) -> None:
    monkeypatch.chdir(tmp_path)
    calls = _chain(monkeypatch, tmp_path, locus)
    result = CliRunner().invoke(
        delegate_command,
        [
            "Summarise this.",
            "--task-type",
            "document",
            "--session-model",
            SESSION_MODEL,
            *flags,
            "--state-root",
            str(tmp_path / "state"),
        ],
    )
    assert result.exit_code == 0, result.output
    (call,) = calls
    assert call["payload"]["metadata"]["session_model"] == SESSION_MODEL
    assert f"session model: {SESSION_MODEL}" in result.stderr
