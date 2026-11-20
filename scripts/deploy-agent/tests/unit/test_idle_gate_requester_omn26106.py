# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Every idle path names its requester to the recreate window gate (OMN-26106).

omnibase_infra#6080 (OMN-25912) made ``_idle_converge_gate_open`` hand its ``who`` to
the recreate window, whose request model refuses an empty ``requested_by``. Two callers
passed nothing: the stability follow and the forward migration converge. On the dev
host the stability follow raised ``ValidationError`` on every idle tick, so
``deploy-agent-dev.service`` restarted 28 times and :8098 refused connections.

The unit-test conftest admits any agent that carries no gate, whatever ``who`` is, which
is how the empty requester passed the suite. These tests stand in for the window with
one that refuses exactly what the journal recorded
(``fixtures/dev_agent_requested_by_crash_omn26106.txt``), so a caller that names no
requester fails here the way it failed on the host.
"""

from __future__ import annotations

import inspect
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest
from deploy_agent import agent as agent_mod
from deploy_agent.agent import DeployAgent
from deploy_agent.idle_converge import (
    IDLE_CONVERGE_REQUESTER,
    STABILITY_FOLLOW_REQUESTER,
)

from .test_stability_follows_dev_tag_omn19998 import LAB_FENCE, _lab_agent

pytestmark = pytest.mark.unit

JOURNAL = (
    Path(__file__).parent / "fixtures" / "dev_agent_requested_by_crash_omn26106.txt"
)


class _RecordedRefusalError(ValueError):
    """What the journal recorded the recreate window raising for an empty requester."""


def _recorded_refusal() -> tuple[str, str]:
    """The model, the field and the refused input the journal recorded."""
    text = JOURNAL.read_text(encoding="utf-8")
    model = re.search(r"validation error for (\w+)\n(\w+)\n", text)
    refused = re.search(r"input_value=('[^']*')", text)
    assert model is not None and refused is not None, (
        "the journal fixture lost its error"
    )
    return f"{model.group(1)}.{model.group(2)}", refused.group(1)


@pytest.fixture
def window_asks(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """The recreate window, refusing the input the journal recorded; it admits the rest."""
    field, refused = _recorded_refusal()
    asked: list[str] = []
    real = agent_mod.load_dev_boot_guard

    def loader(effect: str = "handler_dev_boot_guard") -> Any:
        if effect != agent_mod.DEV_ROLL_EVENTS:
            return real(effect)

        def window_admits(agent: object, who: str) -> bool:
            asked.append(who)
            if repr(who) == refused:
                raise _RecordedRefusalError(f"{field}: input_value={refused}")
            return True

        return SimpleNamespace(window_admits=window_admits)

    monkeypatch.setattr(agent_mod, "load_dev_boot_guard", loader)
    return asked


def test_the_journal_fixture_records_the_empty_requester() -> None:
    assert _recorded_refusal() == ("ModelDevRecreateRequest.requested_by", "''")
    text = JOURNAL.read_text(encoding="utf-8")
    assert "in _maybe_stability_follow" in text
    assert "in _idle_converge_gate_open" in text
    assert "string_too_short" in text


def test_the_gate_takes_its_requester_with_no_default() -> None:
    who = inspect.signature(DeployAgent._idle_converge_gate_open).parameters["who"]
    assert who.default is inspect.Parameter.empty


def test_the_stability_follow_names_its_requester_to_the_window(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, window_asks: list[str]
) -> None:
    agent = _lab_agent(tmp_path, monkeypatch, fence=LAB_FENCE)

    agent._maybe_idle_converge()

    assert window_asks == [STABILITY_FOLLOW_REQUESTER]
    assert len(list((tmp_path / "jobs").glob("*.json"))) == 1


def test_the_migration_converge_names_its_requester_to_the_window(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, window_asks: list[str]
) -> None:
    monkeypatch.setenv("DEPLOY_AGENT_ALLOWED_LANES", "dev")
    monkeypatch.setenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:19092")
    monkeypatch.setattr(agent_mod, "STATE_DIR", tmp_path / "jobs")
    monkeypatch.setattr(
        agent_mod, "read_checkout_migration_fingerprint", Mock(return_value="new")
    )
    monkeypatch.setattr(
        agent_mod, "read_applied_migration_fingerprint", Mock(return_value="old")
    )
    agent = DeployAgent(skip_self_update=True)
    converge = Mock(return_value="new")
    monkeypatch.setattr(agent.job_store, "has_active_job", Mock(return_value=False))
    monkeypatch.setattr(agent.executor, "converge_forward_migration", converge)
    try:
        agent._maybe_converge_forward_migration()
    finally:
        agent._job_pool.shutdown(wait=True)
        agent._lag_pool.shutdown(wait=True)
        agent._settle_pool.shutdown(wait=True)

    assert window_asks == [IDLE_CONVERGE_REQUESTER]
    converge.assert_called_once()
