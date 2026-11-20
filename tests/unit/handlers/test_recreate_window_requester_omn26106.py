# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The real recreate window accepts every requester the deploy agent's idle paths name (OMN-26106).

The window's request model refuses an empty ``requested_by``. The agent's stability
follow and migration converge named none, and ``deploy-agent-dev.service`` crash-looped
on the dev host. The agent-side callers are tested in
``scripts/deploy-agent/tests/unit/test_idle_gate_requester_omn26106.py``; this file holds
the half that needs the real model and gate, which the deploy-agent test environment
(the published package) does not carry.

Run with ``-k recreate_window_requester``.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest
from deploy_agent import agent as agent_mod
from deploy_agent.idle_converge import (
    IDLE_CONVERGE_REQUESTER,
    IDLE_RESTORE_REQUESTER,
    STABILITY_FOLLOW_REQUESTER,
)
from pydantic import ValidationError

from omnibase_infra.handlers.dev_recreate_window.handler_dev_recreate_window import (
    DevRecreateWindowGate,
)
from omnibase_infra.models.model_dev_recreate_announcement import (
    ModelDevRecreateAnnouncement,
)
from omnibase_infra.models.model_dev_recreate_policy import ModelDevRecreatePolicy
from omnibase_infra.models.model_dev_recreate_request import ModelDevRecreateRequest
from omnibase_infra.models.model_unit_file_digest import ModelUnitFileDigest

REPO_ROOT = Path(__file__).resolve().parents[3]
POLICY_ENV = REPO_ROOT / "docker" / "runtime-policy.env"
JOURNAL = (
    REPO_ROOT
    / "scripts"
    / "deploy-agent"
    / "tests"
    / "unit"
    / "fixtures"
    / "dev_agent_requested_by_crash_omn26106.txt"
)
#: The tracked prod deploy agent unit and drop-in as they stood when OMN-26106 was
#: written. A deliberate edit of either file updates its digest here in the same change.
PROD_UNIT_DIR = REPO_ROOT / "scripts" / "deploy-agent" / "deploy"
PROD_UNIT_DIGESTS = (
    (
        PROD_UNIT_DIR / "deploy-agent.service",
        ModelUnitFileDigest(
            name="deploy-agent.service",
            sha256="8c1d10d778cdf226838f6d3210efa27182ac6373a12b0492b31dde4edcedf8be",
        ),
    ),
    (
        PROD_UNIT_DIR / "deploy-agent.service.d" / "override.conf",
        ModelUnitFileDigest(
            name="deploy-agent.service.d/override.conf",
            sha256="dd026e66e81897631d8cab300c0cc6afcf006261ddb756740731f69ef41354b1",
        ),
    ),
)
REQUESTERS = (
    IDLE_CONVERGE_REQUESTER,
    IDLE_RESTORE_REQUESTER,
    STABILITY_FOLLOW_REQUESTER,
)


def _declared_policy() -> ModelDevRecreatePolicy:
    env: dict[str, str] = {}
    for line in POLICY_ENV.read_text(encoding="utf-8").splitlines():
        key, sep, value = line.partition("=")
        if sep and not key.startswith("#"):
            env[key] = value
    return ModelDevRecreatePolicy.from_env(env)


def _agent_with_gate(clock: datetime) -> agent_mod.DeployAgent:
    def bus(announcement: ModelDevRecreateAnnouncement) -> bool:
        return True

    agent = agent_mod.DeployAgent.__new__(agent_mod.DeployAgent)
    agent.__dict__.update(
        _load_gate=None,
        _load_gate_last_verdict=None,
        _recreate_gate=DevRecreateWindowGate(
            _declared_policy(), bus, clock=lambda: clock
        ),
    )
    return agent


def test_recreate_window_requester_empty_ask_is_the_recorded_journal_error() -> None:
    """The model's own refusal of '' reads as the journal recorded it on the host."""
    with pytest.raises(ValidationError) as refused:
        ModelDevRecreateRequest(
            policy=_declared_policy(),
            now=datetime(2026, 11, 19, 19, 16, 13, tzinfo=UTC),
            requested_by="",
        )
    recorded = JOURNAL.read_text(encoding="utf-8")
    for line in (
        "1 validation error for ModelDevRecreateRequest",
        "String should have at least 1 character [type=string_too_short, input_value='', input_type=str]",
    ):
        assert line in recorded
        assert line in str(refused.value)


@pytest.mark.parametrize("requester", REQUESTERS)
def test_recreate_window_requester_every_idle_requester_gets_a_decision(
    requester: str,
) -> None:
    agent = _agent_with_gate(datetime(2026, 11, 19, 19, 16, 13, tzinfo=UTC))
    assert agent._idle_converge_gate_open(requester) in (True, False)


def test_recreate_window_requester_the_gate_refuses_a_bare_call() -> None:
    agent = _agent_with_gate(datetime(2026, 11, 19, 19, 16, 13, tzinfo=UTC))
    no_requester: tuple[str, ...] = ()
    with pytest.raises(TypeError):
        agent._idle_converge_gate_open(*no_requester)


def test_recreate_window_requester_prod_unit_keeps_its_port_and_names_no_dev_key() -> (
    None
):
    """The prod agent's tracked unit and drop-in stay apart from the dev unit and its key."""
    unit_dir = REPO_ROOT / "scripts" / "deploy-agent" / "deploy"
    prod = (unit_dir / "deploy-agent.service").read_text(encoding="utf-8")
    drop_in = (unit_dir / "deploy-agent.service.d" / "override.conf").read_text(
        encoding="utf-8"
    )
    assert "Environment=DEPLOY_AGENT_PORT=8099" in prod
    assert "deploy-agent-dev-signing-key" not in prod + drop_in


@pytest.mark.parametrize(("path", "recorded"), PROD_UNIT_DIGESTS)
def test_recreate_window_requester_prod_unit_digest_is_untouched(
    path: Path, recorded: ModelUnitFileDigest
) -> None:
    """AC4: the tracked prod unit and drop-in are byte-for-byte the recorded ones.

    The prod agent's signing key is not in the repository, so its before and after
    read is the host read the contract's local-gate entry makes.
    """
    assert not recorded.reads_as_changed(path.read_bytes()), recorded.name


@pytest.mark.parametrize(("path", "recorded"), PROD_UNIT_DIGESTS)
def test_recreate_window_requester_prod_unit_digest_reads_a_one_byte_change(
    path: Path, recorded: ModelUnitFileDigest, tmp_path: Path
) -> None:
    """The digest read can fail: a copy with one byte changed reads as changed."""
    original = path.read_bytes()
    copy = tmp_path / path.name
    copy.write_bytes(original)
    assert not recorded.reads_as_changed(copy.read_bytes())
    copy.write_bytes(original[:-1] + bytes([original[-1] ^ 0x01]))
    assert recorded.reads_as_changed(copy.read_bytes()), recorded.name
