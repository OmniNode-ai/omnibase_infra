# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""``onex delegate`` never sends a run with no caller lane (OMN-20299).

On the lab dev lane, 651 of 699 ``delegation_events`` rows in the 24 hours to
2026-10-01T12:24Z carried a null ``caller_lane``. Every one of the largest
groups came through ``onex delegate``: the worktree-prune classifier and
triage (launchd jobs on the operator machine), the spread and liveness
probes, CI-red classifiers and Claude Code sessions with a session id but no
lane variable. The resolver ended in "otherwise none", so each of those runs
attributed to nobody and the delegation share could not tell an internal run
from an external one.

The resolver now ends in a lane that always exists, chosen most explicit
first after the existing three rules: the launchd job label, the GitHub
Actions workflow, the Claude Code session, and last ``unattributed:<host>``.
A null ``caller_lane`` on the projection therefore means a caller that did not
go through this CLI. The derived names never collide with a ledger lane, so
they never count as a lane's own delegation.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from omnibase_infra.cli.delegate_caller import resolve_delegate_caller

pytestmark = pytest.mark.unit

_ELSEWHERE = Path("/work/omni_home")
_SESSION = "15501d1a-5cfc-422b-a688-0ace6bfe0f21"
_LANE_TOKEN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")


def _resolve(environ: dict[str, str], host: str = "omnipc2") -> tuple[str, str]:
    caller = resolve_delegate_caller(None, cwd=_ELSEWHERE, environ=environ, host=host)
    assert caller.lane is not None, "a CLI run must never send a null caller lane"
    assert _LANE_TOKEN.fullmatch(caller.lane), caller.lane
    return caller.lane, caller.lane_source


def test_caller_lane_is_never_null_with_nothing_set() -> None:
    assert _resolve({}) == ("unattributed:omnipc2", "fallback host")


def test_caller_lane_names_the_launchd_job() -> None:
    lane, source = _resolve({"XPC_SERVICE_NAME": "ai.omninode.worktree-prune"})
    assert (lane, source) == (
        "launchd:ai.omninode.worktree-prune",
        "fallback env XPC_SERVICE_NAME",
    )


@pytest.mark.parametrize(
    "value", ["0", "application.com.apple.Terminal.1A2B", "", "not a label"]
)
def test_caller_lane_ignores_an_xpc_name_that_is_not_a_launchd_job(
    value: str,
) -> None:
    assert _resolve({"XPC_SERVICE_NAME": value})[0] == "unattributed:omnipc2"


def test_caller_lane_names_the_github_actions_workflow() -> None:
    lane, source = _resolve(
        {
            "GITHUB_ACTIONS": "true",
            "GITHUB_REPOSITORY": "OmniNode-ai/omnibase_infra",
            "GITHUB_WORKFLOW": "Hostile Reviewer",
        }
    )
    assert (lane, source) == (
        "gha:omnibase_infra:hostile-reviewer",
        "fallback env GITHUB_WORKFLOW",
    )


def test_caller_lane_names_the_claude_code_session() -> None:
    lane, source = _resolve({"CLAUDE_CODE_SESSION_ID": _SESSION.upper()})
    assert (lane, source) == (f"session:{_SESSION}", "fallback session")


def test_caller_lane_prefers_a_stated_lane_over_every_fallback() -> None:
    lane, source = _resolve(
        {
            "ONEX_LANE": "omn20299-server-reconcile",
            "XPC_SERVICE_NAME": "ai.omninode.worktree-prune",
            "CLAUDE_CODE_SESSION_ID": _SESSION,
        }
    )
    assert (lane, source) == ("omn20299-server-reconcile", "env ONEX_LANE")


def test_caller_lane_prefers_the_job_over_the_session() -> None:
    lane, _ = _resolve(
        {
            "XPC_SERVICE_NAME": "ai.omninode.worktree-prune",
            "CLAUDE_CODE_SESSION_ID": _SESSION,
        }
    )
    assert lane == "launchd:ai.omninode.worktree-prune"


@pytest.mark.parametrize("host", ["", "   ", "host with spaces!", "-x", "a" * 300])
def test_caller_lane_host_fallback_is_always_a_lane_token(host: str) -> None:
    lane, source = _resolve({}, host=host)
    assert lane.startswith("unattributed:")
    assert source == "fallback host"


def test_caller_lane_fallback_names_no_ledger_lane() -> None:
    """A derived name always carries a kind prefix a ledger lane never starts with."""
    for environ in (
        {},
        {"XPC_SERVICE_NAME": "ai.omninode.worktree-prune"},
        {"CLAUDE_CODE_SESSION_ID": _SESSION},
        {
            "GITHUB_ACTIONS": "true",
            "GITHUB_REPOSITORY": "o/r",
            "GITHUB_WORKFLOW": "ci",
        },
    ):
        lane, _ = _resolve(environ)
        assert lane.split(":", 1)[0] in {"unattributed", "launchd", "gha", "session"}
