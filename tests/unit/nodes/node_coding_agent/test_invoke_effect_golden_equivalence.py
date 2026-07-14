# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Golden-equivalence proof for the invoke-effect def-B flip (OMN-14589).

RSD mechanical-wave canary (path (b)): the first canon-shape flip
(``scripts/ci/canonical_handler_shape.py``, OMN-14355) outside omnibase_core /
omnimarket. ``HandlerCodingAgentInvoke.handle()`` moved from
``handle(envelope) -> ModelHandlerOutput[None]`` (def-A) to
``handle(command) -> ModelCodingAgentResult`` (def-B); the runtime adapter now
supplies the already-validated typed command and unwraps the bare
``ModelCodingAgentResult`` return into the EFFECT's ``events[]`` itself.

The goldens under
``tests/fixtures/golden/node_coding_agent_invoke_effect/*.json`` were recorded by
replaying each scenario through BOTH the pre-flip envelope-wrapping ``handle()``
body and the post-flip ``handle(command)`` and confirming byte-equivalent output
(module ``duration_ms`` — wall-clock, non-deterministic). This test is the
durable regression: it drives the SAME scenarios through the live def-B
``handle()`` and asserts the recorded golden output still reproduces, so a
future edit to this handler cannot silently change behavior without failing
here.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from omnibase_infra.models.coding_agent.model_coding_agent_invoke_command import (
    ModelCodingAgentInvokeCommand,
)
from omnibase_infra.models.coding_agent.model_coding_agent_result import (
    ModelCodingAgentResult,
)
from omnibase_infra.models.coding_agent.model_subprocess_outcome import (
    ModelSubprocessOutcome,
)
from omnibase_infra.nodes.node_coding_agent_invoke_effect.handlers.handler_coding_agent_invoke import (
    HandlerCodingAgentInvoke,
)
from scripts.ci.compute_golden import compare_output

pytestmark = pytest.mark.unit

_GOLDEN_DIR = (
    Path(__file__).resolve().parents[4]
    / "tests"
    / "fixtures"
    / "golden"
    / "node_coding_agent_invoke_effect"
)
_CRED_HOME = "/home/omniinfra"


def _golden_files() -> list[Path]:
    files = sorted(_GOLDEN_DIR.glob("*.json"))
    assert files, f"no golden fixtures found under {_GOLDEN_DIR}"
    return files


class _Spy:
    def __init__(self, outcome: ModelSubprocessOutcome) -> None:
        self.outcome = outcome
        self.calls: list[object] = []

    def __call__(self, invocation: object) -> ModelSubprocessOutcome:
        self.calls.append(invocation)
        return self.outcome


def _handler_for_scenario(scenario: str) -> HandlerCodingAgentInvoke:
    """Rebuild the exact mocked-seam handler each golden was recorded against."""
    if scenario == "success_claude_read_only":
        return HandlerCodingAgentInvoke(
            run_subprocess=_Spy(
                ModelSubprocessOutcome(
                    returncode=0, stdout="done", stderr="", timed_out=False
                )
            ),
            probe_head_sha=lambda _cwd: "abc1234",
            capture_diff=lambda _cwd: (("foo.py",), "diff --git a/foo.py b/foo.py"),
            which=lambda _b: "/usr/bin/claude",
            agent_credential_home=_CRED_HOME,
        )
    if scenario == "success_codex_workspace_write":
        return HandlerCodingAgentInvoke(
            run_subprocess=_Spy(
                ModelSubprocessOutcome(
                    returncode=0, stdout="done", stderr="", timed_out=False
                )
            ),
            probe_head_sha=lambda _cwd: "def5678",
            capture_diff=lambda _cwd: ((), ""),
            which=lambda _b: "/usr/bin/codex",
            agent_credential_home=_CRED_HOME,
        )
    if scenario == "unavailable_cred_home_unset":
        return HandlerCodingAgentInvoke(
            run_subprocess=_Spy(
                ModelSubprocessOutcome(
                    returncode=0, stdout="x", stderr="", timed_out=False
                )
            ),
            which=lambda _b: "/usr/bin/claude",
        )
    if scenario == "unavailable_binary_missing":
        return HandlerCodingAgentInvoke(
            run_subprocess=_Spy(
                ModelSubprocessOutcome(
                    returncode=0, stdout="x", stderr="", timed_out=False
                )
            ),
            which=lambda _b: None,
            agent_credential_home=_CRED_HOME,
        )
    if scenario == "timeout":
        return HandlerCodingAgentInvoke(
            run_subprocess=_Spy(
                ModelSubprocessOutcome(
                    returncode=-9, stdout="", stderr="", timed_out=True
                )
            ),
            probe_head_sha=lambda _cwd: None,
            capture_diff=lambda _cwd: ((), ""),
            which=lambda _b: "/usr/bin/claude",
            agent_credential_home=_CRED_HOME,
        )
    if scenario == "subprocess_error":
        return HandlerCodingAgentInvoke(
            run_subprocess=_Spy(
                ModelSubprocessOutcome(
                    returncode=1, stdout="", stderr="boom", timed_out=False
                )
            ),
            probe_head_sha=lambda _cwd: None,
            capture_diff=lambda _cwd: ((), ""),
            which=lambda _b: "/usr/bin/claude",
            agent_credential_home=_CRED_HOME,
        )
    if scenario == "empty_response":
        return HandlerCodingAgentInvoke(
            run_subprocess=_Spy(
                ModelSubprocessOutcome(
                    returncode=0, stdout="   ", stderr="", timed_out=False
                )
            ),
            probe_head_sha=lambda _cwd: None,
            capture_diff=lambda _cwd: ((), ""),
            which=lambda _b: "/usr/bin/claude",
            agent_credential_home=_CRED_HOME,
        )
    raise AssertionError(f"no seam mapping for scenario {scenario!r}")


@pytest.mark.parametrize("golden_path", _golden_files(), ids=lambda p: p.stem)
async def test_handle_reproduces_recorded_golden(golden_path: Path) -> None:
    """def-B ``handle(command)`` on the recorded input == the recorded output."""
    golden: dict[str, Any] = json.loads(golden_path.read_text(encoding="utf-8"))
    command = ModelCodingAgentInvokeCommand.model_validate(golden["input"])
    handler = _handler_for_scenario(golden["scenario"])
    fresh_output = await handler.handle(command)
    assert isinstance(fresh_output, ModelCodingAgentResult)
    diffs = compare_output(golden, fresh_output)
    assert diffs == [], f"{golden_path.name}: handle() output diverged: {diffs}"


def test_golden_fixture_count_matches_expected_candidate_pool() -> None:
    """Regression guard: the recorded scenario pool has a known, reviewed size."""
    assert len(_golden_files()) == 7
