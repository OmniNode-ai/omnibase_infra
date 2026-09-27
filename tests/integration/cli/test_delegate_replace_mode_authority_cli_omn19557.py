# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""End-to-end CLI coverage for the OMN-19557 replace-mode authority refusal.

The unit module beside this one (``tests/unit/cli/test_delegate_replace_mode_authority_omn19557.py``)
drives ``_validate_replace_mode_authority`` directly. This one goes through
``click`` — the real command, the real option parsing, the real refusal path —
because the defect this ticket fixes was only reachable from a command line: a
Codex lane typed ``--criteria-mode replace-task-class`` with a reject-only
criteria set and climbed a full three-tier ladder to a refused terminal
(lane ``exercise-delegation-shadow``, ledger 4034). The fix moves that refusal
in front of dispatch, at the flag.

Follows the same seam as ``tests/integration/cli/test_delegate_criteria_cli.py``:
this environment has no omnimarket (repo layering forbids the dependency, and
the OMN-15620 venv-purity gate refuses to run this suite with it present), so
the acceptance-capable set and the vocabulary are supplied by monkeypatch
exactly as the real installed omnimarket's quality gate would resolve them.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from click.testing import CliRunner

from omnibase_infra.cli import cli_delegate
from omnibase_infra.cli.cli_delegate import delegate_command

pytestmark = pytest.mark.integration

#: What omnimarket's gate reports as acceptance-capable on 2026-09-25 (dispatch
#: venv, omnimarket 0.4.219): each of these, alone in a replacing set, holds
#: adequacy authority. Every other declared slug is reject-only.
_CAPABLE = frozenset(
    {
        "compiles_without_errors",
        "docstring_present",
        "final_artifact_only",
        "uses_pytest_mark_unit",
    }
)
_VOCABULARY = _CAPABLE | frozenset(
    {"concise", "task_completed", "plain_text_only", "no_refusal"}
)

#: The exact flags the Codex lane passed in run 7d83a5df-8824-4817-a70d-e5997b1a7af3.
_CODEX_C2_CRITERIA = ("concise", "task_completed", "plain_text_only")


@pytest.fixture
def _gate_resolvable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli_delegate, "load_supported_criteria", lambda: _VOCABULARY)
    monkeypatch.setattr(
        cli_delegate, "load_acceptance_capable_criteria", lambda: _CAPABLE
    )


def _invoke(args: list[str]) -> object:
    """Run the real command with option parsing, stopping before dispatch."""
    return CliRunner().invoke(delegate_command, args, catch_exceptions=False)


@pytest.mark.usefixtures("_gate_resolvable")
class TestReplaceModeWithoutAcceptanceAuthorityIsRefusedAtTheFlag:
    def test_the_measured_codex_flags_are_refused_before_dispatch(
        self, tmp_path: Path
    ) -> None:
        """RED before the fix: this reached dispatch and climbed the ladder.

        Exit code 2, the refusal names every acceptance-capable slug so the
        caller's next attempt can pick one, and no run directory is created —
        the whole point is refusing before the (expensive) dispatch call.
        """
        args = ["draft a pull request body"]
        for criterion in _CODEX_C2_CRITERIA:
            args += ["--criteria", criterion]
        args += [
            "--criteria-mode",
            "replace-task-class",
            "--task-type",
            "document",
            "--state-root",
            str(tmp_path),
        ]
        result = _invoke(args)
        assert result.exit_code == 2
        assert "can never be accepted" in result.output
        for slug in sorted(_CAPABLE):
            assert slug in result.output, f"refusal must name {slug!r}"
        assert "--response-contract" in result.output
        assert "omnimarket is NOT INSTALLED" not in result.output, (
            "the authority refusal must precede the drift guard"
        )
        assert not (tmp_path / "runs").exists(), "no run may be created"

    def test_declaring_a_response_contract_is_its_own_authority_and_proceeds_past_the_check(
        self, tmp_path: Path
    ) -> None:
        """Positive control: a declared --response-contract is not refused here.

        The gate validates against the contract instead of the class DoD, so
        this specific check must not raise; the command may still fail later
        (no live dispatch target in this suite), but not on this guard.
        """
        args = ["draft a pull request body"]
        for criterion in _CODEX_C2_CRITERIA:
            args += ["--criteria", criterion]
        args += [
            "--criteria-mode",
            "replace-task-class",
            "--response-contract",
            '{"type": "object"}',
            "--task-type",
            "document",
            "--state-root",
            str(tmp_path),
        ]
        result = _invoke(args)
        assert "can never be accepted" not in result.output

    def test_extend_mode_keeps_the_task_class_authority_and_is_not_refused(
        self, tmp_path: Path
    ) -> None:
        """Positive control: the same reject-only set under the default mode.

        Adding these criteria to the class bar (rather than replacing it)
        keeps the class's own acceptance authority, so this guard passes.
        """
        args = ["draft a pull request body"]
        for criterion in _CODEX_C2_CRITERIA:
            args += ["--criteria", criterion]
        args += [
            "--task-type",
            "document",
            "--state-root",
            str(tmp_path),
        ]
        result = _invoke(args)
        assert "can never be accepted" not in result.output
