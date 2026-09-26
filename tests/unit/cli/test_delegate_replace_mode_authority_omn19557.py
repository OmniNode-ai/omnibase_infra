# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""A replacing criteria set that can never accept is refused at the flag (OMN-19557).

THE DEFECT. ``--criteria-mode replace-task-class`` makes the caller's criteria
the WHOLE bar: the quality gate drops the task class's own definition of done,
and with it the class's acceptance authority (for prose classes, the
``semantic_adequacy`` judge). Most declared criterion slugs are reject-only
under OMN-13370: ``concise``, ``task_completed``, ``plain_text_only``,
``no_refusal``, ``response_non_empty`` and the rest can FAIL an answer but can
never ACCEPT one. A replacing set made only of them therefore has no acceptance
authority, and the gate refuses every answer with ``TASK_MISMATCH: no
deterministic acceptance or judge adequacy authority``, however good it is.

Measured, not hypothesised:

* lane ``exercise-delegation-shadow`` (ledger 4034, 2026-09-25): ten full-ladder
  runs, every local rung scored 1.0 and was refused, the ladder climbed through
  ``cheap_frontier`` and ``cheap_cloud``, and every run ended failed;
* a Codex lane (codex-batch-83 c2, run ``7d83a5df-8824-4817-a70d-e5997b1a7af3``)
  passed ``--task-type document --criteria concise --criteria task_completed
  --criteria plain_text_only --criteria-mode replace-task-class``: twelve
  attempts across three tiers, quality 1.0, terminal ``quality_gate_refused``.
  Re-running the gate in process on that run's own answer with the default
  ``extend`` mode accepts it.

The outcome is fixed before the first call, so the CLI refuses it before the
first call, and names what WOULD accept.

The acceptance-capable set is resolved from the installed omnimarket's quality
gate, never copied here: this suite runs without omnimarket (OMN-15620 venv
purity), so the resolver is supplied by monkeypatch exactly as the vocabulary
is in ``tests/integration/cli/test_delegate_criteria_cli.py``.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from click.testing import CliRunner

from omnibase_infra.cli import cli_delegate
from omnibase_infra.cli.cli_delegate import (
    _validate_replace_mode_authority,
    delegate_command,
)

pytestmark = pytest.mark.unit

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

#: The exact flags the Codex lane passed in run 7d83a5df.
_CODEX_C2_CRITERIA = ("concise", "task_completed", "plain_text_only")


@pytest.fixture
def _gate_resolvable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli_delegate, "load_supported_criteria", lambda: _VOCABULARY)
    monkeypatch.setattr(
        cli_delegate, "load_acceptance_capable_criteria", lambda: _CAPABLE
    )


@pytest.mark.usefixtures("_gate_resolvable")
class TestAnUnpassableReplacingSetIsRefused:
    def test_the_measured_codex_flags_are_refused_naming_what_can_accept(
        self,
    ) -> None:
        """RED before the fix: this set reached dispatch and climbed the ladder."""
        with pytest.raises(ValueError, match="can never be accepted") as exc:
            _validate_replace_mode_authority(
                _CODEX_C2_CRITERIA, "replace-task-class", response_contract=None
            )
        message = str(exc.value)
        for slug in sorted(_CAPABLE):
            assert slug in message, f"the refusal must name {slug!r} as able to accept"
        assert "--response-contract" in message
        assert "OMN-19557" in message

    def test_replace_mode_with_no_criteria_at_all_is_refused(self) -> None:
        """An empty replacing set leaves the gate its legacy path, which never accepts."""
        with pytest.raises(ValueError, match="can never be accepted"):
            _validate_replace_mode_authority(
                (), "replace-task-class", response_contract=None
            )

    def test_the_refusal_reaches_the_command_line_as_a_usage_error(
        self, tmp_path: Path
    ) -> None:
        """Through click: exit 2, before dispatch and before the drift guard."""
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
        result = CliRunner().invoke(delegate_command, args, catch_exceptions=False)
        assert result.exit_code == 2
        assert "can never be accepted" in result.output
        assert "omnimarket is NOT INSTALLED" not in result.output
        assert not (tmp_path / "runs").exists(), "no run may be created"


@pytest.mark.usefixtures("_gate_resolvable")
class TestPositiveControls:
    """The refusal is narrow: every passable shape still goes through."""

    def test_one_acceptance_capable_criterion_is_enough(self) -> None:
        criteria = (*_CODEX_C2_CRITERIA, "final_artifact_only")
        _validate_replace_mode_authority(
            criteria, "replace-task-class", response_contract=None
        )  # must not raise

    def test_a_declared_response_contract_is_its_own_authority(self) -> None:
        """The gate validates against the contract and skips the DoD path entirely."""
        _validate_replace_mode_authority(
            _CODEX_C2_CRITERIA,
            "replace-task-class",
            response_contract={"type": "object"},
        )  # must not raise

    @pytest.mark.parametrize("mode", [None, "extend-task-class"])
    def test_extending_keeps_the_task_class_authority(self, mode: str | None) -> None:
        """The same reject-only set ADDED to the class bar keeps the class's judge."""
        _validate_replace_mode_authority(
            _CODEX_C2_CRITERIA, mode, response_contract=None
        )  # must not raise


def test_an_unresolvable_gate_passes_through(monkeypatch: pytest.MonkeyPatch) -> None:
    """No omnimarket, no classification: never refuse on a guess.

    Mirrors ``load_supported_criteria``: without the co-install this command
    cannot dispatch anyway, and the next guard reports the real cause.
    """
    monkeypatch.setattr(cli_delegate, "load_acceptance_capable_criteria", lambda: None)
    _validate_replace_mode_authority(
        _CODEX_C2_CRITERIA, "replace-task-class", response_contract=None
    )  # must not raise


def test_the_help_says_criteria_are_slugs_and_warns_about_replace() -> None:
    """The Codex lanes read --help and then passed free-text criteria.

    Three of four measured Codex attempts wrote sentences into ``--criteria``
    because the help described it as "an acceptance criterion this answer must
    meet". The help must say it takes declared slugs, and must say that
    replacing with reject-only slugs cannot accept.
    """
    help_text = " ".join(
        str(CliRunner().invoke(delegate_command, ["--help"]).output).split()
    )
    assert "declared criterion slug" in help_text
    assert "not free text" in help_text
    assert "reject-only" in help_text
