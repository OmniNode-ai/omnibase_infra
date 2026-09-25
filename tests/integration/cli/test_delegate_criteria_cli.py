# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""End-to-end CLI coverage for caller-stated criteria (OMN-18305 residual).

The unit module beside this one drives ``_write_payload`` and the helpers
directly. This one goes through ``click`` — the real command, the real option
parsing, the real refusal path — because every defect this ticket's residual
covers was reachable only from a command line, and two of them were invisible
until a flag was actually typed.

Measured 2026-09-15, live, before this change:

* a drafting prompt that no predicate claimed fell back to ``research`` and was
  refused on every rung for missing source citations;
* the first attempt to state criteria instead failed 265 ms in with a pydantic
  ``ValidationError``, zero rungs attempted, and no mention of which flag was
  wrong — because ``acceptance_criteria`` is a closed slug set and nothing said
  so at the flag.

Neither is observable from the payload writer alone.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from omnibase_infra.cli.cli_delegate import delegate_command

pytestmark = pytest.mark.integration


def _invoke(args: list[str]) -> object:
    """Run the real command with option parsing, stopping before dispatch.

    Dispatch itself needs a co-installed omnimarket and a live model endpoint,
    neither of which belongs in this gate. Everything under test here happens
    in front of dispatch: option parsing, the alias collapse, criterion
    validation, and response-contract loading.
    """
    return CliRunner().invoke(delegate_command, args, catch_exceptions=False)


class TestTheCriteriaFlagIsReachableAndChecked:
    def test_a_free_text_criterion_is_refused_with_a_usable_message(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """RED before the fix: this reached dispatch and died in pydantic.

        The refusal must name the flag and say what IS allowed, because the
        caller's next action is to pick a different value for this flag. It
        must also fire BEFORE the omnimarket drift guard, which is the next
        thing in this command's path and which reports something unrelated.

        The vocabulary is supplied here rather than resolved: this environment
        has no omnimarket (repo layering forbids the dependency and the
        OMN-15620 venv-purity gate refuses to run the suite with it present),
        so the real resolution returns ``None`` and criteria pass through by
        design. What is under test is the refusal, not the co-install.
        """
        from omnibase_infra.cli import cli_delegate

        monkeypatch.setattr(
            cli_delegate,
            "load_supported_criteria",
            lambda: frozenset({"task_completed", "concise"}),
        )
        result = _invoke(
            [
                "draft two paragraphs of rationale prose",
                "--criteria",
                "under 400 words",
                "--state-root",
                str(tmp_path),
            ]
        )
        assert result.exit_code != 0
        assert (
            "--criteria takes declared criterion slugs, not free text" in result.output
        )
        assert "'under 400 words'" in result.output
        assert "task_completed" in result.output, "the refusal must list the vocabulary"
        assert "omnimarket is NOT INSTALLED" not in result.output, (
            "the criterion refusal must precede the drift guard, or the caller "
            "is told about a co-install when their flag value is the problem"
        )

    def test_without_a_resolvable_vocabulary_criteria_pass_through(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Positive control for the refusal above, and the documented seam.

        No omnimarket means no vocabulary. Refusing here would turn a missing
        co-install into a misleading complaint about the caller's criteria, so
        the command proceeds and fails on its own next guard instead.
        """
        from omnibase_infra.cli import cli_delegate

        monkeypatch.setattr(cli_delegate, "load_supported_criteria", lambda: None)
        result = _invoke(
            [
                "draft two paragraphs",
                "--criteria",
                "under 400 words",
                "--state-root",
                str(tmp_path),
            ]
        )
        assert "--criteria takes declared criterion slugs" not in result.output

    def test_the_flags_exist_and_are_documented(self) -> None:
        """A flag nobody can discover is a flag nobody uses."""
        help_text = str(_invoke(["--help"]).output)
        for flag in (
            "--criteria",
            "--criteria-mode",
            "--response-contract",
            "--system-prompt",
            "--task-class",
        ):
            assert flag in help_text


class TestTheTwoSpellingsOfTheClassFlag:
    def test_passing_both_spellings_is_refused(self, tmp_path: Path) -> None:
        result = _invoke(
            [
                "anything",
                "--task-type",
                "document",
                "--task-class",
                "research",
                "--state-root",
                str(tmp_path),
            ]
        )
        assert result.exit_code != 0
        assert "pass one, not both" in result.output


class TestTheResponseContractFlag:
    def test_a_non_object_contract_is_refused_naming_the_flag(
        self, tmp_path: Path
    ) -> None:
        result = _invoke(
            [
                "anything",
                "--response-contract",
                "[1, 2, 3]",
                "--state-root",
                str(tmp_path),
            ]
        )
        assert result.exit_code != 0
        assert "--response-contract must be a JSON object" in result.output

    def test_an_unreadable_json_path_is_refused_naming_the_flag(
        self, tmp_path: Path
    ) -> None:
        result = _invoke(
            [
                "anything",
                "--response-contract",
                str(tmp_path / "absent.json"),
                "--state-root",
                str(tmp_path),
            ]
        )
        assert result.exit_code != 0
        assert "not a readable file" in result.output

    def test_a_readable_contract_file_parses(self, tmp_path: Path) -> None:
        """Positive control: the refusals above are not refusing everything."""
        from omnibase_infra.cli.cli_delegate import _load_response_contract

        path = tmp_path / "shape.json"
        path.write_text(json.dumps({"type": "object"}), encoding="utf-8")
        assert _load_response_contract(str(path)) == {"type": "object"}


class TestTheFallbackIsAnnouncedInHelp:
    def test_help_names_the_fallback_class(self) -> None:
        """A class is being chosen for the caller; the help says which.

        It named ``research`` before this change. That is the class whose
        ``cites_sources`` floor refused a drafting prompt on every rung.
        """
        from omnibase_infra.cli.task_class_selection import DEFAULT_TASK_TYPE

        help_text = str(_invoke(["--help"]).output)
        assert DEFAULT_TASK_TYPE in help_text
        assert "the fallback when none" in help_text


class TestAReplaceModeBarThatCannotAcceptIsRefusedAtTheFlag:
    """OMN-18925: run 7d83a5df climbed six rungs on a bar that could accept nothing."""

    def test_the_measured_command_is_refused_before_dispatch(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from omnibase_infra.cli import cli_delegate

        monkeypatch.setattr(
            cli_delegate,
            "load_supported_criteria",
            lambda: frozenset(
                {"concise", "task_completed", "plain_text_only", "final_artifact_only"}
            ),
        )
        monkeypatch.setattr(
            cli_delegate,
            "load_criteria_adequacy_authority",
            lambda: lambda criteria: "final_artifact_only" in criteria,
        )
        args = [
            "draft a PR body",
            "--task-type",
            "document",
            "--criteria",
            "concise",
            "--criteria",
            "task_completed",
            "--criteria",
            "plain_text_only",
            "--criteria-mode",
            "replace-task-class",
            "--state-root",
            str(tmp_path),
        ]
        result = _invoke(args)
        assert result.exit_code == 2, result.output
        assert "no adequacy authority" in result.output
        assert "omnimarket is NOT INSTALLED" not in result.output, (
            "the bar refusal must precede the drift guard"
        )
        assert not (tmp_path / "runs").exists(), "no run may be created"

    def test_the_same_criteria_in_extend_mode_are_not_refused(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Positive control: extend mode keeps the class's own authority."""
        from omnibase_infra.cli import cli_delegate

        monkeypatch.setattr(
            cli_delegate,
            "load_supported_criteria",
            lambda: frozenset({"concise", "task_completed", "plain_text_only"}),
        )
        monkeypatch.setattr(
            cli_delegate,
            "load_criteria_adequacy_authority",
            lambda: lambda criteria: False,
        )
        result = _invoke(
            [
                "draft a PR body",
                "--criteria",
                "concise",
                "--state-root",
                str(tmp_path),
            ]
        )
        assert "no adequacy authority" not in result.output
