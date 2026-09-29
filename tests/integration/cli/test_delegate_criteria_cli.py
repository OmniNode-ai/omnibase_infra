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
from tests.helpers.cli_registry_stand_in import (
    install_stand_in_registry,
    wiring_authority,
)
from tests.helpers.cli_registry_stand_in.node_delegate_stand_in.model_stand_in_delegate_request import (
    STAND_IN_CRITERIA,
)

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
        self, tmp_path: Path
    ) -> None:
        """RED before OMN-18305's residual: this reached dispatch and died in pydantic.

        The refusal must name the field and say what IS allowed, and it must
        fire BEFORE the omnimarket drift guard, which is the next thing in
        this command's path and which reports something unrelated. OMN-19407:
        the vocabulary is the delegate contract's input model, read through
        the registry -- here the stand-in node's model (see conftest).
        """
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
        assert "acceptance_criteria" in result.output
        assert "under 400 words" in result.output
        for slug in STAND_IN_CRITERIA:
            assert slug in result.output, "the refusal must list the vocabulary"
        assert "omnimarket is NOT INSTALLED" not in result.output, (
            "the criterion refusal must precede the drift guard, or the caller "
            "is told about a co-install when their flag value is the problem"
        )

    def test_a_declared_criterion_passes_the_contract_check(
        self, tmp_path: Path
    ) -> None:
        """Positive control: the refusal above is not refusing every criterion."""
        slug = sorted(STAND_IN_CRITERIA)[0]
        result = _invoke(
            [
                "draft two paragraphs",
                "--criteria",
                slug,
                "--state-root",
                str(tmp_path),
            ]
        )
        assert "acceptance_criteria" not in result.output

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
    def test_help_names_the_contracts_fallback_class(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A class is being chosen for the caller; the help says which.

        It named ``research`` before OMN-18305's residual, the class whose
        ``cites_sources`` floor refused a drafting prompt on every rung. Since
        OMN-19407 the fallback is the task-class contract's own declaration,
        read when help is rendered, so the help names whatever it declares.
        """
        authority = wiring_authority().model_copy(update={"fallback": "summarization"})
        install_stand_in_registry(monkeypatch, authority)
        help_text = " ".join(str(_invoke(["--help"]).output).split())
        assert "selection_fallback (summarization)" in help_text
