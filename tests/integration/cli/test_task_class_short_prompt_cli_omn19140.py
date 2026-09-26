# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""CLI-level coverage for short-prompt task-class selection (OMN-19140).

The unit suite beside this one
(``tests/unit/cli/test_task_class_selection_omn19140.py``) drives
``resolve_task_type`` directly. The defect this ticket fixes was visible from
the command line: a short request opening with a summarization imperative was
announced as ``document`` rather than ``summarization``. This test goes through
the real ``onex delegate`` command -- ``click`` option parsing, the real
``run_delegate`` contract resolution and the real classification call --
against the production-mirror contract of OMN-18831, and replays every prompt
in the verbatim OMN-19140 shadow corpus.

The command is stopped right after classification by requesting ``--bus
kafka`` with no ``--lane``/``--kafka-bootstrap``: a deterministic,
network-free ``UsageError`` raised after the "task class: ..." line is on
stderr (the same technique as ``test_task_class_request_scope_cli_omn19523``).
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner, Result

from omnibase_infra.cli import cli_delegate
from omnibase_infra.cli.cli_delegate import delegate_command

pytestmark = pytest.mark.integration

_FIXTURES = Path(__file__).resolve().parents[2] / "fixtures" / "delegation"
_MIRROR = _FIXTURES / "omn18831" / "task_class_selection_production_mirror.yaml"
_CORPUS = _FIXTURES / "omn19140" / "shadow_corpus.yaml"

_CAPTURED_SHORT = "Summarize in one sentence: the broker accepted this request."
_INCIDENTAL_NOUN = "Review this one-line summary of the broker change please."
_THIN_PROMPT = "Summarize this."


def _invoke(prompt: str, monkeypatch: pytest.MonkeyPatch) -> Result:
    """Run the real CLI command against the production-mirror contract."""
    monkeypatch.setattr(
        cli_delegate, "resolve_task_class_contract_path", lambda: _MIRROR
    )
    return CliRunner().invoke(
        delegate_command,
        [prompt, "--bus", "kafka"],
        catch_exceptions=False,
    )


def _corpus_rows() -> list[dict[str, object]]:
    raw = yaml.safe_load(_CORPUS.read_text(encoding="utf-8"))
    rows: list[dict[str, object]] = raw["rows"]
    return rows


_ROWS = _corpus_rows()


class TestTheCliAnnouncesTheShortPromptClass:
    def test_the_corpus_has_short_summarization_rows(self) -> None:
        """Positive control: the replay includes the below-floor defect."""
        assert _ROWS
        below_floor = [
            row
            for row in _ROWS
            if row["expected_after"] == "summarization" and int(str(row["words"])) < 120
        ]
        assert below_floor

    @pytest.mark.parametrize("row", _ROWS, ids=[f"row-{row['row']}" for row in _ROWS])
    def test_each_shadow_corpus_row_is_announced_as_expected(
        self, row: dict[str, object], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        result = _invoke(str(row["prompt"]), monkeypatch)
        assert result.exit_code != 0  # the deliberate post-classification stop
        assert f"task class: {row['expected_after']} (" in result.output

    def test_only_the_opening_summarization_verb_admits_a_short_request(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        captured = _invoke(_CAPTURED_SHORT, monkeypatch)
        incidental_noun = _invoke(_INCIDENTAL_NOUN, monkeypatch)

        assert captured.exit_code != 0
        assert "task class: summarization (" in captured.output
        assert incidental_noun.exit_code != 0
        # Positive control: a class WAS announced, so the absence below is real.
        assert "task class: " in incidental_noun.output
        assert "task class: summarization (" not in incidental_noun.output

    def test_a_request_below_the_short_prompt_floor_stays_out(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        result = _invoke(_THIN_PROMPT, monkeypatch)
        assert result.exit_code != 0
        # Positive control: a class WAS announced, so the absence below is real.
        assert "task class: " in result.output
        assert "task class: summarization (" not in result.output
