# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""CLI-level coverage for request-scoped task-class selection (OMN-19523).

The unit suite beside this one
(``tests/unit/cli/test_task_class_selection_payload_scope.py``) drives
``resolve_task_type`` directly. The misroutes this ticket fixes were observed
from the command line: the class ``onex delegate`` announces on stderr is the
class a caller's request is graded against. This test goes through the real
``onex delegate`` command -- ``click`` option parsing, the real
``run_delegate`` contract resolution and the real classification call --
against the production-mirror contract of OMN-18831, and replays the recorded
misrouted prompts committed verbatim under
``tests/fixtures/delegation/omn19523/``.

The command is stopped right after classification by requesting ``--bus
kafka`` with no ``--lane``/``--kafka-bootstrap``: a deterministic,
network-free ``UsageError`` raised after the "task class: ..." line is on
stderr (the same technique as ``test_task_class_qualified_phrases_cli_omn18831``).
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner

from omnibase_infra.cli import cli_delegate
from omnibase_infra.cli.cli_delegate import delegate_command

pytestmark = pytest.mark.integration

_FIXTURES = Path(__file__).resolve().parents[2] / "fixtures" / "delegation"
_MIRROR = _FIXTURES / "omn18831" / "task_class_selection_production_mirror.yaml"
_CORPUS = _FIXTURES / "omn19523" / "misrouted_prompts.yaml"


def _invoke(prompt: str, monkeypatch: pytest.MonkeyPatch) -> object:
    monkeypatch.setattr(
        cli_delegate, "resolve_task_class_contract_path", lambda: _MIRROR
    )
    return CliRunner().invoke(
        delegate_command,
        [prompt, "--bus", "kafka"],
        catch_exceptions=False,
    )


def _routable_rows() -> list[dict[str, object]]:
    raw = yaml.safe_load(_CORPUS.read_text(encoding="utf-8"))
    rows: list[dict[str, object]] = raw["prompts"]
    return [row for row in rows if row.get("needed_class")]


_ROWS = _routable_rows()


class TestTheCliAnnouncesTheRequestScopedClass:
    def test_the_corpus_has_routable_rows(self) -> None:
        """Positive control: an empty corpus would pass the replay vacuously."""
        assert _ROWS

    @pytest.mark.parametrize(
        "row", _ROWS, ids=[str(row["file"]).rsplit("/", 1)[-1] for row in _ROWS]
    )
    def test_a_recorded_misrouted_prompt_is_announced_as_the_needed_class(
        self, row: dict[str, object], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        prompt = (_CORPUS.parent / str(row["file"])).read_text(encoding="utf-8")
        result = _invoke(prompt, monkeypatch)
        assert result.exit_code != 0  # the deliberate post-classification stop
        assert f"task class: {row['needed_class']} (" in result.output
        assert f"task class: {row['was']} (" not in result.output

    def test_a_phrase_only_inside_a_fence_does_not_change_the_announced_class(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        request = (
            "Resolve this git merge conflict in a hooks file and output only "
            "the resolved text in one block."
        )
        fenced = (
            f"{request}\n\n```yaml\nPlease write a test for the parser module.\n```\n"
        )
        bare = _invoke(f"{request}\n", monkeypatch)
        with_block = _invoke(fenced, monkeypatch)
        announced = [
            line.split(" (", 1)[0]
            for line in bare.output.splitlines()
            if line.startswith("task class:")
        ]
        assert announced, bare.output
        assert f"{announced[0]} (" in with_block.output
