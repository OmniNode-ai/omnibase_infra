# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""CLI-level coverage for the short-prompt admission rule (OMN-19140).

The unit suite beside this one
(``tests/unit/cli/test_task_class_selection_omn19140.py``) drives
``resolve_task_type`` directly against the digest-pinned production mirror.
This test goes through the real ``onex delegate`` command -- ``click`` option
parsing, the real ``run_delegate`` contract resolution, the real
``resolve_task_class`` call -- because the task class the CLI announces on
stderr is the class a caller's request is actually graded against: the
defect was that a one-sentence summary request was announced as the
``document`` fallback.

The command is stopped immediately after classification by requesting
``--bus kafka`` with no ``--lane``/``--kafka-bootstrap``, the same
deterministic, network-free ``UsageError`` the qualifier-gated CLI test uses.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from click.testing import CliRunner, Result

from omnibase_infra.cli import cli_delegate
from omnibase_infra.cli.cli_delegate import delegate_command

pytestmark = pytest.mark.integration

_MIRROR = (
    Path(__file__).resolve().parents[2]
    / "fixtures"
    / "delegation"
    / "omn18831"
    / "task_class_selection_production_mirror.yaml"
)

#: The captured 9-word prompt the OMN-19136 report used as its positive control.
_CAPTURED_SHORT = "Summarize in one sentence: the broker accepted this request."


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


class TestTheCliAnnouncesTheShortSummaryRequest:
    def test_the_captured_short_request_is_announced_as_summarization(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        result = _invoke(_CAPTURED_SHORT, monkeypatch)
        assert result.exit_code != 0  # the deliberate post-classification stop
        assert "task class: summarization" in result.output
        assert "at the start of a 9-word prompt" in result.output

    def test_the_padded_control_is_announced_as_summarization_too(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """AC2: the fix does not trade the long direction for the short one."""
        padded = _CAPTURED_SHORT + " The broker accepted this request." * 25
        result = _invoke(padded, monkeypatch)
        assert result.exit_code != 0
        assert "task class: summarization" in result.output
        assert "at the start" not in result.output


class TestTheFloorStillKeepsIncidentalUsesOut:
    def test_a_thin_request_is_not_announced_as_summarization(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """AC4: below the short-prompt floor there is nothing to summarise."""
        result = _invoke("Summarize the standup.", monkeypatch)
        assert result.exit_code != 0
        assert "task class: summarization" not in result.output

    def test_the_verb_mid_sentence_in_a_code_request_is_announced_as_code(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """AC3: the verb must OPEN a short prompt to admit it."""
        result = _invoke(
            "Implement a function that can summarize log lines.", monkeypatch
        )
        assert result.exit_code != 0
        assert "task class: code_generation" in result.output
