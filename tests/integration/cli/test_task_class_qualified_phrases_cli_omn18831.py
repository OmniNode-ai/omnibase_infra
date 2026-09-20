# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""CLI-level coverage for qualifier-gated selection phrases (OMN-18831).

The unit suite beside this one (``tests/unit/cli/test_task_class_selection_omn18831.py``)
drives ``resolve_task_type`` directly against a hand-built and a mirrored
contract. This test goes through the real ``onex delegate`` command --
``click`` option parsing, the real ``run_delegate`` contract resolution, the
real ``resolve_task_class`` call -- because the defect this ticket fixes was
only ever observed from the command line: the task class the CLI announces on
stderr is the class a caller's request is actually graded against, and
nothing in the unit suite exercises the announcement path itself.

The command is stopped immediately after classification by requesting
``--bus kafka`` with no ``--lane``/``--kafka-bootstrap``, which is a
``UsageError`` that fires only after the "task class: ..." line is already on
stderr (see ``run_delegate`` in ``cli_delegate.py``) -- so this reaches the
real classification path without a live model endpoint or a co-installed
omnimarket runtime.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from click.testing import CliRunner

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


def _invoke(
    prompt: str, monkeypatch: pytest.MonkeyPatch, *, mirror: Path = _MIRROR
) -> object:
    """Run the real CLI command against the production-mirror contract.

    ``--bus kafka`` with neither ``--lane`` nor ``--kafka-bootstrap`` is a
    deterministic, network-free ``UsageError`` raised AFTER classification, so
    the run reaches the real ``resolve_task_class`` path and stops cleanly
    before dispatch.
    """
    monkeypatch.setattr(
        cli_delegate, "resolve_task_class_contract_path", lambda: mirror
    )
    return CliRunner().invoke(
        delegate_command,
        [prompt, "--bus", "kafka"],
        catch_exceptions=False,
    )


class TestTheCliAnnouncesTheQualifierGatedClass:
    def test_an_unqualified_ambiguous_phrase_routes_to_the_prose_class(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """'write a' with no code object nearby must not reach a class whose
        acceptance bar is deterministic (OMN-18831's defect)."""
        result = _invoke(
            "Write a GitHub PR body in markdown from these facts.",
            monkeypatch,
        )
        assert result.exit_code != 0  # the deliberate post-classification stop
        assert "task class: document" in result.output

    def test_the_same_phrase_qualified_by_a_code_object_still_routes_to_code(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The positive control: 'write a' WITH a declared qualifier nearby
        must keep claiming the prompt, proving this is a word-sense fix and
        not a deletion of the phrase."""
        result = _invoke(
            "Write a parser for the lane manifest file.",
            monkeypatch,
        )
        assert result.exit_code != 0
        assert "task class: code_generation" in result.output

    def test_the_ordinary_english_sense_of_assertion_routes_to_the_prose_class(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        result = _invoke(
            "Weigh whether that assertion about the outage holds up.",
            monkeypatch,
        )
        assert result.exit_code != 0
        assert "task class: document" in result.output
