# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""CLI-level coverage for the prose-output veto (OMN-18831, the residual).

The unit suite beside this one
(``tests/unit/cli/test_task_class_prose_output_veto_omn18831.py``) drives
``resolve_task_type`` directly against the digest-pinned production mirror.
This test goes through the real ``onex delegate`` command -- ``click`` option
parsing, the real ``run_delegate`` contract resolution, the real
``resolve_task_class`` call -- because the task class the CLI announces on
stderr is the class a caller's request is actually graded against, and the
veto's reason line is only useful if it reaches that announcement.

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

#: Run 1a67961a-664e-4a98-8680-378e8c393921, verbatim from its run.json: a
#: markdown table for a pull request description whose first row counts
#: "collectable pytest modules". Before the veto it resolved to ``test``.
_RECORDED_TABLE_REQUEST = (
    "Reformat these measured numbers into a markdown table of exactly four data "
    "rows plus a header, for a pull request description. Columns: Metric, "
    "Before, After, Delta. Output only the table, no prose. Numbers: collectable "
    "pytest modules summed over the last 20 merged pull requests, before 7014, "
    "after 7054, delta plus 40 which is plus 0.57 percent. Pull requests whose "
    "shard count changed, before 0, after 0, delta 0. Pull requests escalated to "
    "the whole suite, before 0, after 0, delta 0. Worst-case selector wall time "
    "in seconds, before 0.072, after 0.344, delta plus 0.272."
)


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


class TestTheCliAnnouncesTheVetoedClass:
    def test_the_recorded_table_request_is_announced_as_prose(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The recorded misroute: 'pytest' inside a PR-description table."""
        result = _invoke(_RECORDED_TABLE_REQUEST, monkeypatch)
        assert result.exit_code != 0  # the deliberate post-classification stop
        assert "task class: document" in result.output
        assert "vetoed: 'test' matched" in result.output
        assert "'pull request description'" in result.output

    def test_a_pr_body_listing_the_tests_that_ran_is_announced_as_prose(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        result = _invoke(
            "Draft a GitHub PR body in markdown. Sections: Summary, Tests. "
            "Tests: ran the unit tests for the handler and ruff; all pass. "
            "Output only the body.",
            monkeypatch,
        )
        assert result.exit_code != 0
        assert "task class: document" in result.output
        assert "vetoed: 'test' matched" in result.output


class TestGenuineCodeRequestsAreStillAnnouncedAsCode:
    def test_a_real_test_request_is_announced_as_test(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The positive control: the veto must not touch a request for tests."""
        result = _invoke(
            "Write unit tests for the retry helper in retry.py using pytest.",
            monkeypatch,
        )
        assert result.exit_code != 0
        assert "task class: test" in result.output
        assert "vetoed" not in result.output

    def test_a_code_request_that_says_no_prose_is_announced_as_code(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """'no prose' describes the answer's form; it is deliberately not a veto."""
        result = _invoke(
            "Write a POSIX-bash function named parse_reset_epoch. Input: one "
            "string argument, a refusal banner line. Output: print epoch seconds "
            "of the reset moment on stdout and return 0, or print nothing and "
            "return 1 when no reset time can be found. Return only the bash "
            "function body, no prose, no markdown fence.",
            monkeypatch,
        )
        assert result.exit_code != 0
        assert "task class: code_generation" in result.output
        assert "vetoed" not in result.output
