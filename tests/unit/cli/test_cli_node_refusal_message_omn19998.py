# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Message precedence of the ``SkillRoutingError`` envelope (OMN-19998).

``run_node_by_name`` reports ``runtime.last_error`` first, then a refusing
handler's own ``error_message``, then a generic line. A stub runtime stands in
for ``ExecutionGraphHostingRuntime`` so each branch is driven directly.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from click.testing import CliRunner

from omnibase_core.enums.enum_workflow_result import EnumWorkflowResult
from omnibase_infra.cli import cli_node
from omnibase_infra.cli.cli_node import run_node_by_name

pytestmark = pytest.mark.unit

_NODE = "node_ci_live_contact_effect"


def _envelope_message(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    last_error: str | None,
    handler_result: object,
) -> str:
    class _StubRuntime:
        exit_code = 1

        def __init__(self, **_: object) -> None:
            self.last_error = last_error
            self.handler_result = handler_result

        def run(self) -> EnumWorkflowResult:
            return EnumWorkflowResult.FAILED

    monkeypatch.setattr(cli_node, "ExecutionGraphHostingRuntime", _StubRuntime)
    result = CliRunner().invoke(
        run_node_by_name, [_NODE, "--state-root", str(tmp_path / "state")]
    )
    assert result.exit_code == 1, result.output
    message: str = json.loads(result.output[result.output.rfind("{") :])["message"]
    return message


def test_last_error_wins_over_the_handler_error_message(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    message = _envelope_message(
        monkeypatch,
        tmp_path,
        last_error="runtime failure",
        handler_result=SimpleNamespace(error_message="handler refusal"),
    )

    assert message == "runtime failure"


def test_handler_error_message_is_used_when_last_error_is_unset(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    message = _envelope_message(
        monkeypatch,
        tmp_path,
        last_error=None,
        handler_result=SimpleNamespace(error_message="handler refusal"),
    )

    assert message == "handler refusal"


@pytest.mark.parametrize(
    "handler_result", [None, SimpleNamespace(), SimpleNamespace(error_message="")]
)
def test_no_message_falls_back_to_the_generic_line(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, handler_result: object
) -> None:
    message = _envelope_message(
        monkeypatch, tmp_path, last_error=None, handler_result=handler_result
    )

    assert message == f"Node '{_NODE}' run did not complete (result=failed)."
