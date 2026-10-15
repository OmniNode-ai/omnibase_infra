# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Named-test metadata survives the real in-process runtime (OMN-17427).

The CLI, payload file, contract loader, in-memory bus, typed handler request,
terminal event and receipt writer all run. Only omnimarket's handler is
replaced with a metadata echo, so no model endpoint or broker is needed.
This checks transport of the test specification, not execution of the tests
by omnimarket or support for named tests on the deployed lane.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner
from pydantic import Field

from omnibase_infra.cli import cli_delegate
from omnibase_infra.cli.cli_delegate import delegate_command
from tests.fixtures.handler_correlated_noop import (
    HandlerCorrelatedNoop,
    ModelCorrelatedNoopRequest,
    ModelDelegateSkillFixtureTerminal,
)

pytestmark = pytest.mark.integration


class ModelNamedTestEchoRequest(ModelCorrelatedNoopRequest):
    """Accept the metadata at the runtime's typed request boundary."""

    metadata: dict[str, str] = Field(default_factory=dict)


class HandlerNamedTestEcho:
    """Return the received metadata through the correlated terminal event."""

    def handle(
        self, request: ModelNamedTestEchoRequest
    ) -> ModelDelegateSkillFixtureTerminal:
        return (
            HandlerCorrelatedNoop()
            .handle(request)
            .model_copy(
                update={"response": json.dumps(request.metadata, sort_keys=True)}
            )
        )


@pytest.mark.parametrize("explicit_node_ids", [False, True])
def test_named_tests_reach_the_handler_and_persisted_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    explicit_node_ids: bool,
) -> None:
    module = __name__
    contract = tmp_path / "contract.yaml"
    contract.write_text(
        "name: named_test_echo\n"
        "node_type: compute\n"
        "terminal_event: onex.evt.proof.correlated-noop-completed.v1\n"
        f"input_model: {module}.ModelNamedTestEchoRequest\n"
        "handler:\n"
        f"  module: {module}\n"
        "  class: HandlerNamedTestEcho\n"
        f"  input_model: {module}.ModelNamedTestEchoRequest\n"
        "handler_routing:\n"
        f"  default_handler: {module}:HandlerNamedTestEcho\n",
        encoding="utf-8",
    )
    # Use the same offline contract substitution as the neighbouring CLI
    # integration suites; dispatch and receipt_mode remain the real functions.
    monkeypatch.delenv("OMNI_HOME", raising=False)
    monkeypatch.setattr(cli_delegate, "check_omnimarket_drift", lambda **_: None)
    monkeypatch.setattr(cli_delegate, "_resolve_packaged_contract", lambda _: contract)
    monkeypatch.setenv("ONEX_ARTIFACT_STORE_ROOT", str(tmp_path / "artifacts"))
    monkeypatch.chdir(tmp_path)

    sources = {
        "test_add.py": "from solution import add\n\ndef test_add():\n    assert add(1, 2) == 3\n",
        "test_zero.py": "from solution import add\n\ndef test_zero():\n    assert add(0, 0) == 0\n",
    }
    flags: list[str] = []
    for path, source in sources.items():
        (tmp_path / path).write_text(source, encoding="utf-8")
        flags.extend(("--test-file", path))
    if explicit_node_ids:
        expected_ids = "test_zero.py::test_zero test_add.py::test_add"
        for node_id in expected_ids.split():
            flags.extend(("--named-test", node_id))
    else:
        expected_ids = "test_add.py test_zero.py"

    state_root = tmp_path / "state"
    result = CliRunner().invoke(
        delegate_command,
        [
            "Write add(a, b).",
            "--task-type",
            "code_generation",
            "--ticket",
            "OMN-17427",
            *flags,
            "--answer-path",
            "solution.py",
            "--bus",
            "inmemory",
            "--locus",
            "in-process",
            "--state-root",
            str(state_root),
            "--emit-socket",
            str(tmp_path / "no-daemon.sock"),
            "--json",
        ],
        catch_exceptions=False,
    )

    assert result.exit_code == 0, result.output
    assert f"named tests: {expected_ids}" in result.stderr
    (workflow_path,) = (state_root / "runs").glob("*/workflow_result.json")
    workflow = json.loads(workflow_path.read_text(encoding="utf-8"))
    received = json.loads(workflow["handler_result"]["response"])
    assert received["ticket_id"] == "OMN-17427"
    assert received["named_tests"] == expected_ids
    assert received["named_tests_answer_path"] == "solution.py"
    assert json.loads(received["named_tests_files"]) == sources

    # The caller's artifact must carry the same handler response, rather than
    # checking only the input JSON before the runtime has consumed it.
    receipt = json.loads(
        (workflow_path.parent / "receipt.json").read_text(encoding="utf-8")
    )
    assert receipt["status"] == "success"
    assert (workflow_path.parent / "result.txt").read_text(encoding="utf-8") == (
        workflow["handler_result"]["response"]
    )
    (payload_path,) = (state_root / "tmp").glob("delegate-input-*.json")
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    assert receipt["correlation_id"] == payload["correlation_id"]
    assert receipt["run_id"] == workflow_path.parent.name
