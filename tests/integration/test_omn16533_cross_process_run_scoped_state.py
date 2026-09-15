# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""OMN-16533 across real process boundaries.

The reported defect was between PROCESSES: separate ``onex delegate``
invocations, each its own interpreter, echoing one another's recorded state.
The unit suite drives the same code in one interpreter through ``CliRunner``,
which cannot observe a boundary it never crosses — two real processes racing
for one filename is the shape that actually failed.

These tests spawn the CLI as a subprocess, twice, against ONE state root, and
assert each process keeps the result it produced. Nothing is mocked: a real
interpreter, a real contract, a real runtime, a real filesystem.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import uuid
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration

_PROOF_NOOP_CONTRACT = (
    "---\n"
    "name: proof_noop\n"
    "node_type: compute\n"
    "terminal_event: onex.evt.proof.noop-completed.v1\n"
    "handler:\n"
    "  module: tests.fixtures.handler_proof_noop\n"
    "  class: HandlerProofNoop\n"
    "  input_model: tests.fixtures.handler_proof_noop.ModelProofNoopRequest\n"
    "handler_routing:\n"
    "  default_handler: tests.fixtures.handler_proof_noop:HandlerProofNoop\n"
)

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _invoke_in_subprocess(
    *,
    workspace: Path,
    state_root: Path,
    cwd: Path,
    name: str,
) -> dict[str, object]:
    """Run ``onex node ... --output receipt`` as its own process.

    Returns the parsed receipt. ``cwd`` is deliberately a caller-chosen
    directory: the default state root is a relative path, so where the process
    stands is part of what is under test.
    """
    contract_path = workspace / f"{name}-contract.yaml"
    contract_path.write_text(_PROOF_NOOP_CONTRACT, encoding="utf-8")
    input_path = workspace / f"{name}-input.json"
    input_path.write_text(json.dumps({"name": name, "count": 3}), encoding="utf-8")

    env = dict(os.environ)
    env["ONEX_ARTIFACT_STORE_ROOT"] = str(workspace / "artifacts")
    # The fixture handler is importable only from the repo root, so the repo
    # root is PREPENDED rather than assigned: CI resolves `omnibase_infra`
    # itself through the inherited PYTHONPATH, and replacing the value made
    # the child process unable to import the package under test.
    inherited_path = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = os.pathsep.join(
        part for part in (str(_REPO_ROOT), inherited_path) if part
    )
    # The omnimarket co-install drift guard (OMN-13930) binds to $OMNI_HOME and
    # governs dispatch of MARKET-provided nodes. This test dispatches a fixture
    # contract written moments ago in tmp_path — no market node is reachable
    # from it — so the guard has nothing to say here, and CI runs with the
    # variable unset anyway. Dropping it makes a developer's box behave as CI
    # does rather than depending on whose workspace the suite runs in.
    env.pop("OMNI_HOME", None)

    # The click command is invoked by import rather than through the `onex`
    # console script on purpose: which package's `onex` wins the PATH differs
    # between a developer venv and the CI environment, and a test that spawns
    # a real process should fail on the behaviour under test, never on which
    # script happened to be installed. This is still a real process — its own
    # interpreter, its own working directory, its own filesystem view.
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "from omnibase_infra.cli.cli_node import run_node_by_name; "
            "run_node_by_name()",
            "proof_noop",
            "--contract",
            str(contract_path),
            "--input",
            str(input_path),
            "--state-root",
            str(state_root),
            "--output",
            "receipt",
            "--emit-socket",
            str(workspace / "no-daemon.sock"),
        ],
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    assert completed.returncode == 0, (
        f"{name} exited {completed.returncode}\n"
        f"stdout: {completed.stdout}\nstderr: {completed.stderr}"
    )
    parsed: object = json.loads(completed.stdout.strip())
    assert isinstance(parsed, dict)
    return parsed


def _result_files(state_root: Path) -> list[Path]:
    return sorted(state_root.rglob("workflow_result.json"))


@pytest.mark.integration
class TestCrossProcessRunScopedState:
    def test_two_processes_sharing_one_state_root_keep_their_own_results(
        self, tmp_path: Path
    ) -> None:
        """The reported shape: separate processes, one state root.

        Before the fix these two processes wrote the same filename, so the
        second silently destroyed the first's record. Each now owns a path the
        other cannot name.
        """
        state_root = tmp_path / "shared-state"
        cwd = tmp_path / "cwd"
        cwd.mkdir()

        first = _invoke_in_subprocess(
            workspace=tmp_path, state_root=state_root, cwd=cwd, name="process-one"
        )
        after_first = _result_files(state_root)
        assert len(after_first) == 1
        first_bytes = after_first[0].read_bytes()

        second = _invoke_in_subprocess(
            workspace=tmp_path, state_root=state_root, cwd=cwd, name="process-two"
        )

        assert first["run_id"] != second["run_id"]
        assert first["correlation_id"] != second["correlation_id"]

        after_second = _result_files(state_root)
        assert len(after_second) == 2, (
            f"the second process overwrote the first's record: {after_second}"
        )
        assert after_first[0].read_bytes() == first_bytes

        # Each receipt's run_id names the directory holding that run's result,
        # and each result file agrees about whose it is.
        for receipt in (first, second):
            run_id = receipt["run_id"]
            assert isinstance(run_id, str)
            uuid.UUID(run_id)
            own = state_root / "runs" / run_id / "workflow_result.json"
            assert own.exists(), f"run {run_id} has no result of its own"
            assert json.loads(own.read_text(encoding="utf-8"))["run_id"] == run_id

        assert not (state_root / "workflow_result.json").exists()

    def test_each_process_reports_its_own_result_not_the_others(
        self, tmp_path: Path
    ) -> None:
        """Content, not just paths: neither receipt carries the other's work.

        The fixture handler echoes the name it was given, so a receipt that
        picked up the peer's record would say so in its own body.
        """
        state_root = tmp_path / "shared-state"
        cwd = tmp_path / "cwd"
        cwd.mkdir()

        alpha = _invoke_in_subprocess(
            workspace=tmp_path, state_root=state_root, cwd=cwd, name="alpha"
        )
        bravo = _invoke_in_subprocess(
            workspace=tmp_path, state_root=state_root, cwd=cwd, name="bravo"
        )

        alpha_body = alpha["result"]
        bravo_body = bravo["result"]
        assert isinstance(alpha_body, dict) and isinstance(bravo_body, dict)
        assert alpha_body.get("echoed_name") == "alpha"
        assert bravo_body.get("echoed_name") == "bravo"
        assert alpha["status"] == "success"
        assert bravo["status"] == "success"

    def test_an_explicit_state_root_keeps_state_out_of_the_working_directory(
        self, tmp_path: Path
    ) -> None:
        """``--state-root`` is the whole contract for where state lands.

        Checked from a real process because the default state root is
        relative, so only a process with its own working directory can prove
        nothing leaked into one.
        """
        cwd = tmp_path / "cwd"
        cwd.mkdir()
        _invoke_in_subprocess(
            workspace=tmp_path,
            state_root=tmp_path / "explicit-state",
            cwd=cwd,
            name="cwd-probe",
        )
        assert list(cwd.iterdir()) == [], (
            "an explicit --state-root must leave the working directory untouched"
        )
