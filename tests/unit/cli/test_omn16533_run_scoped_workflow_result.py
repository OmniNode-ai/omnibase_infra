# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""OMN-16533: the workflow result belongs to ONE invocation, not to a directory.

``RuntimeLocal`` serialises its workflow result to
``<its state_root>/workflow_result.json`` — one FIXED filename. Every
``onex node`` / ``onex skill`` / ``onex delegate`` invocation that shared a
state root therefore shared one file, and the default state root is the
relative string ``.onex_state``, which resolves against whatever working
directory the process happens to be in. Two unrelated invocations from the
same directory collided by accident rather than by choice.

Two landed joins already stop the collision being served as truth: the
OMN-15449 ``run_id`` anchor join and the OMN-17295 correlation join. They are
DETECTION — they refuse a foreign file. The run that wrote a perfectly good
result still loses it, and reports "no receipt found" for work it actually
completed. That is the residual this module pins.

The fix is structural: the runtime is handed a state root private to this
invocation (``<state root>/runs/<run_id>``), so the result file is keyed by
the run that wrote it and no peer can address it at all. The joins stay as
defence in depth.
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path

import pytest
from click.testing import CliRunner, Result

from omnibase_infra.cli.cli_node import run_node_by_name
from omnibase_infra.cli.receipt_mode import WORKFLOW_RESULT_FILENAME

pytestmark = pytest.mark.unit

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


def _run_against(
    *,
    workspace: Path,
    state_root: Path,
    monkeypatch: pytest.MonkeyPatch,
    name: str,
) -> Result:
    """One real CLI invocation in receipt mode against ``state_root``."""
    contract_path = workspace / f"{name}-contract.yaml"
    contract_path.write_text(_PROOF_NOOP_CONTRACT, encoding="utf-8")
    input_path = workspace / f"{name}-input.json"
    input_path.write_text(json.dumps({"name": name, "count": 3}), encoding="utf-8")
    monkeypatch.setenv("ONEX_ARTIFACT_STORE_ROOT", str(workspace / "artifacts"))
    return CliRunner().invoke(
        run_node_by_name,
        [
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
        catch_exceptions=False,
    )


def _result_files(state_root: Path) -> list[Path]:
    return sorted(state_root.rglob(WORKFLOW_RESULT_FILENAME))


def _receipt(stdout: str) -> dict[str, object]:
    parsed: object = json.loads(stdout.strip())
    assert isinstance(parsed, dict)
    return parsed


class TestWorkflowResultIsRunScoped:
    def test_result_is_keyed_by_this_runs_own_run_id(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The file lives under this run's key, never at the shared root.

        A path no peer invocation can address is the only thing that makes
        the collision structurally impossible rather than merely detected.
        """
        state_root = tmp_path / "state"
        result = _run_against(
            workspace=tmp_path,
            state_root=state_root,
            monkeypatch=monkeypatch,
            name="solo",
        )
        assert result.exit_code == 0, result.stdout

        assert not (state_root / WORKFLOW_RESULT_FILENAME).exists(), (
            "the shared, un-keyed path is the collision surface — nothing "
            "may be written there"
        )
        written = _result_files(state_root)
        assert len(written) == 1, f"expected exactly one result file, got {written}"
        run_dir = written[0].parent
        stored = json.loads(written[0].read_text(encoding="utf-8"))
        assert stored["run_id"] == run_dir.name, (
            "the directory key and the stamped run_id must be the same run"
        )
        assert run_dir.parent == state_root / "runs"
        uuid.UUID(run_dir.name)

    def test_a_second_invocation_cannot_clobber_the_first_runs_result(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Two runs, one state root, two intact results.

        This is the reported shape with the timing removed: the second run
        does not have to be concurrent to destroy the first's record, it only
        has to reuse the directory.
        """
        state_root = tmp_path / "state"
        first = _run_against(
            workspace=tmp_path,
            state_root=state_root,
            monkeypatch=monkeypatch,
            name="first",
        )
        assert first.exit_code == 0, first.stdout
        after_first = _result_files(state_root)
        assert len(after_first) == 1
        first_bytes = after_first[0].read_bytes()

        second = _run_against(
            workspace=tmp_path,
            state_root=state_root,
            monkeypatch=monkeypatch,
            name="second",
        )
        assert second.exit_code == 0, second.stdout

        after_second = _result_files(state_root)
        assert len(after_second) == 2, (
            f"the second run overwrote the first's record: {after_second}"
        )
        assert after_first[0].read_bytes() == first_bytes, (
            "the first run's own result must survive a later invocation"
        )
        run_ids = {
            json.loads(path.read_text(encoding="utf-8"))["run_id"]
            for path in after_second
        }
        assert len(run_ids) == 2

    def test_a_peer_write_cannot_cost_this_run_the_result_it_produced(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A peer's write must not turn a completed run into "no receipt".

        The OMN-15449 join correctly refuses foreign content, but when the
        foreign content lands at the address this run reads from — a peer
        invocation finishing in the window between this run's write and this
        run's read — the refusal costs the run the result it genuinely
        produced. Reading a run-keyed path makes the peer's file unreachable
        rather than merely untrusted.

        The peer is simulated exactly as the OMN-15449 suite simulates it:
        the write is planted at the shared root immediately before the read.
        """
        from omnibase_infra.cli import receipt_mode as receipt_mode_module

        state_root = tmp_path / "state"
        real_load = receipt_mode_module._load_workflow_data

        def _plant_peer_write_then_load(read_root: Path) -> dict[str, object]:
            state_root.mkdir(parents=True, exist_ok=True)
            (state_root / WORKFLOW_RESULT_FILENAME).write_text(
                json.dumps(
                    {
                        "result": "completed",
                        "exit_code": 0,
                        "workflow": "a-concurrent-unrelated-invocation",
                        "run_id": str(uuid.uuid4()),
                        "handler_result": {
                            "status": "success",
                            "echoed_name": "someone-elses-run",
                            "echoed_count": 999,
                        },
                    }
                ),
                encoding="utf-8",
            )
            return real_load(read_root)

        monkeypatch.setattr(
            receipt_mode_module, "_load_workflow_data", _plant_peer_write_then_load
        )

        result = _run_against(
            workspace=tmp_path,
            state_root=state_root,
            monkeypatch=monkeypatch,
            name="victim",
        )

        assert result.exit_code == 0, (
            "a completed run must report its own result, not a refusal caused "
            f"by a stranger's file: {result.stdout}"
        )
        payload = _receipt(result.stdout)
        body = payload["result"]
        assert isinstance(body, dict)
        assert body.get("echoed_name") == "victim"
        assert "anchor-join refusal" not in json.dumps(payload)

    def test_nothing_is_written_relative_to_the_working_directory(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``--state-root`` is the whole contract for where state lands.

        A CWD-relative write is how unrelated invocations came to share a
        state root without anybody choosing one.
        """
        cwd = tmp_path / "cwd"
        cwd.mkdir()
        monkeypatch.chdir(cwd)
        result = _run_against(
            workspace=tmp_path,
            state_root=tmp_path / "state",
            monkeypatch=monkeypatch,
            name="cwd-probe",
        )
        assert result.exit_code == 0, result.stdout
        assert not (cwd / ".onex_state").exists()
        assert list(cwd.iterdir()) == []
