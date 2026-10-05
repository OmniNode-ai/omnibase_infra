# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Every way ``onex delegate`` ends writes a ``receipt.json`` (OMN-17427).

Built from the 2026-10-03 delegation consistency reconciliation: four runs on
the ``.201`` lane had a ``delegation_events`` row and no ``receipt.json``
because the caller's own 300 s limit ended the process before a terminal
arrived, and two more failed with a receipt that held no terminal payload and
left nothing on disk. Every lane is told to read the outcome from
``.onex_state/runs/<run_id>/receipt.json``, so on those runs the instruction
named a file that did not exist.

* a forced short timeout writes a ``timeout`` receipt
  -> :class:`TestForcedTimeoutWritesATimeoutReceipt`
* a caller that ends the process with ``SIGTERM`` writes an ``interrupted``
  receipt, and one that uses ``SIGKILL`` leaves the in-flight receipt filed
  before the wait -> :class:`TestCallerTimeoutLeavesAReceipt`
* a failed run writes its terminal cause -> :class:`TestFailedRunWritesItsCause`
"""

from __future__ import annotations

import json
import logging
import os
import signal
import time
import uuid
from pathlib import Path

import pytest

from omnibase_core.enums.enum_skill_result_status import EnumSkillResultStatus
from omnibase_core.models.dispatch.model_skill_result import ModelSkillResult
from omnibase_infra.cli import cli_delegate
from omnibase_infra.cli.cli_delegate import _write_local_run_files, run_delegate
from omnibase_infra.cli.delegate_terminal_resolver import (
    DelegateTerminalUnresolvedError,
)
from omnibase_infra.cli.model_delegate_run_addressing import (
    ModelDelegateRunAddressing,
)
from omnibase_infra.cli.model_receipt_runtime_summary import (
    ModelReceiptRuntimeSummary,
)
from omnibase_infra.enums.enum_delegate_locus import EnumDelegateLocus
from omnibase_infra.runtime_identity import collect_runtime_identity
from tests.helpers.cli_registry_stand_in import (
    install_stand_in_registry,
    wiring_authority,
)

pytestmark = pytest.mark.unit

_ADDRESSING = ModelDelegateRunAddressing(
    locus=EnumDelegateLocus.DEPLOYED_LANE,
    bus="kafka",
    lane="dev",
    dispatch_target="onex.cmd.omnimarket.delegate-skill.v1 via broker.example:19092",
)
_CONTRACT = (
    "/site-packages/omnimarket/nodes/node_delegate_skill_orchestrator/contract.yaml"
)


def _only_run_dir(state_root: Path) -> Path:
    runs = list((state_root / "runs").iterdir())
    assert len(runs) == 1, f"expected one run directory, found {runs}"
    return runs[0]


def _read_receipt(run_dir: Path) -> dict[str, object]:
    return json.loads((run_dir / "receipt.json").read_text(encoding="utf-8"))


def _prepare(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    monkeypatch.setattr(
        cli_delegate,
        "_resolve_packaged_contract",
        lambda _name: tmp_path / "contract.yaml",
    )
    install_stand_in_registry(
        monkeypatch, wiring_authority(terminal_delivery_margin_seconds=1)
    )
    # Shrink the grace window so a forced timeout does not wait out the default.
    monkeypatch.setattr(cli_delegate, "_HARD_TIMEOUT_GRACE_SECONDS", 1)
    return tmp_path / "state"


def _run(state_root: Path, tmp_path: Path) -> int:
    return run_delegate(
        prompt="research the routing architecture",
        task_type=None,
        max_tokens=None,
        state_root=state_root,
        timeout=1,
        verbose=False,
        emit_socket=tmp_path / "no-daemon.sock",
    )


def _swallowing_hang(**_kwargs: object) -> int:
    """Stands in for ``run_receipt_mode``: the real one wraps the hang in a broad except."""
    try:
        time.sleep(10)
    except Exception:
        logging.getLogger(__name__).exception("receipt_mode: runtime raised")
    return 1


class TestForcedTimeoutWritesATimeoutReceipt:
    def test_hard_timeout_leaves_a_timeout_receipt_naming_the_run(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        state_root = _prepare(monkeypatch, tmp_path)
        monkeypatch.setattr(cli_delegate, "run_receipt_mode", _swallowing_hang)

        exit_code = _run(state_root, tmp_path)
        captured = capsys.readouterr()

        assert exit_code == 1
        run_dir = _only_run_dir(state_root)
        receipt = _read_receipt(run_dir)
        assert receipt["terminal_class"] == "timeout"
        assert receipt["status"] == EnumSkillResultStatus.FAILED.value
        assert receipt["terminal_recorded"] is False
        assert receipt["declared_timeout_seconds"] == 2
        assert receipt["run_id"] == run_dir.name
        # The ids on disk are the ones the caller was shown on stdout.
        printed = json.loads(captured.out.strip().splitlines()[-1])
        assert receipt["correlation_id"] == printed["correlation_id"]
        assert receipt["run_id"] == printed["run_id"]
        for name in ("result.txt", "receipt.json", "run.json"):
            assert (run_dir / name).exists(), f"{name} was not written"


class TestCallerTimeoutLeavesAReceipt:
    def test_sigterm_from_the_caller_writes_an_interrupted_receipt(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        state_root = _prepare(monkeypatch, tmp_path)

        def _caller_ends_the_process(**_kwargs: object) -> int:
            os.kill(os.getpid(), signal.SIGTERM)
            time.sleep(5)
            return 0

        monkeypatch.setattr(cli_delegate, "run_receipt_mode", _caller_ends_the_process)

        exit_code = _run(state_root, tmp_path)

        assert exit_code == 128 + signal.SIGTERM
        receipt = _read_receipt(_only_run_dir(state_root))
        assert receipt["terminal_class"] == "interrupted"
        assert receipt["signal"] == "SIGTERM"
        assert receipt["terminal_recorded"] is False
        assert receipt["correlation_id"]

    def test_signal_handlers_are_restored_after_the_run(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        state_root = _prepare(monkeypatch, tmp_path)
        monkeypatch.setattr(cli_delegate, "run_receipt_mode", lambda **_k: 0)
        before = signal.getsignal(signal.SIGTERM)

        _run(state_root, tmp_path)

        assert signal.getsignal(signal.SIGTERM) is before

    def test_a_receipt_is_on_disk_before_the_wait_so_a_sigkill_leaves_one(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """SIGKILL cannot be answered; what exists mid-wait is what a kill leaves."""
        state_root = _prepare(monkeypatch, tmp_path)
        seen: dict[str, object] = {}

        def _observe_mid_wait(**kwargs: object) -> int:
            run_id = str(kwargs["run_id"])
            seen["run_id"] = run_id
            seen["receipt"] = _read_receipt(state_root / "runs" / run_id)
            return 1

        monkeypatch.setattr(cli_delegate, "run_receipt_mode", _observe_mid_wait)

        _run(state_root, tmp_path)

        in_flight = seen["receipt"]
        assert isinstance(in_flight, dict)
        assert in_flight["terminal_class"] == "in_flight"
        assert in_flight["status"] == EnumSkillResultStatus.PENDING.value
        assert in_flight["terminal_recorded"] is False
        assert in_flight["run_id"] == seen["run_id"]
        assert in_flight["correlation_id"]

    def test_an_in_flight_receipt_is_settled_when_no_writer_replaced_it(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        state_root = _prepare(monkeypatch, tmp_path)
        monkeypatch.setattr(cli_delegate, "run_receipt_mode", lambda **_k: 1)

        exit_code = _run(state_root, tmp_path)

        assert exit_code == 1
        receipt = _read_receipt(_only_run_dir(state_root))
        assert receipt["terminal_class"] == "failed"
        assert receipt["status"] == EnumSkillResultStatus.FAILED.value
        assert receipt["exit_code"] == 1

    def test_a_terminal_writers_receipt_is_left_as_written(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        state_root = _prepare(monkeypatch, tmp_path)

        def _writer_replaces_it(**kwargs: object) -> int:
            run_dir = state_root / "runs" / str(kwargs["run_id"])
            (run_dir / "receipt.json").write_text(
                json.dumps({"status": "success", "model": "answered"}),
                encoding="utf-8",
            )
            return 0

        monkeypatch.setattr(cli_delegate, "run_receipt_mode", _writer_replaces_it)

        assert _run(state_root, tmp_path) == 0

        assert _read_receipt(_only_run_dir(state_root)) == {
            "status": "success",
            "model": "answered",
        }


def _summary_receipt(
    *,
    workflow_result: str,
    error: str,
    error_type: str,
    published: bool = True,
) -> ModelSkillResult[ModelReceiptRuntimeSummary]:
    """A receipt shaped as ``run_receipt_mode`` builds one for a run with no terminal."""
    return ModelSkillResult[ModelReceiptRuntimeSummary](
        skill_name=cli_delegate.DELEGATE_NODE_NAME,
        node_name=cli_delegate.DELEGATE_NODE_NAME,
        status=EnumSkillResultStatus.FAILED,
        correlation_id=uuid.uuid4(),
        run_id=uuid.uuid4(),
        exit_code=1,
        duration_ms=300_000,
        result=ModelReceiptRuntimeSummary(
            workflow_result=workflow_result,
            exit_code=1,
            workflow=_CONTRACT,
            terminal_payload=None,
            handler_result=None,
            wire_correlation_id=uuid.uuid4() if published else None,
            error=error,
            runtime_error_type=error_type,
        ),
        result_model=(
            "omnibase_infra.cli.model_receipt_runtime_summary."
            "ModelReceiptRuntimeSummary"
        ),
        runtime_identity=collect_runtime_identity(config_source="test"),
    )


def _write(receipt: object, state_root: Path) -> None:
    _write_local_run_files(
        receipt=receipt,
        state_root=state_root,
        addressing=_ADDRESSING,
        prompt="Reply with exactly the word READY",
        task_type="summarization",
        task_type_resolution="explicit",
    )


class TestFailedRunWritesItsCause:
    def test_a_failed_run_with_no_terminal_payload_writes_its_terminal_cause(
        self, tmp_path: Path
    ) -> None:
        receipt = _summary_receipt(
            workflow_result="failed",
            error="provider returned HTTP 503 for every rung",
            error_type="DelegationFailed",
        )

        with pytest.raises(DelegateTerminalUnresolvedError):
            _write(receipt, tmp_path)

        written = _read_receipt(tmp_path / "runs" / str(receipt.run_id))
        assert written["terminal_class"] == "failed"
        assert written["terminal_recorded"] is False
        assert written["correlation_id"] == str(receipt.correlation_id)
        assert written["runtime_error_type"] == "DelegationFailed"
        assert written["runtime_error"] == "provider returned HTTP 503 for every rung"
        assert written["terminal_payload"] is None
        assert "no resolvable delegation terminal" in str(written["failure_reason"])

    def test_a_run_refused_before_publish_writes_its_cause(
        self, tmp_path: Path
    ) -> None:
        receipt = _summary_receipt(
            workflow_result="failed",
            error="payload refused by the request model",
            error_type="DelegationFailed",
            published=False,
        )

        with pytest.raises(DelegateTerminalUnresolvedError):
            _write(receipt, tmp_path)

        written = _read_receipt(tmp_path / "runs" / str(receipt.run_id))
        assert written["terminal_class"] == "failed"
        assert "failed before publish" in str(written["failure_reason"])

    def test_a_runtime_timeout_without_a_terminal_is_a_timeout_receipt(
        self, tmp_path: Path
    ) -> None:
        receipt = _summary_receipt(
            workflow_result="timeout", error="", error_type="TimeoutError"
        )

        with pytest.raises(DelegateTerminalUnresolvedError):
            _write(receipt, tmp_path)

        written = _read_receipt(tmp_path / "runs" / str(receipt.run_id))
        assert written["terminal_class"] == "timeout"
        assert written["workflow_result"] == "timeout"
