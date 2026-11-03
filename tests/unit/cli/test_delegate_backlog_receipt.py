# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""A send that ends while its command is still queued records a typed terminal (OMN-22211).

On the deployed dev lane a delegation whose command sat behind a backlog on the
delegate-skill consumer ended ``terminal_class: timeout`` with no attempt and no
rung, which is byte-identical to a run the consumer lost. The queue depth the
CLI observed was in the receipt, but the class said nothing: a caller had to
open ``queue_depth`` to learn the command had never started.

* a hard-timeout end with records ahead records ``queued_not_started`` and names
  the depth -> :class:`TestQueuedNotStartedHardTimeout`
* the runtime's own cooperative timeout ends the same way ->
  :class:`TestQueuedNotStartedRuntimeTimeout`
* the receipt of run 91fda261, scrubbed -> :class:`TestReceipt91fda261`
* no records ahead, or an unobserved depth, stays a bare ``timeout`` ->
  :class:`TestAnythingElseStaysATimeout`
"""

from __future__ import annotations

import json
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
from omnibase_infra.cli.model_delegate_queue_depth import ModelDelegateQueueDepth
from omnibase_infra.cli.model_delegate_run_addressing import (
    ModelDelegateRunAddressing,
)
from omnibase_infra.cli.model_receipt_runtime_summary import (
    ModelReceiptRuntimeSummary,
)
from omnibase_infra.enums.enum_delegate_incomplete_class import (
    EnumDelegateIncompleteClass,
)
from omnibase_infra.enums.enum_delegate_locus import EnumDelegateLocus
from omnibase_infra.runtime_identity import collect_runtime_identity
from tests.helpers.cli_registry_stand_in import (
    install_stand_in_registry,
    wiring_authority,
)

pytestmark = pytest.mark.unit

_FIXTURE = (
    Path(__file__).resolve().parent.parent.parent
    / "fixtures"
    / "omn22211"
    / "receipt_91fda261_scrubbed.json"
)
_ADDRESSING = ModelDelegateRunAddressing(
    locus=EnumDelegateLocus.DEPLOYED_LANE,
    bus="kafka",
    lane="dev",
    dispatch_target="delegate-command-topic via broker.example:19092",
)
_CONTRACT = (
    "/site-packages/omnimarket/nodes/node_delegate_skill_orchestrator/contract.yaml"
)
_GROUP = "delegate-consumer-group"


def _depth(records_ahead: int) -> ModelDelegateQueueDepth:
    return ModelDelegateQueueDepth(
        records_ahead=records_ahead,
        backlog_records=records_ahead + 1,
        consumer_group=_GROUP,
    )


def _only_run_dir(state_root: Path) -> Path:
    runs = list((state_root / "runs").iterdir())
    assert len(runs) == 1, f"expected one run directory, found {runs}"
    return runs[0]


def _read_receipt(run_dir: Path) -> dict[str, object]:
    receipt: dict[str, object] = json.loads(
        (run_dir / "receipt.json").read_text(encoding="utf-8")
    )
    return receipt


def _prepare(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    monkeypatch.setattr(
        cli_delegate,
        "_resolve_packaged_contract",
        lambda _name: tmp_path / "contract.yaml",
    )
    install_stand_in_registry(
        monkeypatch, wiring_authority(terminal_delivery_margin_seconds=1)
    )
    monkeypatch.setattr(cli_delegate, "_HARD_TIMEOUT_GRACE_SECONDS", 1)
    return tmp_path / "state"


def _hang(**_kwargs: object) -> int:
    time.sleep(10)
    return 1


def _send_that_hits_the_hard_timeout(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    depth: ModelDelegateQueueDepth,
    *,
    json_output: bool = False,
) -> tuple[int, Path]:
    state_root = _prepare(monkeypatch, tmp_path)
    monkeypatch.setattr(cli_delegate, "run_receipt_mode", _hang)
    monkeypatch.setattr(
        cli_delegate, "observe_delegate_queue_depth", lambda **_kwargs: depth
    )
    exit_code = run_delegate(
        prompt="2^100 mod 997",
        task_type=None,
        max_tokens=None,
        state_root=state_root,
        timeout=1,
        verbose=False,
        emit_socket=tmp_path / "no-daemon.sock",
        json_output=json_output,
    )
    return exit_code, _only_run_dir(state_root)


def _runtime_timeout_receipt() -> ModelSkillResult[ModelReceiptRuntimeSummary]:
    """A receipt shaped as ``run_receipt_mode`` builds one when its own wait expires."""
    return ModelSkillResult[ModelReceiptRuntimeSummary](
        skill_name=cli_delegate.DELEGATE_NODE_NAME,
        node_name=cli_delegate.DELEGATE_NODE_NAME,
        status=EnumSkillResultStatus.FAILED,
        correlation_id=uuid.uuid4(),
        run_id=uuid.uuid4(),
        exit_code=1,
        duration_ms=300_000,
        result=ModelReceiptRuntimeSummary(
            workflow_result="timeout",
            exit_code=1,
            workflow=_CONTRACT,
            terminal_payload=None,
            handler_result=None,
            wire_correlation_id=uuid.uuid4(),
            error="",
            runtime_error_type="TimeoutError",
        ),
        result_model=(
            "omnibase_infra.cli.model_receipt_runtime_summary."
            "ModelReceiptRuntimeSummary"
        ),
        runtime_identity=collect_runtime_identity(config_source="test"),
    )


def _write_runtime_timeout(
    receipt: ModelSkillResult[ModelReceiptRuntimeSummary],
    state_root: Path,
    depth: ModelDelegateQueueDepth,
) -> dict[str, object]:
    with pytest.raises(DelegateTerminalUnresolvedError):
        _write_local_run_files(
            receipt=receipt,
            state_root=state_root,
            addressing=_ADDRESSING,
            prompt="2^100 mod 997",
            task_type="reasoning",
            task_type_resolution="explicit",
            queue_depth_observer=lambda: depth,
        )
    run_id = str(receipt.run_id)
    return _read_receipt(state_root / "runs" / run_id)


class TestQueuedNotStartedHardTimeout:
    def test_queued_not_started_names_the_backlog_depth(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        exit_code, run_dir = _send_that_hits_the_hard_timeout(
            monkeypatch, tmp_path, _depth(50)
        )

        receipt = _read_receipt(run_dir)
        assert exit_code == 1
        assert receipt["terminal_class"] == "queued_not_started"
        assert receipt["terminal_class"] != EnumDelegateIncompleteClass.TIMEOUT.value
        assert receipt["status"] == EnumSkillResultStatus.FAILED.value
        assert receipt["terminal_recorded"] is False
        assert receipt["route_attributed"] is False
        assert receipt["attempts"] == []
        queue_depth = receipt["queue_depth"]
        assert isinstance(queue_depth, dict)
        assert queue_depth["records_ahead"] == 50
        assert queue_depth["backlog_records"] == 51
        assert queue_depth["consumer_group"] == _GROUP
        reason = str(receipt["failure_reason"])
        assert "50 record(s)" in reason
        assert "still queued" in reason

    def test_queued_not_started_stdout_result_still_names_the_depth(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        _send_that_hits_the_hard_timeout(
            monkeypatch, tmp_path, _depth(50), json_output=True
        )

        printed = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
        assert printed["result"]["queue_depth"]["records_ahead"] == 50


class TestQueuedNotStartedRuntimeTimeout:
    def test_queued_not_started_when_the_runtime_wait_expires(
        self, tmp_path: Path
    ) -> None:
        receipt = _runtime_timeout_receipt()

        written = _write_runtime_timeout(receipt, tmp_path, _depth(50))

        assert written["terminal_class"] == "queued_not_started"
        assert written["terminal_recorded"] is False
        assert written["route_attributed"] is False
        assert written["workflow_result"] == "timeout"
        queue_depth = written["queue_depth"]
        assert isinstance(queue_depth, dict)
        assert queue_depth["records_ahead"] == 50
        assert "50 record(s)" in str(written["failure_reason"])


class TestReceipt91fda261:
    """The recorded receipt of run 91fda261: 50 records ahead, a backlog of 51."""

    def _recorded_depth(self) -> ModelDelegateQueueDepth:
        recorded = json.loads(_FIXTURE.read_text(encoding="utf-8"))
        return ModelDelegateQueueDepth.model_validate(recorded["queue_depth"])

    def test_receipt_91fda261_fixture_is_the_recorded_backlog(self) -> None:
        depth = self._recorded_depth()

        assert depth.records_ahead == 50
        assert depth.backlog_records == 51

    def test_receipt_91fda261_hard_timeout_is_queued_not_started(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        recorded = json.loads(_FIXTURE.read_text(encoding="utf-8"))
        _, run_dir = _send_that_hits_the_hard_timeout(
            monkeypatch, tmp_path, self._recorded_depth()
        )

        receipt = _read_receipt(run_dir)
        assert receipt["terminal_class"] == recorded["expected_terminal_class"]
        assert receipt["queue_depth"] == recorded["queue_depth"]
        assert receipt["attempts"] == []
        assert receipt["route_attributed"] is False

    def test_receipt_91fda261_runtime_timeout_is_queued_not_started(
        self, tmp_path: Path
    ) -> None:
        written = _write_runtime_timeout(
            _runtime_timeout_receipt(), tmp_path, self._recorded_depth()
        )

        assert written["terminal_class"] == "queued_not_started"
        queue_depth = written["queue_depth"]
        assert isinstance(queue_depth, dict)
        assert queue_depth["records_ahead"] == 50


class TestAnythingElseStaysATimeout:
    """Positive controls: the new class is not what every timeout becomes."""

    def test_no_records_ahead_is_a_bare_timeout(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _, run_dir = _send_that_hits_the_hard_timeout(monkeypatch, tmp_path, _depth(0))

        assert _read_receipt(run_dir)["terminal_class"] == "timeout"

    def test_an_unobserved_depth_is_a_bare_timeout(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        unobserved = ModelDelegateQueueDepth(
            unresolved_reason="the broker did not answer a lag query"
        )

        _, run_dir = _send_that_hits_the_hard_timeout(monkeypatch, tmp_path, unobserved)

        assert _read_receipt(run_dir)["terminal_class"] == "timeout"

    def test_a_runtime_timeout_with_no_records_ahead_is_a_bare_timeout(
        self, tmp_path: Path
    ) -> None:
        written = _write_runtime_timeout(
            _runtime_timeout_receipt(), tmp_path, _depth(0)
        )

        assert written["terminal_class"] == "timeout"
