# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""A delegation that fails before publish is reported as itself (OMN-19131).

The measured defect: every ``onex delegate --timeout`` run between
2026-09-21 and 2026-09-22 failed inside ``RuntimeLocal._build_initial_payload``,
because the CLI wrote ``requested_timeout_seconds`` into a request whose
installed model forbade extra fields. Nothing was published. The operator was
then told the receipt "carries no resolvable delegation terminal", which is
the sentence for a bus-side outage, and the real error (field, model) existed
only in a capture log the CLI never named. One lane spent four attempts and a
wrong diagnosis on it.

These tests force exactly that failure through the real command, the real
``run_receipt_mode`` and the real ``RuntimeLocal``, against a stand-in request
model that forbids extras and declares no timeout field, and read what the
operator is shown:

* AC3: the terminal-resolution sentence never appears for a run that never
  published.
* AC4: the operator-visible error names the offending field, the rejecting
  model's import path, and the capture log path, on stderr and in the receipt.

The positive control runs the identical command without ``--timeout`` and
completes, so the stand-in contract is valid and the one difference is the
field the model refuses.
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path

import pytest
from click.testing import CliRunner, Result
from pydantic import BaseModel, ConfigDict, Field

from omnibase_core.enums.enum_skill_result_status import EnumSkillResultStatus
from omnibase_core.models.dispatch.model_skill_result import ModelSkillResult
from omnibase_infra.cli import cli_delegate
from omnibase_infra.cli.cli_delegate import (
    _delegate_receipt_evidence_error,
    _delegation_result,
    delegate_command,
)
from omnibase_infra.cli.delegate_caller import CALLER_LANE_ENV_VARS
from omnibase_infra.cli.delegate_pre_publish_failure import (
    DelegatePrePublishFailureError,
    describe_pre_publish_failure,
    pre_publish_failure_from_receipt,
)
from omnibase_infra.cli.delegate_terminal_resolver import (
    DelegateReplyTimeoutError,
    DelegateTerminalUnresolvedError,
)
from omnibase_infra.cli.model_receipt_runtime_summary import (
    ModelReceiptRuntimeSummary,
)
from omnibase_infra.runtime_identity import collect_runtime_identity
from tests.fixtures.handler_correlated_noop import (
    HandlerCorrelatedNoop,
    ModelCorrelatedNoopRequest,
    ModelDelegateSkillFixtureTerminal,
)

pytestmark = pytest.mark.integration

_TERMINAL_SENTENCE = "no resolvable delegation terminal"


class ModelTimeoutlessDelegateRequest(BaseModel):
    """Stand-in for the pre-OMN-18852 released request model.

    Forbids extra fields and declares no ``requested_timeout_seconds``, which
    is the exact shape of ``omnimarket`` 0.4.185's ``ModelDelegateSkillRequest``
    that refused every flagged run. It accepts ``metadata``, the map every
    released request model declares, because the command always sends a
    caller lane there (OMN-20299), even with every caller variable cleared.
    It accepts ``tenant_id`` for the same reason: omnimarket has declared it
    since v0.4.7 (OMN-14349), and the command stamps it (OMN-17427).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    correlation_id: str = ""
    prompt: str = ""
    task_type: str = ""
    source: str = ""
    metadata: dict[str, str] = Field(default_factory=dict)
    tenant_id: str | None = None


class HandlerTimeoutlessNoop(HandlerCorrelatedNoop):
    """The correlated no-op handler, fed from the timeout-less stand-in model."""

    def handle(  # type: ignore[override]
        self, request: ModelTimeoutlessDelegateRequest
    ) -> ModelDelegateSkillFixtureTerminal:
        return super().handle(
            ModelCorrelatedNoopRequest(
                correlation_id=request.correlation_id,
                prompt=request.prompt,
                task_type=request.task_type,
            )
        )


_MODEL_IMPORT_PATH = f"{__name__}.ModelTimeoutlessDelegateRequest"

_TIMEOUTLESS_CONTRACT = (
    "---\n"
    "name: correlated_noop\n"
    "node_type: compute\n"
    "terminal_event: onex.evt.proof.correlated-noop-completed.v1\n"
    f"input_model: {_MODEL_IMPORT_PATH}\n"
    "handler:\n"
    f"  module: {__name__}\n"
    "  class: HandlerTimeoutlessNoop\n"
    f"  input_model: {_MODEL_IMPORT_PATH}\n"
    "handler_routing:\n"
    f"  default_handler: {__name__}:HandlerTimeoutlessNoop\n"
)


@pytest.fixture(autouse=True)
def stand_in_contract(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Resolve the stand-in contract and keep the host's workspace out of it."""
    monkeypatch.delenv("OMNI_HOME", raising=False)
    monkeypatch.delenv("KAFKA_BOOTSTRAP_SERVERS", raising=False)
    monkeypatch.delenv("ONEX_CONTRACTS_DIR", raising=False)
    monkeypatch.setattr(cli_delegate, "check_omnimarket_drift", lambda **_: None)
    # Named for the delegate node, as the packaged contract is: the receipt
    # writer scopes itself to delegate runs by the workflow path.
    contract_path = tmp_path / cli_delegate.DELEGATE_NODE_NAME / "contract.yaml"
    contract_path.parent.mkdir()
    contract_path.write_text(_TIMEOUTLESS_CONTRACT, encoding="utf-8")
    monkeypatch.setattr(
        cli_delegate, "_resolve_packaged_contract", lambda _name: contract_path
    )
    monkeypatch.setenv("ONEX_ARTIFACT_STORE_ROOT", str(tmp_path / "artifacts"))
    return contract_path


#: The caller attribution the command derives from its environment and working
#: directory (OMN-19514's ticket, OMN-19860's lane and session). The stand-in
#: request model above predates both, so a run started from a ticket worktree
#: or a Claude Code session would be refused for them and the positive control
#: would fail for a reason that is not the one under test. Cleared here, and
#: the command runs from a directory that is not a ticket worktree.
_CALLER_ENV_CLEARED: dict[str, str | None] = dict.fromkeys(
    (*CALLER_LANE_ENV_VARS, "CLAUDE_CODE_SESSION_ID", "ONEX_LANE_REGISTRY_ROOT")
)


def _invoke(tmp_path: Path, *extra: str) -> tuple[Result, Path]:
    state_root = tmp_path / "state"
    runner = CliRunner(env=_CALLER_ENV_CLEARED)
    with runner.isolated_filesystem(temp_dir=tmp_path):
        result = runner.invoke(
            delegate_command,
            [
                "Reply with exactly the word READY",
                "--task-type",
                "summarization",
                "--bus",
                "inmemory",
                "--locus",
                "in-process",
                "--state-root",
                str(state_root),
                "--emit-socket",
                str(tmp_path / "no-daemon.sock"),
                *extra,
            ],
            catch_exceptions=False,
        )
    return result, state_root


def _sole_capture_log(state_root: Path) -> Path:
    logs = sorted((state_root / "captures").glob("*.log"))
    assert len(logs) == 1, f"expected exactly one capture log, got {logs}"
    return logs[0].resolve()


class TestPrePublishValidationFailureThroughTheCommand:
    """AC3 and AC4, end to end through the real command line."""

    def test_positive_control_without_the_field_completes(self, tmp_path: Path) -> None:
        """Same contract, same model, no ``--timeout``: the run completes.

        Without this, a stand-in contract that fails for any other reason would
        make every assertion below pass for the wrong reason.
        """
        result, _state_root = _invoke(tmp_path)

        assert result.exit_code == 0, result.stderr
        receipt = json.loads(result.stdout.strip())
        assert receipt["status"] == EnumSkillResultStatus.SUCCESS.value

    def test_the_terminal_sentence_never_reaches_the_operator(
        self, tmp_path: Path
    ) -> None:
        """AC3: a run that never published is not described as a lost terminal."""
        result, _state_root = _invoke(tmp_path, "--timeout", "60")

        assert result.exit_code != 0, "a refused payload is not a successful run"
        assert _TERMINAL_SENTENCE not in result.stderr
        assert _TERMINAL_SENTENCE not in result.stdout

    def test_stderr_names_field_model_and_capture_log(self, tmp_path: Path) -> None:
        """AC4: the reader never has to open the capture log to learn the field."""
        result, state_root = _invoke(tmp_path, "--timeout", "60")
        capture_log = _sole_capture_log(state_root)

        assert "requested_timeout_seconds" in result.stderr
        assert "supplied by `--timeout`" in result.stderr
        assert "extra_forbidden" in result.stderr
        assert _MODEL_IMPORT_PATH in result.stderr
        assert str(capture_log) in result.stderr
        assert "before publish" in result.stderr
        # The capture log named is the one holding the runtime's own refusal.
        assert "extra_forbidden" in capture_log.read_text(encoding="utf-8")

    def test_the_receipt_on_stdout_carries_the_same_cause(self, tmp_path: Path) -> None:
        """The typed receipt a caller parses says the same thing stderr does."""
        result, state_root = _invoke(tmp_path, "--timeout", "60")
        capture_log = _sole_capture_log(state_root)

        receipt = ModelSkillResult[ModelReceiptRuntimeSummary].model_validate(
            json.loads(result.stdout.strip())
        )
        assert receipt.status is EnumSkillResultStatus.FAILED
        assert receipt.exit_code != 0
        error = receipt.result.error
        assert "requested_timeout_seconds" in error
        assert "supplied by `--timeout`" in error
        assert _MODEL_IMPORT_PATH in error
        assert str(capture_log) in error
        assert receipt.result.wire_correlation_id is None


def _summary_receipt(
    *,
    workflow_result: str,
    wire_correlation_id: uuid.UUID | None,
    runtime_error_is_transport: bool = False,
    runtime_error_type: str = "",
) -> dict[str, object]:
    receipt = ModelSkillResult[ModelReceiptRuntimeSummary](
        skill_name=cli_delegate.DELEGATE_NODE_NAME,
        node_name=cli_delegate.DELEGATE_NODE_NAME,
        status=EnumSkillResultStatus.FAILED,
        correlation_id=uuid.uuid4(),
        run_id=uuid.uuid4(),
        exit_code=1,
        duration_ms=7,
        result=ModelReceiptRuntimeSummary(
            workflow_result=workflow_result,
            exit_code=1,
            workflow=(
                "/site-packages/omnimarket/nodes/"
                "node_delegate_skill_orchestrator/contract.yaml"
            ),
            wire_correlation_id=wire_correlation_id,
            runtime_error_is_transport=runtime_error_is_transport,
            runtime_error_type=runtime_error_type,
        ),
        result_model=(
            "omnibase_infra.cli.model_receipt_runtime_summary."
            "ModelReceiptRuntimeSummary"
        ),
        runtime_identity=collect_runtime_identity(config_source="test"),
    )
    dumped = receipt.model_dump(mode="json")
    assert isinstance(dumped, dict)
    return dumped


class TestPrePublishClassification:
    """Which receipts are pre-publish failures, and which stay what they were."""

    def test_failed_with_no_wire_correlation_is_pre_publish(self) -> None:
        envelope = _summary_receipt(workflow_result="failed", wire_correlation_id=None)

        assert pre_publish_failure_from_receipt(envelope) is not None
        with pytest.raises(DelegatePrePublishFailureError) as raised:
            _delegation_result(envelope)
        assert _TERMINAL_SENTENCE not in str(raised.value)
        # Still a DelegateTerminalUnresolvedError, so every existing catch site
        # keeps catching it.
        assert isinstance(raised.value, DelegateTerminalUnresolvedError)

    def test_published_run_without_a_terminal_keeps_the_terminal_sentence(
        self,
    ) -> None:
        """A run that DID publish and got nothing back is the bus-side case."""
        envelope = _summary_receipt(
            workflow_result="timeout", wire_correlation_id=uuid.uuid4()
        )

        assert pre_publish_failure_from_receipt(envelope) is None
        with pytest.raises(DelegateTerminalUnresolvedError) as raised:
            _delegation_result(envelope)
        assert not isinstance(raised.value, DelegatePrePublishFailureError)
        assert _TERMINAL_SENTENCE in str(raised.value)
        # OMN-20386: and it names the cause, so the caller is not left to infer it.
        assert isinstance(raised.value, DelegateReplyTimeoutError)
        assert "cause timeout" in str(raised.value)

    def test_completed_run_without_a_terminal_is_not_pre_publish(self) -> None:
        """OMN-18569's completed-but-empty shape keeps failing closed as before."""
        envelope = _summary_receipt(
            workflow_result="completed", wire_correlation_id=None
        )

        assert pre_publish_failure_from_receipt(envelope) is None

    def test_transport_failure_is_left_to_the_transport_refusal(self) -> None:
        """OMN-18925 owns a broker that never answered; this does not claim it."""
        envelope = _summary_receipt(
            workflow_result="error",
            wire_correlation_id=None,
            runtime_error_is_transport=True,
            runtime_error_type="InfraConnectionError",
        )

        assert pre_publish_failure_from_receipt(envelope) is None

    def test_validator_returns_the_cause_instead_of_raising(self) -> None:
        """The validator turns the receipt FAILED; it must not erase it by raising."""
        envelope = _summary_receipt(workflow_result="failed", wire_correlation_id=None)

        class _Receipt:
            def model_dump(self, *, mode: str) -> dict[str, object]:
                return envelope

        error = _delegate_receipt_evidence_error(_Receipt())
        assert error is not None
        assert "before publish" in error
        assert _TERMINAL_SENTENCE not in error


class TestDescribeNeverGuesses:
    """When no field-level refusal exists, the message says so instead of naming one."""

    def test_a_payload_the_model_accepts_names_no_field(
        self, tmp_path: Path, stand_in_contract: Path
    ) -> None:
        payload = tmp_path / "payload.json"
        payload.write_text(json.dumps({"prompt": "READY"}), encoding="utf-8")
        envelope = _summary_receipt(workflow_result="failed", wire_correlation_id=None)

        message = describe_pre_publish_failure(
            envelope=envelope,
            contract_path=stand_in_contract,
            payload_path=payload,
            capture_log_path=tmp_path / "capture.log",
        )

        assert "not a field-level refusal" in message
        assert _MODEL_IMPORT_PATH in message
        assert "capture.log" in message

    def test_an_unreadable_contract_names_no_model(self, tmp_path: Path) -> None:
        payload = tmp_path / "payload.json"
        payload.write_text("{}", encoding="utf-8")
        envelope = _summary_receipt(workflow_result="failed", wire_correlation_id=None)

        message = describe_pre_publish_failure(
            envelope=envelope,
            contract_path=tmp_path / "absent.yaml",
            payload_path=payload,
            capture_log_path=tmp_path / "capture.log",
        )

        assert "could not be read" in message
        assert "Cause: the request payload was refused" not in message
