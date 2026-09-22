# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID

import pytest

from omnibase_core.artifacts.artifact_store import ArtifactStore
from omnibase_core.enums.artifacts.enum_artifact_retention_class import (
    EnumArtifactRetentionClass,
)
from omnibase_core.enums.enum_execution_locus_kind import EnumExecutionLocusKind
from omnibase_core.enums.enum_package_source_kind import EnumPackageSourceKind
from omnibase_core.enums.enum_skill_result_status import EnumSkillResultStatus
from omnibase_core.models.dispatch.model_skill_result import ModelSkillResult
from omnibase_core.models.runtime.model_package_identity import ModelPackageIdentity
from omnibase_core.models.runtime.model_runtime_identity import ModelRuntimeIdentity
from omnibase_infra.cli.cli_delegate import _write_issued_delegation_evidence
from omnibase_infra.cli.model_delegate_locus_decision import ModelDelegateLocusDecision
from omnibase_infra.cli.model_delegate_run_addressing import ModelDelegateRunAddressing
from omnibase_infra.cli.model_receipt_runtime_summary import ModelReceiptRuntimeSummary
from omnibase_infra.enums.enum_delegate_locus import EnumDelegateLocus

RUN_ID = UUID("00000000-0000-0000-0000-000000000001")
CORRELATION_ID = UUID("00000000-0000-0000-0000-000000000002")


def _receipt(
    *, status: EnumSkillResultStatus, failure_code: str | None = None
) -> ModelSkillResult[object]:
    store = ArtifactStore()
    artifact = store.write_blob(
        b"actual handler artifact",
        media_type="application/json",
        artifact_kind="handler_result",
        source_system="onex_cli",
        scope_ref="node_delegate_skill_orchestrator",
        correlation_id=str(CORRELATION_ID),
        retention_class=EnumArtifactRetentionClass.SESSION,
    )
    terminal = {"attempts": [], "response": "caller-visible bytes"}
    result: object = terminal
    result_model = "omnimarket.models.ModelDelegateSkillResponse"
    if not status.is_success_like:
        terminal["response"] = ""
        terminal["failure_code"] = failure_code
        result = ModelReceiptRuntimeSummary(
            workflow_result="failed",
            exit_code=1,
            workflow="contracts/node_delegate_skill_orchestrator.yaml",
            terminal_payload=terminal,
        )
        result_model = (
            "omnibase_infra.cli.model_receipt_runtime_summary."
            "ModelReceiptRuntimeSummary"
        )
    return ModelSkillResult[object](
        skill_name="delegate",
        node_name="node_delegate_skill_orchestrator",
        status=status,
        correlation_id=CORRELATION_ID,
        run_id=RUN_ID,
        exit_code=0 if status.is_success_like else 1,
        duration_ms=12,
        result=result,
        result_model=result_model,
        artifact_refs=[artifact],
        runtime_identity=ModelRuntimeIdentity(
            host="proof-host",
            locus_kind=EnumExecutionLocusKind.CONTAINER,
            execution_locus="proof-container",
            interpreter="/app/.venv/bin/python",
            packages={
                "omnimarket": ModelPackageIdentity(
                    name="omnimarket",
                    version="0.4.188",
                    commit="a" * 40,
                    source=EnumPackageSourceKind.VCS,
                )
            },
            stamped_at=datetime(2026, 9, 22, tzinfo=UTC),
        ),
    )


def _setup_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    status: EnumSkillResultStatus = EnumSkillResultStatus.SUCCESS,
    failure_code: str | None = None,
) -> tuple[
    ModelSkillResult[object],
    Path,
    ModelDelegateRunAddressing,
    ModelDelegateLocusDecision,
]:
    artifact_root = tmp_path / "artifact-store"
    monkeypatch.setenv("ONEX_ARTIFACT_STORE_ROOT", str(artifact_root))
    receipt = _receipt(status=status, failure_code=failure_code)
    state_root = tmp_path / "state"
    run_dir = state_root / "runs" / str(RUN_ID)
    run_dir.mkdir(parents=True)
    request = state_root / "tmp" / f"delegate-input-{RUN_ID}.json"
    request.parent.mkdir()
    request.write_bytes(b'{"prompt":"exact caller prompt"}')
    (run_dir / "result.txt").write_bytes(
        b"caller-visible bytes" if status.is_success_like else b""
    )
    addressing = ModelDelegateRunAddressing(
        locus=EnumDelegateLocus.IN_PROCESS,
        bus="inmemory",
    )
    decision = ModelDelegateLocusDecision(
        locus=EnumDelegateLocus.IN_PROCESS,
        resolved_from="tier0",
        orchestrator_contract="/app/contract.yaml",
        orchestrator_distribution="omnimarket 0.4.188 (/app)",
        command_topic="onex.cmd.delegate.v1",
        broker="",
        lane_consumer_groups=(),
    )
    (run_dir / "run.json").write_text(
        json.dumps(
            {
                "run_id": str(RUN_ID),
                "correlation_id": str(CORRELATION_ID),
                **addressing.as_run_file_fields(),
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "receipt.json").write_text(
        json.dumps(
            {
                "run_id": str(RUN_ID),
                "correlation_id": str(CORRELATION_ID),
                "receipt": receipt.model_dump(mode="json"),
            }
        ),
        encoding="utf-8",
    )
    return receipt, request, addressing, decision


def test_callback_captures_emitted_bytes_and_hash_verified_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    receipt, request, addressing, decision = _setup_run(tmp_path, monkeypatch)
    emitted = (receipt.model_dump_json() + "\n").encode("utf-8")
    _write_issued_delegation_evidence(
        receipt=receipt,
        receipt_bytes=emitted,
        payload_path=request,
        state_root=tmp_path / "state",
        addressing=addressing,
        locus_decision=decision,
    )
    evidence = tmp_path / "state" / "runs" / str(RUN_ID) / "evidence"
    manifest = json.loads((evidence / "manifest.json").read_text(encoding="utf-8"))
    assert (evidence / "request.bin").read_bytes() == request.read_bytes()
    assert (evidence / "caller_response.bin").read_bytes() == b"caller-visible bytes"
    assert (evidence / "stdout-receipt.json").read_bytes() == emitted
    assert manifest["artifact_refs"][0]["sha256"] == receipt.artifact_refs[0].hex_digest
    assert manifest["addressing"] == addressing.model_dump(mode="json")
    assert manifest["terminal"]["json_pointer"] == "/result"
    assert (
        manifest["tenant_proof"]["state"] == "pending_authoritative_projection_readback"
    )
    raw_receipt = json.loads(
        (evidence / "stdout-receipt.json").read_text(encoding="utf-8")
    )
    assert (
        json.loads((evidence / "terminal-payload.json").read_text(encoding="utf-8"))
        == raw_receipt["result"]
    )
    assert (evidence.stat().st_mode & 0o777) == 0o700


def test_callback_rejects_address_mismatch_without_partial_capture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    receipt, request, addressing, decision = _setup_run(tmp_path, monkeypatch)
    run_path = tmp_path / "state" / "runs" / str(RUN_ID) / "run.json"
    run_payload = json.loads(run_path.read_text(encoding="utf-8"))
    run_payload["bus"] = "kafka"
    run_path.write_text(json.dumps(run_payload), encoding="utf-8")
    with pytest.raises(ValueError, match="disagrees on bus"):
        _write_issued_delegation_evidence(
            receipt=receipt,
            receipt_bytes=(receipt.model_dump_json() + "\n").encode("utf-8"),
            payload_path=request,
            state_root=tmp_path / "state",
            addressing=addressing,
            locus_decision=decision,
        )
    assert not (run_path.parent / "evidence").exists()


def test_callback_rejects_bytes_that_do_not_match_typed_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    receipt, request, addressing, decision = _setup_run(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="bytes disagree with typed receipt"):
        _write_issued_delegation_evidence(
            receipt=receipt,
            receipt_bytes=b'{"run_id":"wrong"}\n',
            payload_path=request,
            state_root=tmp_path / "state",
            addressing=addressing,
            locus_decision=decision,
        )
    assert not (tmp_path / "state" / "runs" / str(RUN_ID) / "evidence").exists()


@pytest.mark.parametrize("response_code", ["429", "503"])
def test_callback_captures_failed_summary_and_terminal_carrier(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, response_code: str
) -> None:
    receipt, request, addressing, decision = _setup_run(
        tmp_path,
        monkeypatch,
        status=EnumSkillResultStatus.FAILED,
        failure_code=response_code,
    )
    _write_issued_delegation_evidence(
        receipt=receipt,
        receipt_bytes=(receipt.model_dump_json() + "\n").encode("utf-8"),
        payload_path=request,
        state_root=tmp_path / "state",
        addressing=addressing,
        locus_decision=decision,
    )
    evidence = tmp_path / "state" / "runs" / str(RUN_ID) / "evidence"
    manifest = json.loads((evidence / "manifest.json").read_text(encoding="utf-8"))
    assert (evidence / "caller_response.bin").read_bytes() == b""
    assert manifest["terminal"]["json_pointer"] == "/result/terminal_payload"
    terminal = json.loads(
        (evidence / "terminal-payload.json").read_text(encoding="utf-8")
    )
    assert terminal["failure_code"] == response_code


def test_callback_rejects_tampered_artifact_without_partial_capture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    receipt, request, addressing, decision = _setup_run(tmp_path, monkeypatch)
    artifact = receipt.artifact_refs[0]
    blob = tmp_path / "artifact-store" / artifact.hex_digest[:2] / artifact.hex_digest
    blob.write_bytes(b"tampered")
    with pytest.raises(Exception, match="hash"):
        _write_issued_delegation_evidence(
            receipt=receipt,
            receipt_bytes=(receipt.model_dump_json() + "\n").encode("utf-8"),
            payload_path=request,
            state_root=tmp_path / "state",
            addressing=addressing,
            locus_decision=decision,
        )
    evidence = tmp_path / "state" / "runs" / str(RUN_ID) / "evidence"
    assert not evidence.exists()
