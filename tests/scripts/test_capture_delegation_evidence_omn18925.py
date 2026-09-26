# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import json
import subprocess
import sys
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from uuid import UUID

import pytest

from omnibase_core.enums.enum_execution_locus_kind import EnumExecutionLocusKind
from omnibase_core.enums.enum_package_source_kind import EnumPackageSourceKind
from omnibase_core.enums.enum_skill_result_status import EnumSkillResultStatus
from omnibase_core.models.artifacts.model_artifact_ref import ModelArtifactRef
from omnibase_core.models.dispatch.model_skill_result import ModelSkillResult
from omnibase_core.models.runtime.model_package_identity import ModelPackageIdentity
from omnibase_core.models.runtime.model_runtime_identity import ModelRuntimeIdentity
from omnibase_infra.cli.model_receipt_runtime_summary import ModelReceiptRuntimeSummary

SCRIPT = Path(__file__).parents[2] / "scripts" / "capture_delegation_evidence.py"
RUN_ID = UUID("00000000-0000-0000-0000-000000000001")
CORRELATION_ID = UUID("00000000-0000-0000-0000-000000000002")
COMMIT = "a" * 40


def _runtime_identity() -> ModelRuntimeIdentity:
    return ModelRuntimeIdentity(
        host="isolated-proof-host",
        locus_kind=EnumExecutionLocusKind.CONTAINER,
        execution_locus="capture-proof-container",
        interpreter="/app/.venv/bin/python",
        packages={
            "omnimarket": ModelPackageIdentity(
                name="omnimarket",
                version="0.4.188",
                commit=COMMIT,
                source=EnumPackageSourceKind.VCS,
            )
        },
        stamped_at=datetime(2026, 9, 21, tzinfo=UTC),
    )


def _receipt(
    *, status: EnumSkillResultStatus, terminal: dict[str, object]
) -> ModelSkillResult[object]:
    result: object
    result_model: str
    if status.is_success_like:
        result = {"answer": "caller-visible response"}
        result_model = "builtins.dict"
    else:
        result = ModelReceiptRuntimeSummary(
            workflow_result="failed",
            exit_code=1,
            workflow="contracts/OMN-19013.yaml",
            terminal_payload=terminal,
        )
        result_model = "omnibase_infra.cli.model_receipt_runtime_summary.ModelReceiptRuntimeSummary"
    return ModelSkillResult[object](
        skill_name="delegate",
        node_name="node_delegate_skill_orchestrator",
        status=status,
        correlation_id=CORRELATION_ID,
        run_id=RUN_ID,
        exit_code=0 if status.is_success_like else 1,
        duration_ms=123,
        result=result,
        result_model=result_model,
        artifact_refs=[ModelArtifactRef.from_bytes(b"actual receipt artifact")],
        runtime_identity=_runtime_identity(),
    )


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")


def _fixture(
    tmp_path: Path,
    *,
    status: EnumSkillResultStatus = EnumSkillResultStatus.SUCCESS,
    terminal_status: str = "completed",
) -> tuple[Path, Path]:
    request = tmp_path / "request.bin"
    response = tmp_path / "response.bin"
    terminal_path = tmp_path / "terminal.json"
    receipt_path = tmp_path / "actual-receipt.json"
    source_readback = tmp_path / "source-readback.json"
    route_readback = tmp_path / "route-readback.json"
    descriptor = tmp_path / "descriptor.json"
    terminal = {
        "correlation_id": str(CORRELATION_ID),
        "payload": {"tenant_id": "tenant-1", "status": terminal_status},
    }
    request.write_bytes(b"exact prompt bytes\x00")
    response.write_bytes(
        b"caller-visible response bytes" if status.is_success_like else b""
    )
    _write_json(terminal_path, terminal)
    receipt = _receipt(status=status, terminal=terminal)
    receipt_path.write_text(receipt.model_dump_json(), encoding="utf-8")
    _write_json(
        source_readback,
        {
            "run_id": str(RUN_ID),
            "correlation_id": str(CORRELATION_ID),
            "package_name": "omnimarket",
            "repository": "OmniNode-ai/omnimarket",
            "commit_sha": COMMIT,
            "build_identity": "sha256:build-provenance",
        },
    )
    _write_json(
        route_readback,
        {
            "run_id": str(RUN_ID),
            "correlation_id": str(CORRELATION_ID),
            "lane": "isolated-lab",
            "command_topic": "onex.cmd.delegate.v1",
            "broker": "redpanda-1",
            "consumer": "delegation-handler",
        },
    )
    _write_json(
        descriptor,
        {
            "run_id": str(RUN_ID),
            "correlation_id": str(CORRELATION_ID),
            "tenant_id": "tenant-1",
            "actual_receipt_json": str(receipt_path),
            "source_readback_json": str(source_readback),
            "route_readback_json": str(route_readback),
            "evidence": {
                "request": str(request),
                "caller_response": str(response),
                "terminal_payload": str(terminal_path),
            },
        },
    )
    return descriptor, tmp_path / "packed"


def _run(descriptor: Path, output: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), str(descriptor), str(output)],
        capture_output=True,
        text=True,
        check=False,
    )


def _json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_success_handler_result_receipt_binds_exact_raw_bytes_and_provenance(
    tmp_path: Path,
) -> None:
    descriptor, output = _fixture(tmp_path)
    result = _run(descriptor, output)
    assert result.returncode == 0, result.stderr
    manifest = _json(output / "manifest.json")
    assert manifest["correlation_id"] == str(CORRELATION_ID)
    assert manifest["tenant_id"] == "tenant-1"
    assert manifest["source"]["commit_sha"] == COMMIT
    assert manifest["route"]["consumer"] == "delegation-handler"
    assert manifest["receipt"]["path"] == "receipt.json"
    assert manifest["readbacks"]["source"]["path"] == "source-readback.json"
    assert (output / "receipt.json").read_bytes() == (
        tmp_path / "actual-receipt.json"
    ).read_bytes()
    assert (
        manifest["evidence"]["request"]["sha256"]
        == sha256(b"exact prompt bytes\x00").hexdigest()
    )
    assert (
        output / "caller-response.bin"
    ).read_bytes() == b"caller-visible response bytes"
    assert (output.stat().st_mode & 0o777) == 0o700


@pytest.mark.parametrize("terminal_status", ["429", "503"])
def test_failure_summary_receipt_preserves_empty_response_and_terminal_payload(
    tmp_path: Path, terminal_status: str
) -> None:
    descriptor, output = _fixture(
        tmp_path,
        status=EnumSkillResultStatus.FAILED,
        terminal_status=terminal_status,
    )
    result = _run(descriptor, output)
    assert result.returncode == 0, result.stderr
    manifest = _json(output / "manifest.json")
    assert manifest["status"] == "failed"
    assert manifest["evidence"]["caller_response"]["bytes"] == 0
    assert (output / "terminal-payload.bin").read_bytes() == (
        tmp_path / "terminal.json"
    ).read_bytes()


@pytest.mark.parametrize(
    ("change", "expected"),
    [
        ("tenant", "does not match terminal payload"),
        ("receipt", "actual_receipt_json"),
        ("source", "source_readback_json"),
    ],
)
def test_missing_or_mismatched_identity_is_rejected_without_output(
    tmp_path: Path, change: str, expected: str
) -> None:
    descriptor, output = _fixture(tmp_path)
    payload = _json(descriptor)
    if change == "tenant":
        payload["tenant_id"] = "wrong-tenant"
    else:
        payload.pop(
            "actual_receipt_json" if change == "receipt" else "source_readback_json"
        )
    _write_json(descriptor, payload)
    result = _run(descriptor, output)
    assert result.returncode == 2
    assert expected in result.stderr
    assert not output.exists()


def test_source_readback_must_match_stamped_package_commit(tmp_path: Path) -> None:
    descriptor, output = _fixture(tmp_path)
    payload = _json(descriptor)
    source = Path(str(payload["source_readback_json"]))
    source_payload = _json(source)
    source_payload["commit_sha"] = "b" * 40
    _write_json(source, source_payload)
    result = _run(descriptor, output)
    assert result.returncode == 2
    assert "does not match receipt package commit" in result.stderr
    assert not output.exists()


def test_terminal_without_tenant_is_rejected_for_tenant_scoped_k1_k6(
    tmp_path: Path,
) -> None:
    descriptor, output = _fixture(tmp_path)
    payload = _json(descriptor)
    terminal = Path(str(payload["evidence"]["terminal_payload"]))
    terminal_payload = _json(terminal)
    terminal_payload["payload"].pop("tenant_id")
    _write_json(terminal, terminal_payload)
    result = _run(descriptor, output)
    assert result.returncode == 2
    assert "requires terminal payload tenant_id" in result.stderr
    assert not output.exists()


def test_missing_evidence_is_rejected_without_reconstruction(tmp_path: Path) -> None:
    descriptor, output = _fixture(tmp_path)
    payload = _json(descriptor)
    payload["evidence"].pop("terminal_payload")
    _write_json(descriptor, payload)
    result = _run(descriptor, output)
    assert result.returncode == 2
    assert "evidence.terminal_payload" in result.stderr
    assert not output.exists()


def test_existing_output_is_never_overwritten(tmp_path: Path) -> None:
    descriptor, output = _fixture(tmp_path)
    output.mkdir()
    sentinel = output / "sentinel"
    sentinel.write_bytes(b"keep")
    result = _run(descriptor, output)
    assert result.returncode == 2
    assert sentinel.read_bytes() == b"keep"


def test_expected_hash_mismatch_is_rejected(tmp_path: Path) -> None:
    descriptor, output = _fixture(tmp_path)
    payload = _json(descriptor)
    payload["expected_sha256"] = {"request": "0" * 64}
    _write_json(descriptor, payload)
    result = _run(descriptor, output)
    assert result.returncode == 2
    assert "evidence hash mismatch" in result.stderr
    assert not output.exists()
