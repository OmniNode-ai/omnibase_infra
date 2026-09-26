# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import json
import subprocess
import sys
from hashlib import sha256
from pathlib import Path

SCRIPT = (
    Path(__file__).parents[2] / "scripts" / "finalize_delegation_projection_evidence.py"
)


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _command(code: str) -> list[str]:
    return [sys.executable, "-c", code, "{correlation_id}"]


def _fixture(tmp_path: Path) -> tuple[Path, Path]:
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    captured: dict[str, dict[str, str]] = {}
    for name in (
        "request",
        "caller_response",
        "run_readback",
        "persisted_receipt",
        "stdout_receipt",
    ):
        path = evidence / f"{name}.bin"
        data = f"{name}-bytes".encode()
        path.write_bytes(data)
        captured[name] = {"path": path.name, "sha256": sha256(data).hexdigest()}
    _write_json(
        evidence / "manifest.json",
        {
            "run_id": "run-1",
            "correlation_id": "11111111-1111-1111-1111-111111111111",
            "evidence": captured,
            "tenant_proof": {"state": "pending_authoritative_projection_readback"},
        },
    )
    plan = tmp_path / "plan.json"
    _write_json(
        plan,
        {
            "captured_run_id": "run-1",
            "correlation_id": "11111111-1111-1111-1111-111111111111",
            "terminal_readback_command": _command(
                "import json, sys; print(json.dumps({'correlation_id': sys.argv[1], "
                "'_envelope': {'tenant_id': 'tenant-1'}}))"
            ),
            "projection_readback_command": _command(
                "import json, sys; print(json.dumps({'rows': [{'correlation_id': "
                "sys.argv[1], 'tenant_id': 'tenant-1'}]}))"
            ),
            "issued_lane_tenant_identity_command": _command(
                "import json; print(json.dumps({'tenant_id': 'tenant-1'}))"
            ),
        },
    )
    return evidence, plan


def _run(evidence: Path, plan: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), str(evidence), str(plan)],
        capture_output=True,
        text=True,
        check=False,
    )


def test_finalizer_runs_correlation_bound_readbacks_and_binds_them(
    tmp_path: Path,
) -> None:
    evidence, plan = _fixture(tmp_path)

    result = _run(evidence, plan)

    assert result.returncode == 0, result.stderr
    proof_dir = evidence / "tenant-projection-proof"
    proof = json.loads((proof_dir / "tenant-proof.json").read_text())
    assert proof["correlation_id"] == "11111111-1111-1111-1111-111111111111"
    assert proof["execution_plan"]["sha256"] == sha256(plan.read_bytes()).hexdigest()
    assert not (evidence / ".tenant-readbacks.tmp").exists()


def test_finalizer_rejects_failed_readback_without_partial_proof(
    tmp_path: Path,
) -> None:
    evidence, plan = _fixture(tmp_path)
    payload = json.loads(plan.read_text())
    payload["projection_readback_command"] = [
        sys.executable,
        "-c",
        "raise SystemExit(7)",
    ]
    _write_json(plan, payload)

    result = _run(evidence, plan)

    assert result.returncode == 2
    assert "projection readback command failed with exit 7" in result.stderr
    assert not (evidence / "tenant-projection-proof").exists()
    assert not (evidence / ".tenant-readbacks.tmp").exists()


def test_actual_port_sink_retains_raw_terminal_then_finalizes_projection_proof(
    tmp_path: Path,
) -> None:
    """The final workflow binds the port's exact raw event before readbacks."""

    evidence, plan = _fixture(tmp_path)
    raw_terminal = json.dumps(
        {
            "correlation_id": "11111111-1111-1111-1111-111111111111",
            "tenant_id": "tenant-1",
            "payload": {"content": "actual terminal carrier"},
        },
        separators=(",", ":"),
    ).encode()
    driver = tmp_path / "run_sink.py"
    driver.write_text(
        "\n".join(
            (
                "import asyncio",
                "import sys",
                "from pathlib import Path",
                "from uuid import UUID",
                f"sys.path.insert(0, {str(SCRIPT.parent)!r})",
                "from finalize_delegation_projection_evidence import (",
                "    build_final_k1_k6_terminal_evidence_sink,",
                ")",
                "from omnibase_infra.runtime.models.model_delegation_terminal_evidence import (",
                "    ModelDelegationTerminalEvidence,",
                ")",
                "sink = build_final_k1_k6_terminal_evidence_sink(",
                f"    Path({str(evidence)!r}), Path({str(plan)!r})",
                ")",
                "asyncio.run(sink(ModelDelegationTerminalEvidence(",
                "    correlation_id=UUID('11111111-1111-1111-1111-111111111111'),",
                "    tenant_id='tenant-1',",
                "    topic='onex.evt.omnimarket.delegation-completed.v1',",
                f"    raw_envelope={raw_terminal!r},",
                "    encoding='utf-8', partition=4, offset='91',",
                "))) ",
                "asyncio.run(sink.finalize_after_dispatch())",
            )
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [sys.executable, str(driver)], capture_output=True, text=True, check=False
    )

    assert result.returncode == 0, result.stderr
    retained = evidence / "actual-dispatch-terminal"
    assert (retained / "terminal-envelope.json").read_bytes() == raw_terminal
    cursor = json.loads((retained / "transport-cursor.json").read_text())
    assert cursor["topic"] == "onex.evt.omnimarket.delegation-completed.v1"
    assert cursor["partition"] == 4
    assert cursor["offset"] == "91"
    proof = json.loads(
        (evidence / "tenant-projection-proof" / "tenant-proof.json").read_text()
    )
    assert proof["terminal_envelope"]["sha256"] == sha256(raw_terminal).hexdigest()
