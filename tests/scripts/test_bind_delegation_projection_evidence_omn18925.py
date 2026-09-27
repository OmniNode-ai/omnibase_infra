# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import json
import subprocess
import sys
from hashlib import sha256
from pathlib import Path

SCRIPT = (
    Path(__file__).parents[2] / "scripts" / "bind_delegation_projection_evidence.py"
)


def _json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
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
    _json(
        evidence / "manifest.json",
        {
            "run_id": "run-1",
            "correlation_id": "corr-1",
            "evidence": captured,
            "tenant_proof": {"state": "pending_authoritative_projection_readback"},
        },
    )
    terminal = tmp_path / "terminal.json"
    projection = tmp_path / "projection.json"
    identity = tmp_path / "tenant.json"
    _json(
        terminal, {"correlation_id": "corr-1", "_envelope": {"tenant_id": "tenant-1"}}
    )
    _json(projection, {"rows": [{"correlation_id": "corr-1", "tenant_id": "tenant-1"}]})
    _json(identity, {"tenant_id": "tenant-1"})
    return evidence, terminal, projection, identity


def _run(
    evidence: Path, terminal: Path, projection: Path, identity: Path
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            str(evidence),
            str(terminal),
            str(projection),
            str(identity),
        ],
        capture_output=True,
        text=True,
        check=False,
    )


def test_binds_hash_verified_first_stage_to_single_matching_tenant_row(
    tmp_path: Path,
) -> None:
    evidence, terminal, projection, identity = _fixture(tmp_path)
    result = _run(evidence, terminal, projection, identity)
    assert result.returncode == 0, result.stderr
    proof = json.loads(
        (evidence / "tenant-projection-proof" / "tenant-proof.json").read_text()
    )
    assert proof["tenant_id"] == "tenant-1"


def test_rejects_matching_wrong_terminal_and_projection_tenant(tmp_path: Path) -> None:
    evidence, terminal, projection, identity = _fixture(tmp_path)
    _json(terminal, {"correlation_id": "corr-1", "_tenant_id": "wrong-tenant"})
    _json(
        projection,
        {"rows": [{"correlation_id": "corr-1", "tenant_id": "wrong-tenant"}]},
    )
    result = _run(evidence, terminal, projection, identity)
    assert result.returncode == 2
    assert "issued lane tenant identity" in result.stderr
    assert not (evidence / "tenant-projection-proof").exists()


def test_rejects_missing_terminal_tenant(tmp_path: Path) -> None:
    evidence, terminal, projection, identity = _fixture(tmp_path)
    _json(terminal, {"correlation_id": "corr-1"})

    result = _run(evidence, terminal, projection, identity)

    assert result.returncode == 2
    assert "terminal tenant identity" in result.stderr
    assert not (evidence / "tenant-projection-proof").exists()


def test_accepts_the_actual_broker_envelope_root_tenant_field(tmp_path: Path) -> None:
    evidence, terminal, projection, identity = _fixture(tmp_path)
    _json(terminal, {"correlation_id": "corr-1", "tenant_id": "tenant-1"})

    result = _run(evidence, terminal, projection, identity)

    assert result.returncode == 0, result.stderr


def test_rejects_contradictory_terminal_tenant_carriers(tmp_path: Path) -> None:
    evidence, terminal, projection, identity = _fixture(tmp_path)
    _json(
        terminal,
        {
            "correlation_id": "corr-1",
            "tenant_id": "tenant-1",
            "_envelope": {"tenant_id": "wrong-tenant"},
        },
    )

    result = _run(evidence, terminal, projection, identity)

    assert result.returncode == 2
    assert "contradictory tenant identities" in result.stderr
    assert not (evidence / "tenant-projection-proof").exists()


def test_rejects_ambiguous_projection_rows(tmp_path: Path) -> None:
    evidence, terminal, projection, identity = _fixture(tmp_path)
    _json(
        projection,
        {"rows": [{"correlation_id": "corr-1", "tenant_id": "tenant-1"}] * 2},
    )
    result = _run(evidence, terminal, projection, identity)
    assert result.returncode == 2
    assert "exactly one row" in result.stderr
    assert not (evidence / "tenant-projection-proof").exists()


def test_rejects_tampered_first_stage_bytes(tmp_path: Path) -> None:
    evidence, terminal, projection, identity = _fixture(tmp_path)
    (evidence / "request.bin").write_bytes(b"tampered")

    result = _run(evidence, terminal, projection, identity)

    assert result.returncode == 2
    assert "capture evidence hash mismatch: request" in result.stderr
    assert not (evidence / "tenant-projection-proof").exists()
