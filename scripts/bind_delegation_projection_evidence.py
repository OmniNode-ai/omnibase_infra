#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Bind one captured delegation run to its authoritative tenant projection row.

The generic delegate CLI has no tenant field. The final K1-K6 proof therefore
supplies the raw terminal envelope and the correlation-scoped Market/projection
readback here. This command copies both private inputs and refuses a missing or
mismatched correlation/tenant pair; it never amends the first-stage manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Any


def _read_json_bytes(path: Path, label: str) -> tuple[dict[str, Any], bytes]:
    try:
        data = path.read_bytes()
        value = json.loads(data.decode("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read {label}: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value, data


def _nonempty_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"missing non-empty {label}")
    return value


def _terminal_identity(envelope: dict[str, Any]) -> tuple[str, str]:
    """Extract only the two identities the actual terminal-envelope model owns."""
    correlation_id = _nonempty_string(
        envelope.get("correlation_id"), "terminal correlation_id"
    )
    # A broker-retained ``ModelEventEnvelope`` owns tenant_id at its root.  The
    # older delegate terminal carriers put the same fact under one of the two
    # following shapes.  All are actual terminal representations; none reads a
    # tenant from a reconstructed CLI request.  If more than one is present,
    # agreement is required: choosing a precedence would turn contradictory
    # terminal evidence into a false tenant proof.
    candidates: list[str] = []
    for value in (
        envelope.get("tenant_id"),
        envelope.get("_tenant_id"),
        (
            envelope.get("_envelope", {}).get("tenant_id")
            if isinstance(envelope.get("_envelope"), dict)
            else None
        ),
    ):
        if value is not None:
            candidates.append(_nonempty_string(value, "terminal tenant identity"))
    if not candidates:
        raise ValueError("missing non-empty terminal tenant identity")
    if len(set(candidates)) != 1:
        raise ValueError("terminal carries contradictory tenant identities")
    return correlation_id, candidates[0]


def _copy_private(destination: Path, data: bytes) -> dict[str, Any]:
    with destination.open("xb") as handle:
        handle.write(data)
    destination.chmod(0o600)
    return {
        "path": destination.name,
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def _verify_first_stage(manifest: dict[str, Any], evidence_dir: Path) -> None:
    evidence = manifest.get("evidence")
    if not isinstance(evidence, dict):
        raise ValueError("capture manifest lacks first-stage evidence hashes")
    capture_kind = manifest.get("capture_kind", "delegate_cli")
    if capture_kind == "delegate_cli":
        required = (
            "request",
            "caller_response",
            "run_readback",
            "persisted_receipt",
            "stdout_receipt",
        )
    elif capture_kind == "runtime_delegation_port":
        required = (
            "request",
            "caller_response",
            "terminal_envelope",
            "terminal_cursor",
            "dispatch_plan",
        )
    else:
        raise ValueError("capture manifest has unknown capture_kind")
    for name in required:
        entry = evidence.get(name)
        if not isinstance(entry, dict):
            raise ValueError(f"capture manifest lacks {name} evidence")
        path_name = _nonempty_string(entry.get("path"), f"{name} evidence path")
        expected_hash = _nonempty_string(entry.get("sha256"), f"{name} evidence hash")
        path = (evidence_dir / path_name).resolve()
        try:
            path.relative_to(evidence_dir.resolve())
        except ValueError as exc:
            raise ValueError(f"capture evidence path is invalid: {path_name}") from exc
        if not path.is_file():
            raise ValueError(f"capture evidence path is invalid: {path_name}")
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected_hash:
            raise ValueError(f"capture evidence hash mismatch: {name}")


def bind(
    evidence_dir: Path,
    terminal_path: Path,
    projection_path: Path,
    tenant_identity_path: Path,
    *,
    execution_plan_path: Path | None = None,
) -> Path:
    manifest_path = evidence_dir / "manifest.json"
    manifest, manifest_bytes = _read_json_bytes(manifest_path, "capture manifest")
    run_id = _nonempty_string(manifest.get("run_id"), "manifest run_id")
    correlation_id = _nonempty_string(
        manifest.get("correlation_id"), "manifest correlation_id"
    )
    tenant_proof = manifest.get("tenant_proof")
    if not isinstance(tenant_proof, dict) or tenant_proof.get("state") != (
        "pending_authoritative_projection_readback"
    ):
        raise ValueError("capture manifest is not awaiting authoritative tenant proof")
    _verify_first_stage(manifest, evidence_dir)

    terminal, terminal_bytes = _read_json_bytes(terminal_path, "terminal envelope")
    projection, projection_bytes = _read_json_bytes(
        projection_path, "projection readback"
    )
    tenant_identity, tenant_identity_bytes = _read_json_bytes(
        tenant_identity_path, "issued lane tenant identity"
    )
    execution_plan_bytes: bytes | None = None
    if execution_plan_path is not None:
        _, execution_plan_bytes = _read_json_bytes(
            execution_plan_path, "finalization execution plan"
        )
    terminal_correlation, terminal_tenant = _terminal_identity(terminal)
    rows = projection.get("rows")
    if not isinstance(rows, list) or len(rows) != 1 or not isinstance(rows[0], dict):
        raise ValueError("projection readback must contain exactly one row")
    projection_row = rows[0]
    projection_correlation = _nonempty_string(
        projection_row.get("correlation_id"), "projection correlation_id"
    )
    projection_tenant = _nonempty_string(
        projection_row.get("tenant_id"), "projection tenant_id"
    )
    expected_tenant = _nonempty_string(
        tenant_identity.get("tenant_id"), "issued lane tenant identity"
    )
    if terminal_correlation != correlation_id:
        raise ValueError("terminal correlation_id does not match captured run")
    if projection_correlation != correlation_id:
        raise ValueError("projection correlation_id does not match captured run")
    if terminal_tenant != projection_tenant:
        raise ValueError("terminal and projection tenant identities do not match")
    if terminal_tenant != expected_tenant:
        raise ValueError("terminal tenant does not match issued lane tenant identity")

    proof_dir = evidence_dir / "tenant-projection-proof"
    if proof_dir.exists():
        raise ValueError(f"refusing to overwrite tenant proof directory {proof_dir}")
    staging = evidence_dir / f".{proof_dir.name}.tmp"
    if staging.exists():
        raise ValueError(f"refusing to reuse tenant proof staging directory {staging}")
    staging.mkdir(mode=0o700)
    try:
        terminal_capture = _copy_private(
            staging / "terminal-envelope.json", terminal_bytes
        )
        projection_capture = _copy_private(
            staging / "projection-readback.json", projection_bytes
        )
        tenant_identity_capture = _copy_private(
            staging / "issued-lane-tenant-identity.json", tenant_identity_bytes
        )
        proof = {
            "schema_version": "1.0.0",
            "captured_run_id": run_id,
            "correlation_id": correlation_id,
            "tenant_id": terminal_tenant,
            "capture_manifest": {
                "path": "../manifest.json",
                "sha256": hashlib.sha256(manifest_bytes).hexdigest(),
            },
            "terminal_envelope": terminal_capture,
            "projection_readback": projection_capture,
            "issued_lane_tenant_identity": tenant_identity_capture,
        }
        if execution_plan_bytes is not None:
            proof["execution_plan"] = _copy_private(
                staging / "execution-plan.json", execution_plan_bytes
            )
        with (staging / "tenant-proof.json").open("x", encoding="utf-8") as handle:
            json.dump(proof, handle, indent=2, sort_keys=True)
            handle.write("\n")
        (staging / "tenant-proof.json").chmod(0o600)
        staging.replace(proof_dir)
        return proof_dir / "tenant-proof.json"
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evidence_dir", type=Path)
    parser.add_argument("terminal_envelope", type=Path)
    parser.add_argument("projection_readback", type=Path)
    parser.add_argument("issued_lane_tenant_identity", type=Path)
    args = parser.parse_args(argv)
    try:
        print(
            bind(
                args.evidence_dir,
                args.terminal_envelope,
                args.projection_readback,
                args.issued_lane_tenant_identity,
            )
        )
    except ValueError as exc:
        print(f"TENANT_PROOF_REJECTED: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
