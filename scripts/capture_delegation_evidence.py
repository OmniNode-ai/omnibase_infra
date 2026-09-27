#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Pack one delegation run's caller-visible evidence without reconstructing it.

The descriptor points at bytes already produced by the caller/runtime.  The
packer copies those bytes into a private, write-once evidence directory and
writes a manifest containing their hashes and the identities from the receipt.
It deliberately refuses incomplete or mismatched evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

from omnibase_core.models.dispatch.model_skill_result import ModelSkillResult
from omnibase_infra.cli.model_receipt_runtime_summary import (
    ModelReceiptRuntimeSummary,
)

_HEX = set("0123456789abcdef")
_REQUIRED_SOURCE = ("repository", "commit_sha", "build_identity")
_REQUIRED_ROUTE = ("lane", "command_topic", "broker", "consumer")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read {label}: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object: {path}")
    return value


def _required_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"missing non-empty {name}")
    return value


def _actual_receipt(path: Path) -> tuple[ModelSkillResult[Any], dict[str, Any]]:
    """Load the emitted receipt envelope, never a reconstructed substitute."""
    raw = _read_json(path, "actual receipt")
    try:
        return ModelSkillResult[Any].model_validate(raw), raw
    except ValueError as exc:
        raise ValueError(f"actual receipt is not a ModelSkillResult: {exc}") from exc


def _readback(
    path: Path,
    label: str,
    *,
    run_id: str,
    correlation_id: str,
) -> dict[str, Any]:
    """Read an independently persisted runtime readback and bind its join keys."""
    readback = _read_json(path, label)
    if readback.get("run_id") != run_id:
        raise ValueError(f"{label}.run_id does not match actual receipt")
    if readback.get("correlation_id") != correlation_id:
        raise ValueError(f"{label}.correlation_id does not match actual receipt")
    return readback


def _source_provenance(
    readback: dict[str, Any], receipt: ModelSkillResult[Any]
) -> dict[str, Any]:
    """Bind a source readback to the package identity stamped by the receipt."""
    for field in (*_REQUIRED_SOURCE, "package_name"):
        _required_string(readback.get(field), f"source readback.{field}")
    if receipt.runtime_identity is None:
        raise ValueError("actual receipt has no runtime_identity")
    package_name = str(readback["package_name"])
    package = receipt.runtime_identity.package(package_name)
    if package is None:
        raise ValueError(
            f"source readback.package_name is absent from receipt runtime_identity: {package_name}"
        )
    if package.commit != readback["commit_sha"]:
        raise ValueError(
            "source readback.commit_sha does not match receipt package commit"
        )
    return {field: readback[field] for field in (*_REQUIRED_SOURCE, "package_name")}


def _route_provenance(readback: dict[str, Any]) -> dict[str, Any]:
    for field in _REQUIRED_ROUTE:
        _required_string(readback.get(field), f"route readback.{field}")
    return {field: readback[field] for field in _REQUIRED_ROUTE}


def _terminal_identity(path: Path, *, correlation_id: str) -> str | None:
    """Read only identity fields from the exact terminal bytes retained as evidence."""
    terminal = _read_json(path, "terminal payload")
    if terminal.get("correlation_id") != correlation_id:
        raise ValueError(
            "terminal payload correlation_id does not match actual receipt"
        )
    payload = terminal.get("payload")
    if not isinstance(payload, dict):
        raise ValueError("terminal payload lacks object payload")
    tenant_id = payload.get("tenant_id")
    if tenant_id is not None and not isinstance(tenant_id, str):
        raise ValueError("terminal payload tenant_id must be a string or explicit null")
    return tenant_id


def _sha_field(value: Any, name: str) -> str:
    digest = _required_string(value, name)
    if len(digest) != 64 or any(char not in _HEX for char in digest):
        raise ValueError(f"{name} must be lowercase SHA-256 hex")
    return digest


def _copy_once(source: Path, destination: Path) -> tuple[int, str]:
    try:
        data = source.read_bytes()
    except OSError as exc:
        raise ValueError(f"cannot read evidence bytes {source}: {exc}") from exc
    try:
        with destination.open("xb") as handle:
            handle.write(data)
    except FileExistsError as exc:
        raise ValueError(f"refusing to overwrite evidence file {destination}") from exc
    destination.chmod(0o600)
    return len(data), _sha256(data)


def pack(descriptor_path: Path, output_dir: Path) -> Path:
    descriptor = _read_json(descriptor_path, "descriptor")
    receipt_path = Path(
        _required_string(descriptor.get("actual_receipt_json"), "actual_receipt_json")
    )
    receipt, _receipt_raw = _actual_receipt(receipt_path)
    run_id = str(receipt.run_id)
    correlation_id = str(receipt.correlation_id)
    status = receipt.status.value

    if descriptor.get("run_id") != run_id:
        raise ValueError("descriptor.run_id does not match actual receipt")
    if descriptor.get("correlation_id") != correlation_id:
        raise ValueError("descriptor.correlation_id does not match actual receipt")

    source_readback_path = Path(
        _required_string(descriptor.get("source_readback_json"), "source_readback_json")
    )
    route_readback_path = Path(
        _required_string(descriptor.get("route_readback_json"), "route_readback_json")
    )
    source = _source_provenance(
        _readback(
            source_readback_path,
            "source readback",
            run_id=run_id,
            correlation_id=correlation_id,
        ),
        receipt,
    )
    route = _route_provenance(
        _readback(
            route_readback_path,
            "route readback",
            run_id=run_id,
            correlation_id=correlation_id,
        )
    )

    paths = descriptor.get("evidence")
    if not isinstance(paths, dict):
        raise ValueError("descriptor.evidence must be an object")
    names = {
        "request": "request.bin",
        "caller_response": "caller-response.bin",
        "terminal_payload": "terminal-payload.bin",
    }
    for key in names:
        _required_string(paths.get(key), f"evidence.{key}")

    tenant_id = _terminal_identity(
        Path(paths["terminal_payload"]), correlation_id=correlation_id
    )
    if tenant_id is None:
        raise ValueError(
            "tenant-scoped K1-K6 evidence requires terminal payload tenant_id"
        )
    if descriptor.get("tenant_id") != tenant_id:
        raise ValueError("descriptor.tenant_id does not match terminal payload")

    if output_dir.exists():
        raise ValueError(
            f"refusing to overwrite existing evidence directory {output_dir}"
        )
    output_dir.mkdir(mode=0o700, parents=True)
    try:
        captured: dict[str, dict[str, Any]] = {}
        all_paths = {
            **names,
            "actual_receipt": "receipt.json",
            "source_readback": "source-readback.json",
            "route_readback": "route-readback.json",
        }
        source_paths: dict[str, Path] = {
            **{key: Path(paths[key]) for key in names},
            "actual_receipt": receipt_path,
            "source_readback": source_readback_path,
            "route_readback": route_readback_path,
        }
        for key, filename in all_paths.items():
            length, digest = _copy_once(source_paths[key], output_dir / filename)
            captured[key] = {"path": filename, "bytes": length, "sha256": digest}

        expected_hashes = descriptor.get("expected_sha256", {})
        if not isinstance(expected_hashes, dict):
            raise ValueError("expected_sha256 must be an object")
        for key, expected in expected_hashes.items():
            if key not in captured:
                raise ValueError(f"expected_sha256 names unknown evidence: {key}")
            if captured[key]["sha256"] != _sha_field(
                expected, f"expected_sha256.{key}"
            ):
                raise ValueError(f"evidence hash mismatch for {key}")

        response = captured["caller_response"]
        if not receipt.status.is_success_like:
            if response["bytes"] != 0:
                response["failure_response_bytes"] = True
            else:
                response["failure_response_bytes"] = False
        elif response["bytes"] == 0:
            raise ValueError("successful run requires non-empty caller_response bytes")

        manifest = {
            "schema_version": "1.0.0",
            "run_id": run_id,
            "correlation_id": correlation_id,
            "tenant_id": tenant_id,
            "status": status,
            "source": source,
            "route": route,
            "receipt": captured["actual_receipt"],
            "readbacks": {
                "source": captured["source_readback"],
                "route": captured["route_readback"],
            },
            "evidence": captured,
        }
        manifest_path = output_dir / "manifest.json"
        with manifest_path.open("x", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2, sort_keys=True)
            handle.write("\n")
        manifest_path.chmod(0o600)
        return manifest_path
    except Exception:
        shutil.rmtree(output_dir)
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("descriptor", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args(argv)
    try:
        print(pack(args.descriptor, args.output_dir))
    except (OSError, ValueError) as exc:
        print(f"EVIDENCE_REJECTED: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
