# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Evidence bundle writer utility.

Writes a standard evidence bundle to disk in a tamper-evident, atomicity-sentinel
layout. The proof_summary.md file is written LAST and serves as the completeness
sentinel: a bundle is complete if and only if the sentinel exists AND all declared
artifact hashes in artifact_manifest.json match recomputation.

Write order is encoded in artifact_manifest.json as monotonically increasing
``write_order`` integers. Never use filesystem timestamps for ordering proof —
mtime is unreliable across FSes and NFS.

Evidence root convention: ``docs/evidence/<plan-slug>/<correlation-id>/``

Ticket: OMN-11207
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Protocol, cast, runtime_checkable

from omnibase_core.types import JsonType

logger = logging.getLogger(__name__)

_ARTIFACT_MANIFEST_FILENAME = "artifact_manifest.json"
_PROOF_SUMMARY_FILENAME = "proof_summary.md"
_RUN_MANIFEST_FILENAME = "run_manifest.json"
_CONTRACT_SNAPSHOT_FILENAME = "contract_snapshot.json"
_INPUT_FILENAME = "input.json"
_OUTPUT_FILENAME = "output.json"
_VERIFIER_RESULT_FILENAME = "verifier_result.json"

JsonPayload = dict[str, JsonType]


@runtime_checkable
class ProtocolEvidenceBundle(Protocol):
    """Structural protocol for evidence bundles accepted by the writer.

    Implementations may be Pydantic models or plain dicts. The writer
    uses duck typing so it works before and after the canonical
    omnibase_core.models.evidence_bundle PR lands.
    """

    @property
    def correlation_id(self) -> str:
        """Unique correlation ID for this run; used as the bundle directory name."""
        ...

    @property
    def run_manifest(self) -> JsonPayload:
        """Run-level metadata (node, version, timestamps, plan slug, etc.)."""
        ...

    @property
    def contract_snapshot(self) -> JsonPayload | None:
        """Snapshot of the contract YAML at execution time, or None."""
        ...

    @property
    def input(self) -> JsonPayload | None:
        """Node input payload, or None."""
        ...

    @property
    def output(self) -> JsonPayload | None:
        """Node output payload, or None."""
        ...

    @property
    def verifier_result(self) -> JsonPayload | None:
        """Optional verifier / DoD check result, or None."""
        ...


def _sha256_file(path: Path) -> str:
    """Return the hex-encoded SHA-256 digest of a file's contents."""
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def _write_json(path: Path, data: JsonPayload) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True, ensure_ascii=True))


def write_evidence_bundle(evidence_root: Path, bundle: object) -> Path:
    """Write a standard evidence bundle to disk.

    Accepts any object that exposes the ProtocolEvidenceBundle interface
    (including plain dataclasses or dicts with matching attributes).

    Write sequence:
    1. run_manifest.json
    2. contract_snapshot.json (if present)
    3. input.json (if present)
    4. output.json (if present)
    5. verifier_result.json (if present)
    6. artifact_manifest.json  (hashes + write_order for every preceding file)
    7. proof_summary.md        (atomicity sentinel — LAST)

    Completeness invariant: bundle is complete iff proof_summary.md exists AND
    all hashes in artifact_manifest.json match recomputation.

    Args:
        evidence_root: Root directory under which the bundle directory is created.
        bundle: Any object satisfying ProtocolEvidenceBundle, or a dict with
            equivalent keys.

    Returns:
        Path to the created bundle directory.

    Raises:
        ValueError: If the bundle has no correlation_id.
        OSError: On filesystem failures.
    """
    adapted: ProtocolEvidenceBundle
    if isinstance(bundle, dict):
        adapted = DictBundleAdapter(bundle)
    elif isinstance(bundle, ProtocolEvidenceBundle):
        adapted = bundle
    else:
        adapted = DictBundleAdapter({})

    correlation_id: str = adapted.correlation_id
    if not correlation_id:
        raise ValueError("bundle.correlation_id must be non-empty")

    bundle_dir = evidence_root / correlation_id
    bundle_dir.mkdir(parents=True, exist_ok=True)

    artifact_entries: list[JsonPayload] = []
    write_counter = 0

    def _write_artifact(filename: str, data: JsonPayload) -> None:
        nonlocal write_counter
        path = bundle_dir / filename
        _write_json(path, data)
        sha256 = _sha256_file(path)
        artifact_entries.append(
            {
                "filename": filename,
                "sha256": sha256,
                "write_order": write_counter,
            }
        )
        write_counter += 1
        logger.debug("Wrote artifact %s (sha256=%s)", filename, sha256[:12])

    # 1. run_manifest.json — always present
    _write_artifact(_RUN_MANIFEST_FILENAME, adapted.run_manifest)

    # 2-5. Optional artifacts — written only when present
    optional_artifacts: list[tuple[str, JsonPayload | None]] = [
        (_CONTRACT_SNAPSHOT_FILENAME, adapted.contract_snapshot),
        (_INPUT_FILENAME, adapted.input),
        (_OUTPUT_FILENAME, adapted.output),
        (_VERIFIER_RESULT_FILENAME, adapted.verifier_result),
    ]
    for filename, payload in optional_artifacts:
        if payload is not None:
            _write_artifact(filename, payload)

    # 6. artifact_manifest.json — records all preceding files with hashes
    # cast: list[JsonPayload] is list[JsonType] but mypy can't see through the PEP 695
    # recursive type alias, so we cast to satisfy the dict[str, JsonType] value constraint.
    manifest_path = bundle_dir / _ARTIFACT_MANIFEST_FILENAME
    _write_json(
        manifest_path,
        {
            "correlation_id": correlation_id,
            "artifact_count": len(artifact_entries),
            "artifacts": cast("list[JsonType]", artifact_entries),
        },
    )
    manifest_sha256 = _sha256_file(manifest_path)
    logger.debug("Wrote artifact_manifest.json (sha256=%s)", manifest_sha256[:12])

    # 7. proof_summary.md — atomicity sentinel, written LAST
    proof_path = bundle_dir / _PROOF_SUMMARY_FILENAME
    proof_lines = [
        "# Evidence Bundle — Proof Summary",
        "",
        f"**correlation_id**: `{correlation_id}`",
        f"**artifact_count**: {len(artifact_entries)}",
        f"**artifact_manifest_sha256**: `{manifest_sha256}`",
        "",
        "## Artifacts",
        "",
    ]
    for entry in artifact_entries:
        proof_lines.append(
            f"- `{entry['filename']}` (write_order={entry['write_order']}, "
            f"sha256=`{entry['sha256']}`)"
        )
    proof_lines += [
        "",
        "## Completeness",
        "",
        "This file is the atomicity sentinel. Its presence means all declared "
        "artifacts were written. Verify by recomputing SHA-256 for each file "
        "listed above and comparing against artifact_manifest.json entries.",
        "",
    ]
    proof_path.write_text("\n".join(proof_lines))
    logger.info(
        "Evidence bundle complete: %s (%d artifacts)", bundle_dir, len(artifact_entries)
    )

    return bundle_dir


class DictBundleAdapter:
    """Wraps a plain dict to expose the ProtocolEvidenceBundle interface."""

    def __init__(self, data: dict[str, JsonType]) -> None:
        self._data = data

    @property
    def correlation_id(self) -> str:
        value = self._data.get("correlation_id", "")
        return str(value) if value is not None else ""

    @property
    def run_manifest(self) -> JsonPayload:
        value = self._data.get("run_manifest", {})
        return value if isinstance(value, dict) else {}

    @property
    def contract_snapshot(self) -> JsonPayload | None:
        value = self._data.get("contract_snapshot")
        return value if isinstance(value, dict) else None

    @property
    def input(self) -> JsonPayload | None:
        value = self._data.get("input")
        return value if isinstance(value, dict) else None

    @property
    def output(self) -> JsonPayload | None:
        value = self._data.get("output")
        return value if isinstance(value, dict) else None

    @property
    def verifier_result(self) -> JsonPayload | None:
        value = self._data.get("verifier_result")
        return value if isinstance(value, dict) else None


__all__: list[str] = [
    "DictBundleAdapter",
    "ProtocolEvidenceBundle",
    "write_evidence_bundle",
]
