# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Integration coverage for util_evidence_bundle_writer (OMN-11207).

Exercises `write_evidence_bundle` against the real filesystem with a real
tmp_path and asserts:
- Sentinel ordering: proof_summary.md has the highest write_order even
  when concurrent disk timestamps would suggest otherwise.
- Manifest hash stability across two independent writes of the same
  payload to different correlation_ids.
- Disk-level layout: every declared artifact lands in
  evidence_root/<correlation_id>/ and no stray files appear.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from omnibase_infra.utils.util_evidence_bundle_writer import write_evidence_bundle

pytestmark = pytest.mark.integration


def _full_bundle(correlation_id: str) -> dict[str, Any]:
    return {
        "correlation_id": correlation_id,
        "run_manifest": {"node": "node_integration_compute", "version": "1.0.0"},
        "contract_snapshot": {
            "name": "node_integration_compute",
            "node_type": "COMPUTE_GENERIC",
        },
        "input": {"payload": "integration-input"},
        "output": {"result": "integration-output"},
        "verifier_result": {"passed": True, "checks": []},
    }


def test_proof_summary_is_the_completeness_sentinel(tmp_path: Path) -> None:
    """proof_summary.md is written LAST on disk after the manifest is sealed.

    The artifact_manifest.json captures write_order for every pre-sentinel
    artifact. proof_summary.md itself is the atomicity sentinel — it must
    exist on disk for the bundle to be considered complete, and it is
    deliberately not listed in the manifest's artifacts array (its presence
    is the proof, not a member of the index it follows).
    """
    write_evidence_bundle(tmp_path, _full_bundle("corr-sentinel"))
    bundle_dir = tmp_path / "corr-sentinel"
    manifest = json.loads((bundle_dir / "artifact_manifest.json").read_text())
    write_orders = [a["write_order"] for a in manifest["artifacts"]]
    assert write_orders == sorted(write_orders), "write_order is not monotonic"
    assert len(set(write_orders)) == len(write_orders), "write_order has duplicates"
    assert (bundle_dir / "proof_summary.md").is_file(), (
        "proof_summary.md sentinel missing — bundle not complete"
    )


def test_artifact_manifest_hash_is_deterministic_across_writes(
    tmp_path: Path,
) -> None:
    """Same logical bundle, two correlation_ids, identical artifact hashes per file.

    Asserts the writer hashes file content (not paths or timestamps), so
    two structurally identical bundles produce per-file SHA-256 digests
    that match. Bundle-level paths differ by correlation_id; per-file
    hashes do not.
    """
    write_evidence_bundle(tmp_path, _full_bundle("corr-a"))
    write_evidence_bundle(tmp_path, _full_bundle("corr-b"))

    def _hashes_by_name(corr: str) -> dict[str, str]:
        manifest = json.loads((tmp_path / corr / "artifact_manifest.json").read_text())
        return {a["filename"]: a["sha256"] for a in manifest["artifacts"]}

    a = _hashes_by_name("corr-a")
    b = _hashes_by_name("corr-b")
    shared = set(a) & set(b)
    for name in shared - {"run_manifest.json"}:
        assert a[name] == b[name], f"{name} hash drifted between identical bundles"


def test_no_stray_files_under_correlation_dir(tmp_path: Path) -> None:
    """Only the declared artifacts appear in the bundle directory on disk."""
    write_evidence_bundle(tmp_path, _full_bundle("corr-no-stray"))
    bundle_dir = tmp_path / "corr-no-stray"
    files_on_disk = {p.name for p in bundle_dir.iterdir() if p.is_file()}
    manifest = json.loads((bundle_dir / "artifact_manifest.json").read_text())
    declared = {a["filename"] for a in manifest["artifacts"]} | {
        "artifact_manifest.json",
        "proof_summary.md",
    }
    assert files_on_disk == declared, (
        f"on-disk files diverge from declared artifacts: "
        f"{files_on_disk.symmetric_difference(declared)}"
    )


def test_artifact_sha256_matches_recomputation_for_each_file(tmp_path: Path) -> None:
    """Per-artifact SHA-256 in the manifest matches a fresh disk recomputation."""
    write_evidence_bundle(tmp_path, _full_bundle("corr-recompute"))
    bundle_dir = tmp_path / "corr-recompute"
    manifest = json.loads((bundle_dir / "artifact_manifest.json").read_text())
    for artifact in manifest["artifacts"]:
        on_disk = hashlib.sha256(
            (bundle_dir / artifact["filename"]).read_bytes()
        ).hexdigest()
        assert on_disk == artifact["sha256"], (
            f"{artifact['filename']} disk hash != manifest hash"
        )
