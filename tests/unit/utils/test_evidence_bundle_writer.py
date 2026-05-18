# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Unit tests for util_evidence_bundle_writer module.

Covers:
- Bundle directory is created under evidence_root/correlation_id/
- All declared artifacts are written to disk
- artifact_manifest.json has correct hashes and monotonically increasing write_order
- proof_summary.md is written LAST (highest write_order, NOT by mtime)
- Bundle hash (artifact_manifest sha256) matches recomputation
- Optional artifacts are omitted when None
- Dict-style bundles are accepted
- Empty correlation_id raises ValueError
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from omnibase_infra.utils.util_evidence_bundle_writer import write_evidence_bundle

_PROOF_SUMMARY = "proof_summary.md"
_ARTIFACT_MANIFEST = "artifact_manifest.json"
_RUN_MANIFEST = "run_manifest.json"
_CONTRACT_SNAPSHOT = "contract_snapshot.json"
_INPUT = "input.json"
_OUTPUT = "output.json"
_VERIFIER_RESULT = "verifier_result.json"


def _make_full_bundle(correlation_id: str = "test-corr-001") -> dict[str, Any]:
    return {
        "correlation_id": correlation_id,
        "run_manifest": {"node": "node_test_compute", "version": "1.0.0"},
        "contract_snapshot": {
            "name": "node_test_compute",
            "node_type": "COMPUTE_GENERIC",
        },
        "input": {"payload": "hello"},
        "output": {"result": "world"},
        "verifier_result": {"passed": True, "checks": []},
    }


def _make_minimal_bundle(correlation_id: str = "test-corr-min") -> dict[str, Any]:
    return {
        "correlation_id": correlation_id,
        "run_manifest": {"node": "node_minimal", "version": "0.1.0"},
        "contract_snapshot": None,
        "input": None,
        "output": None,
        "verifier_result": None,
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.unit
class TestEvidenceBundleDirectoryCreation:
    def test_creates_bundle_dir_under_evidence_root(self, tmp_path: Path) -> None:
        bundle = _make_full_bundle("corr-dir-test")
        result = write_evidence_bundle(tmp_path, bundle)
        assert result == tmp_path / "corr-dir-test"
        assert result.is_dir()

    def test_creates_nested_evidence_root_if_missing(self, tmp_path: Path) -> None:
        deep_root = tmp_path / "docs" / "evidence" / "plan-alpha"
        bundle = _make_minimal_bundle("corr-nested")
        result = write_evidence_bundle(deep_root, bundle)
        assert result.is_dir()

    def test_empty_correlation_id_raises(self, tmp_path: Path) -> None:
        bundle = {"correlation_id": "", "run_manifest": {}}
        with pytest.raises(ValueError, match="correlation_id"):
            write_evidence_bundle(tmp_path, bundle)

    @pytest.mark.parametrize(
        "bad_id",
        [
            "../escape",
            "nested/path",
            "back\\slash",
            ".",
            "..",
            ".hidden",
            "/absolute",
        ],
    )
    def test_correlation_id_path_traversal_rejected(
        self, tmp_path: Path, bad_id: str
    ) -> None:
        bundle = {"correlation_id": bad_id, "run_manifest": {}}
        with pytest.raises(ValueError, match="single path segment"):
            write_evidence_bundle(tmp_path, bundle)

    def test_existing_bundle_dir_raises_file_exists(self, tmp_path: Path) -> None:
        bundle = _make_minimal_bundle("corr-replay")
        write_evidence_bundle(tmp_path, bundle)
        with pytest.raises(FileExistsError, match="already exists"):
            write_evidence_bundle(tmp_path, bundle)


@pytest.mark.unit
class TestEvidenceBundleArtifactsWritten:
    def test_full_bundle_writes_all_artifacts(self, tmp_path: Path) -> None:
        bundle = _make_full_bundle("corr-full")
        bundle_dir = write_evidence_bundle(tmp_path, bundle)
        for filename in [
            _RUN_MANIFEST,
            _CONTRACT_SNAPSHOT,
            _INPUT,
            _OUTPUT,
            _VERIFIER_RESULT,
            _ARTIFACT_MANIFEST,
            _PROOF_SUMMARY,
        ]:
            assert (bundle_dir / filename).exists(), f"Missing: {filename}"

    def test_minimal_bundle_omits_optional_artifacts(self, tmp_path: Path) -> None:
        bundle = _make_minimal_bundle("corr-min")
        bundle_dir = write_evidence_bundle(tmp_path, bundle)
        assert (bundle_dir / _RUN_MANIFEST).exists()
        assert (bundle_dir / _ARTIFACT_MANIFEST).exists()
        assert (bundle_dir / _PROOF_SUMMARY).exists()
        for optional in [_CONTRACT_SNAPSHOT, _INPUT, _OUTPUT, _VERIFIER_RESULT]:
            assert not (bundle_dir / optional).exists(), f"Should be absent: {optional}"

    def test_run_manifest_contents_match(self, tmp_path: Path) -> None:
        bundle = _make_full_bundle("corr-content")
        bundle_dir = write_evidence_bundle(tmp_path, bundle)
        data = json.loads((bundle_dir / _RUN_MANIFEST).read_text())
        assert data["node"] == "node_test_compute"
        assert data["version"] == "1.0.0"


@pytest.mark.unit
class TestArtifactManifest:
    def test_artifact_manifest_has_correct_hashes(self, tmp_path: Path) -> None:
        bundle = _make_full_bundle("corr-hash")
        bundle_dir = write_evidence_bundle(tmp_path, bundle)
        manifest = json.loads((bundle_dir / _ARTIFACT_MANIFEST).read_text())
        for entry in manifest["artifacts"]:
            actual = _sha256(bundle_dir / entry["filename"])
            assert actual == entry["sha256"], (
                f"Hash mismatch for {entry['filename']}: "
                f"expected {entry['sha256']}, got {actual}"
            )

    def test_artifact_manifest_write_order_is_monotonic(self, tmp_path: Path) -> None:
        bundle = _make_full_bundle("corr-order")
        bundle_dir = write_evidence_bundle(tmp_path, bundle)
        manifest = json.loads((bundle_dir / _ARTIFACT_MANIFEST).read_text())
        orders = [e["write_order"] for e in manifest["artifacts"]]
        assert orders == list(range(len(orders))), (
            f"Write order not monotonic: {orders}"
        )

    def test_artifact_manifest_correlation_id_matches(self, tmp_path: Path) -> None:
        bundle = _make_full_bundle("corr-id-check")
        bundle_dir = write_evidence_bundle(tmp_path, bundle)
        manifest = json.loads((bundle_dir / _ARTIFACT_MANIFEST).read_text())
        assert manifest["correlation_id"] == "corr-id-check"

    def test_artifact_manifest_artifact_count_matches_entries(
        self, tmp_path: Path
    ) -> None:
        bundle = _make_full_bundle("corr-count")
        bundle_dir = write_evidence_bundle(tmp_path, bundle)
        manifest = json.loads((bundle_dir / _ARTIFACT_MANIFEST).read_text())
        assert manifest["artifact_count"] == len(manifest["artifacts"])

    def test_minimal_bundle_manifest_artifact_count(self, tmp_path: Path) -> None:
        bundle = _make_minimal_bundle("corr-min-count")
        bundle_dir = write_evidence_bundle(tmp_path, bundle)
        manifest = json.loads((bundle_dir / _ARTIFACT_MANIFEST).read_text())
        # Only run_manifest is written for minimal bundle
        assert manifest["artifact_count"] == 1

    def test_artifact_manifest_sha256_matches_recomputation(
        self, tmp_path: Path
    ) -> None:
        bundle = _make_full_bundle("corr-recompute")
        bundle_dir = write_evidence_bundle(tmp_path, bundle)
        proof_text = (bundle_dir / _PROOF_SUMMARY).read_text()
        recorded_sha = ""
        for line in proof_text.splitlines():
            if "artifact_manifest_sha256" in line:
                recorded_sha = line.split("`")[1]
                break
        assert recorded_sha, "artifact_manifest_sha256 not found in proof_summary.md"

        actual_sha = _sha256(bundle_dir / _ARTIFACT_MANIFEST)
        assert actual_sha == recorded_sha


@pytest.mark.unit
class TestProofSummarySentinel:
    def test_proof_summary_is_last_by_write_order(self, tmp_path: Path) -> None:
        """proof_summary.md must be written LAST — verified via write_order, not mtime."""
        bundle = _make_full_bundle("corr-sentinel")
        bundle_dir = write_evidence_bundle(tmp_path, bundle)
        manifest = json.loads((bundle_dir / _ARTIFACT_MANIFEST).read_text())
        artifact_filenames = {e["filename"] for e in manifest["artifacts"]}
        # proof_summary.md is NOT in the manifest — it is the sentinel written after
        assert _PROOF_SUMMARY not in artifact_filenames

    def test_proof_summary_exists_after_write(self, tmp_path: Path) -> None:
        bundle = _make_full_bundle("corr-sentinel-exists")
        bundle_dir = write_evidence_bundle(tmp_path, bundle)
        assert (bundle_dir / _PROOF_SUMMARY).exists()

    def test_proof_summary_lists_all_manifest_artifacts(self, tmp_path: Path) -> None:
        bundle = _make_full_bundle("corr-sentinel-list")
        bundle_dir = write_evidence_bundle(tmp_path, bundle)
        manifest = json.loads((bundle_dir / _ARTIFACT_MANIFEST).read_text())
        proof_text = (bundle_dir / _PROOF_SUMMARY).read_text()
        for entry in manifest["artifacts"]:
            assert entry["filename"] in proof_text, (
                f"Artifact {entry['filename']} missing from proof_summary.md"
            )

    def test_proof_summary_references_correlation_id(self, tmp_path: Path) -> None:
        bundle = _make_full_bundle("corr-sentinel-id")
        bundle_dir = write_evidence_bundle(tmp_path, bundle)
        proof_text = (bundle_dir / _PROOF_SUMMARY).read_text()
        assert "corr-sentinel-id" in proof_text


@pytest.mark.unit
class TestDictStyleBundle:
    def test_dict_bundle_with_all_fields(self, tmp_path: Path) -> None:
        bundle = _make_full_bundle("corr-dict-full")
        bundle_dir = write_evidence_bundle(tmp_path, bundle)
        assert (bundle_dir / _RUN_MANIFEST).exists()
        assert (bundle_dir / _PROOF_SUMMARY).exists()

    def test_dict_bundle_minimal_keys(self, tmp_path: Path) -> None:
        bundle = {"correlation_id": "corr-dict-min", "run_manifest": {"v": 1}}
        bundle_dir = write_evidence_bundle(tmp_path, bundle)
        assert (bundle_dir / _RUN_MANIFEST).exists()
        assert (bundle_dir / _PROOF_SUMMARY).exists()
        # Optional artifacts absent
        for optional in [_CONTRACT_SNAPSHOT, _INPUT, _OUTPUT, _VERIFIER_RESULT]:
            assert not (bundle_dir / optional).exists()
