# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Unit tests for artifact storage (OMN-2151).

Tests:
- Directory structure creation
- Plan/result/verdict/attribution write and read
- Artifact file writing (text and bytes)
- Latest-by-pattern symlink management
- Candidate and run listing
- Path traversal protection
- Configuration
"""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import pytest

from omnibase_infra.validation.artifact_store import (
    ArtifactStore,
    ModelArtifactStoreConfig,
)

pytestmark = pytest.mark.unit


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def store(tmp_path: Path) -> ArtifactStore:
    """Create an artifact store rooted in a temporary directory."""
    config = ModelArtifactStoreConfig(root_dir=str(tmp_path))
    return ArtifactStore(config)


# ============================================================================
# Directory Structure
# ============================================================================


class TestDirectoryStructure:
    """Tests for artifact directory structure creation."""

    def test_candidate_dir(self, store: ArtifactStore) -> None:
        """candidate_dir returns the expected path."""
        cid = uuid4()
        path = store.candidate_dir(cid)
        assert path == store.root / str(cid)

    def test_run_dir(self, store: ArtifactStore) -> None:
        """run_dir returns the expected nested path."""
        cid = uuid4()
        rid = uuid4()
        path = store.run_dir(cid, rid)
        assert path == store.root / str(cid) / str(rid)

    def test_artifacts_dir(self, store: ArtifactStore) -> None:
        """artifacts_dir returns the expected nested path."""
        cid = uuid4()
        rid = uuid4()
        path = store.artifacts_dir(cid, rid)
        assert path == store.root / str(cid) / str(rid) / "artifacts"

    def test_logs_dir(self, store: ArtifactStore) -> None:
        """logs_dir returns the expected nested path."""
        cid = uuid4()
        rid = uuid4()
        path = store.logs_dir(cid, rid)
        assert path == store.root / str(cid) / str(rid) / "artifacts" / "logs"

    def test_latest_by_pattern_dir(self, store: ArtifactStore) -> None:
        """latest_by_pattern_dir returns the expected path."""
        path = store.latest_by_pattern_dir()
        assert path == store.root / "latest_by_pattern"

    def test_ensure_run_dirs_creates_all(self, store: ArtifactStore) -> None:
        """ensure_run_dirs creates the full directory tree."""
        cid = uuid4()
        rid = uuid4()
        run_path = store.ensure_run_dirs(cid, rid)

        assert run_path.is_dir()
        assert store.artifacts_dir(cid, rid).is_dir()
        assert store.logs_dir(cid, rid).is_dir()
        assert store.latest_by_pattern_dir().is_dir()


# ============================================================================
# Write and Read
# ============================================================================


class TestWriteAndRead:
    """Tests for writing and reading artifacts."""

    def test_write_and_read_plan(self, store: ArtifactStore) -> None:
        """write_plan persists YAML and read_plan returns it."""
        cid = uuid4()
        plan_data = {"plan_id": str(uuid4()), "checks": ["CHECK-PY-001"]}

        path = store.write_plan(cid, plan_data)
        assert path.is_file()
        assert path.name == "plan.yaml"

        read_data = store.read_plan(cid)
        assert read_data is not None
        assert read_data["checks"] == ["CHECK-PY-001"]

    def test_read_plan_missing_returns_none(self, store: ArtifactStore) -> None:
        """read_plan returns None when no plan exists."""
        assert store.read_plan(uuid4()) is None

    def test_write_result(self, store: ArtifactStore) -> None:
        """write_result creates result.yaml in the run directory."""
        cid = uuid4()
        rid = uuid4()
        result_data = {"pass_count": 10, "fail_count": 2}

        path = store.write_result(cid, rid, result_data)
        assert path.is_file()
        assert path.name == "result.yaml"
        assert path.parent == store.run_dir(cid, rid)

    def test_write_verdict(self, store: ArtifactStore) -> None:
        """write_verdict creates verdict.yaml in the run directory."""
        cid = uuid4()
        rid = uuid4()
        verdict_data = {"verdict": "pass", "score": 0.95}

        path = store.write_verdict(cid, rid, verdict_data)
        assert path.is_file()
        assert path.name == "verdict.yaml"

    def test_write_and_read_verdict(self, store: ArtifactStore) -> None:
        """write_verdict and read_verdict round-trip correctly."""
        cid = uuid4()
        rid = uuid4()
        verdict_data = {"verdict": "fail", "blocking": ["CHECK-PY-001"]}

        store.write_verdict(cid, rid, verdict_data)
        read_data = store.read_verdict(cid, rid)
        assert read_data is not None
        assert read_data["verdict"] == "fail"
        assert read_data["blocking"] == ["CHECK-PY-001"]

    def test_read_verdict_missing_returns_none(self, store: ArtifactStore) -> None:
        """read_verdict returns None when no verdict exists."""
        assert store.read_verdict(uuid4(), uuid4()) is None

    def test_write_attribution(self, store: ArtifactStore) -> None:
        """write_attribution creates attribution.yaml in the run directory."""
        cid = uuid4()
        rid = uuid4()
        attr_data = {"agent": "test-agent", "correlation_id": str(uuid4())}

        path = store.write_attribution(cid, rid, attr_data)
        assert path.is_file()
        assert path.name == "attribution.yaml"

    def test_write_artifact_text(self, store: ArtifactStore) -> None:
        """write_artifact creates a text file in the artifacts directory."""
        cid = uuid4()
        rid = uuid4()

        path = store.write_artifact(cid, rid, "junit.xml", "<testsuites/>")
        assert path.is_file()
        assert path.read_text() == "<testsuites/>"
        assert path.parent == store.artifacts_dir(cid, rid)

    def test_write_artifact_bytes(self, store: ArtifactStore) -> None:
        """write_artifact creates a binary file in the artifacts directory."""
        cid = uuid4()
        rid = uuid4()

        path = store.write_artifact(cid, rid, "coverage.bin", b"\x00\x01\x02")
        assert path.is_file()
        assert path.read_bytes() == b"\x00\x01\x02"


# ============================================================================
# Symlinks
# ============================================================================


class TestSymlinks:
    """Tests for latest_by_pattern symlink management."""

    def test_update_latest_symlink_creates_symlink(self, store: ArtifactStore) -> None:
        """update_latest_symlink creates a symlink to the run directory."""
        cid = uuid4()
        rid = uuid4()
        pid = uuid4()

        # Ensure the target directory exists
        store.ensure_run_dirs(cid, rid)

        symlink_path = store.update_latest_symlink(pid, cid, rid)
        assert symlink_path.is_symlink()
        assert symlink_path.name == str(pid)

    def test_update_latest_symlink_replaces_existing(
        self, store: ArtifactStore
    ) -> None:
        """update_latest_symlink replaces an existing symlink."""
        cid = uuid4()
        rid1 = uuid4()
        rid2 = uuid4()
        pid = uuid4()

        store.ensure_run_dirs(cid, rid1)
        store.ensure_run_dirs(cid, rid2)

        store.update_latest_symlink(pid, cid, rid1)
        store.update_latest_symlink(pid, cid, rid2)

        symlink_path = store.latest_by_pattern_dir() / str(pid)
        assert symlink_path.is_symlink()
        # The symlink target should point to rid2
        target = symlink_path.readlink()
        assert str(rid2) in str(target)

    def test_resolve_latest_returns_path(self, store: ArtifactStore) -> None:
        """resolve_latest returns the resolved path for an existing symlink."""
        cid = uuid4()
        rid = uuid4()
        pid = uuid4()

        store.ensure_run_dirs(cid, rid)
        store.update_latest_symlink(pid, cid, rid)

        resolved = store.resolve_latest(pid)
        assert resolved is not None

    def test_resolve_latest_returns_none_when_missing(
        self, store: ArtifactStore
    ) -> None:
        """resolve_latest returns None when no symlink exists."""
        assert store.resolve_latest(uuid4()) is None


# ============================================================================
# Listing
# ============================================================================


class TestListing:
    """Tests for listing candidates and runs."""

    def test_list_candidates_empty(self, store: ArtifactStore) -> None:
        """list_candidates returns empty list for empty store."""
        assert store.list_candidates() == []

    def test_list_candidates_with_data(self, store: ArtifactStore) -> None:
        """list_candidates returns candidate IDs."""
        cid = uuid4()
        store.write_plan(cid, {"test": True})

        candidates = store.list_candidates()
        assert str(cid) in candidates

    def test_list_candidates_excludes_latest_by_pattern(
        self, store: ArtifactStore
    ) -> None:
        """list_candidates excludes the latest_by_pattern directory."""
        cid = uuid4()
        store.ensure_run_dirs(cid, uuid4())

        candidates = store.list_candidates()
        assert "latest_by_pattern" not in candidates

    def test_list_runs_empty(self, store: ArtifactStore) -> None:
        """list_runs returns empty list for unknown candidate."""
        assert store.list_runs(uuid4()) == []

    def test_list_runs_with_data(self, store: ArtifactStore) -> None:
        """list_runs returns run IDs for a candidate."""
        cid = uuid4()
        rid = uuid4()
        store.ensure_run_dirs(cid, rid)

        runs = store.list_runs(cid)
        assert str(rid) in runs


# ============================================================================
# Path Traversal Protection
# ============================================================================


class TestPathTraversalProtection:
    """Tests for path traversal validation in write_artifact."""

    def test_write_artifact_rejects_parent_traversal(
        self, store: ArtifactStore
    ) -> None:
        """write_artifact raises ValueError for ../../ traversal."""
        cid = uuid4()
        rid = uuid4()

        with pytest.raises(ValueError, match="Path traversal detected"):
            store.write_artifact(cid, rid, "../../etc/passwd", "malicious")

    def test_write_artifact_rejects_absolute_path(self, store: ArtifactStore) -> None:
        """write_artifact raises ValueError for absolute paths."""
        cid = uuid4()
        rid = uuid4()

        with pytest.raises(ValueError, match="Path traversal detected"):
            store.write_artifact(cid, rid, "/etc/passwd", "malicious")

    def test_write_artifact_rejects_single_parent_traversal(
        self, store: ArtifactStore
    ) -> None:
        """write_artifact raises ValueError for single ../ traversal."""
        cid = uuid4()
        rid = uuid4()

        with pytest.raises(ValueError, match="Path traversal detected"):
            store.write_artifact(cid, rid, "../escape.txt", "malicious")

    def test_write_artifact_rejects_deeply_nested_traversal(
        self, store: ArtifactStore
    ) -> None:
        """write_artifact raises ValueError for nested-then-escape paths."""
        cid = uuid4()
        rid = uuid4()

        with pytest.raises(ValueError, match="Path traversal detected"):
            store.write_artifact(cid, rid, "subdir/../../escape.txt", "malicious")

    def test_write_artifact_allows_nested_subdir(self, store: ArtifactStore) -> None:
        """write_artifact allows legitimate nested filenames."""
        cid = uuid4()
        rid = uuid4()

        path = store.write_artifact(cid, rid, "logs/check.log", "log data")
        assert path.is_file()
        assert path.read_text() == "log data"
        # Must be within the artifacts directory
        artifacts = store.artifacts_dir(cid, rid)
        assert path.is_relative_to(artifacts)

    def test_write_artifact_allows_simple_filename(self, store: ArtifactStore) -> None:
        """write_artifact allows simple filenames without subdirs."""
        cid = uuid4()
        rid = uuid4()

        path = store.write_artifact(cid, rid, "output.json", '{"ok": true}')
        assert path.is_file()
        assert path.read_text() == '{"ok": true}'

    def test_write_artifact_no_dirs_created_on_traversal(
        self, store: ArtifactStore, tmp_path: Path
    ) -> None:
        """Path traversal must not create directories outside artifacts."""
        cid = uuid4()
        rid = uuid4()

        escape_target = tmp_path / "escaped"
        assert not escape_target.exists()

        with pytest.raises(ValueError, match="Path traversal detected"):
            store.write_artifact(cid, rid, "../../escaped/pwned.txt", "bad")

        # The escaped directory must NOT have been created
        assert not escape_target.exists()

    def test_validate_path_within_static_method(self, tmp_path: Path) -> None:
        """_validate_path_within rejects traversal as a standalone check."""
        root = tmp_path / "safe"
        root.mkdir()

        # Valid path
        result = ArtifactStore._validate_path_within("file.txt", root)
        assert result.is_relative_to(root)

        # Traversal
        with pytest.raises(ValueError, match="Path traversal detected"):
            ArtifactStore._validate_path_within("../escape.txt", root)


# ============================================================================
# Configuration
# ============================================================================


class TestModelArtifactStoreConfig:
    """Tests for ModelArtifactStoreConfig model."""

    def test_default_config(self) -> None:
        """Default config uses home directory."""
        config = ModelArtifactStoreConfig()
        assert ".claude/validation" in config.root_dir
        assert config.create_dirs is True

    def test_frozen(self) -> None:
        """Config is frozen."""
        config = ModelArtifactStoreConfig()
        with pytest.raises(Exception):
            config.create_dirs = False  # type: ignore[misc]

    def test_extra_forbid(self) -> None:
        """Extra fields are forbidden."""
        with pytest.raises(Exception):
            ModelArtifactStoreConfig(unknown_field="x")  # type: ignore[call-arg]
