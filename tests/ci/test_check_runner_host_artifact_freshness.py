# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Coverage for the runner host artifact-freshness detector (OMN-15114).

OMN-15104 closed *container-vs-repo* image drift but left a distinct gap
unclosed: the operator-maintained checkout on the runner host
(``~/.omnibase/runners/``, rsynced by ``deploy-runners.sh``) sat 19 days
stale relative to ``origin/dev`` because the rsync step only runs inside
the full (disruptive) deploy pipeline, which operators avoid for small
fixes. These tests hold the pure comparison/reporting/parsing logic to the
fail-closed contract established by the sibling
``check_runner_fleet_image_drift.py`` suite: an unreadable/unknown remote
path must be reported, never silently treated as in-sync.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_CI = REPO_ROOT / "scripts" / "ci"
FRESHNESS_SCRIPT = SCRIPTS_CI / "check_runner_host_artifact_freshness.py"


def _load_freshness_module() -> Any:
    spec = importlib.util.spec_from_file_location(
        "check_runner_host_artifact_freshness", FRESHNESS_SCRIPT
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(SCRIPTS_CI))
    try:
        spec.loader.exec_module(module)
    finally:
        if str(SCRIPTS_CI) in sys.path:
            sys.path.remove(str(SCRIPTS_CI))
    return module


def test_parse_sync_paths_extracts_literal_quoted_entries() -> None:
    module = _load_freshness_module()
    fixture = (
        "SYNC_PATHS=(\n"
        '    "${RUNNER_FLEET_CONFIG}"\n'
        '    "pyproject.toml"\n'
        '    "docker/runners/runner-image.lock.json"\n'
        ")\n"
    )
    assert module.parse_sync_paths(fixture) == [
        "pyproject.toml",
        "docker/runners/runner-image.lock.json",
    ]


def test_parse_sync_paths_raises_when_array_missing() -> None:
    module = _load_freshness_module()
    with pytest.raises(ValueError, match="SYNC_PATHS"):
        module.parse_sync_paths("no array here\n")


def test_find_stale_paths_no_findings_when_all_match() -> None:
    module = _load_freshness_module()
    findings = module.find_stale_paths(
        local_hashes={"a": "abc123", "b": "def456"},
        remote_hashes={"a": "abc123", "b": "def456"},
    )
    assert findings == []


def test_find_stale_paths_reports_hash_mismatch() -> None:
    module = _load_freshness_module()
    findings = module.find_stale_paths(
        local_hashes={"docker/runners/runner-image.lock.json": "newhash"},
        remote_hashes={"docker/runners/runner-image.lock.json": "oldhash"},
    )
    assert len(findings) == 1
    assert findings[0].path == "docker/runners/runner-image.lock.json"
    assert findings[0].local_sha256 == "newhash"
    assert findings[0].remote_sha256 == "oldhash"


def test_find_stale_paths_reports_unreadable_remote_not_skipped() -> None:
    """Fail-closed: a path we cannot hash on the remote must be flagged,
    never silently treated as in-sync (the exact failure mode this check
    exists to close — 19 days of no signal while the host checkout drifted)."""
    module = _load_freshness_module()
    findings = module.find_stale_paths(
        local_hashes={"config/runner_fleet.yaml": "abc"},
        remote_hashes={"config/runner_fleet.yaml": None},
    )
    assert len(findings) == 1
    assert findings[0].remote_sha256 is None
    assert "UNREADABLE" in findings[0].as_line()


def test_render_report_ok_when_no_findings() -> None:
    module = _load_freshness_module()
    report = module.render_report([])
    assert "OK" in report


def test_render_report_lists_every_finding() -> None:
    module = _load_freshness_module()
    findings = module.find_stale_paths(
        local_hashes={"a": "1", "b": "2", "c": "3"},
        remote_hashes={"a": "1", "b": "different", "c": "3"},
    )
    report = module.render_report(findings)
    finding_lines = report.split("\n")[1:]
    assert any("b" in line for line in finding_lines)
    assert not any("a:" in line or "c:" in line for line in finding_lines)
    assert "1 path(s) stale" in report


def test_compute_local_hashes_reads_real_files(tmp_path: Path) -> None:
    module = _load_freshness_module()
    (tmp_path / "sub").mkdir()
    target = tmp_path / "sub" / "file.txt"
    target.write_bytes(b"hello world")
    hashes = module.compute_local_hashes(tmp_path, ["sub/file.txt"])
    import hashlib

    assert hashes["sub/file.txt"] == hashlib.sha256(b"hello world").hexdigest()


def test_compute_remote_hashes_uses_remote_sha256(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_freshness_module()
    calls: list[tuple[str, str]] = []

    def _fake_remote(ssh_host: str, remote_path: str) -> str | None:
        calls.append((ssh_host, remote_path))
        return {"host_dir/a": "hash-a", "host_dir/b": None}[remote_path]

    monkeypatch.setattr(module, "_remote_sha256", _fake_remote)
    result = module.compute_remote_hashes("myhost", "host_dir", ["a", "b"])
    assert result == {"a": "hash-a", "b": None}
    assert calls == [("myhost", "host_dir/a"), ("myhost", "host_dir/b")]


def test_live_deploy_script_sync_paths_parses_without_error() -> None:
    """Regression guard: the real deploy-runners.sh SYNC_PATHS array must
    stay parseable, and must still include the runner-image lock file --
    the exact artifact whose drift went undetected for 19 days."""
    module = _load_freshness_module()
    deploy_script = REPO_ROOT / "scripts" / "deploy-runners.sh"
    paths = module.parse_sync_paths(deploy_script.read_text(encoding="utf-8"))
    assert "docker/runners/runner-image.lock.json" in paths
    assert "docker/runners/Dockerfile" in paths


def test_live_deploy_script_sync_paths_excludes_the_checker_itself() -> None:
    """Regression guard (OMN-15114 follow-up): the freshness checker must
    never re-add itself to SYNC_PATHS.

    The checker only ever runs off-host (a local cron on the operator/dev
    machine, per install_host_artifact_freshness_cron in deploy-runners.sh)
    -- it ssh's OUT to the runner host, it never runs ON it. Listing it in
    SYNC_PATHS is a self-referential bug: any host that has not yet
    received the file reports the checker's own absence as drift on its
    very first real invocation, independent of whether every other synced
    artifact is actually fresh. This is the exact failure a prior PR
    shipped and claimed (falsely, without re-verification) was a clean
    exit-0 run.
    """
    module = _load_freshness_module()
    deploy_script = REPO_ROOT / "scripts" / "deploy-runners.sh"
    paths = module.parse_sync_paths(deploy_script.read_text(encoding="utf-8"))
    assert "scripts/ci/check_runner_host_artifact_freshness.py" not in paths
