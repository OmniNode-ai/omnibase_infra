# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Tests for the runner image validation script (OMN-12568).

OMN-12568 is the acceptance of the OMN-12567 versioned runner image contract.
``scripts/ci/validate_runner_image.py`` runs on a freshly recreated runner and
asserts three things:

1. the image identity binding is present and consistent (baked-vs-recorded),
2. the happy path resolves zero ``uv sync`` (prebuilt env baked + UV_NO_SYNC=1),
3. the runner has the tooling the Receipt-Gate path needs.

These tests drive the validation logic against synthetic lock + env fixtures so
the acceptance contract is regression-locked independently of any live runner.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.ci.validate_runner_image import (
    RECEIPT_GATE_TOOLS,
    check_image_identity,
    check_receipt_gate_readiness,
    check_zero_uv_sync,
    resolve_lock_path,
    validate_runner,
)

pytestmark = pytest.mark.unit


_LOCK = {
    "base_image_digest": "sha256:deadbeef",
    "gh_version": "2.67.0",
    "identity_digest": "6a27412c6da1841cc48a0ef7051ac28b",
    "image_version": 1,
    "kubectl_version": "1.32.1",
    "python_version": "3.12",
    "runner_version": "2.323.0",
    "shared_env_digest": "3b7238ac6e3b8b503e12dfb7",
    "shared_env_install_args": "--frozen --all-extras --all-groups --no-install-project",
    "uv_version": "0.6.14",
}


def _write_lock(tmp_path: Path, overrides: dict[str, object] | None = None) -> Path:
    payload = dict(_LOCK)
    if overrides:
        payload.update(overrides)
    lock_path = tmp_path / "runner-image.lock.json"
    lock_path.write_text(json.dumps(payload), encoding="utf-8")
    return lock_path


def _make_ready_env(tmp_path: Path, repo: str = "omnibase_infra") -> Path:
    """Create a ready prebuilt env: <root>/<repo>/<digest>/{manifest.json,.venv}."""
    env_root = tmp_path / "ci-envs"
    digest_dir = env_root / repo / _LOCK["shared_env_digest"]  # type: ignore[operator]
    (digest_dir / ".venv" / "bin").mkdir(parents=True)
    (digest_dir / ".venv" / "bin" / "python").write_text("", encoding="utf-8")
    (digest_dir / "manifest.json").write_text("{}", encoding="utf-8")
    return env_root


class TestImageIdentity:
    def test_missing_lock_fails(self, tmp_path: Path) -> None:
        result = check_image_identity(None, tmp_path, baked_identity=None)
        assert not result.passed
        assert "no runner image lock file" in result.detail

    def test_consistent_lock_passes_repo_checkout(self, tmp_path: Path) -> None:
        lock = _write_lock(tmp_path)
        # No baked identity env => repo-checkout validation; recorded digest only.
        result = check_image_identity(lock, tmp_path, baked_identity=None)
        assert result.passed, result.detail
        assert "verified" in result.detail

    def test_baked_identity_match_passes(self, tmp_path: Path) -> None:
        lock = _write_lock(tmp_path)
        result = check_image_identity(
            lock,
            tmp_path,
            baked_identity=_LOCK["identity_digest"],  # type: ignore[arg-type]
        )
        assert result.passed, result.detail
        assert "matches baked image" in result.detail

    def test_baked_identity_drift_fails(self, tmp_path: Path) -> None:
        lock = _write_lock(tmp_path)
        result = check_image_identity(
            lock, tmp_path, baked_identity="0000000000000000deadbeefdeadbeef"
        )
        assert not result.passed
        assert "drift" in result.detail

    def test_unbound_baked_identity_fails(self, tmp_path: Path) -> None:
        lock = _write_lock(tmp_path)
        result = check_image_identity(lock, tmp_path, baked_identity="unbound")
        assert not result.passed
        assert "unbound" in result.detail

    def test_empty_identity_digest_fails(self, tmp_path: Path) -> None:
        lock = _write_lock(tmp_path, overrides={"identity_digest": ""})
        result = check_image_identity(lock, tmp_path, baked_identity=None)
        assert not result.passed
        assert "identity_digest" in result.detail


class TestZeroUvSync:
    def test_missing_prebuilt_env_fails(self, tmp_path: Path) -> None:
        result = check_zero_uv_sync(
            tmp_path / "absent", "omnibase_infra", uv_no_sync="1"
        )
        assert not result.passed
        assert "prebuilt env" in result.detail

    def test_ready_env_with_no_sync_passes(self, tmp_path: Path) -> None:
        env_root = _make_ready_env(tmp_path)
        result = check_zero_uv_sync(env_root, "omnibase_infra", uv_no_sync="1")
        assert result.passed, result.detail

    def test_ready_env_without_uv_no_sync_in_validator_env_passes(
        self, tmp_path: Path
    ) -> None:
        # UV_NO_SYNC is published into $GITHUB_ENV at job time, not necessarily set
        # in the validator's own shell; absence is acceptable, the env being baked
        # is the load-bearing signal.
        env_root = _make_ready_env(tmp_path)
        result = check_zero_uv_sync(env_root, "omnibase_infra", uv_no_sync=None)
        assert result.passed, result.detail
        assert "canary" in result.detail

    def test_falsy_uv_no_sync_fails(self, tmp_path: Path) -> None:
        env_root = _make_ready_env(tmp_path)
        result = check_zero_uv_sync(env_root, "omnibase_infra", uv_no_sync="0")
        assert not result.passed
        assert "uv sync" in result.detail

    def test_env_without_manifest_marker_fails(self, tmp_path: Path) -> None:
        env_root = tmp_path / "ci-envs"
        digest_dir = env_root / "omnibase_infra" / "abc123"
        (digest_dir / ".venv" / "bin").mkdir(parents=True)
        (digest_dir / ".venv" / "bin" / "python").write_text("", encoding="utf-8")
        # No manifest.json => env not marked ready.
        result = check_zero_uv_sync(env_root, "omnibase_infra", uv_no_sync="1")
        assert not result.passed


class TestReceiptGateReadiness:
    def test_all_tools_present_passes(self) -> None:
        # python3 is guaranteed present in the test environment.
        result = check_receipt_gate_readiness(tools=["python3"])
        assert result.passed, result.detail

    def test_missing_tool_fails(self) -> None:
        result = check_receipt_gate_readiness(
            tools=["definitely-not-a-real-binary-omn12568"]
        )
        assert not result.passed
        assert "missing Receipt-Gate tooling" in result.detail

    def test_default_tool_set_includes_gh_and_uv(self) -> None:
        assert "gh" in RECEIPT_GATE_TOOLS
        assert "uv" in RECEIPT_GATE_TOOLS
        assert "python3" in RECEIPT_GATE_TOOLS


class TestResolveLockPath:
    def test_explicit_path_wins(self, tmp_path: Path) -> None:
        lock = _write_lock(tmp_path)
        assert resolve_lock_path(lock) == lock

    def test_missing_explicit_path_returns_none(self, tmp_path: Path) -> None:
        assert resolve_lock_path(tmp_path / "nope.json") is None


class TestValidateRunnerAggregate:
    def test_all_green(self, tmp_path: Path) -> None:
        lock = _write_lock(tmp_path)
        env_root = _make_ready_env(tmp_path)
        report = validate_runner(
            repo_root=tmp_path,
            lock_path=lock,
            env_root=env_root,
            env_repo="omnibase_infra",
            baked_identity=_LOCK["identity_digest"],  # type: ignore[arg-type]
            uv_no_sync="1",
            tools=["python3"],
        )
        assert report.ok, report.to_dict()
        assert {c.name for c in report.checks} == {
            "image_identity",
            "zero_uv_sync",
            "receipt_gate_readiness",
        }

    def test_one_failure_makes_report_red(self, tmp_path: Path) -> None:
        lock = _write_lock(tmp_path)
        # No prebuilt env => zero_uv_sync fails => report red.
        report = validate_runner(
            repo_root=tmp_path,
            lock_path=lock,
            env_root=tmp_path / "absent",
            baked_identity=_LOCK["identity_digest"],  # type: ignore[arg-type]
            uv_no_sync="1",
            tools=["python3"],
        )
        assert not report.ok
        report_dict = report.to_dict()
        assert report_dict["ok"] is False
