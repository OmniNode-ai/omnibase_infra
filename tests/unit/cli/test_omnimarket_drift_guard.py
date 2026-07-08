# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Tests for the omnimarket pre-flight drift guard (OMN-14060).

Covers both resolver functions (``installed_omnimarket_commit``,
``canonical_local_omnimarket_commit``) in isolation, and the combined
``check_omnimarket_drift`` fail-open / fail-closed behavior. The canonical-clone
resolver is exercised against a REAL throwaway git repo (not mocked) so the
`git -C <root> rev-parse HEAD` invocation is proven, not assumed.
"""

from __future__ import annotations

import json
import subprocess
from importlib.metadata import PackageNotFoundError
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from omnibase_infra.cli.omnimarket_drift_guard import (
    OmnimarketDriftError,
    canonical_local_omnimarket_commit,
    check_omnimarket_drift,
    installed_omnimarket_commit,
)

pytestmark = pytest.mark.unit

_FAKE_SHA_A = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
_FAKE_SHA_B = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"


def _make_git_repo(root: Path) -> str:
    """Init a throwaway git repo at ``root`` with one commit; return its HEAD sha."""
    subprocess.run(["git", "init", "--quiet"], cwd=root, check=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"], cwd=root, check=True
    )
    subprocess.run(["git", "config", "user.name", "Test"], cwd=root, check=True)
    (root / "README.md").write_text("x", encoding="utf-8")
    subprocess.run(["git", "add", "README.md"], cwd=root, check=True)
    subprocess.run(["git", "commit", "--quiet", "-m", "init"], cwd=root, check=True)
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


# ---------------------------------------------------------------------------
# installed_omnimarket_commit
# ---------------------------------------------------------------------------


def test_installed_commit_none_when_package_absent() -> None:
    with patch(
        "omnibase_infra.cli.omnimarket_drift_guard.distribution",
        side_effect=PackageNotFoundError,
    ):
        assert installed_omnimarket_commit() is None


def test_installed_commit_none_when_not_vcs_install() -> None:
    # A PyPI wheel install has no direct_url.json at all -- OMN-14064's case.
    fake_dist = MagicMock()
    fake_dist.read_text.return_value = None
    with patch(
        "omnibase_infra.cli.omnimarket_drift_guard.distribution",
        return_value=fake_dist,
    ):
        assert installed_omnimarket_commit() is None


def test_installed_commit_none_when_direct_url_has_no_vcs_info() -> None:
    # e.g. a local path install (file:// direct_url with no vcs_info key).
    fake_dist = MagicMock()
    fake_dist.read_text.return_value = json.dumps({"url": "file:///some/path"})
    with patch(
        "omnibase_infra.cli.omnimarket_drift_guard.distribution",
        return_value=fake_dist,
    ):
        assert installed_omnimarket_commit() is None


def test_installed_commit_none_when_direct_url_malformed() -> None:
    fake_dist = MagicMock()
    fake_dist.read_text.return_value = "{not json"
    with patch(
        "omnibase_infra.cli.omnimarket_drift_guard.distribution",
        return_value=fake_dist,
    ):
        assert installed_omnimarket_commit() is None


def test_installed_commit_returns_sha_from_vcs_install() -> None:
    fake_dist = MagicMock()
    fake_dist.read_text.return_value = json.dumps(
        {
            "url": "https://github.com/OmniNode-ai/omnimarket.git",
            "vcs_info": {"vcs": "git", "commit_id": _FAKE_SHA_A},
        }
    )
    with patch(
        "omnibase_infra.cli.omnimarket_drift_guard.distribution",
        return_value=fake_dist,
    ):
        assert installed_omnimarket_commit() == _FAKE_SHA_A


# ---------------------------------------------------------------------------
# canonical_local_omnimarket_commit
# ---------------------------------------------------------------------------


def test_canonical_none_when_omni_home_unset() -> None:
    assert canonical_local_omnimarket_commit(omni_home="") is None


def test_canonical_none_when_clone_absent(tmp_path: Path) -> None:
    # $OMNI_HOME exists but has no omnimarket/.git subdirectory.
    assert canonical_local_omnimarket_commit(omni_home=str(tmp_path)) is None


def test_canonical_reads_real_local_clone_head(tmp_path: Path) -> None:
    # Real git repo, not mocked -- proves the `git -C <root> rev-parse HEAD`
    # invocation actually works end-to-end.
    omnimarket_root = tmp_path / "omnimarket"
    omnimarket_root.mkdir()
    head_sha = _make_git_repo(omnimarket_root)
    assert canonical_local_omnimarket_commit(omni_home=str(tmp_path)) == head_sha


def test_canonical_none_when_git_invocation_fails(tmp_path: Path) -> None:
    # A directory with a .git *file* (not a real repo) trips `git rev-parse`.
    omnimarket_root = tmp_path / "omnimarket"
    omnimarket_root.mkdir()
    (omnimarket_root / ".git").write_text("not a real git dir", encoding="utf-8")
    assert canonical_local_omnimarket_commit(omni_home=str(tmp_path)) is None


# ---------------------------------------------------------------------------
# check_omnimarket_drift
# ---------------------------------------------------------------------------


def test_drift_check_fails_open_when_not_installed() -> None:
    with patch(
        "omnibase_infra.cli.omnimarket_drift_guard.installed_omnimarket_commit",
        return_value=None,
    ):
        check_omnimarket_drift()  # must not raise


def test_drift_check_fails_open_when_no_canonical_clone() -> None:
    with (
        patch(
            "omnibase_infra.cli.omnimarket_drift_guard.installed_omnimarket_commit",
            return_value=_FAKE_SHA_A,
        ),
        patch(
            "omnibase_infra.cli.omnimarket_drift_guard.canonical_local_omnimarket_commit",
            return_value=None,
        ),
    ):
        check_omnimarket_drift()  # must not raise -- can't determine canonical


def test_drift_check_passes_when_commits_match() -> None:
    with (
        patch(
            "omnibase_infra.cli.omnimarket_drift_guard.installed_omnimarket_commit",
            return_value=_FAKE_SHA_A,
        ),
        patch(
            "omnibase_infra.cli.omnimarket_drift_guard.canonical_local_omnimarket_commit",
            return_value=_FAKE_SHA_A,
        ),
    ):
        check_omnimarket_drift()  # must not raise


def test_drift_check_raises_on_mismatch_with_actionable_message() -> None:
    with (
        patch(
            "omnibase_infra.cli.omnimarket_drift_guard.installed_omnimarket_commit",
            return_value=_FAKE_SHA_A,
        ),
        patch(
            "omnibase_infra.cli.omnimarket_drift_guard.canonical_local_omnimarket_commit",
            return_value=_FAKE_SHA_B,
        ),
    ):
        with pytest.raises(OmnimarketDriftError) as exc_info:
            check_omnimarket_drift()
    message = str(exc_info.value)
    assert _FAKE_SHA_A[:12] in message
    assert _FAKE_SHA_B[:12] in message
    assert "check-omnimarket-venv-drift.sh --repair" in message
