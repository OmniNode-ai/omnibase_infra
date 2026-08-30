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
import logging
import subprocess
from importlib.metadata import PackageNotFoundError
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from omnibase_infra.cli.omnimarket_drift_guard import (
    DRIFT_OVERRIDE_ENV,
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


def test_drift_check_fails_open_when_not_installed_and_no_canonical_clone() -> None:
    # Neither side determinable (e.g. CI runner, no $OMNI_HOME) -- fails open.
    with (
        patch(
            "omnibase_infra.cli.omnimarket_drift_guard.installed_omnimarket_commit",
            return_value=None,
        ),
        patch(
            "omnibase_infra.cli.omnimarket_drift_guard.canonical_local_omnimarket_commit",
            return_value=None,
        ),
    ):
        check_omnimarket_drift()  # must not raise


def test_drift_check_raises_when_not_installed_but_canonical_clone_present() -> None:
    # OMN-14531: the actual aislop_sweep-blind regression -- omnimarket
    # silently reverted from a git co-install to completely absent while a
    # canonical $OMNI_HOME/omnimarket clone was reachable. This is a
    # DETERMINABLE, actionable state and must now raise loudly instead of
    # falling through the old unconditional "installed is None -> return"
    # fail-open path.
    with (
        patch(
            "omnibase_infra.cli.omnimarket_drift_guard.installed_omnimarket_commit",
            return_value=None,
        ),
        patch(
            "omnibase_infra.cli.omnimarket_drift_guard.canonical_local_omnimarket_commit",
            return_value=_FAKE_SHA_A,
        ),
    ):
        with pytest.raises(OmnimarketDriftError) as exc_info:
            check_omnimarket_drift()
    message = str(exc_info.value)
    assert "NOT INSTALLED" in message
    assert _FAKE_SHA_A[:12] in message
    assert "install-node-skill-package.sh --execute" in message


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


def test_drift_check_names_full_path_repair_command_when_omni_home_known(
    tmp_path: Path,
) -> None:
    # The refusal fires mid-dispatch, not necessarily from inside
    # $OMNI_HOME/omnibase_infra -- the named repair command must be a full,
    # copy-pasteable path, not a cwd-relative one that only resolves by luck.
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
            check_omnimarket_drift(omni_home=str(tmp_path))
    message = str(exc_info.value)
    expected_repair = str(
        tmp_path / "omnibase_infra" / "scripts" / "check-omnimarket-venv-drift.sh"
    )
    expected_install = str(
        tmp_path / "omnibase_infra" / "scripts" / "install-node-skill-package.sh"
    )
    assert expected_repair in message
    assert expected_install in message


# ---------------------------------------------------------------------------
# Default-ON refusal + the named override escape hatch (OMN-13930)
# ---------------------------------------------------------------------------
#
# The guard shipped fail-closed with NO escape hatch and NO env named in its
# message: an operator who hit it in a legitimate edge case (deliberately
# testing an unmerged omnimarket branch, a detached canonical clone mid-
# rebase) had no supported way past it and no string to search for. The
# unsupported workarounds are worse than the drift -- unsetting $OMNI_HOME
# silently disables the guard EVERYWHERE with no warning, and editing the
# guard source is untracked. These tests pin the contract: refusal stays the
# default, the override is opt-in, and its name is printed in the refusal
# itself so the escape hatch is discoverable from the failure alone.
#
# The env var is READ at the CLI boundary (click ``envvar=``), never here --
# this module stays a pure function of its arguments. The env-to-argument
# binding is proven at the CLI seam in ``test_cli_skill.py``; an unbound
# option would be exactly the OMN-14531 silent-no-op trap.


def _patch_drift(monkeypatch: pytest.MonkeyPatch, installed: str | None) -> None:
    """Force a determinable drift state: canonical present, installed as given."""
    monkeypatch.setattr(
        "omnibase_infra.cli.omnimarket_drift_guard.installed_omnimarket_commit",
        lambda: installed,
    )
    monkeypatch.setattr(
        "omnibase_infra.cli.omnimarket_drift_guard.canonical_local_omnimarket_commit",
        lambda omni_home=None: _FAKE_SHA_B,
    )


@pytest.mark.parametrize("installed", [None, _FAKE_SHA_A])
def test_refusal_message_names_the_override_env(
    monkeypatch: pytest.MonkeyPatch, installed: str | None
) -> None:
    """BOTH refusal paths (absent install, stale install) name the override env.

    An escape hatch nobody can find is not an escape hatch. Asserting on the
    exported constant rather than a copied literal keeps the message and the
    binding from drifting apart.
    """
    _patch_drift(monkeypatch, installed)
    with pytest.raises(OmnimarketDriftError) as exc_info:
        check_omnimarket_drift()
    assert DRIFT_OVERRIDE_ENV in str(exc_info.value)


@pytest.mark.parametrize("installed", [None, _FAKE_SHA_A])
def test_refusal_is_default_on_when_override_not_requested(
    monkeypatch: pytest.MonkeyPatch, installed: str | None
) -> None:
    """Refusal is the DEFAULT: ``allow_drift`` defaults False at every call site.

    Keyword-only with a False default means a call site added later that
    forgets the argument fails CLOSED rather than silently disabling the
    guard.
    """
    _patch_drift(monkeypatch, installed)
    with pytest.raises(OmnimarketDriftError):
        check_omnimarket_drift()


@pytest.mark.parametrize("installed", [None, _FAKE_SHA_A])
def test_override_downgrades_refusal_to_a_loud_warning(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    installed: str | None,
) -> None:
    """An explicit override dispatches, but never silently.

    The warning is the point: a silent bypass would reintroduce the exact
    invisible-drift failure mode this guard exists to end, so the override
    must be LOUD on every dispatch and must still name the variable that
    caused it.
    """
    _patch_drift(monkeypatch, installed)
    with caplog.at_level(logging.WARNING):
        check_omnimarket_drift(allow_drift=True)  # must not raise
    combined = " ".join(record.getMessage() for record in caplog.records)
    assert DRIFT_OVERRIDE_ENV in combined
    assert _FAKE_SHA_B[:12] in combined


def test_override_does_not_warn_when_there_is_no_drift(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """With the override set but NO drift present, nothing is warned.

    Guards against an implementation that warns on the override's mere
    presence: that would train operators to ignore the warning, defeating it
    for the run where drift is real.
    """
    _patch_drift(monkeypatch, _FAKE_SHA_B)  # installed == canonical
    with caplog.at_level(logging.WARNING):
        check_omnimarket_drift(allow_drift=True)
    assert not [r for r in caplog.records if DRIFT_OVERRIDE_ENV in r.getMessage()]
