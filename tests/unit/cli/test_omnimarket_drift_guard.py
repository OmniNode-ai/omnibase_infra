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
import sys
from importlib.metadata import PackageNotFoundError
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from omnibase_infra.cli import omnimarket_drift_guard as guard
from omnibase_infra.cli.omnimarket_drift_guard import (
    DRIFT_OVERRIDE_ENV,
    OmnimarketDriftError,
    canonical_local_omnimarket_commit,
    check_omnimarket_drift,
    installed_omnimarket_commit,
)
from omnibase_infra.cli.workspace_reconcile import (
    ModelReconcileOutcome,
    make_workspace_reconciler,
    reconcile_workspace_venvs,
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


def test_not_installed_refusal_names_the_running_interpreter() -> None:
    """OMN-17190: "omnimarket is not installed" is ambiguous between two very
    different faults -- the CLI venv lost its provider layer, or this is not
    the CLI venv at all.

    The second is what actually happened during OMN-17190 verification:
    ``uv run --project X onex`` silently resolves ``onex`` from the inherited
    PATH whenever the project entrypoint is not resolvable, and that other
    interpreter (a uv-tool env with a PyPI omnimarket and a pre-OMN-17190
    guard) refuses identically whether the real venv is drifted or IN_SYNC.
    Ten of fifteen verification dispatches died that way and the refusal text
    gave no way to tell. Naming ``sys.executable`` makes the next occurrence a
    one-line diagnosis, and pointing at the wrapper names the fix.
    """
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
    assert sys.executable in message
    assert "scripts/onex" in message
    assert "docs/runbooks/onex-cli-invocation.md" in message


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


# --------------------------------------------------------------------------- #
# Self-healing drift refusal (OMN-17190)
# --------------------------------------------------------------------------- #
class _Reconciler:
    """A recording stand-in for the bound workspace reconciler.

    ``on_success`` lets a test model the real effect the script has: it mutates
    site-packages, so the guard's re-probe must see a DIFFERENT answer than its
    first probe. A test double that changes nothing could not distinguish
    "re-checked" from "assumed".
    """

    def __init__(
        self,
        *,
        ok: bool,
        detail: str = "",
        on_success: object = None,
    ) -> None:
        self.outcome = ModelReconcileOutcome(
            ok=ok, command="bash /w/reconcile-workspace-venvs.sh", detail=detail
        )
        self._on_success = on_success
        self.calls = 0

    def __call__(self) -> ModelReconcileOutcome:
        self.calls += 1
        if self.outcome.ok and callable(self._on_success):
            self._on_success()
        return self.outcome


@pytest.fixture
def canonical_head(monkeypatch: pytest.MonkeyPatch) -> str:
    """Pin the canonical clone HEAD so only the installed side varies."""
    head = "a" * 40
    monkeypatch.setattr(
        guard, "canonical_local_omnimarket_commit", lambda omni_home=None: head
    )
    return head


def test_successful_reconcile_lets_the_dispatch_proceed(
    monkeypatch: pytest.MonkeyPatch, canonical_head: str
) -> None:
    """The whole point: drift is repaired in flight, not handed to a human."""
    state = {"installed": "b" * 40}
    monkeypatch.setattr(
        guard, "installed_omnimarket_commit", lambda: state["installed"]
    )

    def _repair() -> None:
        state["installed"] = canonical_head

    reconciler = _Reconciler(ok=True, on_success=_repair)
    guard.check_omnimarket_drift(omni_home="/w", reconcile=reconciler)

    assert reconciler.calls == 1


def test_reconcile_runs_exactly_once(
    monkeypatch: pytest.MonkeyPatch, canonical_head: str
) -> None:
    """No retry loop on the CLI hot path.

    A reconcile that ran and left the venv drifted is reporting something an
    identical second attempt will not fix; looping would turn a clear refusal
    into a hang.
    """
    monkeypatch.setattr(guard, "installed_omnimarket_commit", lambda: "b" * 40)
    reconciler = _Reconciler(ok=True)

    with pytest.raises(guard.OmnimarketDriftError):
        guard.check_omnimarket_drift(omni_home="/w", reconcile=reconciler)

    assert reconciler.calls == 1


def test_failed_reconcile_refuses_and_names_the_exact_command(
    monkeypatch: pytest.MonkeyPatch, canonical_head: str
) -> None:
    monkeypatch.setattr(guard, "installed_omnimarket_commit", lambda: None)
    reconciler = _Reconciler(ok=False, detail="uv sync did not complete")

    with pytest.raises(guard.OmnimarketDriftError) as excinfo:
        guard.check_omnimarket_drift(omni_home="/w", reconcile=reconciler)

    message = str(excinfo.value)
    assert reconciler.calls == 1
    assert "reconcile-workspace-venvs.sh" in message
    assert "uv sync did not complete" in message


def test_failed_reconcile_keeps_the_original_diagnosis_and_the_override(
    monkeypatch: pytest.MonkeyPatch, canonical_head: str
) -> None:
    """A failed reconcile ADDS to the refusal; it must not replace it.

    An earlier revision of this behaviour raised a refusal that named only the
    reconcile failure. That reads as an improvement and is not one: it discards
    the two facts a reader actually needs -- WHAT drifted, and the repair
    command for it -- and hands them a second-order failure to debug in place
    of the first-order one. Three pre-existing dispatch-surface tests
    (`test_drift_guard_fires_before_delegate_dispatch`,
    `test_drift_guard_fires_before_unknown_node_lookup`,
    `test_drift_override_env_unset_still_refuses`) assert exactly that content,
    and they are right to.

    The OMN-13930 override is named for the same reason it is named on every
    other refusal in this module: it is evaluated BEFORE the reconcile, so it
    genuinely works from this state, and a documented escape hatch withheld
    from the message does not stop being used -- it just turns the failure into
    a dead end, which is the argument this module's own docstring makes for
    naming it at all. The refusal still says plainly that a broken venv is to
    be FIXED, not worked around.
    """
    monkeypatch.setattr(guard, "installed_omnimarket_commit", lambda: None)

    with pytest.raises(guard.OmnimarketDriftError) as excinfo:
        guard.check_omnimarket_drift(
            omni_home="/w", reconcile=_Reconciler(ok=False, detail="boom")
        )

    message = str(excinfo.value)
    # The original diagnosis survives.
    assert "NOT INSTALLED" in message
    assert canonical_head[:12] in message
    assert "install-node-skill-package.sh --execute" in message
    # The reconcile failure is added, with the command to reproduce it.
    assert "boom" in message
    assert "reconcile-workspace-venvs.sh" in message
    # And the documented override is still discoverable from the failure alone.
    assert guard.DRIFT_OVERRIDE_ENV in message


def test_reconcile_reporting_success_while_still_drifted_still_refuses(
    monkeypatch: pytest.MonkeyPatch, canonical_head: str
) -> None:
    """Trust the re-check, never the reconciler's own say-so."""
    monkeypatch.setattr(guard, "installed_omnimarket_commit", lambda: "b" * 40)

    with pytest.raises(guard.OmnimarketDriftError) as excinfo:
        guard.check_omnimarket_drift(omni_home="/w", reconcile=_Reconciler(ok=True))

    assert "STILL drifted" in str(excinfo.value)


def test_no_reconciler_preserves_the_pure_refusal(
    monkeypatch: pytest.MonkeyPatch, canonical_head: str
) -> None:
    """Omitting ``reconcile`` must keep the guard a pure detect-and-refuse.

    Every non-CLI caller depends on this: a guard that silently shelled out by
    default would be an astonishing thing to import.
    """
    monkeypatch.setattr(guard, "installed_omnimarket_commit", lambda: "b" * 40)

    with pytest.raises(guard.OmnimarketDriftError) as excinfo:
        guard.check_omnimarket_drift(omni_home="/w")

    assert guard.DRIFT_OVERRIDE_ENV in str(excinfo.value)


def test_reconciler_is_not_invoked_when_there_is_no_drift(
    monkeypatch: pytest.MonkeyPatch, canonical_head: str
) -> None:
    monkeypatch.setattr(guard, "installed_omnimarket_commit", lambda: canonical_head)
    reconciler = _Reconciler(ok=True)

    guard.check_omnimarket_drift(omni_home="/w", reconcile=reconciler)

    assert reconciler.calls == 0


def test_allow_drift_short_circuits_before_any_reconcile(
    monkeypatch: pytest.MonkeyPatch, canonical_head: str
) -> None:
    """The operator override means 'I accept this build', not 'repair it'.

    Silently reinstalling under an explicit accept-as-is would change the very
    build the operator chose to run against.
    """
    monkeypatch.setattr(guard, "installed_omnimarket_commit", lambda: "b" * 40)
    reconciler = _Reconciler(ok=True)

    guard.check_omnimarket_drift(omni_home="/w", allow_drift=True, reconcile=reconciler)

    assert reconciler.calls == 0


def test_make_workspace_reconciler_returns_none_without_omni_home() -> None:
    assert make_workspace_reconciler(None) is None
    assert make_workspace_reconciler("") is None


def test_missing_reconcile_script_is_a_failed_outcome_not_a_raise(
    tmp_path: Path,
) -> None:
    """The guard turns outcomes into refusals; the adapter must never raise."""
    outcome = reconcile_workspace_venvs(str(tmp_path))

    assert outcome.ok is False
    assert "reconcile-workspace-venvs.sh" in outcome.command
    assert "not found" in outcome.detail
