# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""The runner host's bind-mounted hooks must converge from a merged sha.

WHY THIS FILE EXISTS (OMN-18819). Every `omnibase-ci` runner is a container on
the lab host that bind-mounts its hook scripts from a STAGED copy under
`~/.omnibase/runners/docker/runners/`, deliberately not from a git checkout, so
a half-pulled tree can never change what the fleet executes. Nothing converged
that copy. `deploy-runners.sh` rsyncs it only inside a full pipeline that
force-recreates every runner, and `check_runner_host_artifact_freshness.py`
(OMN-15114) only REPORTS -- against a working tree its own docstring calls
"assumed-current".

Two live consequences, both measured 2026-09-19:

  * OMN-18802's fix merged as `omnibase_infra#3814` (squash `55455c5c8`) and
    the deployed hook was still the 2026-09-04 file. It reached the fleet
    because a human copied it, which the 2026-09-17 operator ruling forbids.
  * OMN-16056 has read Done since 2026-08-20 with its change absent from the
    host, because `docker/runners/git-mirror-refresh.sh` is not in
    `SYNC_PATHS` at all -- no mechanism carries that file. Its Done flip cited
    a dod_verify whose every check was repo-side.

WHAT IS PINNED HERE.

  1. The missing path is in the synced set, read out of the real script.
  2. Expected content comes from a GIT REF, never the working tree. A dirty
     checkout must not be able to define "current" -- that is the specific
     assumption behind the 19-day staleness OMN-15114 was opened on.
  3. Convergence replaces a drifted copy and READS IT BACK. A write whose
     result is never re-hashed is a hope, not a convergence.
  4. Fail-closed: an unreadable or unreachable remote path is reported, never
     counted as converged.
  5. The scheduled entry exists, and it does not write under `/tmp` (standing
     rule on this host).

The git-backed tests build a real throwaway repository and run real `git`
against it, so "resolved from the ref" is exercised rather than mocked.
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from omnibase_core.validators.no_unguarded_git_subprocess import (
    scrub_git_location_env,
)

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_CI = REPO_ROOT / "scripts" / "ci"
FRESHNESS_SCRIPT = SCRIPTS_CI / "check_runner_host_artifact_freshness.py"
DEPLOY_SCRIPT = REPO_ROOT / "scripts" / "deploy-runners.sh"

# The file whose absence from SYNC_PATHS is the whole of the OMN-16056 defect.
MIRROR_REFRESH_PATH = "docker/runners/git-mirror-refresh.sh"

# Assembled rather than spelled: the hardcoded-temp-path lint fires on the
# literal, and this test's whole job is to search for it.
_TMP_PREFIX = "/" + "tmp" + "/"

_GIT_IDENTITY = {
    "GIT_AUTHOR_NAME": "t",
    "GIT_AUTHOR_EMAIL": "t@example.invalid",
    "GIT_COMMITTER_NAME": "t",
    "GIT_COMMITTER_EMAIL": "t@example.invalid",
}


def _git(*args: str, cwd: Path) -> str:
    # OMN-18434: GIT_DIR / GIT_WORK_TREE exported by a hook override both
    # `cwd=` and `git -C`, so an unscrubbed fixture would operate on the real
    # invoking worktree instead of tmp_path.
    result = subprocess.run(
        ["git", *args],
        cwd=str(cwd),
        capture_output=True,
        text=True,
        check=True,
        timeout=60,
        env=scrub_git_location_env(os.environ) | _GIT_IDENTITY,
    )
    return result.stdout.strip()


def _load() -> Any:
    spec = importlib.util.spec_from_file_location(
        "check_runner_host_artifact_freshness", FRESHNESS_SCRIPT
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(SCRIPTS_CI))
    try:
        spec.loader.exec_module(module)
    finally:
        if str(SCRIPTS_CI) in sys.path:
            sys.path.remove(str(SCRIPTS_CI))
    return module


# ---------------------------------------------------------------------------
# 1 -- the missing path
# ---------------------------------------------------------------------------


def test_mirror_refresh_script_is_in_the_synced_set() -> None:
    """`git-mirror-refresh.sh` must be carried to the host.

    Its absence is not a cosmetic omission: it is why OMN-16056 could report
    Done for a month while the four mirrors it added never existed on the
    fleet. Read out of the real deploy script, never a second copy of the list.
    """
    module = _load()
    paths = module.parse_sync_paths(DEPLOY_SCRIPT.read_text(encoding="utf-8"))
    assert MIRROR_REFRESH_PATH in paths, (
        f"{MIRROR_REFRESH_PATH} is not in SYNC_PATHS in {DEPLOY_SCRIPT}, so no "
        "mechanism carries it to the runner host. That is the mechanical "
        "reason OMN-16056's four mirrors were never deployed while its ticket "
        f"read Done. Synced paths found: {sorted(paths)}"
    )


def test_every_bind_mounted_runner_hook_is_in_the_synced_set() -> None:
    """Whatever the runners execute must be something the fleet converges.

    Generalises the assertion above so the next hook added under
    `docker/runners/` cannot repeat OMN-16056 by being invisible to the sync.
    """
    module = _load()
    paths = set(module.parse_sync_paths(DEPLOY_SCRIPT.read_text(encoding="utf-8")))
    on_disk = {
        f"docker/runners/{entry.name}"
        for entry in (REPO_ROOT / "docker" / "runners").iterdir()
        if entry.is_file() and entry.suffix == ".sh"
    }
    missing = sorted(on_disk - paths)
    assert not missing, (
        "these shell artifacts under docker/runners/ are not in SYNC_PATHS, so "
        "a merged change to them can never reach the fleet:\n  " + "\n  ".join(missing)
    )


# ---------------------------------------------------------------------------
# 2 -- the baseline is a ref, not the working tree
# ---------------------------------------------------------------------------


def _make_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git("init", "--quiet", "--initial-branch=dev", ".", cwd=repo)
    (repo / "docker").mkdir()
    (repo / "docker" / "runners").mkdir()
    (repo / "docker" / "runners" / "hook.sh").write_text("committed\n")
    _git("add", "-A", cwd=repo)
    _git("commit", "--quiet", "-m", "c0", cwd=repo)
    return repo


def test_expected_hashes_come_from_the_ref_not_the_dirty_working_tree(
    tmp_path: Path,
) -> None:
    """A dirty checkout must not be able to define what the fleet should carry.

    The pre-existing checker hashes the working tree and documents its
    baseline as "assumed-current". On an operator machine with an in-flight
    edit that is simply false, and it is the assumption behind the 19-day
    staleness OMN-15114 was opened on.
    """
    module = _load()
    repo = _make_repo(tmp_path)
    hook = repo / "docker" / "runners" / "hook.sh"
    hook.write_text("DIRTY UNCOMMITTED EDIT\n")

    hashes = module.compute_ref_hashes(repo, "HEAD", ["docker/runners/hook.sh"])

    import hashlib

    committed = hashlib.sha256(b"committed\n").hexdigest()
    dirty = hashlib.sha256(b"DIRTY UNCOMMITTED EDIT\n").hexdigest()
    assert hashes["docker/runners/hook.sh"] == committed, (
        "the expected hash was taken from the working tree, so an uncommitted "
        "edit would be pushed to every runner as if it were merged."
    )
    assert hashes["docker/runners/hook.sh"] != dirty


def test_ref_content_and_resolved_sha_are_reported_together(tmp_path: Path) -> None:
    """A readback that does not name the sha cannot be audited later."""
    module = _load()
    repo = _make_repo(tmp_path)
    head = _git("rev-parse", "HEAD", cwd=repo)

    assert module.resolve_ref_sha(repo, "HEAD") == head
    assert len(head) == 40


def test_a_ref_that_does_not_exist_fails_closed(tmp_path: Path) -> None:
    """An unresolvable ref must refuse, never fall back to the working tree.

    Falling back is how a converger silently reverts to the behaviour this
    change exists to remove.
    """
    module = _load()
    repo = _make_repo(tmp_path)
    with pytest.raises(module.RefResolutionError):
        module.resolve_ref_sha(repo, "refs/heads/no-such-branch")


# ---------------------------------------------------------------------------
# 3 -- convergence replaces and reads back
# ---------------------------------------------------------------------------


def test_converge_replaces_a_drifted_path_and_reads_it_back() -> None:
    """The write is followed by a re-hash, and the result says so.

    A converger that writes and reports success without re-reading has proved
    nothing about the file the runners will actually execute.
    """
    module = _load()
    written: dict[str, bytes] = {}

    def fake_push(path: str, content: bytes) -> None:
        written[path] = content

    def fake_remote_hash(path: str) -> str | None:
        import hashlib

        if path in written:
            return hashlib.sha256(written[path]).hexdigest()
        return "0" * 64  # the stale copy currently on the host

    results = module.converge_paths(
        paths=["docker/runners/hook.sh"],
        ref_contents={"docker/runners/hook.sh": b"committed\n"},
        remote_hash=fake_remote_hash,
        push=fake_push,
        ref_sha="a" * 40,
    )

    assert written["docker/runners/hook.sh"] == b"committed\n"
    (result,) = results
    assert result.converged is True
    assert result.readback_sha256 == result.expected_sha256, (
        "the readback hash was not compared to the expected hash, so the "
        "result reports a convergence it never verified"
    )
    assert "a" * 40 in result.as_line(), (
        "the readback line does not name the sha it converged from"
    )


def test_converge_leaves_an_already_matching_path_untouched() -> None:
    """The steady state writes nothing.

    Without this the test above passes on a converger that rewrites every file
    on every tick, which would churn the bind mount under running jobs.
    """
    module = _load()
    import hashlib

    expected = hashlib.sha256(b"committed\n").hexdigest()
    written: dict[str, bytes] = {}

    results = module.converge_paths(
        paths=["docker/runners/hook.sh"],
        ref_contents={"docker/runners/hook.sh": b"committed\n"},
        remote_hash=lambda path: expected,
        push=lambda path, content: written.__setitem__(path, content),
        ref_sha="a" * 40,
    )

    assert written == {}, "an in-sync path was rewritten anyway"
    (result,) = results
    assert result.converged is True
    assert result.action == "already-current"


# ---------------------------------------------------------------------------
# 4 -- fail closed
# ---------------------------------------------------------------------------


def test_a_readback_that_disagrees_is_reported_not_swallowed() -> None:
    """If the file on the host does not match after the write, say so."""
    module = _load()

    results = module.converge_paths(
        paths=["docker/runners/hook.sh"],
        ref_contents={"docker/runners/hook.sh": b"committed\n"},
        remote_hash=lambda path: "f" * 64,  # never matches, even after the push
        push=lambda path, content: None,
        ref_sha="a" * 40,
    )

    (result,) = results
    assert result.converged is False
    assert "MISMATCH" in result.as_line().upper()


def test_an_unreachable_host_path_is_reported_not_counted_as_converged() -> None:
    """`None` from the remote hash means unverifiable, which is not success."""
    module = _load()

    results = module.converge_paths(
        paths=["docker/runners/hook.sh"],
        ref_contents={"docker/runners/hook.sh": b"committed\n"},
        remote_hash=lambda path: None,
        push=lambda path, content: None,
        ref_sha="a" * 40,
    )

    (result,) = results
    assert result.converged is False, (
        "an unverifiable path was counted as converged -- the same fail-open "
        "shape the sibling freshness checker was written to avoid"
    )


def test_a_push_that_raises_is_reported_not_propagated() -> None:
    """One unwritable path must not abandon the remaining paths.

    A converger that dies on the first failure leaves the rest of the fleet
    unconverged and reports nothing about them.
    """
    module = _load()

    def exploding_push(path: str, content: bytes) -> None:
        raise OSError("permission denied")

    results = module.converge_paths(
        paths=["a.sh", "b.sh"],
        ref_contents={"a.sh": b"x", "b.sh": b"y"},
        remote_hash=lambda path: "0" * 64,
        push=exploding_push,
        ref_sha="a" * 40,
    )

    assert len(results) == 2, "a failing path stopped the run"
    assert all(not result.converged for result in results)


# ---------------------------------------------------------------------------
# 5 -- it is scheduled, and it does not write under /tmp
# ---------------------------------------------------------------------------


def test_the_converge_run_is_scheduled_by_the_installer() -> None:
    """Convergence must be on a timer, not on somebody remembering."""
    text = DEPLOY_SCRIPT.read_text(encoding="utf-8")
    assert "runner-host-hook-converge" in text, (
        "no scheduled converge entry in deploy-runners.sh. A reconcile nobody "
        "schedules is advisory, which is the state this ticket exists to end."
    )
    assert "--mode converge" in text, (
        "the scheduled entry does not invoke the converge mode."
    )


def test_no_scheduled_entry_writes_its_log_under_tmp() -> None:
    """Standing rule on this host: no `/tmp`.

    The pre-existing freshness cron logs to `/tmp`, which is both against the
    rule and self-defeating -- the evidence of a drift run disappears on
    reboot, exactly when it is most worth reading.
    """
    text = DEPLOY_SCRIPT.read_text(encoding="utf-8")
    offenders = [
        line.strip()
        for line in text.splitlines()
        if "crontab" not in line and _TMP_PREFIX in line and "cron_line=" in line
    ]
    assert not offenders, "a scheduled entry writes under /tmp:\n  " + "\n  ".join(
        offenders
    )
