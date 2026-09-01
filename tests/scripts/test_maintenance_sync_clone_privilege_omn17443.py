# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The `:37` host-maintenance sync writes as the clone owner, or not at all (OMN-17443).

THE INCIDENT, AND WHY IT IS A SECOND ONE

OMN-17366 made the workspace *reconciler* write as the surface owner and
repaired 2010 root-owned paths under `.201`'s operator-owned clones. A full root
reconciler tick immediately afterwards left 0 -> 0. Roughly thirty minutes later
the operator could not pull::

    $ git -C /data/omninode/omnibase_infra pull --ff-only origin dev
    fatal: unpack-objects failed

32 root-owned paths were back, all under `omnibase_infra/.git/objects`, and
their mtimes named the hour exactly -- `2026-08-31 23:37:02`. There is one root
job at `:37`::

    # /etc/cron.d/omninode-host-maintenance-sync
    37 * * * * root /data/maintenance/bin/omninode-host-maintenance-sync.sh --check --slack

and it fetched into that clone as root, once an hour, for as long as it existed.
Identical defect to OMN-17366; different job. Every "the host is clean" claim
therefore had a one-hour shelf life.

WHY OMN-17366's GATE DID NOT CATCH IT -- THE FINDING THAT OUTLIVES THE FETCH

`scripts/check_reconciler_privilege.py` discovered the scripts it scans by the
glob `scripts/reconcile*.sh`. This file matches neither that pattern nor that
directory, so the one file still carrying the defect was the one file the gate
never opened.

That is the OMN-17383 lesson repeating verbatim -- a gate whose scope stops at a
filename pattern gives false assurance -- and it has now recurred twice inside
the same epic. So the fix here is two-part and the second part is the larger
one: route this script's write through the shared privilege library, AND make
the gate's DISCOVERY follow the invocation (what a scheduler actually runs)
rather than a filename shape, so a third recurrence needs a new kind of mistake
rather than a new filename.

WHAT IS PROVEN HERE, AND WHAT IS DELIBERATELY PROVEN ELSEWHERE

The behavioural half below drives the real artifact against a real git clone
with a real remote, and reads the *write* directly: `refs/remotes/origin/dev`
and the loose-object count, before and after. "It refused" is a weaker claim
than "it did not write", and 32 objects appearing in `.git/objects` is precisely
what the incident was.

The root -> `runuser` privilege DROP is not re-proven here. It belongs to
`scripts/reconcile_privilege_lib.sh` and is already pinned by
`tests/scripts/test_reconcile_clone_privilege_omn17366.py`, which can shim
`id`/`runuser` onto PATH because `reconcile-host.sh` inherits PATH. This script
deliberately pins its own minimal PATH (it runs from cron, where inheriting is
the bug), so shimming it would require adding an env knob that changes which
binaries a root job executes -- a real hardening regression traded for a test
convenience. Sourcing the ONE library instead of copying its mechanics is what
makes that split legitimate, and `test_the_sync_shares_the_one_privilege_library`
is what keeps it true.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]
SYNC_SCRIPT = REPO_ROOT / "deploy" / "maintenance" / "omninode-host-maintenance-sync.sh"
LIB = REPO_ROOT / "scripts" / "reconcile_privilege_lib.sh"
GATE = REPO_ROOT / "scripts" / "check_reconciler_privilege.py"
VENV_RECONCILER = REPO_ROOT / "scripts" / "reconcile-workspace-venvs.sh"

TRACKED_REL = "deploy/maintenance/omninode-system-slack-report.sh"
TRACKED_BODY = "#!/usr/bin/env bash\necho canonical\n"

#: A user this process demonstrably is not. The library's own contract says
#: callers may set CURRENT_USER; setting it to a stranger models the
#: root-cron-vs-operator split without needing privileges to test a privilege
#: drop, and without shimming a PATH this script pins on purpose.
FOREIGN = "someone-else"


def _git(cwd: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "GIT_CONFIG_GLOBAL": "/dev/null",
            "GIT_CONFIG_SYSTEM": "/dev/null",
        },
    )


def _object_count(clone: Path) -> int:
    """Loose objects under .git/objects -- the thing that actually accumulated.

    The incident was counted this way (32 root-owned paths, "all in
    `omnibase_infra/.git/objects`"), so the assertion is counted the same way
    rather than through a proxy for it.
    """
    objects = clone / ".git" / "objects"
    return sum(1 for path in objects.rglob("*") if path.is_file())


def _origin_dev(clone: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(clone), "rev-parse", "--verify", "refs/remotes/origin/dev"],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip()


@pytest.fixture
def stale_clone(tmp_path: Path) -> Path:
    """A real clone whose `origin/dev` is one commit behind a real remote.

    Staleness is the instrument: a fetch that runs ADVANCES the remote-tracking
    ref and deposits objects, and a fetch that was refused leaves both exactly
    where they were. No shim can fake either reading.
    """
    remote = tmp_path / "remote.git"
    _git(tmp_path, "init", "--bare", "--quiet", "-b", "dev", str(remote))

    seed = tmp_path / "seed"
    (seed / "deploy" / "maintenance").mkdir(parents=True)
    (seed / "scripts").mkdir(parents=True)
    _git(tmp_path, "init", "--quiet", "-b", "dev", str(seed))
    _git(seed, "config", "user.email", "test@omninode.ai")
    _git(seed, "config", "user.name", "test")
    (seed / TRACKED_REL).write_text(TRACKED_BODY, encoding="utf-8")
    # The script sources the privilege library FROM the clone it syncs, because
    # it is installed flat into /data/maintenance/bin with no repo beside it.
    shutil.copy2(LIB, seed / "scripts" / LIB.name)
    _git(seed, "add", "-A")
    _git(seed, "commit", "--quiet", "--no-gpg-sign", "-m", "seed")
    _git(seed, "push", "--quiet", str(remote), "dev")

    clone = tmp_path / "infra-clone"
    _git(tmp_path, "clone", "--quiet", str(remote), str(clone))

    # Move the remote forward so the clone's origin/dev is genuinely behind.
    (seed / "deploy" / "maintenance" / "later.txt").write_text("later\n")
    _git(seed, "add", "-A")
    _git(seed, "commit", "--quiet", "--no-gpg-sign", "-m", "advance the remote")
    _git(seed, "push", "--quiet", str(remote), "dev")
    return clone


def _manifest(tmp_path: Path, hostpath: Path) -> Path:
    path = tmp_path / "manifest.txt"
    path.write_text(f"{TRACKED_REL}|{hostpath}|0755\n", encoding="utf-8")
    return path


def _run_sync(
    clone: Path, tmp_path: Path, *, current_user: str | None = None
) -> subprocess.CompletedProcess[str]:
    """Run the real script with the fetch ENABLED -- that write is the subject."""
    hostpath = tmp_path / "installed.sh"
    hostpath.write_text(TRACKED_BODY, encoding="utf-8")
    env = dict(os.environ)
    env.update(
        {
            "OMNINODE_INFRA_REPO_ROOT": str(clone),
            "OMNINODE_MAINTENANCE_SYNC_MANIFEST": str(_manifest(tmp_path, hostpath)),
            "OMNINODE_ALERT_ENV_FILE": str(tmp_path / "absent.env"),
            "GIT_CONFIG_GLOBAL": "/dev/null",
            "GIT_CONFIG_SYSTEM": "/dev/null",
        }
    )
    env.pop("OMNINODE_MAINTENANCE_SYNC_SKIP_FETCH", None)
    if current_user is not None:
        env["CURRENT_USER"] = current_user
    return subprocess.run(
        ["bash", str(SYNC_SCRIPT), "--check"],
        capture_output=True,
        text=True,
        env=env,
        timeout=120,
        check=False,
    )


# --------------------------------------------------------------------------- #
# AC1 -- it writes as the owner, or it does not write
# --------------------------------------------------------------------------- #
def test_a_clone_this_process_cannot_own_is_refused_before_any_fetch(
    stale_clone: Path, tmp_path: Path
) -> None:
    """RED before the fix: the pre-fix script fetched regardless and exited 0.

    The refusal is the visible half. The load-bearing half is the pair of
    readings around it -- `origin/dev` unmoved and not one new object -- because
    32 new objects under `.git/objects` is exactly what the operator's broken
    `pull` was made of.
    """
    before_ref = _origin_dev(stale_clone)
    before_objects = _object_count(stale_clone)

    result = _run_sync(stale_clone, tmp_path, current_user=FOREIGN)

    assert result.returncode == 2, (
        "a clone this process cannot write as the owner of must be FATAL, never "
        f"a fetch anyway. got exit {result.returncode}: {result.stdout}{result.stderr}"
    )
    assert "cannot become" in result.stderr, result.stderr
    assert _origin_dev(stale_clone) == before_ref, (
        "the sync advanced refs/remotes/origin/dev in a clone it may not write "
        "-- that ref update is one of the 32 root-owned paths from the incident"
    )
    assert _object_count(stale_clone) == before_objects, (
        "the sync deposited git objects into a clone it may not write; that is "
        "the OMN-17443 defect verbatim"
    )


def test_owning_the_clone_still_fetches_normally(
    stale_clone: Path, tmp_path: Path
) -> None:
    """Control, and it must stay green: on every host where the runner IS the
    owner -- a developer machine, the operator running this by hand -- the guard
    is invisible and the fetch happens. A guard that refuses everywhere would
    turn a live drift detector into a job that never checks anything, which is
    strictly worse than the bug it replaces.
    """
    before_ref = _origin_dev(stale_clone)

    result = _run_sync(stale_clone, tmp_path)

    assert result.returncode == 0, result.stdout + result.stderr
    assert "drifted=0" in result.stdout, result.stdout
    assert _origin_dev(stale_clone) != before_ref, (
        "origin/dev did not move, so the fetch never ran and this control is "
        "passing for the wrong reason"
    )


def test_a_missing_privilege_library_is_fatal_not_a_silent_root_fetch(
    stale_clone: Path, tmp_path: Path
) -> None:
    """Fail-closed, matching this script's own stated doctrine.

    Without the library there is no way to know who owns the clone, and "could
    not determine" must never resolve to "fetch as whoever I am" -- that is the
    defect with an extra step. The script's header says a failed fetch or an
    unreadable path is CRITICAL; not knowing who to write as is the same class.
    """
    (stale_clone / "scripts" / LIB.name).unlink()
    before_objects = _object_count(stale_clone)

    result = _run_sync(stale_clone, tmp_path)

    assert result.returncode == 2, result.stdout + result.stderr
    assert LIB.name in result.stderr, result.stderr
    assert _object_count(stale_clone) == before_objects


def test_skip_fetch_needs_no_library_because_it_writes_nothing(
    stale_clone: Path, tmp_path: Path
) -> None:
    """The guard covers the WRITE, not the script.

    ``OMNINODE_MAINTENANCE_SYNC_SKIP_FETCH=1`` is the caller-already-fetched
    path: it performs no clone write at all, so demanding a privilege plan there
    would be ceremony -- and ceremony in a fail-closed script is what gets the
    fail-closed part loosened later. This also keeps the shipped hermetic suite
    in `tests/unit/scripts/test_omninode_host_maintenance_sync.py` honest
    against a fake clone that carries no `scripts/` directory.
    """
    (stale_clone / "scripts" / LIB.name).unlink()
    hostpath = tmp_path / "installed.sh"
    hostpath.write_text(TRACKED_BODY, encoding="utf-8")
    env = dict(os.environ)
    env.update(
        {
            "OMNINODE_INFRA_REPO_ROOT": str(stale_clone),
            "OMNINODE_MAINTENANCE_SYNC_MANIFEST": str(_manifest(tmp_path, hostpath)),
            "OMNINODE_MAINTENANCE_SYNC_SKIP_FETCH": "1",
            "OMNINODE_ALERT_ENV_FILE": str(tmp_path / "absent.env"),
        }
    )
    result = subprocess.run(
        ["bash", str(SYNC_SCRIPT), "--check"],
        capture_output=True,
        text=True,
        env=env,
        timeout=120,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr


def test_the_sync_shares_the_one_privilege_library() -> None:
    """OMN-17366's central requirement, extended to the third writer.

    A second `as_owner` here would be exactly the drift the library exists to
    prevent, and it is the copy nobody watches that rots. This assertion is what
    licenses this module to leave the privilege-drop mechanics to
    `test_reconcile_clone_privilege_omn17366.py` instead of re-proving them.
    """
    source = SYNC_SCRIPT.read_text(encoding="utf-8")
    assert LIB.name in source, f"{SYNC_SCRIPT.name} does not source {LIB.name}"
    assert "as_owner() {" not in source, (
        f"{SYNC_SCRIPT.name} defines its own as_owner instead of sourcing "
        f"{LIB.name} -- that is the second implementation OMN-17366 forbids"
    )


# --------------------------------------------------------------------------- #
# AC3 -- the gate's SCOPE, not just its rule
# --------------------------------------------------------------------------- #
def _gate(repo_root: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["python3", str(GATE), "--repo-root", str(repo_root)],
        capture_output=True,
        text=True,
        check=False,
    )


def _fixture_repo(
    tmp_path: Path,
    *,
    cron_line: str,
    manifest_entries: str,
    host_scripts: dict[str, str],
) -> Path:
    """A minimal repo shaped like this one: reconcilers, a manifest, a cron unit."""
    root = tmp_path / "repo"
    scripts = root / "scripts"
    cron_dir = root / "deploy" / "maintenance" / "cron.d"
    cron_dir.mkdir(parents=True)
    scripts.mkdir(parents=True)

    shutil.copy2(VENV_RECONCILER, scripts / VENV_RECONCILER.name)
    shutil.copy2(LIB, scripts / LIB.name)
    (scripts / "reconcile-host.sh").write_text(
        "#!/usr/bin/env bash\n"
        'source "$SCRIPT_DIR/reconcile_privilege_lib.sh"\n'
        'as_owner git -C "$OMNI_HOME/$repo" fetch --prune origin "$BRANCH"\n',
        encoding="utf-8",
    )
    (root / "deploy" / "maintenance" / SYNC_SCRIPT.name).write_text(
        "#!/usr/bin/env bash\nMANIFEST=(\n" + manifest_entries + ")\n",
        encoding="utf-8",
    )
    for relpath, body in host_scripts.items():
        target = root / relpath
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(body, encoding="utf-8")
    (cron_dir / "unit").write_text(
        "SHELL=/bin/bash\nPATH=/usr/bin:/bin\n\n" + cron_line + "\n", encoding="utf-8"
    )
    return root


_SYNC_ENTRY = (
    '  "deploy/maintenance/omninode-host-maintenance-sync.sh'
    '|/data/maintenance/bin/omninode-host-maintenance-sync.sh|0755"\n'
)


def test_the_gate_passes_on_the_real_repository() -> None:
    result = _gate(REPO_ROOT)
    assert result.returncode == 0, result.stdout + result.stderr


def test_the_gate_scans_a_scheduled_host_script_outside_the_reconciler_glob(
    tmp_path: Path,
) -> None:
    """RED before the fix, and this is the whole ticket.

    The offending file matches neither `scripts/reconcile*.sh` nor `scripts/`,
    so under the old discovery the gate reported OK while a root cron job
    fetched into an operator-owned clone every hour. Discovery must follow what
    a scheduler INVOKES.
    """
    root = _fixture_repo(
        tmp_path,
        cron_line="37 * * * * root /data/maintenance/bin/hourly-sync.sh --check",
        manifest_entries=_SYNC_ENTRY
        + '  "deploy/maintenance/hourly-sync.sh|/data/maintenance/bin/hourly-sync.sh|0755"\n',
        host_scripts={
            "deploy/maintenance/hourly-sync.sh": (
                "#!/usr/bin/env bash\n"
                'git -C "$INFRA_REPO_ROOT" fetch --quiet origin dev\n'
            )
        },
    )

    result = _gate(root)

    assert result.returncode == 1, (
        "the gate passed a scheduled host script whose git fetch runs as "
        f"whoever cron is: {result.stdout}{result.stderr}"
    )
    assert "hourly-sync.sh" in result.stderr, result.stderr
    assert "as_owner" in result.stderr, result.stderr


def test_the_gate_accepts_a_scheduled_host_script_that_writes_as_the_owner(
    tmp_path: Path,
) -> None:
    """The shape the fix takes must actually pass, or the gate is unsatisfiable."""
    root = _fixture_repo(
        tmp_path,
        cron_line="37 * * * * root /data/maintenance/bin/hourly-sync.sh --check",
        manifest_entries=_SYNC_ENTRY
        + '  "deploy/maintenance/hourly-sync.sh|/data/maintenance/bin/hourly-sync.sh|0755"\n',
        host_scripts={
            "deploy/maintenance/hourly-sync.sh": (
                "#!/usr/bin/env bash\n"
                'source "$INFRA_REPO_ROOT/scripts/reconcile_privilege_lib.sh"\n'
                'rp_plan_privileges "$INFRA_REPO_ROOT"\n'
                'as_owner git -C "$INFRA_REPO_ROOT" fetch --quiet origin dev\n'
            )
        },
    )

    result = _gate(root)

    assert result.returncode == 0, result.stdout + result.stderr


def test_a_scheduled_command_outside_the_manifest_cannot_dodge_the_gate(
    tmp_path: Path,
) -> None:
    """The anti-dodge property, and the reason AC3 is worth more than the fetch.

    Discovery that followed the invocation but silently gave up when the invoked
    path mapped to no repo file would restore the original hole through a new
    door: install an unlisted script, schedule it as root, and the gate reports
    OK on a file it once again never opened. Unmapped is FATAL, not skipped --
    the same fail-closed posture the sync script applies to its own manifest.
    """
    root = _fixture_repo(
        tmp_path,
        cron_line="19 * * * * root /data/maintenance/bin/not-in-the-manifest.sh",
        manifest_entries=_SYNC_ENTRY,
        host_scripts={},
    )

    result = _gate(root)

    assert result.returncode == 1, result.stdout + result.stderr
    assert "not-in-the-manifest.sh" in result.stderr, result.stderr


def test_a_manifest_entry_pointing_at_no_repo_file_is_fatal(tmp_path: Path) -> None:
    """The mapping must resolve to something the gate can actually read.

    A manifest row naming a deleted repo path would otherwise satisfy the
    lookup above while leaving the scheduled command unscanned -- "could not
    determine" resolving to "fine", which this family of scripts exists to end.
    """
    root = _fixture_repo(
        tmp_path,
        cron_line="19 * * * * root /data/maintenance/bin/deleted.sh",
        manifest_entries=_SYNC_ENTRY
        + '  "deploy/maintenance/deleted.sh|/data/maintenance/bin/deleted.sh|0755"\n',
        host_scripts={},
    )

    result = _gate(root)

    assert result.returncode == 1, result.stdout + result.stderr
    assert "deploy/maintenance/deleted.sh" in result.stderr, result.stderr


def test_the_gate_rejects_the_captured_root_fetching_sync(tmp_path: Path) -> None:
    """Incident replay (OMN-15547 registry): the REAL bytes, not a stand-in.

    Everything above proves the new discovery rule against fixtures this module
    wrote. That is exactly the weakness OMN-15547 exists to close -- a guard fed
    a synthetic input it was designed alongside can be green while enforcing
    nothing. So this case drives the shipped gate over
    ``omninode-host-maintenance-sync.root-fetch.sh.captured``: the file as it
    stood on dev at ``eddf4a9d``, which is what root ran at ``:37``, alongside
    the real, unmodified cron unit that runs it. The cron unit is copied from
    the working tree rather than captured because this fix does not touch it --
    those are the same bytes either way.

    The accept control at the end is load-bearing. Without it a gate that
    refused every scheduled host script would satisfy the reject half and look
    correct, while making the fix unshippable.
    """
    captured = (
        REPO_ROOT
        / "tests"
        / "fixtures"
        / "omn17443"
        / "omninode-host-maintenance-sync.root-fetch.sh.captured"
    )

    def _repo(sync_body: str, where: Path) -> Path:
        scripts = where / "scripts"
        scripts.mkdir(parents=True)
        shutil.copy2(VENV_RECONCILER, scripts / VENV_RECONCILER.name)
        shutil.copy2(LIB, scripts / LIB.name)
        shutil.copy2(REPO_ROOT / "scripts" / "reconcile-host.sh", scripts)
        maintenance = where / "deploy" / "maintenance"
        (maintenance / "cron.d").mkdir(parents=True)
        (maintenance / SYNC_SCRIPT.name).write_text(sync_body, encoding="utf-8")
        shutil.copy2(
            REPO_ROOT
            / "deploy"
            / "maintenance"
            / "cron.d"
            / "omninode-host-maintenance-sync",
            maintenance / "cron.d" / "omninode-host-maintenance-sync",
        )
        return where

    shipped_at_the_time = _gate(
        _repo(captured.read_text(encoding="utf-8"), tmp_path / "before")
    )
    assert shipped_at_the_time.returncode == 1, (
        "the gate accepted the exact file that fetched as root into an "
        "operator-owned clone every hour -- a scope regression back to the "
        f"scripts/reconcile*.sh glob: {shipped_at_the_time.stdout}"
        f"{shipped_at_the_time.stderr}"
    )
    assert SYNC_SCRIPT.name in shipped_at_the_time.stderr
    assert "as_owner" in shipped_at_the_time.stderr

    fixed = _gate(
        _repo(SYNC_SCRIPT.read_text(encoding="utf-8"), tmp_path / "after"),
    )
    assert fixed.returncode == 0, (
        "the FIXED file fails the same gate in the same repo shape, so the "
        f"rejection above proves nothing: {fixed.stdout}{fixed.stderr}"
    )


def test_the_real_cron_units_are_all_covered_by_discovery() -> None:
    """Every scheduled command this repo ships must be a file the gate opens.

    Asserted against the real tree rather than a fixture, because the property
    that matters is about THIS repo's cron units -- a new one added without a
    manifest entry is the OMN-15525 condition, and the point of AC3 is that it
    can no longer be introduced quietly.
    """
    import sys

    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    from check_reconciler_privilege import (
        discover_scheduled_host_scripts,
    )

    scripts, failures = discover_scheduled_host_scripts(REPO_ROOT)
    assert not failures, failures
    names = {path.name for path in scripts}
    assert SYNC_SCRIPT.name in names, (
        f"the :37 sync is not reached by discovery; scanned: {sorted(names)}"
    )
    assert "omninode-workspace-reconcile.sh" in names, sorted(names)
    assert "omninode-system-slack-report.sh" in names, sorted(names)
