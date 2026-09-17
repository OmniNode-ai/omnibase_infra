# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-18567 -- the deploy runner's private clone tree must be converged on a schedule.

WHAT IS UNDER TEST
    ``deploy/maintenance/omninode-runner-tree-converge.sh`` and the cron unit
    that invokes it. These tests drive the artifact that actually runs on the
    host, not a re-implementation (memory ``feedback_test_the_artifact_that_runs``).

WHY IT EXISTS
    ``/data/omninode/runner_omni_home`` is the deploy runner's private
    ``OMNI_HOME``: the build source the dev-lane refresh, the stability-lane
    refresh and the release-train tag cut all read. Nothing converged it. On
    2026-09-17 five of its six clones were on detached HEADs and every one was
    behind ``origin/dev`` -- ``omnibase_infra`` by 294 commits,
    ``onex_change_control`` by 1120. The one-time repair is done; this tick is
    the recurrence, which is what OMN-18567 is actually for.

    Two mechanisms already run on that host and neither reaches this tree: the
    host-maintenance sync converges host-resident FILES and touches no clone,
    and the workspace reconciler is scoped to the shared operator-owned tree.
    Per CLAUDE.md rule 5, a repair nobody schedules is advisory.

THE THREE ACCEPTANCE CRITERIA, AND WHERE EACH IS PINNED
    AC1 (it converges, on a schedule, as the owning uid)
        ``test_idle_tick_converges_every_clone``,
        ``test_second_tick_on_a_converged_tree_reports_in_sync``,
        ``test_cron_unit_schedules_the_converge_verb`` and the manifest tests.
    AC2 (never concurrent with a deploy job on the tree)
        the five refusal tests. Each one asserts the clone did NOT move, not
        merely that the word REFUSED was printed -- a guard that prints a
        refusal and converges anyway is the failure this class exists to end.
    AC3 (no file changes owner)
        ``test_every_tree_touching_invocation_runs_as_the_owner``. This is the
        static half; the dynamic half is a live readback on ``.201``, recorded
        on the ticket, because a hermetic test runs every command as one user
        and so cannot observe an ownership change at all.

HERMETICITY
    Every test builds a throwaway tree of real git clones under ``tmp_path``
    and drives the script against it. Nothing reads or writes
    ``/data/omninode``, ``/data/maintenance`` or ``/etc/cron.d``. The two
    host-only probes -- the runner's worker process list and the procfs scan
    for a process sitting in the tree -- are reached through declared command
    seams, so the refusal LOGIC is tested here while the default probe strings
    are pinned by ``test_default_probes_are_host_local_and_named`` and proven
    by the live readback on the ticket.
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

import pytest

from omnibase_core.validators.no_unguarded_git_subprocess import (
    scrub_git_location_env,
)

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]
MAINTENANCE = REPO_ROOT / "deploy" / "maintenance"
TICK = MAINTENANCE / "omninode-runner-tree-converge.sh"
CRON_UNIT = MAINTENANCE / "cron.d" / "omninode-runner-tree-converge"
SYNC_SCRIPT = MAINTENANCE / "omninode-host-maintenance-sync.sh"
REPORTER = MAINTENANCE / "omninode-system-slack-report.sh"

# The converge script the tick delegates to lives in omniclaude, which is not
# checked out beside this repo in CI. Every test supplies a stand-in whose
# behaviour is the part this tick depends on: re-attach and hard-reset one
# clone onto its upstream. What the real script does BEYOND that (preserving
# patches, writing evidence, appending a ledger row) is omniclaude's to test
# and is already covered by its ten hermetic cases.
FAKE_CONVERGE = """#!/usr/bin/env bash
set -euo pipefail
repo=""
execute=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --execute) execute=1 ;;
    --ticket|--lane|--to-branch) shift ;;
    --*) ;;
    *) repo="$1" ;;
  esac
  shift
done
clone="${OMNI_HOME:?}/${repo}"
branch="$(git -C "$clone" symbolic-ref -q --short HEAD || echo dev)"
git -C "$clone" fetch --quiet origin
target="$(git -C "$clone" rev-parse "origin/${branch}")"
if (( execute )); then
  git -C "$clone" checkout --force --quiet "$branch"
  git -C "$clone" reset --hard --quiet "$target"
fi
echo "converge stub: $repo -> ${target:0:12}"
"""

VERDICT_RE = re.compile(
    r"^runner-tree-converge\|(?P<verdict>[A-Z_]+)\|ts=(?P<ts>\S+?)\|"
    r"tree=(?P<tree>\S+?)\|clones=(?P<clones>\d+)\|in_sync=(?P<in_sync>\d+)\|"
    r"converged=(?P<converged>\d+)\|failed=(?P<failed>\d+)\|skipped=(?P<skipped>\d+)\|"
    r"reason=(?P<reason>[^|]*)\|last_success=(?P<last_success>\S+?)\|"
    r"detail=(?P<detail>.*)$"
)


# --------------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------------- #
def _git(cwd: Path, *args: str) -> str:
    env = scrub_git_location_env(os.environ)
    env.update(
        {
            "GIT_CONFIG_GLOBAL": "/dev/null",
            "GIT_CONFIG_SYSTEM": "/dev/null",
            "GIT_AUTHOR_NAME": "t",
            "GIT_AUTHOR_EMAIL": "t@t",
            "GIT_COMMITTER_NAME": "t",
            "GIT_COMMITTER_EMAIL": "t@t",
        }
    )
    # OMN-18434: the scrub is repeated in the `env=` expression, not only where
    # `env` was built. Git exports GIT_DIR / GIT_WORK_TREE / GIT_INDEX_FILE into
    # every hook environment and those OVERRIDE `cwd=`, so under a pre-push hook
    # an unscrubbed fixture mutates the REAL invoking worktree rather than
    # tmp_path. Scrubbing at the call site is what makes the guarantee legible
    # to the static guard instead of resting on a helper several frames up.
    return subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
        env=scrub_git_location_env(env),
    ).stdout.strip()


def _make_clone(tree: Path, origins: Path, name: str) -> Path:
    """One clone under `tree`, tracking `dev` on a bare origin, one commit behind."""
    origin = origins / f"{name}.git"
    origin.mkdir(parents=True)
    _git(origin, "init", "--bare", "--initial-branch=dev", ".")

    seed = origins / f"{name}-seed"
    seed.mkdir()
    _git(seed, "init", "--initial-branch=dev", ".")
    (seed / "a.txt").write_text("one\n", encoding="utf-8")
    _git(seed, "add", "a.txt")
    _git(seed, "commit", "-m", "one")
    _git(seed, "remote", "add", "origin", str(origin))
    _git(seed, "push", "-u", "origin", "dev")

    clone = tree / name
    _git(tree, "clone", "--quiet", str(origin), name)
    _git(clone, "remote", "set-head", "origin", "-a")

    # Advance the origin so every clone starts BEHIND, which is the live state
    # this tick exists to repair.
    (seed / "a.txt").write_text("two\n", encoding="utf-8")
    _git(seed, "commit", "-am", "two")
    _git(seed, "push", "origin", "dev")
    return clone


@pytest.fixture
def tree(tmp_path: Path) -> Path:
    """A stand-in for the runner's private OMNI_HOME: two clones, both behind."""
    tree_dir = tmp_path / "runner_omni_home"
    tree_dir.mkdir()
    origins = tmp_path / "origins"
    origins.mkdir()
    _make_clone(tree_dir, origins, "omnibase_infra")
    detached = _make_clone(tree_dir, origins, "onex_change_control")
    # The live state: detached at its own current commit.
    _git(detached, "checkout", "--quiet", "--detach", "HEAD")
    return tree_dir


@pytest.fixture
def converge_stub(tmp_path: Path) -> Path:
    stub = tmp_path / "converge-canonical-clone.sh"
    stub.write_text(FAKE_CONVERGE, encoding="utf-8")
    stub.chmod(0o755)
    return stub


def _run(
    tree: Path,
    converge_stub: Path,
    tmp_path: Path,
    *args: str,
    worker_probe: str = "echo 0",
    proc_scan: str = "true",
) -> subprocess.CompletedProcess[str]:
    state = tmp_path / "state"
    state.mkdir(exist_ok=True)
    env = scrub_git_location_env(os.environ)
    env.update(
        {
            "GIT_CONFIG_GLOBAL": "/dev/null",
            "GIT_CONFIG_SYSTEM": "/dev/null",
            "OMNINODE_RUNNER_TREE": str(tree),
            "OMNINODE_RUNNER_TREE_CONVERGE_SCRIPT": str(converge_stub),
            "OMNINODE_RUNNER_TREE_STATE_DIR": str(state),
            # The privilege library is read from this repo, which is where it
            # ships; the host default points at the `.201` deploy clone.
            "OMNINODE_INFRA_REPO_ROOT": str(REPO_ROOT),
            "OMNINODE_RUNNER_TREE_WORKER_PROBE_CMD": worker_probe,
            "OMNINODE_RUNNER_TREE_PROC_SCAN_CMD": proc_scan,
            "OMNINODE_RUNNER_TREE_SKIP_SCRIPT_REFRESH": "1",
        }
    )
    return subprocess.run(
        ["bash", str(TICK), *args],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )


def _verdict(result: subprocess.CompletedProcess[str]) -> re.Match[str]:
    lines = [
        line
        for line in result.stdout.splitlines()
        if line.startswith("runner-tree-converge|")
    ]
    assert lines, (
        "the tick printed no verdict line. 'it failed' and 'nobody ran it' have "
        f"to stay distinguishable from the output alone.\nstdout:\n{result.stdout}"
        f"\nstderr:\n{result.stderr}"
    )
    matched = VERDICT_RE.match(lines[-1])
    assert matched, f"verdict line does not match the declared grammar: {lines[-1]}"
    return matched


def _behind(clone: Path) -> int:
    # Fetch first: the clone's remote-tracking ref is stale until something
    # advances it, and reading a stale ref is the false-green this tick exists
    # to remove -- a test that made the same mistake would prove nothing.
    _git(clone, "fetch", "--quiet", "origin")
    return int(_git(clone, "rev-list", "--count", "HEAD..origin/dev"))


def _attached(clone: Path) -> bool:
    proc = subprocess.run(
        ["git", "-C", str(clone), "symbolic-ref", "--quiet", "HEAD"],
        capture_output=True,
        text=True,
        env=scrub_git_location_env(os.environ),
        check=False,
    )
    return proc.returncode == 0 and proc.stdout.startswith("refs/")


# --------------------------------------------------------------------------- #
# AC2 -- the busy gate. Every case asserts the tree did NOT move.
# --------------------------------------------------------------------------- #
def test_a_running_job_worker_refuses_and_moves_nothing(
    tree: Path, converge_stub: Path, tmp_path: Path
) -> None:
    """A live job in the container that mounts the tree is a hard refusal.

    The deploy scripts read this tree as their build source. A `reset --hard`
    underneath a running job is the collision AC2 names.
    """
    before = {c.name: _git(c, "rev-parse", "HEAD") for c in sorted(tree.iterdir())}

    result = _run(tree, converge_stub, tmp_path, "--converge", worker_probe="echo 2")

    matched = _verdict(result)
    assert matched.group("verdict") == "REFUSED"
    assert "busy" in matched.group("reason")
    assert result.returncode == 0, (
        "a refusal is the guard WORKING, not a fault. Reddening cron on every "
        "busy tick is how the channel stops being read."
    )
    for clone in sorted(tree.iterdir()):
        assert _git(clone, "rev-parse", "HEAD") == before[clone.name]


def test_an_unreadable_worker_probe_fails_closed(
    tree: Path, converge_stub: Path, tmp_path: Path
) -> None:
    """Probe error is UNKNOWN, and UNKNOWN refuses."""
    before = {c.name: _git(c, "rev-parse", "HEAD") for c in sorted(tree.iterdir())}

    result = _run(tree, converge_stub, tmp_path, "--converge", worker_probe="exit 7")

    matched = _verdict(result)
    assert matched.group("verdict") == "REFUSED"
    assert "unknown" in matched.group("reason")
    for clone in sorted(tree.iterdir()):
        assert _git(clone, "rev-parse", "HEAD") == before[clone.name]


def test_an_empty_worker_probe_answer_is_unknown_not_idle(
    tree: Path, converge_stub: Path, tmp_path: Path
) -> None:
    """The exact fail-open shape `deploy-runners.sh:1141` documents.

    `docker top ... | grep -c X || true` exits 0 with EMPTY output when the
    container is gone or the daemon errors. Reading that as "zero workers" is
    how a fail-open busy check killed a live job once already, so an empty
    answer must be UNKNOWN and must refuse -- not be coerced to 0.
    """
    before = {c.name: _git(c, "rev-parse", "HEAD") for c in sorted(tree.iterdir())}

    result = _run(tree, converge_stub, tmp_path, "--converge", worker_probe="true")

    matched = _verdict(result)
    assert matched.group("verdict") == "REFUSED"
    assert "unknown" in matched.group("reason")
    for clone in sorted(tree.iterdir()):
        assert _git(clone, "rev-parse", "HEAD") == before[clone.name]


def test_a_process_sitting_in_the_tree_refuses(
    tree: Path, converge_stub: Path, tmp_path: Path
) -> None:
    """The out-of-band writer the container probe cannot see.

    Anyone running a lane refresh or the tag cut by hand on `.201` with
    OMNI_HOME pointed at this tree is invisible to the runner's process list.
    A process whose cwd or open fd resolves inside the tree is the only signal
    that sees them.
    """
    before = {c.name: _git(c, "rev-parse", "HEAD") for c in sorted(tree.iterdir())}

    result = _run(
        tree,
        converge_stub,
        tmp_path,
        "--converge",
        proc_scan=f"echo '/proc/4242/cwd' '{tree}/omnibase_infra'",
    )

    matched = _verdict(result)
    assert matched.group("verdict") == "REFUSED"
    assert "in_tree_process" in matched.group("reason")
    for clone in sorted(tree.iterdir()):
        assert _git(clone, "rev-parse", "HEAD") == before[clone.name]


def test_an_unreadable_proc_scan_fails_closed(
    tree: Path, converge_stub: Path, tmp_path: Path
) -> None:
    result = _run(tree, converge_stub, tmp_path, "--converge", proc_scan="exit 3")
    matched = _verdict(result)
    assert matched.group("verdict") == "REFUSED"
    assert "unknown" in matched.group("reason")


def test_an_in_flight_git_operation_refuses(
    tree: Path, converge_stub: Path, tmp_path: Path
) -> None:
    """An orphaned child of a dead job still holds the object store.

    `Runner.Worker` can be gone while a git process it spawned is mid-checkout.
    The marker set is the one `scripts/git-gc-auto.sh:76` already uses to decide
    "do not touch this object store right now".
    """
    (tree / "omnibase_infra" / ".git" / "index.lock").write_text("", encoding="utf-8")
    before = _git(tree / "omnibase_infra", "rev-parse", "HEAD")

    result = _run(tree, converge_stub, tmp_path, "--converge")

    matched = _verdict(result)
    assert matched.group("verdict") == "REFUSED"
    assert "git_operation_in_flight" in matched.group("reason")
    assert _git(tree / "omnibase_infra", "rev-parse", "HEAD") == before


# --------------------------------------------------------------------------- #
# AC1 -- it actually converges
# --------------------------------------------------------------------------- #
def test_idle_tick_converges_every_clone(
    tree: Path, converge_stub: Path, tmp_path: Path
) -> None:
    for clone in sorted(tree.iterdir()):
        assert _behind(clone) > 0

    result = _run(tree, converge_stub, tmp_path, "--converge")

    matched = _verdict(result)
    assert matched.group("verdict") == "CONVERGED", result.stdout + result.stderr
    assert matched.group("converged") == "2"
    assert matched.group("failed") == "0"
    assert result.returncode == 0
    for clone in sorted(tree.iterdir()):
        assert _behind(clone) == 0, f"{clone.name} is still behind its upstream"
        assert _attached(clone), f"{clone.name} is still detached"


def test_the_verdict_names_every_clone_with_before_and_after_shas(
    tree: Path, converge_stub: Path, tmp_path: Path
) -> None:
    """A receipt that does not name what moved cannot be audited after the fact."""
    before = {c.name: _git(c, "rev-parse", "HEAD") for c in sorted(tree.iterdir())}

    result = _run(tree, converge_stub, tmp_path, "--converge")
    detail = _verdict(result).group("detail")

    entries = dict(part.split(":", 1) for part in detail.split(";") if ":" in part)
    assert set(entries) == {"omnibase_infra", "onex_change_control"}
    for name, moved in entries.items():
        head_before, _, head_after = moved.partition("->")
        assert head_before == before[name][:12], f"{name}: wrong before-sha"
        after = _git(tree / name, "rev-parse", "HEAD")
        assert head_after == after[:12], f"{name}: wrong after-sha"
        assert head_before != head_after


def test_second_tick_on_a_converged_tree_reports_in_sync(
    tree: Path, converge_stub: Path, tmp_path: Path
) -> None:
    """An in-sync tick writes nothing and says so.

    "nothing drifted" and "nothing ran" must be distinguishable from the output
    alone -- the same property `--converge` gives the host maintenance sync.
    """
    first = _run(tree, converge_stub, tmp_path, "--converge")
    assert _verdict(first).group("verdict") == "CONVERGED"

    second = _run(tree, converge_stub, tmp_path, "--converge")

    matched = _verdict(second)
    assert matched.group("verdict") == "IN_SYNC"
    assert matched.group("in_sync") == "2"
    assert matched.group("converged") == "0"
    assert second.returncode == 0


def test_check_mode_reports_the_drift_and_moves_nothing(
    tree: Path, converge_stub: Path, tmp_path: Path
) -> None:
    before = {c.name: _git(c, "rev-parse", "HEAD") for c in sorted(tree.iterdir())}

    result = _run(tree, converge_stub, tmp_path, "--check")

    matched = _verdict(result)
    assert matched.group("verdict") == "DRIFTED"
    for clone in sorted(tree.iterdir()):
        assert _git(clone, "rev-parse", "HEAD") == before[clone.name]


def test_a_clone_the_converge_script_refuses_is_a_failure_not_a_silence(
    tree: Path, converge_stub: Path, tmp_path: Path
) -> None:
    """A converge that could not repair must redden, and must name the clone."""
    refusing = tmp_path / "refusing-converge.sh"
    refusing.write_text(
        "#!/usr/bin/env bash\necho 'REFUSED: no upstream' >&2\nexit 2\n",
        encoding="utf-8",
    )
    refusing.chmod(0o755)

    result = _run(tree, refusing, tmp_path, "--converge")

    matched = _verdict(result)
    assert matched.group("verdict") == "FAILED"
    assert matched.group("failed") == "2"
    assert result.returncode == 1, "a converge that could not repair must redden"
    assert "omnibase_infra" in matched.group("detail")


def test_clone_set_is_discovered_from_the_tree_not_hardcoded(
    tree: Path, converge_stub: Path, tmp_path: Path
) -> None:
    """A seventh clone must be covered the moment it appears.

    A hand-maintained repo list is the OMN-15137 defect: `omnibase_spi` was
    added to one sibling list and forgotten in the other, and the gap surfaced
    three deploy hops later. The tree itself is the only list that cannot drift
    from the tree.
    """
    origins = tmp_path / "origins"
    _make_clone(tree, origins, "omnimarket")
    (tree / "not_a_clone").mkdir()
    (tree / ".onex_state").mkdir()

    result = _run(tree, converge_stub, tmp_path, "--converge")

    matched = _verdict(result)
    assert matched.group("clones") == "3"
    assert "omnimarket" in matched.group("detail")
    assert "not_a_clone" not in matched.group("detail")


# --------------------------------------------------------------------------- #
# AC3 -- ownership, and the privilege rule
# --------------------------------------------------------------------------- #
def test_every_tree_touching_invocation_runs_as_the_owner() -> None:
    """The tree is owned by the runner uid; the cron runs as root.

    A root `reset --hard` rewrites working-tree files as root and leaves a tree
    the runner cannot write -- converting a stale clone into a broken one, which
    is strictly worse than the drift. `scripts/check_reconciler_privilege.py`
    already governs bare `git` writes in this file; what it cannot see is the
    delegation to the converge script, which is where every reset actually
    happens. That line is pinned here.
    """
    source = TICK.read_text(encoding="utf-8")
    assert "reconcile_privilege_lib.sh" in source, (
        "the tick must source the ONE privilege library, never re-implement it"
    )
    assert not re.search(r"^\s*as_owner\s*\(\s*\)\s*\{", source, re.MULTILINE), (
        "two copies of a privilege rule drift, and the half that drifts is the "
        "half nobody is watching (OMN-17366)"
    )

    invocations = [
        line
        for line in source.splitlines()
        if "CONVERGE_SCRIPT" in line
        and re.search(r"(?:bash|sh|exec|env)\s", line)
        and not line.strip().startswith("#")
        and not re.match(r"^\s*(?:say|log|echo|printf|record|fail|die)\b", line.strip())
    ]
    assert invocations, "no invocation of the converge script was found to check"
    for line in invocations:
        assert "as_owner" in line, (
            f"the converge script is executed without as_owner: {line.strip()}"
        )


def test_default_probes_are_host_local_and_named() -> None:
    """The seams the tests drive must default to real host-local probes.

    Each refusal test above overrides a probe, so the DEFAULT strings are the
    one thing those tests cannot reach. Pinning them here is what stops the
    seam from quietly becoming the implementation.
    """
    source = TICK.read_text(encoding="utf-8")
    assert "docker top" in source and "Runner.Worker" in source, (
        "the idle proof is the absence of a job worker in the one container "
        "that mounts the tree"
    )
    assert "omninode-deploy-runner" in source
    assert "-lname" in source, "the procfs scan must match symlink targets"
    assert "gh api" not in source, (
        "the GitHub busy flag is deliberately NOT consulted: root on `.201` "
        "holds no credential with that scope (GH_PAT returns 403 on the org "
        "runners endpoint, verified 2026-09-17), so requiring it would make "
        "every tick refuse forever and AC1 unreachable"
    )


# --------------------------------------------------------------------------- #
# the install path -- an artifact nothing installs is the OMN-15525 condition
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("repo_path", "host_path"),
    [
        (
            "deploy/maintenance/omninode-runner-tree-converge.sh",
            "/data/maintenance/bin/omninode-runner-tree-converge.sh",
        ),
        (
            "deploy/maintenance/cron.d/omninode-runner-tree-converge",
            "/etc/cron.d/omninode-runner-tree-converge",
        ),
    ],
)
def test_both_host_artifacts_are_in_the_maintenance_manifest(
    repo_path: str, host_path: str
) -> None:
    """The manifest IS the install path -- no crontab is ever edited by hand.

    The hourly `:37` `--converge` writes every manifest entry that differs from
    `origin/dev` and reads it back. Adding these two rows is therefore the whole
    installation: merge to `dev`, and the next tick installs them.
    """
    manifest = SYNC_SCRIPT.read_text(encoding="utf-8")
    assert f'"{repo_path}|{host_path}|' in manifest, (
        f"{repo_path} is absent from the host-artifact manifest, so nothing "
        "installs it and nothing alarms when it drifts (OMN-15525)"
    )


def test_cron_unit_schedules_the_converge_verb() -> None:
    """`--check` on a timer is a detector wired to no repair (OMN-17898)."""
    unit = CRON_UNIT.read_text(encoding="utf-8")
    commands = [
        line
        for line in unit.splitlines()
        if line.strip() and not line.startswith("#") and "=" not in line.split()[0]
    ]
    assert len(commands) == 1, f"expected exactly one scheduled command, got {commands}"
    assert "--converge" in commands[0]
    assert "--check" not in commands[0]
    assert "/data/maintenance/bin/omninode-runner-tree-converge.sh" in commands[0]


def test_cron_minute_collides_with_no_other_root_job() -> None:
    """Four root jobs on one host must not contend on the same minute.

    :00/:15/:30/:45 is the system report, :19 the workspace reconcile, :37 the
    maintenance sync. This assertion is here rather than in a comment because a
    comment does not survive the next edit.
    """
    taken = {0, 15, 30, 45, 19, 37}
    line = next(
        line
        for line in CRON_UNIT.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#") and "=" not in line.split()[0]
    )
    minute_field = line.split()[0]
    assert not minute_field.startswith("*"), (
        "a wildcard or stepped minute overlaps every other root job on this host"
    )
    minutes = {int(part) for part in minute_field.split(",")}
    assert minutes, "no minute could be read from the cron line"
    assert not minutes & taken, (
        f"minute(s) {sorted(minutes & taken)} already carry a root job on `.201`"
    )


# --------------------------------------------------------------------------- #
# the verdict has to reach a human
# --------------------------------------------------------------------------- #
def test_the_slack_reporter_reads_the_verdict_file() -> None:
    """A verdict nothing reads is a log line (CLAUDE.md rule 5).

    Folding this into the existing reporter rather than building a second
    alerter is the net-negative-surface rule: it inherits that script's Slack
    poster, its state-change de-duplication and its */15 cron. No new cron unit,
    no second Slack integration.
    """
    reporter = REPORTER.read_text(encoding="utf-8")
    assert "check_runner_tree_converge" in reporter, (
        "the reporter does not collect the runner-tree verdict, so a tick that "
        "stops running, or one that FAILS, reaches nobody"
    )
    assert "runner-tree-converge.status" in reporter
    assert re.search(r"check_runner_tree_converge\b", reporter.split("collect()")[1]), (
        "the collector is defined but never called from collect()"
    )


def test_the_tick_writes_a_status_file_the_reporter_can_read(
    tree: Path, converge_stub: Path, tmp_path: Path
) -> None:
    _run(tree, converge_stub, tmp_path, "--converge")
    status = tmp_path / "state" / "runner-tree-converge.status"
    assert status.is_file(), "no status file was written for the reporter to read"
    matched = VERDICT_RE.match(status.read_text(encoding="utf-8").strip())
    assert matched, "the status file does not carry a parseable verdict line"
    assert matched.group("verdict") == "CONVERGED"


def test_last_success_advances_only_on_a_completed_clean_run(
    tree: Path, converge_stub: Path, tmp_path: Path
) -> None:
    """Chronic refusal must be distinguishable from chronic success.

    The tick exits 0 on a refusal, deliberately. That is only safe because
    `last_success` does NOT advance on one -- it is the field the reporter
    ages to decide that a tree has gone unconverged for too long.
    """
    good = _run(tree, converge_stub, tmp_path, "--converge")
    success_ts = _verdict(good).group("last_success")
    assert success_ts != "never"

    refused = _run(tree, converge_stub, tmp_path, "--converge", worker_probe="echo 1")
    matched = _verdict(refused)
    assert matched.group("verdict") == "REFUSED"
    assert matched.group("last_success") == success_ts, (
        "a refusal advanced last_success, so a permanently busy runner would "
        "look exactly like a permanently converged tree"
    )
