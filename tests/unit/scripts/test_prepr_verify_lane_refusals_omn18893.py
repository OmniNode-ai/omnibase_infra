# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Task 4 of epic OMN-18888: the entrypoint's refusals, run rather than read.

``test_prepr_verify_lane_entrypoint_omn18893.py`` pins the policy table, the
overlay and the exit-code literals. It proves three of the plan's Task 4
falsifiers only by reading text or set membership:

* the attribution preflight refuses a pool-lane build with no stated reason
  (it asserts the pool lanes are IN ``REASON_REQUIRED_LANES``, not that a build
  is refused);
* a deliberately dirtied worktree is refused rather than built (it asserts the
  exit-code name appears twice in the script);
* a second concurrent build blocks on the pool build lock rather than starting
  (nothing runs two invocations).

Each test below runs the real ``prepr_verify_lane.sh`` (``--plan-only`` stops
after every refusal and before any connection, build or start) or the real
``pool_build_lock_acquire`` function lifted out of it, and asserts the named
exit code. Every refusal has a positive control: the same invocation with the
defect removed passes, so a refusal cannot be a script that always fails.

The plan's live-slot falsifiers (the running image digest equals the receipt's,
and the host's canonical clones are unchanged across a real build) need a lab
host and are proven there, not here.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import pytest

from omnibase_core.validators.no_unguarded_git_subprocess import scrub_git_location_env

REPO_ROOT = Path(__file__).resolve().parents[3]
RUNTIME_BUILD = REPO_ROOT / "scripts" / "runtime_build"
ENTRYPOINT = RUNTIME_BUILD / "prepr_verify_lane.sh"
LANE_LOCK_PY = RUNTIME_BUILD / "lane_lock.py"
SIBLING_MANIFEST = RUNTIME_BUILD / "sibling_clone_manifest.sh"
POLICY_PATH = RUNTIME_BUILD / "prepr_slot_policy.py"

pytestmark = pytest.mark.unit

_REASON = "OMN-18893 task 4 refusal proof"


def _policy_constants() -> dict[str, int]:
    """Read the exit codes from the module the shell mirrors, not from a copy."""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import json,sys;sys.path.insert(0,sys.argv[1]);import prepr_slot_policy as p;"
            "print(json.dumps({n:getattr(p,n) for n in dir(p) if n.startswith('EXIT_')}))",
            str(RUNTIME_BUILD),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return {k: int(v) for k, v in json.loads(result.stdout).items()}


EXIT = _policy_constants()
POOL_BUILD_LOCK_PROJECT = subprocess.run(
    [
        sys.executable,
        "-c",
        "import sys;sys.path.insert(0,sys.argv[1]);"
        "from prepr_slot_policy import POOL_BUILD_LOCK_PROJECT as p;print(p)",
        str(RUNTIME_BUILD),
    ],
    capture_output=True,
    text=True,
    check=True,
).stdout.strip()


def _vendored_siblings() -> list[str]:
    result = subprocess.run(
        [
            "bash",
            "-c",
            f'source "{SIBLING_MANIFEST}"; printf "%s\\n" "${{SIBLING_VENDORED_REPOS[@]}}"',
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    repos = [line for line in result.stdout.splitlines() if line]
    assert repos, (
        "the sibling manifest names no vendored repo; the fixtures would be vacuous"
    )
    return repos


def _git(cwd: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-c", "user.email=t@example.invalid", "-c", "user.name=t", *args],
        cwd=cwd,
        env=scrub_git_location_env(os.environ),
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def _make_repo(path: Path, message: str) -> str:
    path.mkdir(parents=True)
    _git(path, "init", "-q")
    _git(path, "commit", "-q", "--allow-empty", "-m", message)
    return _git(path, "rev-parse", "HEAD")


class _Fixture:
    """A clean target worktree and a fake OMNI_HOME of clean sibling clones."""

    def __init__(self, tmp_path: Path) -> None:
        self.home = tmp_path / "home"
        self.home.mkdir()
        self.omni_home = tmp_path / "omni_home"
        self.lock_dir = tmp_path / "locks"
        # The worktree sits alone in its directory so no sibling worktree
        # "beside the target" shadows the fake canonical clones.
        self.worktree = tmp_path / "ticket" / "omnibase_infra"
        self.staging_root = tmp_path / "stage"
        self.target_commit = _make_repo(self.worktree, "target commit")
        self.sibling_commits = {
            repo: _make_repo(self.omni_home / repo, f"sibling {repo} commit")
            for repo in _vendored_siblings()
        }

    def run(
        self, *, reason: str | None = _REASON, slot: str = "1"
    ) -> subprocess.CompletedProcess[str]:
        env = {
            "PATH": os.environ["PATH"],
            # The attribution preflight writes its record under ~/.omnibase;
            # point it at the fixture so a test never touches the real one.
            "HOME": str(self.home),
            "OMNI_HOME": str(self.omni_home),
            "ONEX_LANE_LOCK_DIR": str(self.lock_dir),
        }
        args = [
            "bash",
            str(ENTRYPOINT),
            "--slot",
            slot,
            "--worktree",
            str(self.worktree),
            "--staging-root",
            str(self.staging_root),
            "--plan-only",
        ]
        if reason is not None:
            args += ["--reason", reason]
        return subprocess.run(
            args, env=env, capture_output=True, text=True, timeout=120, check=False
        )

    def attribution_records(self) -> list[Path]:
        return sorted(
            (self.home / ".omnibase" / "infra" / "deploy-attribution").glob("*.json")
        )


@pytest.fixture
def fixture(tmp_path: Path) -> _Fixture:
    return _Fixture(tmp_path)


def _plan(stdout: str) -> dict[str, object]:
    # stdout also carries the attribution preflight's one-line record; the plan
    # is the multi-line object that opens on a line of its own.
    lines = stdout.splitlines()
    plan = json.loads("\n".join(lines[lines.index("{") :]))
    assert isinstance(plan, dict)
    return plan


# ---------------------------------------------------------------------------
# Positive control, then the refusals it makes meaningful
# ---------------------------------------------------------------------------


def test_positive_control_a_clean_target_with_a_real_reason_passes_every_refusal(
    fixture: _Fixture,
) -> None:
    result = fixture.run()
    assert result.returncode == EXIT["EXIT_OK"], result.stderr
    plan = _plan(result.stdout)
    assert plan["plan_only"] is True
    assert plan["compose_project"] == "omnibase-infra-prepr-1"
    assert plan["target_commit"] == fixture.target_commit
    records = fixture.attribution_records()
    assert records, (
        "the attribution preflight never ran, so the zero below proves nothing"
    )
    assert json.loads(records[-1].read_text(encoding="utf-8"))["result"] != "REFUSE"


@pytest.mark.parametrize(
    "dirty",
    ["untracked", "modified"],
)
def test_a_dirty_target_worktree_is_refused_with_the_named_exit_code(
    fixture: _Fixture, dirty: str
) -> None:
    if dirty == "untracked":
        (fixture.worktree / "scratch.txt").write_text("uncommitted\n", encoding="utf-8")
    else:
        (fixture.worktree / "tracked.txt").write_text("v1\n", encoding="utf-8")
        _git(fixture.worktree, "add", "tracked.txt")
        _git(fixture.worktree, "commit", "-q", "-m", "add tracked")
        (fixture.worktree / "tracked.txt").write_text("v2\n", encoding="utf-8")
    result = fixture.run()
    assert result.returncode == EXIT["EXIT_REFUSED_DIRTY_WORKTREE"], result.stderr
    assert "DIRTY" in result.stderr
    assert not fixture.staging_root.exists(), "a refused build must stage nothing"
    # Control: the same tree, cleaned, is accepted by the same invocation.
    (fixture.worktree / "scratch.txt").unlink(missing_ok=True)
    if dirty == "modified":
        _git(fixture.worktree, "checkout", "--", "tracked.txt")
    assert fixture.run().returncode == EXIT["EXIT_OK"]


def test_a_dirty_sibling_is_refused_and_named(fixture: _Fixture) -> None:
    sibling = sorted(fixture.sibling_commits)[0]
    (fixture.omni_home / sibling / "scratch.txt").write_text(
        "uncommitted\n", encoding="utf-8"
    )
    result = fixture.run()
    assert result.returncode == EXIT["EXIT_REFUSED_DIRTY_WORKTREE"], result.stderr
    assert f"sibling {sibling}" in result.stderr
    assert not fixture.staging_root.exists()


def test_a_build_with_no_stated_reason_is_refused_before_the_preflight_runs(
    fixture: _Fixture,
) -> None:
    result = fixture.run(reason=None)
    assert result.returncode == EXIT["EXIT_REFUSED_ATTRIBUTION"], result.stderr
    assert "no reason given" in result.stderr
    assert not fixture.attribution_records(), (
        "the shell refused on its own empty check; the preflight should not have run"
    )


def test_a_placeholder_reason_is_refused_by_the_attribution_preflight(
    fixture: _Fixture,
) -> None:
    result = fixture.run(reason="unit")
    assert result.returncode == EXIT["EXIT_REFUSED_ATTRIBUTION"], result.stderr
    records = fixture.attribution_records()
    assert records, "the refusal did not come from the preflight, which writes a record"
    record = json.loads(records[-1].read_text(encoding="utf-8"))
    assert record["result"] == "REFUSE"
    assert record["compose_project"] == "omnibase-infra-prepr-1"


def test_target_and_sibling_commits_are_recorded_separately_not_collapsed(
    fixture: _Fixture,
) -> None:
    """Provenance negative control: a target whose commit differs from every
    sibling clone's head is reported as itself, and each sibling as its own."""
    result = fixture.run()
    assert result.returncode == EXIT["EXIT_OK"], result.stderr
    distinct = {fixture.target_commit, *fixture.sibling_commits.values()}
    assert len(distinct) == 1 + len(fixture.sibling_commits), (
        "the fixture commits collapsed to a single sha, so the control is vacuous"
    )
    assert _plan(result.stdout)["target_commit"] == fixture.target_commit
    for repo, commit in fixture.sibling_commits.items():
        assert re.search(
            rf"sibling {re.escape(repo)} <- .* @ {commit} ", result.stderr
        ), f"{repo} is not logged at its own head {commit}:\n{result.stderr}"


# ---------------------------------------------------------------------------
# The pool build lock: two overlapping invocations, with timestamps
# ---------------------------------------------------------------------------


def _pool_build_lock_acquire_source() -> str:
    """The shipped function, lifted verbatim from the entrypoint."""
    lines = ENTRYPOINT.read_text(encoding="utf-8").splitlines()
    start = next(
        i for i, ln in enumerate(lines) if ln.startswith("pool_build_lock_acquire() {")
    )
    end = next(i for i in range(start, len(lines)) if lines[i] == "}")
    return "\n".join(lines[start : end + 1])


def _lock_harness(hold: bool) -> str:
    tail = (
        'echo "ACQUIRED $(date +%s.%N)"; exec sleep 600' if hold else 'echo "ACQUIRED"'
    )
    return (
        "set -u\n"
        "POOL_LOCK_OWNED=0\n"
        f"{_pool_build_lock_acquire_source()}\n"
        "pool_build_lock_acquire; rc=$?\n"
        '[ "$rc" = 0 ] || { echo "REFUSED rc=$rc"; exit "$rc"; }\n'
        f"{tail}\n"
    )


def _lock_env(lock_dir: Path, lock_timeout: str) -> dict[str, str]:
    return {
        "PATH": os.environ["PATH"],
        "ONEX_LANE_LOCK_DIR": str(lock_dir),
        "PY": sys.executable,
        "LANE_LOCK_PY": str(LANE_LOCK_PY),
        "POOL_BUILD_LOCK_PROJECT": POOL_BUILD_LOCK_PROJECT,
        "LOCK_TIMEOUT": lock_timeout,
        "TARGET_COMMIT": "0" * 40,
        "TARGET_REPO": "omnibase_infra",
        "SLOT": "1",
    }


def test_a_second_concurrent_build_blocks_on_the_pool_build_lock(
    tmp_path: Path,
) -> None:
    lock_dir = tmp_path / "locks"
    holder = subprocess.Popen(
        ["bash", "-c", _lock_harness(hold=True)],
        env=_lock_env(lock_dir, "5"),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        assert holder.stdout is not None
        first = holder.stdout.readline()
        assert first.startswith("ACQUIRED"), first
        holder_acquired_at = time.time()

        wait_budget = 2.0
        contender_started_at = time.time()
        started = time.monotonic()
        contender = subprocess.run(
            ["bash", "-c", _lock_harness(hold=False)],
            env=_lock_env(lock_dir, str(int(wait_budget))),
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        waited = time.monotonic() - started

        # The contender neither started nor returned early: it waited out its
        # whole budget, while the holder was still running.
        assert contender.returncode == 2, (contender.stdout, contender.stderr)
        assert "ACQUIRED" not in contender.stdout
        assert waited >= wait_budget - 0.2, (
            f"gave up after {waited:.2f}s of {wait_budget}s"
        )
        assert holder.poll() is None, (
            "the holder ended, so the lock was not what blocked"
        )
        assert holder_acquired_at <= contender_started_at
        assert f"pid={holder.pid}" in contender.stderr, (
            "the contention message must name the holder"
        )
    finally:
        holder.kill()
        holder.wait(timeout=30)

    # Control: with the holder gone the same invocation acquires at once, so the
    # refusal above was the lock and not a harness that always fails.
    started = time.monotonic()
    after = subprocess.run(
        ["bash", "-c", _lock_harness(hold=False)],
        env=_lock_env(lock_dir, "5"),
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert after.returncode == 0, (after.stdout, after.stderr)
    assert "ACQUIRED" in after.stdout
    assert time.monotonic() - started < 4.0
