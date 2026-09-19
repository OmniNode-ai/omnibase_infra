# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""The runner job-started hook may not spend the job's own timeout budget.

WHY THIS FILE EXISTS (OMN-18802). `docker/runners/runner-job-started.sh` is a
PRE-JOB hook: its wall clock is billed to the job, before the job's first step
runs. Every mirror path in it is "fail-open" and its header says so in as many
words -- "A mirror that is missing, unreachable, or corrupt costs nothing."

That sentence was false. On 2026-09-19T02:50Z the hook spent 325 seconds
failing open against a degenerate `onex_change_control` mirror (153,451 loose
objects, 4.52 GiB, 389 packs) and both OCC publisher jobs were cancelled with
`The job has exceeded the maximum execution time of 5m0s` having run ZERO
steps: `onex_change_control` run 35416848353, jobs 105827069289
(omninode-runner-44, 334s) and 105827145148 (omninode-runner-9, 312s). No
publisher ran, so no companion was minted and the caller's receipt gate
reported `missing_contract`.

The arithmetic that produced it: two mirror probes at `timeout 10` plus a 1s
sleep, a seed fetch at `_C2_SEED_TIMEOUT` (180s), an `ls-remote --symref` at
10s, a detached checkout at another 180s, the C2c `GITHUB_SHA` fetch at 60s,
and sibling probes at 20s apiece. Each guard is individually reasonable; their
SERIAL SUM exceeds the 300s `timeout-minutes: 5` that both OCC publishers
declare. A fail-open path that costs more than the job has is not fail-open.

WHAT IS PINNED HERE. Two tiers, the same split the sibling mirror-rewrite
suite uses:

  1. Structural -- the script declares ONE aggregate budget and every
     `timeout` in the mirror region is clamped against it, so the bound is a
     property of the file rather than of whoever last summed the constants by
     hand. A new unclamped `timeout 300` added later is a red test, not a
     review catch.
  2. Behavioural -- the pre-seed is pointed at a listener that accepts and
     then never answers, and timed. That is the shape of an overflowed
     accept queue, and it is the falsifier that does not care how the script
     is spelled. It carries its own positive control: the same harness
     against a REAL local `git daemon` must still seed the workspace, so a
     green from tier 2 can never be "the function returned early for an
     unrelated reason".
"""

from __future__ import annotations

import contextlib
import os
import re
import shutil
import socket
import subprocess
import tempfile
import threading
import time
from collections.abc import Iterator
from pathlib import Path

import pytest

from omnibase_core.validators.no_unguarded_git_subprocess import (
    scrub_git_location_env,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
HOOK_SCRIPT = REPO_ROOT / "docker" / "runners" / "runner-job-started.sh"
_MAIN_BODY_MARKER = "\nwire_pypi_cache || true\n"

# The smallest `timeout-minutes` any job that lands on this fleet declares.
# Both OCC publishers (omniclaude call-occ-autobind-reusable.yml and
# call-occ-companion-effect-reusable.yml) declare 5, and several guard jobs in
# onex_change_control guards.yml declare 5 as well. The hook's budget is
# measured against this, not against a comfortable number.
SMALLEST_FLEET_JOB_TIMEOUT_SECONDS = 5 * 60

# The hook is an optimisation. It may spend a small fraction of the tightest
# job's budget and no more. A quarter is the line drawn by OMN-18802: enough
# for a healthy pre-seed (omnimarket measured 5s; a repacked OCC mirror is the
# same order) with headroom, far short of anything that can cancel a job.
MAX_HOOK_BUDGET_SECONDS = SMALLEST_FLEET_JOB_TIMEOUT_SECONDS // 4

pytestmark = [pytest.mark.unit]


def _hook_text() -> str:
    return HOOK_SCRIPT.read_text()


def _functions_only_script() -> str:
    """Everything before the hook's main body starts executing.

    Sourcing this rather than the whole file lets a test call the mirror
    functions directly without the script's own `exit` at the bottom killing
    the test process, and without needing GNU `realpath -m`.
    """
    text = _hook_text()
    idx = text.index(_MAIN_BODY_MARKER)
    assert idx > 0, (
        "expected marker 'wire_pypi_cache || true' not found in "
        f"{HOOK_SCRIPT} -- the function-defs/main-body boundary moved; "
        "update _MAIN_BODY_MARKER"
    )
    return text[:idx]


def _mirror_region(text: str) -> str:
    """The part of the hook that talks to the git mirror.

    Bounded by the two banner comments that already delimit it in the file, so
    this helper cannot silently start measuring the disk-admission or pypi
    blocks if those move.
    """
    start = text.index("# OMN-14027 C2 -- local git-mirror pre-seed")
    end = text.index("# OMN-14027 C1 -- devpi PyPI pull-through cache wiring")
    assert start < end, f"mirror-region banners are out of order in {HOOK_SCRIPT}"
    return text[start:end]


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


# ---------------------------------------------------------------------------
# Tier 1 -- structural
# ---------------------------------------------------------------------------


def test_hook_declares_one_aggregate_budget() -> None:
    """A single named budget exists and its default is a small fraction of the
    tightest job on the fleet.

    Without this the only bound on the hook is the sum of its per-call
    timeouts, which is exactly the quantity nobody was tracking when it grew
    past 300s.
    """
    text = _hook_text()
    match = re.search(
        r'^_C2_HOOK_BUDGET_SECONDS="\$\{OMNI_GIT_MIRROR_HOOK_BUDGET:-(\d+)\}"',
        text,
        re.MULTILINE,
    )
    assert match is not None, (
        "no `_C2_HOOK_BUDGET_SECONDS` declaration found in "
        f"{HOOK_SCRIPT}. The mirror block is a pre-job optimisation billed to "
        "the job's own timeout-minutes; it must declare one aggregate "
        "wall-clock budget rather than relying on the sum of its per-call "
        "timeouts."
    )
    budget = int(match.group(1))
    assert 0 < budget <= MAX_HOOK_BUDGET_SECONDS, (
        f"the hook budget default is {budget}s, which is not a small fraction "
        f"of the tightest job on the fleet ({SMALLEST_FLEET_JOB_TIMEOUT_SECONDS}s). "
        f"Keep it at or under {MAX_HOOK_BUDGET_SECONDS}s: a pre-job "
        "optimisation must never be able to cancel the job it optimises."
    )


def test_budget_clock_starts_before_any_mirror_work() -> None:
    """The budget's start timestamp is taken at hook top level, not inside the
    first function that happens to use it.

    A clock started lazily measures only the step that started it, which is
    the bound the script already effectively had.
    """
    text = _hook_text()
    assert re.search(r'^_C2_BUDGET_START="\$\(date \+%s\)"', text, re.MULTILINE), (
        "`_C2_BUDGET_START` must be assigned at top level in "
        f"{HOOK_SCRIPT} so the budget covers the whole hook."
    )
    start_idx = text.index("_C2_BUDGET_START=")
    seed_idx = text.index("seed_workspace_from_mirror() {")
    assert start_idx < seed_idx, (
        "`_C2_BUDGET_START` is assigned after the first mirror function is "
        "defined; the clock must start before any mirror work can run."
    )


def test_every_mirror_timeout_is_clamped_against_the_budget() -> None:
    """No `timeout` in the mirror region carries a bare literal or an
    unclamped variable.

    This is the half that survives the next person. Adding a new probe with
    its own generous constant is how the serial sum grew to 325s in the first
    place; here it fails the suite instead.
    """
    region = _mirror_region(_hook_text())
    offenders = [
        line.strip()
        for line in region.splitlines()
        if re.search(r"(?:^|\s|&&\s*)timeout\s+(?!\"\$\{?_c2_t)", line)
        and "timeout" in line
        and not line.lstrip().startswith("#")
    ]
    assert not offenders, (
        "these `timeout` invocations in the mirror region are not clamped "
        "against the aggregate budget -- each one can spend the job's own "
        "timeout-minutes:\n  " + "\n  ".join(offenders) + "\n"
        "Resolve the duration through `_c2_budget_timeout <want>` into a "
        "local named `_c2_t` and pass that instead."
    )


def test_budget_helper_refuses_rather_than_passing_zero_to_timeout() -> None:
    """An exhausted budget must return non-zero, never the string "0".

    GNU coreutils reads `timeout 0` as "no limit". Handing an exhausted budget
    to `timeout` as a zero would convert the bound into an unbounded wait --
    the precise failure this whole file exists to prevent, reintroduced by a
    plausible-looking helper.
    """
    region = _mirror_region(_hook_text())
    assert "_c2_budget_timeout()" in region, (
        f"no `_c2_budget_timeout` helper found in the mirror region of {HOOK_SCRIPT}"
    )
    helper = region[region.index("_c2_budget_timeout()") :]
    helper = helper[: helper.index("\n}\n") + 3]
    assert re.search(r'\[\[ "\$\{remaining\}" -gt 0 \]\] \|\| return 1', helper), (
        "`_c2_budget_timeout` must `return 1` on an exhausted budget. It "
        "must not print 0: `timeout 0` means NO timeout to coreutils, so a "
        "spent budget would become an unbounded wait.\n" + helper
    )


def test_the_hook_still_documents_that_fail_open_is_bounded() -> None:
    """The header's cost claim is the thing OMN-18802 falsified; it has to
    read true now.

    A comment is not a mechanism, but a comment asserting the opposite of the
    mechanism is how the next reader concludes the sum does not need checking.
    """
    region = _mirror_region(_hook_text())
    assert "OMN-18802" in region, (
        "the mirror region does not cite OMN-18802. The header's "
        "'costs nothing' claim was false for 325 seconds on 2026-09-19; the "
        "correction belongs beside the claim."
    )


# ---------------------------------------------------------------------------
# Tier 2 -- behavioural, with its own positive control
# ---------------------------------------------------------------------------


# Authorship for fixture commits. Merged into a freshly scrubbed environment at
# every call site rather than cached in a local: the OMN-14891 guard reads the
# `env=` expression statically, so it has to SEE `scrub_git_location_env` there.
_GIT_IDENTITY = {
    "GIT_AUTHOR_NAME": "t",
    "GIT_AUTHOR_EMAIL": "t@example.invalid",
    "GIT_COMMITTER_NAME": "t",
    "GIT_COMMITTER_EMAIL": "t@example.invalid",
}


def _git(*args: str, cwd: Path | None = None, timeout: int = 60) -> None:
    """Run a fixture git command against a scrubbed environment.

    OMN-18434: GIT_DIR / GIT_WORK_TREE / GIT_INDEX_FILE exported by a git hook
    override both `cwd=` and `git -C`, so an unscrubbed fixture would operate
    on the real invoking worktree instead of tmp_path.
    """
    subprocess.run(
        ["git", *args],
        cwd=str(cwd) if cwd else None,
        check=True,
        capture_output=True,
        text=True,
        timeout=timeout,
        env=scrub_git_location_env(os.environ) | _GIT_IDENTITY,
    )


def _bash() -> str:
    bash = shutil.which("bash")
    assert bash is not None
    return bash


def _run_seed(
    *,
    workspace: Path,
    mirror_host: str,
    mirror_port: int,
    budget: int,
    repo: str = "onex_change_control",
) -> tuple[subprocess.CompletedProcess[str], float]:
    """Sources the hook's function defs and runs the pre-seed once, timed."""
    script = _functions_only_script() + (
        '\nseed_workspace_from_mirror "$1"\n'
        '\nwire_uv_git_mirror_rewrite "$1" || true\n'
        '\nwire_sibling_checkout_mirror_rewrite "$1" || true\n'
    )
    with tempfile.NamedTemporaryFile(
        "w", suffix=".sh", delete=False, dir=str(workspace.parent)
    ) as handle:
        handle.write(script)
        driver = Path(handle.name)

    # OMN-18434: the hook shells out to git, so an unscrubbed env here would
    # let a pre-push hook's GIT_DIR retarget the hook's own git commands at
    # the real worktree instead of the tmp_path workspace.
    env = scrub_git_location_env(os.environ)
    env.update(
        {
            "GITHUB_REPOSITORY": f"OmniNode-ai/{repo}",
            "GITHUB_SERVER_URL": "https://github.com",
            "GITHUB_SHA": "0" * 40,
            "OMNI_GIT_MIRROR_HOST": mirror_host,
            "OMNI_GIT_MIRROR_PORT": str(mirror_port),
            "OMNI_GIT_MIRROR_RUNNERS": "ALL",
            "OMNI_GIT_MIRROR_HOOK_BUDGET": str(budget),
            "RUNNER_NAME": "omninode-runner-test",
        }
    )
    env.pop("GITHUB_ENV", None)

    started = time.monotonic()
    result = subprocess.run(
        [_bash(), str(driver), str(workspace)],
        capture_output=True,
        text=True,
        # Deliberately far above the budget: the point is to MEASURE how long
        # the hook takes, so the harness must not be the thing that stops it.
        timeout=600,
        check=False,
        env=env,
        cwd=str(workspace),
    )
    elapsed = time.monotonic() - started
    driver.unlink(missing_ok=True)
    return result, elapsed


@contextlib.contextmanager
def _stalling_listener() -> Iterator[int]:
    """A TCP port that ACCEPTS and then never answers.

    This, not a refused connection or a black-holed address, is the shape of
    the production failure. The `.201` git-daemon has a hardcoded accept
    backlog of 5 and had overflowed it 5,110 times by the time OMN-18802 was
    diagnosed: clients connect, sit in or fall out of the accept queue, and
    stall. A refused connection returns instantly and a black-holed address
    fails fast with EHOSTUNREACH on some hosts -- either would make this test
    pass against the unfixed script, which is the one thing it must not do.
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("127.0.0.1", 0))
    sock.listen(16)
    port = int(sock.getsockname()[1])

    stop = threading.Event()
    held: list[socket.socket] = []

    def _accept_forever() -> None:
        sock.settimeout(0.2)
        while not stop.is_set():
            try:
                conn, _ = sock.accept()
            except (TimeoutError, OSError):
                continue
            # Hold the connection open and say nothing at all.
            held.append(conn)

    thread = threading.Thread(target=_accept_forever, daemon=True)
    thread.start()
    try:
        yield port
    finally:
        stop.set()
        thread.join(timeout=5)
        for conn in held:
            conn.close()
        sock.close()


def test_a_stalling_mirror_costs_at_most_the_budget(tmp_path: Path) -> None:
    """The falsifier, and the one assertion here that does not care how the
    script is spelled.

    Against the unfixed hook this costs two full 10s probes plus a 1s sleep
    before the pre-seed even reaches its 180s fetch, so a 5s budget is
    exceeded four times over. Against the fixed hook the first probe is
    clamped to what remains and the rest is skipped.
    """
    workspace = tmp_path / "ws"
    workspace.mkdir()
    budget = 5

    with _stalling_listener() as port:
        result, elapsed = _run_seed(
            workspace=workspace,
            mirror_host="127.0.0.1",
            mirror_port=port,
            budget=budget,
        )

    assert elapsed < budget + 8, (
        f"the hook spent {elapsed:.0f}s against a stalling mirror with a "
        f"{budget}s budget. This is the OMN-18802 defect: a fail-open path "
        "billed to the job's own timeout-minutes, which cancelled both OCC "
        "publishers at 5m0s having run zero steps.\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    assert result.returncode == 0, (
        "the pre-seed must still fail open (exit 0) when the mirror stalls; "
        f"got {result.returncode}.\n{result.stderr}"
    )


def test_positive_control_a_reachable_mirror_still_seeds(tmp_path: Path) -> None:
    """Without this, the test above passes on a hook that does nothing at all.

    Serves a real bare repo over a real `git daemon` on loopback and asserts
    the workspace comes back seeded.
    """
    if shutil.which("git") is None:  # pragma: no cover - git is a hard dep
        pytest.skip("git is not available")

    mirror_root = tmp_path / "mirrors"
    mirror_root.mkdir()
    repo = "onex_change_control"
    bare = mirror_root / f"{repo}.git"

    work = tmp_path / "src"
    work.mkdir()
    _git("init", "--quiet", "--initial-branch=dev", ".", cwd=work, timeout=30)
    (work / "uv.lock").write_text("# empty\n")
    _git("add", "-A", cwd=work, timeout=30)
    _git("commit", "--quiet", "-m", "seed", cwd=work, timeout=30)
    _git("clone", "--quiet", "--bare", str(work), str(bare))
    _git("-C", str(bare), "symbolic-ref", "HEAD", "refs/heads/dev", timeout=30)

    port = _find_free_port()
    daemon = subprocess.Popen(
        [
            "git",
            "daemon",
            "--reuseaddr",
            "--export-all",
            f"--base-path={mirror_root}",
            "--listen=127.0.0.1",
            f"--port={port}",
            str(mirror_root),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env=scrub_git_location_env(os.environ),
    )
    try:
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            probe = subprocess.run(
                ["git", "ls-remote", f"git://127.0.0.1:{port}/{repo}.git"],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
                env=scrub_git_location_env(os.environ),
            )
            if probe.returncode == 0:
                break
            time.sleep(0.3)
        else:  # pragma: no cover - only on a wedged CI host
            pytest.skip("local git daemon did not come up")

        workspace = tmp_path / "ws"
        workspace.mkdir()
        result, elapsed = _run_seed(
            workspace=workspace,
            mirror_host="127.0.0.1",
            mirror_port=port,
            budget=60,
            repo=repo,
        )

        assert result.returncode == 0, result.stderr
        assert "pre-seeded" in result.stdout, (
            "positive control failed: a reachable mirror did not seed the "
            f"workspace, so the unreachable-mirror timing test proves nothing.\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
        assert (workspace / ".git").exists()
        assert elapsed < 60
    finally:
        daemon.terminate()
        try:
            daemon.wait(timeout=10)
        except subprocess.TimeoutExpired:  # pragma: no cover
            daemon.kill()
