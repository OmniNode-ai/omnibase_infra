# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Guards for the orphaned spin-loop reaper in the pre-push gate (OMN-16995).

`tests/unit/scripts/test_heavy_lock.py` leaked one `sh -c while :; do :; done`
per run, reparented to PID 1 and pegging a core forever. Nineteen of them put
`.200` at load1 39.31/24 = 1.64x against the gate's 1.0x-core fitness
threshold, so the governed selector refused every heavy escalation in the lab
-- for lanes and repos several tickets away from the cause. The root cause is
fixed in that test; this reaper is the stopgap that keeps a gate host usable
while the fix propagates, and the standing defense against the next process
that leaks the same shape.

A reaper on a shared gate host is only safe if it is *narrow*, so what is
pinned here is mostly what it must NOT kill. The shipped shell is
extract-and-executed rather than re-implemented in Python, so a matcher that
widens in the script fails here even if this file is untouched.
"""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
import time
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
LIB = REPO_ROOT / "scripts" / "hooks" / "prepush_dispatch.sh"
HOOK = REPO_ROOT / "scripts" / "hooks" / "prepush_smart_tests.sh"

pytestmark = pytest.mark.unit

#: The exact argv the reaper is allowed to match, and nothing else.
SPIN = "while :; do :; done"


def _driver(
    body: str, env: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    """Run BODY with the real library sourced and the hook's deps stubbed."""
    script = f"""
set -uo pipefail
log() {{ printf '[t] %s\\n' "$1" >&2; }}
die() {{ printf 'DIE: %s\\n' "$1" >&2; exit 1; }}
_prepush_timeout_cmd() {{ printf ''; }}
host_load_ratio() {{ return 1; }}
. {LIB}
{body}
"""
    return subprocess.run(
        ["bash", "-c", script],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
        stdin=subprocess.DEVNULL,
        env={**os.environ, **(env or {})},
    )


# =============================================================================
# The matcher -- run the shipped snippet against a synthetic process table
# =============================================================================
# `kill` is a shell BUILTIN, so it cannot be stubbed on PATH. It is shadowed
# with a shell function instead, which is why this table-driven test can prove
# selection without killing anything. The snippet itself is the shipped one,
# read out of the sourced library.


def _select(rows: list[str], tmp_path: Path, min_age: int = 600) -> list[str]:
    """Pids the SHIPPED snippet selects from a synthetic `ps` output.

    The snippet reports each reap on stdout as "<pid> <age>", so selection is
    read from there. `kill` is a shell BUILTIN and cannot be stubbed on PATH;
    it is shadowed with a no-op shell function instead, which is what makes it
    safe to feed this synthetic pids that belong to real processes.
    """
    fixture = tmp_path / "ps.txt"
    fixture.write_text("\n".join(rows) + "\n", encoding="utf-8")
    body = (
        f"export PS_FIXTURE={fixture}\n"
        f"export PREPUSH_SPIN_ORPHAN_MIN_AGE={min_age}\n"
        # The stubs are single-quoted and the shipped snippet carries no single
        # quote, so it concatenates in verbatim -- this runs THE code that
        # ships, never a Python re-implementation of the matcher.
        'sh -c \'ps() { cat "$PS_FIXTURE"; }\n'
        "kill() { :; }\n"
        '\'"$_PREPUSH_SPIN_ORPHAN_REAPER_SH"\n'
    )
    result = _driver(body)
    assert result.returncode == 0, result.stderr
    return [line.split()[0] for line in result.stdout.splitlines() if line.strip()]


def test_an_exact_signature_ppid1_orphan_past_the_age_floor_is_selected(
    tmp_path: Path,
) -> None:
    assert _select([f"  4242     1    02:47:11 sh -c {SPIN}"], tmp_path) == ["4242"]


@pytest.mark.parametrize(
    ("etime", "selected"),
    [
        ("09:59", False),  # 599s -- under the floor
        ("10:00", True),  # 600s -- exactly the floor
        ("02:47:11", True),  # hours form
        ("1-04:00:00", True),  # days form
        ("00:09", False),  # seconds only, nowhere near the floor
    ],
)
def test_the_age_floor_is_enforced_across_every_etime_rendering(
    etime: str, selected: bool, tmp_path: Path
) -> None:
    """`ps` renders elapsed time three different ways and all three must parse.

    A parser that mis-read `1-04:00:00` as 1 second would leave a day-old
    core-burner running; one that mis-read `10:00` as 10 seconds would reap a
    run that is still in flight.
    """
    got = _select([f"  4242     1    {etime} sh -c {SPIN}"], tmp_path)
    assert got == (["4242"] if selected else [])


@pytest.mark.parametrize(
    "row",
    [
        # Not orphaned: something is still supervising it.
        f"  4242    99    02:47:11 sh -c {SPIN}",
        f"  4242     0    02:47:11 sh -c {SPIN}",
        # A loop that does real work, not a no-op spin.
        "  4242     1    02:47:11 sh -c while :; do sleep 1; done",
        "  4242     1    02:47:11 sh -c while :; do work; done",
        # Same shape, different interpreter -- out of scope by construction.
        f"  4242     1    02:47:11 bash -c {SPIN}",
        f"  4242     1    02:47:11 zsh -c {SPIN}",
        # The signature as a SUBSTRING of a larger command line.
        f"  4242     1    02:47:11 sh -c {SPIN}; rm -rf /tmp/x",
        f"  4242     1    02:47:11 sh -c echo hi; {SPIN}",
        f"  4242     1    02:47:11 timeout 60 sh -c {SPIN}",
        # An editor or a grep that merely mentions it.
        f"  4242     1    02:47:11 vim -c {SPIN}",
        f"  4242     1    02:47:11 grep -F {SPIN}",
    ],
)
def test_nothing_but_the_exact_orphaned_no_op_spin_is_touched(
    row: str, tmp_path: Path
) -> None:
    """The blast radius of a wrong match here is someone else's live work.

    This runs on shared gate hosts that also carry CI runners and the .201
    runtime lanes, so the matcher is anchored on the WHOLE argv, not a
    substring, and on PPID exactly 1.
    """
    assert _select([row], tmp_path) == []


def test_selection_is_per_row_not_all_or_nothing(tmp_path: Path) -> None:
    """A real process table is mixed; the reaper must pick out only its own."""
    rows = [
        f"  1001     1    03:00:00 sh -c {SPIN}",  # reap
        f"  1002   500    03:00:00 sh -c {SPIN}",  # supervised
        f"  1003     1    00:00:30 sh -c {SPIN}",  # too young
        "  1004     1    03:00:00 /usr/bin/python3 -m pytest tests/",  # unrelated
        f"  1005     1    11:19:00 sh -c {SPIN}",  # reap
    ]
    assert _select(rows, tmp_path) == ["1001", "1005"]


# =============================================================================
# End to end -- real orphans, real kills
# =============================================================================


def _spawn_orphan(script: str) -> int:
    """Spawn `sh -c SCRIPT` detached so it is reparented, and return its pid.

    The intermediate shell exits immediately, which is exactly how the leaked
    heavy_lock grandchildren reached PPID 1.
    """
    spawn = subprocess.run(
        ["sh", "-c", 'sh -c "$1" > /dev/null 2>&1 & echo $!', "_", script],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    return int(spawn.stdout.strip())


def _ppid_of(pid: int) -> int | None:
    out = subprocess.run(
        ["ps", "-o", "ppid=", "-p", str(pid)],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    ).stdout.strip()
    return int(out) if out else None


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:  # pragma: no cover - not our process
        return True
    return True


def _kill(pid: int) -> None:
    try:
        os.kill(pid, signal.SIGKILL)
    except OSError:
        pass


@pytest.mark.skipif(shutil.which("ps") is None, reason="no ps(1) on this host")
def test_a_real_orphaned_spin_loop_is_killed_and_logged() -> None:
    """The whole point: the core it was burning comes back, and we say so.

    Age floor 0 makes a freshly-orphaned loop eligible; the floor itself is
    proven above against the synthetic table.
    """
    if os.getpid() == 1:
        pytest.skip("pytest is PID 1 here, so PPID-1 says nothing about orphanhood")
    pid = _spawn_orphan(SPIN)
    try:
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline and _ppid_of(pid) not in (1, None):
            time.sleep(0.1)
        if _ppid_of(pid) != 1:
            pytest.skip("this host reparents to a subreaper, not to PID 1")

        result = _driver(
            'reap_spin_loop_orphans ""',
            env={"PREPUSH_SPIN_ORPHAN_MIN_AGE": "0", "PREPUSH_LOAD_OVERRIDE_LOCAL": ""},
        )
        assert result.returncode == 0, result.stderr

        gone_by = time.monotonic() + 10
        while time.monotonic() < gone_by and _alive(pid):
            time.sleep(0.1)
        assert not _alive(pid), f"the orphan at pid {pid} survived the reaper"
        assert f"pid={pid}" in result.stderr, (
            f"a silent reap is a worse bug than the leak: {result.stderr}"
        )
        assert "OMN-16995" in result.stderr
    finally:
        _kill(pid)


@pytest.mark.skipif(shutil.which("ps") is None, reason="no ps(1) on this host")
def test_a_live_supervised_spin_loop_is_not_killed() -> None:
    """A spin loop with a living parent may be an in-flight test, not a leak."""
    proc = subprocess.Popen(
        ["sh", "-c", f"(sleep 30; kill -9 $$) & {SPIN}"],
        start_new_session=True,
    )
    try:
        time.sleep(0.5)
        result = _driver(
            'reap_spin_loop_orphans ""',
            env={"PREPUSH_SPIN_ORPHAN_MIN_AGE": "0"},
        )
        assert result.returncode == 0, result.stderr
        time.sleep(0.5)
        assert proc.poll() is None, "the reaper killed a supervised spin loop"
    finally:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except OSError:  # pragma: no cover
            pass
        proc.wait(timeout=30)


def test_the_reaper_can_be_switched_off_but_is_on_by_default() -> None:
    on = _driver('printf "%s\\n" "${PREPUSH_REAP_SPIN_ORPHANS:-on}"')
    assert on.stdout.strip() == "on"
    off = _driver(
        'reap_spin_loop_orphans "" && printf RAN\\\\n',
        env={"PREPUSH_REAP_SPIN_ORPHANS": "off"},
    )
    assert off.stdout.strip() == "RAN"
    assert "REAPED" not in off.stderr


def test_at_most_one_reap_per_target_per_run() -> None:
    """The picker calls the load probe repeatedly; the reap must not multiply."""
    result = _driver(
        'reap_spin_loop_orphans ""; reap_spin_loop_orphans ""; '
        'printf "%s\\n" "$_PREPUSH_SPIN_REAPED"',
        env={"PREPUSH_SPIN_ORPHAN_MIN_AGE": "0"},
    )
    assert result.stdout.strip() == "|@local|"


# =============================================================================
# Wiring -- the reaper is useless unless it runs before the measurement
# =============================================================================


def test_the_load_probe_reaps_before_it_measures() -> None:
    """Measuring first and reaping after would refuse the very push that fixed
    the host. The call site is pinned to the top of `host_load_ratio`."""
    text = HOOK.read_text(encoding="utf-8")
    body = text[text.index("host_load_ratio() {") :]
    body = body[: body.index("\n}\n")]
    assert "reap_spin_loop_orphans" in body, (
        "host_load_ratio no longer reaps orphaned spin loops (OMN-16995)"
    )
    assert body.index("reap_spin_loop_orphans") < body.index(
        "_PREPUSH_LOAD_PROBE_SH"
    ), (
        "the reap must happen BEFORE the load is read, or the gate still sees "
        "the load the orphans were producing"
    )


def test_the_reaper_stays_interpreter_free() -> None:
    """OMN-14953: every python under scripts/hooks/ must route through `uv run`,
    and `.201` has no `uv` at all -- so the snippet ssh(1) hands to a remote
    login shell may not invoke an interpreter, exactly like the load probe."""
    text = LIB.read_text(encoding="utf-8")
    snippet = text.split("_PREPUSH_SPIN_ORPHAN_REAPER_SH='", 1)[1].split("'\n", 1)[0]
    for banned in ("python", "python3", "perl", "awk", "uv run"):
        assert banned not in snippet, f"the reaper snippet invokes {banned}"
    assert "'" not in snippet, (
        "the snippet is handed to ssh(1) as a single-quoted assignment, so it "
        "cannot contain a single quote"
    )
