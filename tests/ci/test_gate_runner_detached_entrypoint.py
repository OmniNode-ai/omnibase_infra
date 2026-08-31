# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""The sanctioned gate-runner entry point must not depend on a live attach (OMN-17317).

RED-FIRST, and the red is a real kernel behaviour rather than a mock. The first
test in this file is a CONTROL that reproduces the 2026-08-31 wedge mechanism
directly: a process whose stdout is a pipe with an **open-but-undrained** read
end blocks forever on `write(2)` once it has produced more than the kernel's
pipe buffer. Everything after it asserts that the shipped supervisor does not
have that shape.

The incident (OMN-17317), measured live on `omninode-gate-runner`:

* ``pre_commit._run_single_hook`` buffers a hook's ENTIRE output and writes it
  out in one burst after the hook exits. A fail-closed full-suite escalation
  produces megabytes of per-test lines, so the burst crosses 64 KiB within the
  first fraction of a percent of the run.
* A ``docker exec`` session's stdout is a kernel pipe whose only reader is the
  containerd shim. When the exec client detaches — a dockerd bounce, an ssh
  drop, an agent session ending — the shim holds the read end open and nothing
  drains it.
* Result: 3 h 45 m of tests ran GREEN and exited 0, then ``pre-commit`` wedged
  permanently in ``anon_pipe_write`` and ``git push`` parked in ``do_wait``
  behind a hook that could never exit. No log line, no timeout, no diagnosis.
  Recovery required draining the shim's read fd by hand — exactly 65 538 bytes.

The acceptance criterion this file implements, verbatim from the ticket: *"a
hook whose captured output exceeds 64 KiB, written into a pipe with an
open-but-undrained read end, must fail loudly within the timeout instead of
blocking indefinitely."* The shipped supervisor does better than fail loudly —
it does not write the payload into that pipe at all — so the assertion is that
it **completes** within the bound, records its verdict in a receipt, and leaves
the undrained pipe essentially untouched.

Non-vacuity: without ``test_a_naive_attached_run_wedges_on_an_undrained_pipe``
below, every other test here would pass against an implementation that simply
never produced enough output to block. The control proves the harness really
does construct the wedging condition.
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import stat
import subprocess
import time
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SUPERVISOR = REPO_ROOT / "scripts" / "ci" / "gate_runner_supervisor.sh"
LAUNCHER = REPO_ROOT / "scripts" / "ci" / "run_on_gate_runner.sh"

# Comfortably past the kernel pipe buffer on both designated hosts (64 KiB on
# Linux, 16 KiB growing to 64 KiB on macOS) so the control test wedges on
# either, and past the exact 65 538 bytes drained out of the live incident.
PAYLOAD_BYTES = 256 * 1024
PIPE_BUFFER_CEILING = 64 * 1024

# A payload that emits PAYLOAD_BYTES on stdout and then exits 0. `yes` + `head`
# is used rather than a Python one-liner so the test does not depend on an
# interpreter being resolvable inside whatever userland runs it.
_EMIT_PAYLOAD = (
    f"yes 0123456789abcdefghijklmnopqrstuvwxyz | head -c {PAYLOAD_BYTES}; exit 0"
)

pytestmark = pytest.mark.ci


def _read_all_nonblocking(fd: int) -> bytes:
    """Drain everything currently readable on `fd` without ever blocking."""
    os.set_blocking(fd, False)
    chunks: list[bytes] = []
    while True:
        try:
            chunk = os.read(fd, 65536)
        except BlockingIOError:
            break
        if not chunk:
            break
        chunks.append(chunk)
    return b"".join(chunks)


def _receipt(run_dir: Path) -> dict[str, object]:
    receipt_path = run_dir / "receipt.json"
    assert receipt_path.is_file(), f"no receipt at {receipt_path}"
    parsed = json.loads(receipt_path.read_text())
    assert isinstance(parsed, dict)
    return parsed


def _reap_group(proc: subprocess.Popen[bytes]) -> None:
    """Reap the whole process GROUP, not just the direct child (OMN-16995).

    Every spawn in this module is a supervisor that forks its own heartbeat
    loop, so `proc.kill()` alone signals the wrapper and reparents the loop to
    pid 1 — the exact orphan class this file's own
    `test_the_heartbeat_dies_with_its_supervisor_even_under_sigkill` exists to
    forbid, and the one that took `.200` to 1.64x-core load once already. Every
    `Popen` here therefore passes `start_new_session=True` so the group can be
    signalled as a unit; `tests/unit/scripts/test_spawn_grandchild_audit.py`
    enforces that repo-wide and caught this file on its first governed push.
    """
    if proc.poll() is None:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError):  # pragma: no cover
            proc.kill()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:  # pragma: no cover - defensive
        pass


# ---------------------------------------------------------------------------
# The control: prove the wedge is real before asserting the fix avoids it.
# ---------------------------------------------------------------------------


def test_a_naive_attached_run_wedges_on_an_undrained_pipe() -> None:
    """A naive attached run really does block forever — this is the OMN-17317 defect.

    This is the shape a raw `docker exec` gives a governed pre-push: the child
    writes to a pipe whose read end is held open by something that never reads
    it. It must NOT complete. If this test ever starts passing quickly, the
    harness has stopped constructing the wedging condition and every assertion
    below it has gone vacuous.
    """
    read_fd, write_fd = os.pipe()
    proc = subprocess.Popen(
        ["/bin/sh", "-c", _EMIT_PAYLOAD],
        stdout=write_fd,
        stderr=subprocess.DEVNULL,
        start_new_session=True,  # OMN-16995: group-spawned so cleanup can reap the whole tree
    )
    os.close(write_fd)  # only the child holds the write end now
    try:
        with pytest.raises(subprocess.TimeoutExpired):
            proc.wait(timeout=10)
        assert proc.poll() is None, (
            "the naive attached run completed against an undrained pipe, so this "
            "harness no longer reproduces the OMN-17317 wedge and every "
            "assertion below it is vacuous"
        )
    finally:
        _reap_group(proc)
        os.close(read_fd)


# ---------------------------------------------------------------------------
# The fix: the supervisor never writes the payload into the inherited stream.
# ---------------------------------------------------------------------------


def test_supervisor_completes_when_its_stdout_is_an_undrained_pipe(
    tmp_path: Path,
) -> None:
    """OMN-17317 acceptance: >64 KiB into an open-but-undrained pipe must not block."""
    run_dir = tmp_path / "run"
    read_fd, write_fd = os.pipe()
    proc = subprocess.Popen(
        [
            str(SUPERVISOR),
            "--run-dir",
            str(run_dir),
            "--timeout",
            "120",
            "--no-slot",
            "--label",
            "omn17317-undrained-pipe",
            "--heartbeat-interval",
            "1",
            "--",
            "/bin/sh",
            "-c",
            _EMIT_PAYLOAD,
        ],
        stdout=write_fd,
        stderr=write_fd,
        start_new_session=True,  # OMN-16995: group-spawned so cleanup can reap the whole tree
    )
    os.close(write_fd)
    try:
        returncode = proc.wait(timeout=60)
    except subprocess.TimeoutExpired:  # pragma: no cover - the regression itself
        _reap_group(proc)
        os.close(read_fd)
        pytest.fail(
            "the supervisor blocked on an undrained inherited pipe — this is the "
            "OMN-17317 deadlock reintroduced"
        )
    leaked = _read_all_nonblocking(read_fd)
    os.close(read_fd)

    assert returncode == 0
    receipt = _receipt(run_dir)
    assert receipt["status"] == "passed"
    assert receipt["exit_code"] == 0
    assert receipt["schema"] == "onex.gate_runner.receipt.v1"

    log_bytes = (run_dir / "run.log").stat().st_size
    assert log_bytes > PAYLOAD_BYTES, (
        f"the payload's {PAYLOAD_BYTES} bytes must land in the durable log, got {log_bytes}"
    )
    assert len(leaked) < PIPE_BUFFER_CEILING, (
        f"{len(leaked)} bytes reached the inherited pipe; the payload must never be "
        "written into an interactive exec stream"
    )


def test_receipt_records_a_running_status_before_the_payload_finishes(
    tmp_path: Path,
) -> None:
    """The caller pattern is poll-the-receipt, so a receipt must exist while running."""
    run_dir = tmp_path / "run"
    proc = subprocess.Popen(
        [
            str(SUPERVISOR),
            "--run-dir",
            str(run_dir),
            "--timeout",
            "120",
            "--no-slot",
            "--heartbeat-interval",
            "1",
            "--",
            "/bin/sh",
            "-c",
            "sleep 20",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,  # OMN-16995: group-spawned so cleanup can reap the whole tree
    )
    try:
        deadline = time.monotonic() + 30
        observed = ""
        while time.monotonic() < deadline:
            if (run_dir / "receipt.json").is_file():
                observed = str(_receipt(run_dir)["status"])
                if observed == "running":
                    break
            time.sleep(0.2)
        assert observed == "running", f"expected a running receipt, saw {observed!r}"
    finally:
        _reap_group(proc)


def test_heartbeat_distinguishes_alive_and_progressing_from_dead(
    tmp_path: Path,
) -> None:
    """A monitor must be able to tell alive-slow from wedged (OMN-17317 property 3)."""
    run_dir = tmp_path / "run"
    proc = subprocess.Popen(
        [
            str(SUPERVISOR),
            "--run-dir",
            str(run_dir),
            "--timeout",
            "120",
            "--no-slot",
            "--heartbeat-interval",
            "1",
            "--",
            "/bin/sh",
            "-c",
            "i=0; while [ $i -lt 30 ]; do echo line-$i; sleep 0.5; i=$((i+1)); done",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,  # OMN-16995: group-spawned so cleanup can reap the whole tree
    )
    try:
        heartbeat = run_dir / "heartbeat"
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline and not heartbeat.is_file():
            time.sleep(0.2)
        assert heartbeat.is_file(), "no heartbeat file was written"
        first = heartbeat.read_text()
        assert "log_bytes=" in first, (
            f"heartbeat must record log progress, got {first!r}"
        )
        first_bytes = int(first.split("log_bytes=")[1].split()[0])

        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            later = heartbeat.read_text()
            later_bytes = int(later.split("log_bytes=")[1].split()[0])
            if later_bytes > first_bytes and later != first:
                break
            time.sleep(0.5)
        else:  # pragma: no cover - only on a broken heartbeat
            pytest.fail(
                "the heartbeat never advanced while the payload was producing output"
            )
    finally:
        _reap_group(proc)


def test_the_heartbeat_dies_with_its_supervisor_even_under_sigkill(
    tmp_path: Path,
) -> None:
    """A heartbeat that outlives its run LIES, and leaks the OMN-16995 orphan class.

    Found live while landing this ticket: two ``gate_runner_supervisor.sh``
    heartbeat subshells from an earlier run of THIS file were still alive ~15
    minutes later, orphaned to pid 1, burning CPU and — far worse — still
    stamping a fresh timestamp onto the ``heartbeat`` file of a run whose
    supervisor was long dead. The receipt was frozen at ``running`` while the
    heartbeat advanced, i.e. the exact inversion of the diagnostic contract
    this ticket exists to create: the runbook promises "timestamp frozen is
    dead", and an orphaned loop makes a dead run look healthy forever.

    That is also the OMN-16995 class the ``.200`` gate host has already been
    taken down by once (nineteen leaked loops -> 1.64x-core load -> the
    governed selector refuses every heavy escalation). Leaking it from the
    gate-runner entry point itself would be self-defeating.

    SIGKILL is asserted deliberately rather than SIGTERM: it is the case no
    trap anywhere can catch, so passing it proves the loop's own
    parent-liveness check carries the guarantee rather than the trap.
    """
    run_dir = tmp_path / "run"
    proc = subprocess.Popen(
        [
            str(SUPERVISOR),
            "--run-dir",
            str(run_dir),
            "--timeout",
            "600",
            "--no-slot",
            "--heartbeat-interval",
            "1",
            "--",
            "/bin/sh",
            "-c",
            "sleep 600",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,  # OMN-16995: group-spawned so cleanup can reap the whole tree
    )
    # Captured BEFORE the kill: once the group leader is reaped, `getpgid` can
    # no longer resolve it, and this test must not be able to leak the very
    # class it forbids even when it FAILS.
    pgid = os.getpgid(proc.pid)
    heartbeat = run_dir / "heartbeat"
    try:
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline and not heartbeat.is_file():
            time.sleep(0.2)
        assert heartbeat.is_file(), "the supervisor never produced a heartbeat"

        # SIGKILL the SUPERVISOR ONLY — deliberately not the group. Killing the
        # group would reap the heartbeat too and make the assertion vacuous;
        # the whole point is that the loop must notice on its own.
        proc.kill()
        proc.wait(timeout=30)

        # Within a few heartbeat intervals the loop must notice its supervisor
        # is gone and stop. Sample twice, separated by well over the interval.
        time.sleep(6)
        frozen_at = heartbeat.read_text()
        time.sleep(6)
        assert heartbeat.read_text() == frozen_at, (
            "the heartbeat kept advancing after its supervisor was killed — a "
            "dead run now reads as alive forever, inverting the 'timestamp "
            "frozen is dead' contract, and the orphaned loop is the OMN-16995 "
            "leak class"
        )

        survivors = subprocess.run(
            ["pgrep", "-f", f"gate_runner_supervisor.sh --run-dir {run_dir}"],
            capture_output=True,
            text=True,
            check=False,
        ).stdout.split()
        assert not survivors, (
            f"orphaned supervisor/heartbeat processes survived: {survivors}"
        )
    finally:
        try:
            os.killpg(pgid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass


def test_timeout_aborts_loudly_with_a_typed_receipt(tmp_path: Path) -> None:
    """A run past its wall-clock bound aborts LOUDLY — never a silent block."""
    run_dir = tmp_path / "run"
    completed = subprocess.run(
        [
            str(SUPERVISOR),
            "--run-dir",
            str(run_dir),
            "--timeout",
            "2",
            "--no-slot",
            "--heartbeat-interval",
            "1",
            "--",
            "/bin/sh",
            "-c",
            "sleep 300",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        timeout=90,
        check=False,
    )
    assert completed.returncode == 124
    receipt = _receipt(run_dir)
    assert receipt["status"] == "timeout"
    assert receipt["exit_code"] == 124
    assert receipt["reason"] == "wall_clock_timeout_2s"
    log_text = (run_dir / "run.log").read_text()
    assert "TIMEOUT" in log_text
    assert "remediation" in log_text


def test_a_failing_payload_is_reported_as_failed_with_its_own_status(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    completed = subprocess.run(
        [
            str(SUPERVISOR),
            "--run-dir",
            str(run_dir),
            "--timeout",
            "60",
            "--no-slot",
            "--",
            "/bin/sh",
            "-c",
            "echo boom; exit 3",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        timeout=90,
        check=False,
    )
    assert completed.returncode == 3
    receipt = _receipt(run_dir)
    assert receipt["status"] == "failed"
    assert receipt["exit_code"] == 3
    assert receipt["reason"] == "exit_3"


def test_a_missing_timeout_is_refused_rather_than_defaulted(tmp_path: Path) -> None:
    """An unbounded governed run is the defect itself, so it is not expressible."""
    completed = subprocess.run(
        [
            str(SUPERVISOR),
            "--run-dir",
            str(tmp_path / "run"),
            "--no-slot",
            "--",
            "true",
        ],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert completed.returncode == 2
    assert "--timeout is required" in completed.stderr


# ---------------------------------------------------------------------------
# The exclusive heavy-suite slot (OMN-17221 / OMN-16968 option (a)(ii)).
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    shutil.which("flock") is None,
    reason="flock(1) is Linux-only; the slot is a container-side guarantee",
)
def test_the_heavy_suite_slot_is_exclusive(tmp_path: Path) -> None:
    """A second heavy run must be REFUSED, not silently run concurrently."""
    lock = tmp_path / "SLOT.lock"
    holder_dir = tmp_path / "holder"
    contender_dir = tmp_path / "contender"

    holder = subprocess.Popen(
        [
            str(SUPERVISOR),
            "--run-dir",
            str(holder_dir),
            "--timeout",
            "60",
            "--slot-lock",
            str(lock),
            "--heartbeat-interval",
            "1",
            "--",
            "/bin/sh",
            "-c",
            "sleep 30",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,  # OMN-16995: group-spawned so cleanup can reap the whole tree
    )
    try:
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            if (holder_dir / "receipt.json").is_file() and _receipt(holder_dir)[
                "status"
            ] == "running":
                break
            time.sleep(0.2)
        else:  # pragma: no cover - only if the holder never starts
            pytest.fail("the slot holder never reached a running receipt")

        contender = subprocess.run(
            [
                str(SUPERVISOR),
                "--run-dir",
                str(contender_dir),
                "--timeout",
                "60",
                "--slot-lock",
                str(lock),
                "--",
                "/bin/sh",
                "-c",
                "echo should-not-run",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=60,
            check=False,
        )
        assert contender.returncode == 4
        receipt = _receipt(contender_dir)
        assert receipt["status"] == "refused"
        assert receipt["reason"] == "slot_held"
        assert "should-not-run" not in (contender_dir / "run.log").read_text()
    finally:
        _reap_group(holder)


# Every external command gate_runner_supervisor.sh resolves off PATH. Stated
# once, here, because the fail-closed tests below build a sandbox PATH holding
# exactly these minus one — which both proves the refusal and pins the userland
# contract the gate-runner image has to keep satisfying.
_SUPERVISOR_USERLAND = (
    "bash",
    "sh",
    "date",
    "mkdir",
    "mv",
    "sed",
    "awk",
    "wc",
    "tr",
    "basename",
    "hostname",
    "sleep",
    "flock",
    "timeout",
    "gtimeout",
    "echo",
)


def _sandbox_path_without(tmp_path: Path, *omit: str) -> Path:
    """Build a PATH holding the supervisor's whole userland minus `omit`."""
    sandbox = tmp_path / "sandbox-bin"
    sandbox.mkdir(exist_ok=True)
    for tool in _SUPERVISOR_USERLAND:
        if tool in omit:
            continue
        resolved = shutil.which(tool)
        if resolved is not None:
            (sandbox / tool).symlink_to(resolved)
    return sandbox


def test_a_slot_that_cannot_be_taken_fails_closed(tmp_path: Path) -> None:
    """A gate that cannot run must be indistinguishable from a failing gate."""
    sandbox = _sandbox_path_without(tmp_path, "flock")
    run_dir = tmp_path / "run"
    completed = subprocess.run(
        [
            str(SUPERVISOR),
            "--run-dir",
            str(run_dir),
            "--timeout",
            "60",
            "--slot-lock",
            str(tmp_path / "SLOT.lock"),
            "--",
            "/bin/echo",
            "should-not-run",
        ],
        env={"PATH": str(sandbox)},
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        timeout=60,
        check=False,
    )
    assert completed.returncode == 5
    receipt = _receipt(run_dir)
    assert receipt["status"] == "refused"
    assert receipt["reason"] == "flock_unavailable"
    assert "should-not-run" not in (run_dir / "run.log").read_text()


def test_an_unbounded_run_fails_closed_when_timeout_is_unavailable(
    tmp_path: Path,
) -> None:
    """Property 2 is a property, not an intention: no timeout(1), no run."""
    sandbox = _sandbox_path_without(tmp_path, "timeout", "gtimeout")
    run_dir = tmp_path / "run"
    completed = subprocess.run(
        [
            str(SUPERVISOR),
            "--run-dir",
            str(run_dir),
            "--timeout",
            "60",
            "--no-slot",
            "--",
            "/bin/echo",
            "should-not-run",
        ],
        env={"PATH": str(sandbox)},
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        timeout=60,
        check=False,
    )
    assert completed.returncode == 5
    receipt = _receipt(run_dir)
    assert receipt["status"] == "refused"
    assert receipt["reason"] == "timeout_unavailable"
    assert "should-not-run" not in (run_dir / "run.log").read_text()


# ---------------------------------------------------------------------------
# Launcher admission, driven end-to-end against a stubbed container transport.
# ---------------------------------------------------------------------------


def _install_stub_transport(
    tmp_path: Path,
    *,
    cgroup_line: str,
    foreign_pids: str = "",
    mount_root: str = "",
) -> Path:
    """Install stub `docker` and `pgrep` binaries emulating the container transport.

    The stub emulates the container's Linux userland closely enough to drive the
    launcher's real code path: `docker inspect` answers the two format strings
    the launcher uses, `docker exec sh -c <cgroup probe>` returns a canned
    accounting line, and `docker exec -d` runs the command locally in the
    background. `setsid` is dropped when the host has none (macOS), which is the
    one place the stub compensates for not being Linux.
    """
    bin_dir = tmp_path / "stubbin"
    bin_dir.mkdir(exist_ok=True)

    docker = bin_dir / "docker"
    docker.write_text(
        "#!/usr/bin/env bash\n"
        "set -uo pipefail\n"
        'if [ "$1" = "inspect" ]; then\n'
        '  case "$3" in\n'
        "    *State.Running*) echo true ;;\n"
        f'    *Mounts*) printf "%s\\n" "{mount_root}" ;;\n'
        "  esac\n"
        "  exit 0\n"
        "fi\n"
        'if [ "$1" = "exec" ]; then\n'
        "  shift\n"
        "  detached=0\n"
        '  while [ "$#" -gt 0 ]; do\n'
        '    case "$1" in\n'
        "      -d) detached=1; shift ;;\n"
        "      -e | -w) shift 2 ;;\n"
        "      *) break ;;\n"
        "    esac\n"
        "  done\n"
        "  shift  # container name\n"
        '  if [ "${1:-}" = "setsid" ] && ! command -v setsid >/dev/null 2>&1; then shift; fi\n'
        '  if [ "${detached}" -eq 1 ]; then\n'
        '    "$@" >/dev/null 2>&1 &\n'
        "    exit 0\n"
        "  fi\n"
        '  exec "$@"\n'
        "fi\n"
        "exit 0\n"
    )
    docker.chmod(docker.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

    # The cgroup probe reaches the stub as `sh -c '<probe>'`; intercept `sh` so
    # the canned accounting line is returned without a real cgroup filesystem.
    sh_stub = bin_dir / "sh"
    sh_stub.write_text(
        "#!/usr/bin/env bash\n"
        'if [ "${1:-}" = "-c" ] && [ "${2#*cgroup}" != "${2:-}" ]; then\n'
        f'  printf "%s\\n" "{cgroup_line}"\n'
        "  exit 0\n"
        "fi\n"
        'exec /bin/sh "$@"\n'
    )
    sh_stub.chmod(sh_stub.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

    pgrep = bin_dir / "pgrep"
    if foreign_pids:
        pgrep.write_text(
            f"#!/usr/bin/env bash\nprintf '%s\\n' {foreign_pids}\nexit 0\n"
        )
    else:
        pgrep.write_text("#!/usr/bin/env bash\nexit 1\n")
    pgrep.chmod(pgrep.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

    return bin_dir


# A payload with a distinctive, multi-word argument. It is not decoration: the
# launcher's admission probes once parsed their output with `set --`, which
# CLOBBERED the positional parameters still holding the operator's command, so
# the container ran the probe's own output as the payload. Live on `.201` that
# produced a receipt reading `"command":["fit","0.000","0.143"]` and exit 127 —
# a governed run silently executing something other than what was asked. The
# assertions below pin the argv end to end so that class cannot come back.
_MARKER_PAYLOAD = ["/bin/sh", "-c", "echo omn17317 marker payload"]


def _launch(
    bin_dir: Path, worktree: Path, *extra: str, timeout: int = 120
) -> subprocess.CompletedProcess[str]:
    env = {**os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}"}
    return subprocess.run(
        [
            str(LAUNCHER),
            "--detached",
            *extra,
            str(worktree),
            *_MARKER_PAYLOAD,
        ],
        capture_output=True,
        text=True,
        env=env,
        timeout=timeout,
        check=False,
    )


# Four CPUs' worth of quota, one second of wall clock, and the usage delta that
# makes the ratio unambiguous. `cpu.max` is "<quota> <period>" in microseconds,
# so 400000/100000 is the container's real 4.0-CPU cap.
_IDLE_CGROUP = (
    "400000 100000 1000000 1400000 8589934592 1235464192"  # 0.4 core busy -> 0.10x
)
_SATURATED_CGROUP = (
    "400000 100000 1000000 7000000 8589934592 1235464192"  # 6.0 cores -> 1.50x
)
_MEM_PRESSURE_CGROUP = "400000 100000 1000000 1400000 8589934592 8300000000"  # mem 0.97


def test_admission_refuses_a_saturated_container_with_a_typed_message(
    tmp_path: Path,
) -> None:
    """OMN-16446 finding 5: admission reads the CONTAINER's cgroup, not the host's loadavg."""
    worktree = tmp_path / "wt"
    worktree.mkdir()
    bin_dir = _install_stub_transport(tmp_path, cgroup_line=_SATURATED_CGROUP)
    result = _launch(bin_dir, worktree)
    assert result.returncode == 3
    assert "REFUSED_LOAD" in result.stderr
    assert "CPU quota" in result.stderr
    assert "Nothing is queued here" in result.stderr


def test_admission_refuses_on_memory_pressure(tmp_path: Path) -> None:
    worktree = tmp_path / "wt"
    worktree.mkdir()
    bin_dir = _install_stub_transport(tmp_path, cgroup_line=_MEM_PRESSURE_CGROUP)
    result = _launch(bin_dir, worktree)
    assert result.returncode == 3
    assert "REFUSED_LOAD" in result.stderr
    assert "memory limit" in result.stderr


def test_admission_fails_closed_when_the_container_cannot_be_measured(
    tmp_path: Path,
) -> None:
    worktree = tmp_path / "wt"
    worktree.mkdir()
    bin_dir = _install_stub_transport(tmp_path, cgroup_line="")
    result = _launch(bin_dir, worktree)
    assert result.returncode == 5
    assert "REFUSED_PROBE" in result.stderr


def test_admission_refuses_when_a_foreign_heavy_prepush_holds_the_host(
    tmp_path: Path,
) -> None:
    """The container path now consults the same signal the host queue's gate-1 uses."""
    worktree = tmp_path / "wt"
    worktree.mkdir()
    bin_dir = _install_stub_transport(
        tmp_path, cgroup_line=_IDLE_CGROUP, foreign_pids="380429 468477"
    )
    result = _launch(bin_dir, worktree)
    assert result.returncode == 4
    assert "REFUSED_SLOT" in result.stderr
    assert "380429" in result.stderr


def test_a_fit_container_launches_detached_and_returns_a_pollable_run_id(
    tmp_path: Path,
) -> None:
    """The launcher hands back a run-id and a tail command, never a stream."""
    worktree = tmp_path / "wt"
    worktree.mkdir()
    bin_dir = _install_stub_transport(tmp_path, cgroup_line=_IDLE_CGROUP)
    result = _launch(
        bin_dir, worktree, "--no-slot", "--timeout", "60", "--label", "omn17317"
    )
    assert result.returncode == 0, result.stderr
    assert "LAUNCHED (detached)" in result.stderr
    assert "POLL THE RECEIPT, NEVER HOLD THE PIPE" in result.stderr

    run_root = worktree / ".onex_state" / "gate-runner"
    run_dirs = list(run_root.iterdir())
    assert len(run_dirs) == 1
    run_dir = run_dirs[0]

    deadline = time.monotonic() + 60
    receipt: dict[str, object] = {}
    while time.monotonic() < deadline:
        receipt = _receipt(run_dir)
        if receipt["status"] != "running":
            break
        time.sleep(0.5)
    assert receipt["slot"] == "skipped"
    assert receipt["label"] == "omn17317"
    assert receipt["command"] == _MARKER_PAYLOAD, (
        "the receipt must record the operator's own argv; anything else means the "
        "launcher handed the container a different command than it was given"
    )
    assert receipt["status"] == "passed", receipt
    assert receipt["exit_code"] == 0
    assert "omn17317 marker payload" in (run_dir / "run.log").read_text()


def test_no_slot_cannot_be_used_to_run_an_unbounded_suite(tmp_path: Path) -> None:
    """--no-slot is for BOUNDED ad-hoc work, enforced mechanically, not by convention."""
    worktree = tmp_path / "wt"
    worktree.mkdir()
    bin_dir = _install_stub_transport(tmp_path, cgroup_line=_IDLE_CGROUP)
    result = _launch(bin_dir, worktree, "--no-slot", "--timeout", "14400")
    assert result.returncode == 2
    assert "--no-slot requires --timeout <=" in result.stderr


# ---------------------------------------------------------------------------
# The attached path must not remain a working way to reproduce the defect.
# ---------------------------------------------------------------------------


def _launch_attached(
    bin_dir: Path, worktree: Path, *payload: str
) -> subprocess.CompletedProcess[str]:
    env = {**os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}"}
    return subprocess.run(
        [str(LAUNCHER), str(worktree), *payload],
        capture_output=True,
        text=True,
        env=env,
        timeout=120,
        check=False,
    )


@pytest.mark.parametrize(
    "payload",
    [
        pytest.param(["uv", "run", "pytest", "tests/unit", "-q"], id="pytest"),
        pytest.param(
            ["bash", "scripts/hooks/prepush_smart_tests.sh"], id="prepush-hook"
        ),
        pytest.param(["git", "push", "-u", "origin", "HEAD"], id="git-push"),
    ],
)
def test_attached_mode_refuses_the_heavy_shapes_that_wedge(
    tmp_path: Path, payload: list[str]
) -> None:
    """Attached stdout IS the exec pipe that wedged on 2026-08-31 (OMN-17317).

    The launcher documents "attached is for fast probes only". A comment is not
    a mechanism (repo rule 5): without this refusal the sanctioned entry point
    stays a working way to reproduce the exact defect it exists to remove, and
    the operator who reaches for it gets the wedge rather than a diagnosis.
    """
    worktree = tmp_path / "wt"
    worktree.mkdir()
    bin_dir = _install_stub_transport(tmp_path, cgroup_line=_IDLE_CGROUP)
    result = _launch_attached(bin_dir, worktree, *payload)
    assert result.returncode == 2, (
        f"attached heavy payload {payload} was not refused: {result.stdout}"
    )
    assert "--detached" in result.stderr, (
        "the refusal must name the flag that makes the run safe, not merely refuse"
    )


def test_attached_mode_still_allows_a_genuine_fast_probe(tmp_path: Path) -> None:
    """The refusal must be narrow: short interactive probes are why attached exists.

    A guard that also blocked `git rev-parse` would push operators back onto raw
    `docker exec`, which is the un-governed path this entry point replaces.
    """
    worktree = tmp_path / "wt"
    worktree.mkdir()
    bin_dir = _install_stub_transport(tmp_path, cgroup_line=_IDLE_CGROUP)
    result = _launch_attached(bin_dir, worktree, "git", "rev-parse", "HEAD")
    assert result.returncode == 0, (
        f"a fast attached probe must still run: {result.stderr}"
    )


def test_the_word_push_alone_does_not_trip_the_git_push_refusal(
    tmp_path: Path,
) -> None:
    """`push` is refused only as `git push` — an over-broad matcher is its own defect."""
    worktree = tmp_path / "wt"
    worktree.mkdir()
    bin_dir = _install_stub_transport(tmp_path, cgroup_line=_IDLE_CGROUP)
    result = _launch_attached(bin_dir, worktree, "echo", "push")
    assert result.returncode == 0, (
        f"'echo push' is not a git push and must not be refused: {result.stderr}"
    )


# ---------------------------------------------------------------------------
# Policy: the shape that caused the incident must not come back.
# ---------------------------------------------------------------------------


def test_the_detached_launch_attaches_no_stdio_and_starts_a_new_session() -> None:
    """`docker exec -d` + `setsid` is what makes the wedge structurally unreachable."""
    text = LAUNCHER.read_text()
    launch_block = text.split("# --- Launch ---", 1)[1]
    assert "exec -d \\" in launch_block, (
        "the detached launch must pass -d so no stdio is attached"
    )
    assert "setsid" in launch_block, "the detached run must start its own session"


def test_the_supervisor_redirects_its_stdio_before_the_payload_can_write() -> None:
    """Property 1: no payload byte may reach an inherited stream, ever."""
    text = SUPERVISOR.read_text()
    redirect_at = text.index('exec >> "${LOG}" 2>&1')
    payload_at = text.index('"${timeout_cmd}" --signal=TERM')
    assert redirect_at < payload_at, (
        "the log redirect must be established before the payload runs, otherwise an "
        "undrained inherited pipe can still wedge the writer (OMN-17317)"
    )


def test_neither_script_depends_on_an_operator_specific_path() -> None:
    """jonah and lakshman must both be able to use this (OMN-17280 contractor chain).

    `~/push-lanes` is mode 0750 on `.201`, so anything reading it is jonah-only
    by construction. All run state lives under the target worktree and the slot
    lock lives inside the container, which is why a second account in the
    `docker` group can use this entry point unchanged.
    """
    for script in (LAUNCHER, SUPERVISOR):
        # Comments are excluded deliberately: both scripts EXPLAIN why
        # `~/push-lanes` is unusable by a second account, and a gate that
        # forbade naming the hazard would push that reasoning out of the file.
        code = "\n".join(
            line
            for line in script.read_text().splitlines()
            if not line.lstrip().startswith("#")
        )
        for forbidden in ("push-lanes", "/Users/", "/home/jonah"):
            assert forbidden not in code, (
                f"{script.name} references the operator-specific path {forbidden!r}"
            )


def test_admission_uses_the_same_threshold_knob_as_the_governed_selector() -> None:
    """One knob, one meaning: `PREPUSH_LOAD_THRESHOLD` at a busy/limit ratio."""
    text = LAUNCHER.read_text()
    assert 'PREPUSH_LOAD_THRESHOLD="${PREPUSH_LOAD_THRESHOLD:-1.0}"' in text
    hook = (REPO_ROOT / "scripts" / "hooks" / "prepush_smart_tests.sh").read_text()
    assert 'PREPUSH_LOAD_THRESHOLD="${PREPUSH_LOAD_THRESHOLD:-1.0}"' in hook, (
        "the governed selector's threshold default moved; the gate-runner entry point "
        "must move with it or the two surfaces silently disagree"
    )
