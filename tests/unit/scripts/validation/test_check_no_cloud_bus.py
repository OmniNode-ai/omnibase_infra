# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Tests for the OMN-19623 no-cloud-bus here-string deadlock fix.

## Background

``check_no_cloud_bus.sh`` used to feed its violation-reporting loop from a bare
here-string: ``done <<< "$matches"`` where ``$matches`` holds unbounded ``git
grep`` output. Bash 5.1+ writes a here-document/here-string body into a pipe
*before* forking the reader, falling back to a temp file only above ~64KB.
Under macOS pipe-KVA pressure (many concurrent shells -- e.g. many fleet agent
lanes committing at once) the kernel caps a freshly created pipe's buffer well
below the size bash assumes, and the write blocks forever with nothing left to
drain it: no child process, 0% CPU, the same bash holding both the read and
write end of one pipe. This exact signature (state S, 0% CPU, no children,
both pipe fds open per ``lsof``, sampled blocked in ``heredoc_write``->
``write``) was independently sampled three times on this fleet
(``docs/tracking/ROLLING_WORK_LEDGER.md`` FRICTION rows :4295, :4311, :4330)
and matches a documented upstream bash defect (bash-1.1+ heredoc/here-string
pipe race; fixed upstream in the devel branch after bug-bash reports).

## Why this file does not assert a bare-metal deterministic hang

The trigger is a *kernel* condition (macOS's per-pipe buffer shrinking once
system-wide pipe-KVA usage crosses an internal threshold), not a fixed input
size: on a quiescent host the same construct with the same payload does not
hang at all, and the fleet's own incident reports independently recorded
"not reproduced standalone" for the identical defect.

During triage for this fix, concurrent invocation of the vulnerable
construct (200-300 copies at once, on this fleet's own shared host) did
reproduce the hang at a high rate in some runs -- but the rate swung from
0% to over 90% between successive runs with no change to the script, driven
by transient, uncontrollable ambient load from unrelated processes sharing
the machine. At high enough concurrency (150 copies), even the FIXED script
(which no longer touches a pipe at all -- it reads from a real temp file)
measured a 100% timeout rate on this same host, because at that scale the
dominant effect is ordinary fork/scheduling contention, not the specific
bash pipe defect. A "run N copies and assert one times out" test therefore
cannot distinguish the specific defect from generic overload on a shared
host, would be flaky by construction, and -- worse -- can fail on the FIXED
script and look like a regression that isn't one. Rather than ship a test
that produces a demonstrated false signal, this file deliberately omits a
concurrency-based hang assertion.

What anchors this fix instead:

1. ``test_gate_does_not_feed_a_while_read_loop_via_here_string_or_heredoc``: a
   deterministic static ratchet (same technique as the sibling OMN-14988
   ratchet in ``test_check_no_infra_inmemory_import.py``) that fails the
   instant the dangerous construct reappears, on every platform, with zero
   flakiness.
2. The process-level evidence already on record: three independent live
   incidents on this fleet (ledger FRICTION rows :4295, :4311, :4330), each
   sampled via ``ps``/``lsof`` showing the same signature (state S, 0% CPU,
   no children, both pipe fds held by the one process, blocked in
   ``heredoc_write``->``write``), matching a documented upstream bash defect
   (bash 5.1+ heredoc/here-string-via-pipe race, fixed upstream in the devel
   branch after bug-bash reports) rather than a hypothesis invented for this
   ticket.
3. The mechanism itself: a regular file, opened via ``open(2)``, never goes
   through bash's pipe-based heredoc/here-string code path at all, so the
   fix removes the vulnerable code path unconditionally -- it does not need
   to depend on reproducing the specific kernel condition to be correct.

The remaining tests prove the fix carries no functional regression: the
temp-file-fed loop must detect the same violations, honor the same
suppression marker, and preserve the same exit codes as before.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

from omnibase_core.validators.no_unguarded_git_subprocess import (
    scrub_git_location_env,
)

REPO_ROOT = Path(__file__).resolve().parents[4]
SCRIPT = REPO_ROOT / "scripts" / "validation" / "check_no_cloud_bus.sh"


def _git(args: list[str], cwd: Path) -> None:
    """Run a git command with the OMN-18434 canonical env scrub.

    Without this, git's own exported ``GIT_DIR``/``GIT_WORK_TREE``/
    ``GIT_INDEX_FILE`` (present when this suite itself runs inside a git
    hook) would override ``cwd=`` and retarget the command at the real
    invoking worktree instead of the fixture repo.
    """
    subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=True,
        capture_output=True,
        env=scrub_git_location_env(),
    )


def _bash_interpreters() -> list[str]:
    """Every distinct bash on this machine (mirrors the sibling OMN-14988 test)."""
    candidates = [
        "/bin/bash",
        shutil.which("bash"),
        "/opt/homebrew/bin/bash",
        "/usr/local/bin/bash",
    ]
    seen: set[str] = set()
    found: list[str] = []
    for candidate in candidates:
        if not candidate:
            continue
        real = os.path.realpath(candidate)
        if real in seen or not os.access(real, os.X_OK):
            continue
        seen.add(real)
        found.append(candidate)
    return found


BASH_INTERPRETERS = _bash_interpreters()


def _init_git_repo(root: Path) -> None:
    _git(["init", "-q"], cwd=root)
    _git(["config", "user.email", "test@example.com"], cwd=root)
    _git(["config", "user.name", "Test"], cwd=root)


def _run_gate(
    repo: Path,
    *,
    bash: str = "bash",
    timeout: float | None = 30,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [bash, str(SCRIPT), str(repo)],
        text=True,
        capture_output=True,
        check=False,
        timeout=timeout,
    )


# --------------------------------------------------------------------------
# Functional equivalence: the temp-file-fed loop must behave exactly like
# the original here-string-fed loop.
# --------------------------------------------------------------------------


def test_gate_exits_zero_on_clean_tree(tmp_path: Path) -> None:
    _init_git_repo(tmp_path)
    f = tmp_path / "module.py"
    f.write_text("x = 1\n", encoding="utf-8")
    _git(["add", "-A"], cwd=tmp_path)

    result = _run_gate(tmp_path)

    assert result.returncode == 0
    assert "VIOLATION" not in result.stdout


def test_gate_still_detects_unsuppressed_violations(tmp_path: Path) -> None:
    _init_git_repo(tmp_path)
    f = tmp_path / "module.py"
    f.write_text(
        "BROKER = 'redpanda:29092'  # no suppression here\n",
        encoding="utf-8",
    )
    _git(["add", "-A"], cwd=tmp_path)

    result = _run_gate(tmp_path)

    assert result.returncode == 1
    assert "VIOLATION: module.py:1:" in result.stdout
    assert "1 unsuppressed cloud bus (29092) reference(s)" in result.stdout


def test_gate_allows_suppressed_and_clean_input(tmp_path: Path) -> None:
    _init_git_repo(tmp_path)
    f = tmp_path / "module.py"
    f.write_text(
        "BROKER = 'redpanda:29092'  # cloud-bus-ok OMN-3752\n",
        encoding="utf-8",
    )
    _git(["add", "-A"], cwd=tmp_path)

    result = _run_gate(tmp_path)

    assert result.returncode == 0
    assert "VIOLATION" not in result.stdout


def test_gate_detects_many_violations_across_many_files(tmp_path: Path) -> None:
    """A payload shaped like the real fleet incidents: dozens of hits across
    many files, well past the single-line case, exercising the loop over a
    multi-KB match block the way real git-grep output does."""
    _init_git_repo(tmp_path)
    n = 60
    for i in range(n):
        f = tmp_path / f"pkg_{i}" / f"module_{i}.py"
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(f"BROKER_{i} = 'redpanda:29092'  # line {i}\n", encoding="utf-8")
    _git(["add", "-A"], cwd=tmp_path)

    result = _run_gate(tmp_path)

    assert result.returncode == 1
    assert f"Found {n} unsuppressed cloud bus (29092) reference(s)." in result.stdout
    for i in range(n):
        assert f"module_{i}.py:1:" in result.stdout


@pytest.mark.parametrize("bash", BASH_INTERPRETERS)
def test_gate_behaves_identically_across_every_bash_on_this_host(
    tmp_path: Path, bash: str
) -> None:
    _init_git_repo(tmp_path)
    (tmp_path / "clean.py").write_text("x = 1\n", encoding="utf-8")
    (tmp_path / "dirty.py").write_text("BROKER = 'redpanda:29092'\n", encoding="utf-8")
    _git(["add", "-A"], cwd=tmp_path)

    result = _run_gate(tmp_path, bash=bash)

    assert result.returncode == 1, f"{bash}: {result.stdout}{result.stderr}"
    assert "dirty.py:1:" in result.stdout


# --------------------------------------------------------------------------
# OMN-19623: static ratchet against the here-string/heredoc-into-while-read
# construct that deadlocks under bash 5.1+ pipe pressure.
# --------------------------------------------------------------------------


def test_gate_does_not_feed_a_while_read_loop_via_here_string_or_heredoc() -> None:
    """RED against the pre-fix script on every platform, no reproduction of
    the kernel race required. A `while ... read` compound command redirected
    from `<<<` or `<<` goes through bash's pipe-based heredoc/here-string
    machinery regardless of payload size (the deadlock is only sometimes
    live, per the module docstring, but the vulnerable *construct* is always
    present until this ratchet is enforced). The fix must read from a real
    file (`< "$tmp"`) instead.
    """
    executable_lines = [
        line
        for line in SCRIPT.read_text(encoding="utf-8").splitlines()
        if not line.lstrip().startswith("#")
    ]
    offending = [
        line
        for line in executable_lines
        if ("<<<" in line or "<<" in line) and "done" in line
    ]
    assert not offending, (
        "OMN-19623: a `done <<<`/`done <<` here-string/heredoc feed is back on "
        f"a while-read loop; this deadlocks under bash 5.1+ pipe pressure. "
        f"Feed the loop from a temp file instead: {offending}"
    )


def test_gate_reads_the_match_loop_from_a_real_file() -> None:
    """Positive assertion companion to the ratchet above: the fix is present,
    not merely the absence of the old construct."""
    content = SCRIPT.read_text(encoding="utf-8")
    assert "mktemp" in content
    assert 'done < "$tmp_matches"' in content
