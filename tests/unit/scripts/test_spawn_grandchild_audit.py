# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Repo-wide audit of test spawns that can outlive the test (OMN-16995 DoD 2).

`subprocess.Popen(...).kill()` signals ONE process. When the thing spawned is
an intermediary -- a wrapper script, a shell, `flock`, a runner -- the work it
started is a *grandchild*, and killing the intermediary orphans it. That is
exactly how `tests/unit/scripts/test_heavy_lock.py` leaked one
`sh -c while :; do :; done` per run until 19 of them held ~18.6 of the 24
cores on `.200` and the governed pre-push gate refused every heavy escalation
in the lab.

The fix for that test was `start_new_session=True` + `os.killpg`. This file
makes the choice EXPLICIT everywhere else: a test that spawns a process is
either group-spawned, or it is named below with the reason its descendants
cannot become runaways. A new spawn site fails this test until someone makes
that call. The audit is mechanical rather than a paragraph in a PR body
because a paragraph does not survive the next test that spawns a process.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
TESTS = REPO_ROOT / "tests"

pytestmark = pytest.mark.unit

#: Files whose `Popen` sites are NOT group-spawned, each with the reason its
#: descendants cannot outlive the run as a runaway. Reviewed 2026-08-30.
#: Adding a file here is a claim about a real process tree; verify it with
#: `ps -o pid,ppid,args` after an aborted run before you make it.
AUDITED_WITHOUT_PROCESS_GROUPS: dict[str, str] = {
    "tests/ci/test_runner_listener_liveness.py": (
        "the synthetic Runner.Listener is `#!/bin/sh\\nsleep 120` -- a single "
        "self-terminating command with no CPU cost and no grandchild; the "
        "deliberately-orphaned spawns are the OMN-15233 subject under test"
    ),
    "tests/integration/test_runner_listener_liveness_integration.py": (
        "same synthetic `sleep 120` listener: bounded, CPU-free, childless"
    ),
    "tests/scripts/test_runner_job_started_mirror_rewrite.py": (
        "`git daemon` forks a short-lived child per connection and the test "
        "makes at most a handful; every child exits with its connection"
    ),
    "tests/unit/observability/runner_health/test_runner_monitor_auto_bounce.py": (
        "`flock <file> sleep 10` -- the grandchild is a `sleep 10`, bounded "
        "by construction and consuming no CPU"
    ),
    "tests/unit/scripts/test_lane_census_inventory.py": (
        "a python socket server spawned directly: it IS the child, there is "
        "no intermediary and therefore no grandchild"
    ),
    "tests/scripts/test_forward_migration_advisory_lock.py": (
        "integration-only (needs a live postgres): the `/bin/sh <runner>` "
        "grandchildren are `psql` invocations that terminate when their "
        "statement finishes or their connection drops -- none is an "
        "unbounded loop. Group-spawning these is a live upgrade path, not a "
        "runaway risk"
    ),
}


def _popen_sites() -> dict[str, list[tuple[int, bool]]]:
    """Every `subprocess.Popen` call under tests/, with its group-spawn bit."""
    sites: dict[str, list[tuple[int, bool]]] = {}
    for path in sorted(TESTS.rglob("*.py")):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:  # pragma: no cover - would fail collection anyway
            continue
        rel = path.relative_to(REPO_ROOT).as_posix()
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            name = (
                func.attr
                if isinstance(func, ast.Attribute)
                else getattr(func, "id", "")
            )
            if name != "Popen":
                continue
            grouped = any(
                kw.arg == "start_new_session"
                and isinstance(kw.value, ast.Constant)
                and kw.value.value is True
                for kw in node.keywords
            )
            sites.setdefault(rel, []).append((node.lineno, grouped))
    return sites


def test_every_spawning_test_file_is_group_spawned_or_audited() -> None:
    """A new spawn site must make the orphan decision, not inherit it."""
    unaudited = {
        rel: [line for line, grouped in hits if not grouped]
        for rel, hits in _popen_sites().items()
        if any(not grouped for _, grouped in hits)
        and rel not in AUDITED_WITHOUT_PROCESS_GROUPS
    }
    assert unaudited == {}, (
        "subprocess.Popen without start_new_session=True, and not in the "
        "OMN-16995 audit. Either spawn the process group and reap it with "
        "os.killpg (the fix pattern in tests/unit/scripts/test_heavy_lock.py "
        "and tests/integration/docker/conftest.py), or add the file to "
        f"AUDITED_WITHOUT_PROCESS_GROUPS with the reason: {unaudited}"
    )


def test_the_audit_has_no_stale_entries() -> None:
    """An audit entry for a file that no longer spawns is a lie about risk."""
    sites = _popen_sites()
    stale = [
        rel
        for rel in AUDITED_WITHOUT_PROCESS_GROUPS
        if rel not in sites or all(grouped for _, grouped in sites[rel])
    ]
    assert stale == [], f"audit entries no longer describe any spawn site: {stale}"


def test_the_file_that_caused_the_incident_is_fully_group_spawned() -> None:
    """`test_heavy_lock.py` is the reason this audit exists; it stays fixed."""
    sites = _popen_sites()
    heavy = sites.get("tests/unit/scripts/test_heavy_lock.py")
    assert heavy, "the heavy_lock spawn site vanished -- did the file move?"
    assert all(grouped for _, grouped in heavy), (
        "a non-group spawn came back to test_heavy_lock.py, which is how 19 "
        "core-burning orphans reached the .200 gate host (OMN-16995): "
        f"{[line for line, grouped in heavy if not grouped]}"
    )
    assert "tests/unit/scripts/test_heavy_lock.py" not in AUDITED_WITHOUT_PROCESS_GROUPS
