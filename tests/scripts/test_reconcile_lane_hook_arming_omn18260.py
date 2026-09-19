# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The reconciler arms the lane-identity hook and reads it back (OMN-18260).

THE STATE THIS CLOSES. The stamping hook shipped 2026-09-13 and its installer
was repaired on 2026-09-16 (OMN-18273), after which somebody ran
``lane_identity reconcile --execute`` once, by hand, on one Mac. That is where
it stopped. Nothing in the converged workspace installed it, so its presence on
any host was an artifact of a remembered command -- and a host, a rebuilt clone
or a newly added repository was unarmed with nothing anywhere saying so.

Measured over 2026-09-11..09-18 across six repositories: 39 of 4,105 commits
carried the ``Onex-Lane`` trailer, 0.95%, and 0 carried ``Onex-Fence``. That
number is the ceiling on every downstream gate, because both the pre-push
refusal (OMN-18262) and the pull-request check (OMN-18263) read the lane out of
a commit trailer and an unstamped commit never reaches the comparison at all.

WHAT THESE TESTS PIN, and the order matters:

1. An unarmed clone is NAMED and is verdict-bearing -- the RED case. Silence
   over an unarmed clone is the whole defect; a report that mentioned it while
   still printing IN_SYNC would be the same defect with better prose.
2. The repair ARMS and then READS BACK, and the readback is what decides. An
   arming verb that exits 0 is not evidence that a hook exists (OMN-17307).
3. A sweep that inspected NO clone is a finding, never a pass (CLAUDE.md
   rule 16 -- an empty result is not evidence of absence).
4. A workspace where the question cannot be asked at all says so out loud,
   rather than either failing or falling silent.

The lane-identity module is stubbed here for the same reason ``uv`` is stubbed
in the sibling suites: it ships in another repository, and a test that depended
on the real one would be measuring that repository's checkout rather than this
script's behaviour. What the stub reproduces is the module's CONTRACT -- the
``status`` exit codes 0/1/2 and the ``reconcile --execute`` verb -- which is the
only part of it this script is allowed to know about.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from tests.scripts.test_reconcile_workspace_venvs import (
    _SCRIPT,
    _Workspace,
)

pytestmark = pytest.mark.unit

_EXIT_OK = 0
_EXIT_DRIFT = 1

#: The clone names the stub reports on. Arbitrary, and deliberately not this
#: repository's real clone set: the script must read the answer out of the
#: module rather than knowing a list of its own.
_CLONES = ("omnibase_core", "omnibase_infra", "omniclaude")


def _teach_dispatch_python_to_run_programs(ws: _Workspace) -> None:
    """Let the harness's dispatch interpreter execute a real script.

    The arming leg runs the lane-identity module on the DISPATCH interpreter,
    and the shared harness's stand-in for it is a pure echo -- it answers the
    installed-omnimarket here-doc probe and nothing else. This keeps that
    answer and execs a real interpreter for everything else.
    """
    python = ws.dispatch_venv / "bin" / "python"
    python.write_text(
        "#!/usr/bin/env bash\n"
        'if [[ "${1:-}" == "-" ]]; then\n'
        "  cat >/dev/null 2>&1 || true\n"
        f"  printf '%s\\n' '{ws.market_head}'\n"
        "  exit 0\n"
        "fi\n"
        f'exec {sys.executable} "$@"\n',
        encoding="utf-8",
    )
    python.chmod(0o755)


def _install_lane_identity_stub(ws: _Workspace) -> Path:
    """A stand-in for ``lane_identity.py`` honouring its CLI contract.

    State lives in a JSON file the test writes, so a test moves the world by
    describing it rather than by patching the stub. Every invocation is
    appended to an ordered log, which is how the arm-then-read-back ORDER is
    proven: two separate logs can show that each step ran, only a shared one
    can show which ran first.
    """
    script = ws.root / "omniclaude" / "scripts" / "lane_identity.py"
    script.parent.mkdir(parents=True, exist_ok=True)
    script.write_text(
        '''#!/usr/bin/env python3
"""Stub of omniclaude/scripts/lane_identity.py: the status/reconcile contract."""
import json
import os
import sys

state_path = os.environ["LANE_STUB_STATE"]
log_path = os.environ["LANE_STUB_LOG"]
argv = sys.argv[1:]

with open(log_path, "a", encoding="utf-8") as fh:
    fh.write(" ".join(argv) + "\\n")

with open(state_path, encoding="utf-8") as fh:
    state = json.load(fh)


def emit_status() -> int:
    clones = state["clones"]
    unarmed = [c for c in clones if c in state["unarmed"]]
    for clone in clones:
        mark = "UNARMED " if clone in unarmed else "ARMED   "
        print(f"{mark} {clone}: stub")
    print(
        f"lane_identity: {len(clones) - len(unarmed)}/{len(clones)} clones armed, "
        f"{state.get('registrations', 0)} worktree registration(s), "
        "unregistered policy = silent"
    )
    if not clones:
        print(
            "lane_identity: no clones were checked, which is not the same as none "
            "being unarmed -- refusing to report a pass on an empty sweep "
            "(CLAUDE.md rule 16).",
            file=sys.stderr,
        )
        return 2
    if unarmed:
        print(
            f"lane_identity: {len(unarmed)} clone(s) are NOT armed, so commits "
            "there carry no lane identity and the pre-push refusal has nothing "
            "to compare.",
            file=sys.stderr,
        )
        return 1
    return 0


if argv and argv[0] == "status":
    raise SystemExit(emit_status())

if argv and argv[0] == "reconcile":
    if state.get("arming_works", True):
        state["unarmed"] = []
        with open(state_path, "w", encoding="utf-8") as fh:
            json.dump(state, fh)
    print("lane_identity: stub reconcile")
    raise SystemExit(0 if state.get("arming_exit_ok", True) else 1)

raise SystemExit(2)
''',
        encoding="utf-8",
    )
    script.chmod(0o755)
    return script


def _set_state(
    ws: _Workspace,
    *,
    clones: tuple[str, ...] = _CLONES,
    unarmed: tuple[str, ...] = (),
    arming_works: bool = True,
    arming_exit_ok: bool = True,
) -> None:
    (ws.root / "lane-stub-state.json").write_text(
        json.dumps(
            {
                "clones": list(clones),
                "unarmed": list(unarmed),
                "arming_works": arming_works,
                "arming_exit_ok": arming_exit_ok,
                "registrations": 7,
            }
        ),
        encoding="utf-8",
    )


def _stub_log(ws: _Workspace) -> list[str]:
    path = ws.root / "lane-stub.log"
    if not path.exists():
        return []
    return [line for line in path.read_text(encoding="utf-8").splitlines() if line]


def _run(
    ws: _Workspace, *args: str, with_module: bool = True
) -> subprocess.CompletedProcess[str]:
    env = ws.env()
    # The unowned-CLI-venv census reads $HOME/.claude/plugins/..., which on a
    # developer machine is real host state: without this the verdict under test
    # is decided by whatever that venv happens to hold today.
    fake_home = ws.root / "fakehome"
    fake_home.mkdir(exist_ok=True)
    env["HOME"] = str(fake_home)
    env["LANE_STUB_STATE"] = str(ws.root / "lane-stub-state.json")
    env["LANE_STUB_LOG"] = str(ws.root / "lane-stub.log")
    if with_module:
        env["ONEX_LANE_IDENTITY_SCRIPT"] = str(
            ws.root / "omniclaude" / "scripts" / "lane_identity.py"
        )
    else:
        env["ONEX_LANE_IDENTITY_SCRIPT"] = str(ws.root / "no-such" / "lane_identity.py")
    return subprocess.run(
        ["bash", str(_SCRIPT), *args],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )


@pytest.fixture
def ws(tmp_path: Path) -> _Workspace:
    workspace = _Workspace(tmp_path / "omni_home")
    _teach_dispatch_python_to_run_programs(workspace)
    _install_lane_identity_stub(workspace)
    _set_state(workspace)
    return workspace


# --------------------------------------------------------------------------- #
# RED: an unarmed clone is named, and it moves the verdict
# --------------------------------------------------------------------------- #
def test_check_names_an_unarmed_clone_and_the_verdict_is_drift(ws: _Workspace) -> None:
    """THE DEFECT, stated as a test: silence over a clone with no hook."""
    _set_state(ws, unarmed=("omnibase_core",))

    result = _run(ws, "--check")
    out = result.stdout + result.stderr

    assert "omnibase_core" in out, "the unarmed clone is not named anywhere"
    assert "UNARMED" in out
    assert "lane-identity" in out
    assert result.returncode == _EXIT_DRIFT, (
        "a clone carrying no stamping hook produces commits with no lane, and a "
        f"verdict of {result.returncode} reports that as in sync"
    )
    assert "reconcile --execute" in out, "the remedy is not named"


def test_check_is_in_sync_once_every_clone_is_armed(ws: _Workspace) -> None:
    """GREEN: the same sweep, the same output shape, a clean verdict."""
    _set_state(ws, unarmed=())

    result = _run(ws, "--check")
    out = result.stdout + result.stderr

    assert "3/3 clones armed" in out
    assert "UNARMED" not in out
    assert result.returncode == _EXIT_OK, out


def test_check_mode_arms_nothing(ws: _Workspace) -> None:
    """``--check`` is the read-only probe; it may report, never install."""
    _set_state(ws, unarmed=("omniclaude",))

    _run(ws, "--check")

    assert _stub_log(ws), "the module was never consulted at all"
    assert not [c for c in _stub_log(ws) if c.startswith("reconcile")], (
        "check mode invoked the arming verb; the SessionStart line and every "
        "read-only probe run this mode and must mutate nothing"
    )


# --------------------------------------------------------------------------- #
# The repair arms, and then proves it by reading back
# --------------------------------------------------------------------------- #
def test_the_repair_arms_every_clone_and_then_reads_it_back(ws: _Workspace) -> None:
    _set_state(ws, unarmed=("omnibase_core", "omniclaude"))

    result = _run(ws)
    calls = _stub_log(ws)
    out = result.stdout + result.stderr

    assert any(c.startswith("reconcile") for c in calls), "nothing was armed"
    reconcile_at = next(i for i, c in enumerate(calls) if c.startswith("reconcile"))
    status_after = [i for i, c in enumerate(calls) if c.startswith("status")]
    assert any(i > reconcile_at for i in status_after), (
        "the repair never re-read the surface after arming it, so its report is "
        "the install's own exit status rather than the state on disk"
    )
    assert "3/3 clones armed" in out
    assert result.returncode == _EXIT_OK, out


def test_a_repair_whose_readback_still_finds_an_unarmed_clone_does_not_pass(
    ws: _Workspace,
) -> None:
    """The arming verb exits 0 and changes nothing. The readback must catch it.

    This is OMN-17307's shape in a new surface: a repair that reports its own
    exit status as proof cannot see a write that did not land.
    """
    _set_state(ws, unarmed=("omnibase_core",), arming_works=False)

    result = _run(ws)
    out = result.stdout + result.stderr

    assert "omnibase_core" in out
    assert result.returncode == _EXIT_DRIFT, (
        "the arming verb exited 0 while the hook was still absent, and the "
        f"repair reported {result.returncode}"
    )


def test_an_arming_failure_is_reported_but_the_readback_decides(
    ws: _Workspace,
) -> None:
    """A non-zero arming pass that nonetheless armed everything is not fatal.

    The ledger backfill is the second half of that verb and can fail on a host
    with no claim store while every hook installs correctly. Refusing there
    would red a tick whose actual surface is fine.
    """
    _set_state(ws, unarmed=("omniclaude",), arming_works=True, arming_exit_ok=False)

    result = _run(ws)
    out = result.stdout + result.stderr

    assert "the readback decides" in out
    assert "3/3 clones armed" in out
    assert result.returncode == _EXIT_OK, out


# --------------------------------------------------------------------------- #
# An empty sweep, and a question that cannot be asked
# --------------------------------------------------------------------------- #
def test_a_sweep_that_checked_no_clone_is_a_finding_never_a_pass(
    ws: _Workspace,
) -> None:
    """CLAUDE.md rule 16. Zero rows reads exactly like a clean bill of health."""
    _set_state(ws, clones=(), unarmed=())

    result = _run(ws, "--check")
    out = result.stdout + result.stderr

    assert "empty sweep" in out
    assert result.returncode == _EXIT_DRIFT, out


def test_an_absent_lane_identity_module_is_named_rather_than_silent(
    ws: _Workspace,
) -> None:
    """A deploy runner's clone set does not include omniclaude.

    The question is then not askable, which is neither armed nor a failure --
    but it is still printed, with the path that was looked for, because a
    surface nobody mentions is the failure this leg exists to end.
    """
    result = _run(ws, "--check", with_module=False)
    out = result.stdout + result.stderr

    assert "NOT ASKABLE" in out
    assert "lane_identity.py" in out
    assert result.returncode == _EXIT_OK, (
        "a host that cannot be asked this question is not a host with a defect; "
        "failing here would fire on every deploy runner"
    )


def test_an_absent_module_does_not_stop_the_repair_from_succeeding(
    ws: _Workspace,
) -> None:
    result = _run(ws, with_module=False)

    assert "NOT ASKABLE" in result.stdout + result.stderr
    assert result.returncode == _EXIT_OK
    assert not _stub_log(ws), "the module was invoked despite not being resolvable"


# --------------------------------------------------------------------------- #
# Properties of the code itself
# --------------------------------------------------------------------------- #
def test_the_arming_verb_writes_as_the_surface_owner() -> None:
    """The hook lands inside clones the surface owner owns (OMN-17335).

    A root-owned hook file in a user-owned clone is that ticket's hazard in a
    different directory: the owner's own next reconcile cannot replace it.
    """
    source = _SCRIPT.read_text(encoding="utf-8")
    lines = source.splitlines()
    # The INVOCATION, not a message that quotes it: the remedy text names the
    # same command for a human to run by hand and must not satisfy this.
    arming = [
        i
        for i, line in enumerate(lines)
        if '"$LANE_IDENTITY_SCRIPT" reconcile --execute' in line
        and not line.lstrip().startswith(("#", "say "))
    ]
    assert arming, "no line invokes the arming verb; this test has gone stale"
    for i in arming:
        window = "\n".join(lines[max(0, i - 3) : i + 1])
        assert "as_owner" in window, (
            f"the arming verb does not run through as_owner:\n{window}"
        )


def test_the_reconciler_never_overrides_core_hookspath() -> None:
    """CLAUDE.md rule 17: in a worktree that override is a traceless bypass.

    Arming a hook is exactly the place somebody would reach for it, so the
    absence is pinned rather than left to review.
    """
    source = _SCRIPT.read_text(encoding="utf-8")
    offending = [
        line
        for line in source.splitlines()
        if "core.hooksPath" in line
        and "config" in line
        and not line.lstrip().startswith("#")
    ]
    assert not offending, (
        "the reconciler writes core.hooksPath, which in a worktree silently "
        f"disables every hook: {offending}"
    )


def test_the_module_path_has_no_silent_default_outside_the_workspace() -> None:
    """Resolved from ``$OMNI_HOME``, overridable, never guessed elsewhere."""
    source = _SCRIPT.read_text(encoding="utf-8")
    assert 'LANE_IDENTITY_SCRIPT="${ONEX_LANE_IDENTITY_SCRIPT:-$OMNI_HOME/' in source
