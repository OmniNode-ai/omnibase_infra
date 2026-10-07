# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The clone surface writes as its owner, or refuses (OMN-17366).

THE INCIDENT

`/etc/cron.d/omninode-workspace-reconcile` runs as **root**. Every file under
`.201`'s `/data/omninode` is owned by the operator. `reconcile-host.sh` fetched
each clone and ran the clone delegate in-process, as root, into those
operator-owned trees -- so every hourly tick deposited more root-owned objects.
Counted live on 2026-09-01::

    omnibase_infra 572   omnimarket 261   omnibase_compat 150
    omnibase_core  119   omnibase_spi  16          (1118 total)

The resulting failure is intermittent, which is what makes it expensive: a plain
operator `git fetch` breaks only when it needs to write near an object root
owns::

    error: insufficient permission for adding an object to repository database
    .git/objects
    fatal: failed to write object

So the clone looks healthy until it suddenly does not, and the cause is an hour
of cron ticks in the past rather than anything the operator just did.

THIS IS OMN-17335 ONE SURFACE OVER

OMN-17335 established the rule for the **venv** surface: a mutation runs as the
owner of the surface it writes, via `as_owner`, or it refuses. That fix was
still hypothetical when it landed. This one had already materialised.

The rule is therefore shared, not re-implemented: `scripts/reconcile_privilege_lib.sh`
holds the mechanics and both reconcilers source it. Two copies of a privilege
rule drift, and the half that drifts is the half nobody is watching.

WHY `--check` REFUSES HERE, WHILE THE VENV RECONCILER LETS IT THROUGH

The venv reconciler deliberately exempts `--check` from the ownership rule,
reasoning that a read-only probe writes nothing and that "a read-only probe that
refuses teaches people to stop running it."

That reasoning does not transfer, because **this script's check mode is not
read-only**. It calls `fetch_all` before verdicting -- it must, since a verifier
that takes its target from the thing under verification is not a verifier -- and
`git fetch` writes objects, refs and reflogs. A `--check` that deposits
root-owned objects is the very hazard this ticket is about, so here the guard
covers both modes. `test_check_mode_is_also_refused_because_it_fetches` pins
that divergence so it cannot be "tidied" into agreement with the venv script.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

from tests.scripts.test_reconcile_host_omn17307 import (
    EXIT_INDETERMINATE,
    Workspace,
    _lock,
    _make_clone,
    _run,
    _stub,
    build_workspace,
)

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[2]
_LIB = _REPO_ROOT / "scripts" / "reconcile_privilege_lib.sh"
_GATE = _REPO_ROOT / "scripts" / "check_reconciler_privilege.py"

_FOREIGN = "someone-else"


def _shim_dir(ws: Workspace, name: str = "shims") -> Path:
    d = ws.root.parent / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def _foreign_owner_stat(shims: Path) -> None:
    """A ``stat`` ahead of the real one that reports a foreign owner.

    The only way to model the root-cron-vs-operator split without actually being
    root. Both the GNU and BSD spellings are answered because the library tries
    them in that order.
    """
    real = shutil.which("stat") or "/usr/bin/stat"
    shim = shims / "stat"
    shim.write_text(
        "#!/usr/bin/env bash\n"
        f'if [[ "$1" == "-c" && "$2" == "%U" ]]; then echo {_FOREIGN}; exit 0; fi\n'
        f'if [[ "$1" == "-f" && "$2" == "%Su" ]]; then echo {_FOREIGN}; exit 0; fi\n'
        f'exec {real} "$@"\n',
        encoding="utf-8",
    )
    shim.chmod(0o755)


def _recording_git(shims: Path, witness: Path) -> None:
    """A ``git`` that logs its argv and then behaves exactly like the real one.

    Needed because "it refused" is a weaker claim than "it never wrote". The
    assertion these tests actually care about is that no ``fetch`` reached a
    clone this process does not own.
    """
    real = shutil.which("git") or "/usr/bin/git"
    shim = shims / "git"
    shim.write_text(
        f'#!/usr/bin/env bash\nprintf \'%s\\n\' "$*" >> {witness}\nexec {real} "$@"\n',
        encoding="utf-8",
    )
    shim.chmod(0o755)


def _fake_root(shims: Path, witness: Path) -> None:
    """Make the script believe it is root, with a working ``runuser``.

    ``id -u`` answers 0 and ``id -un`` answers root, because the library reads
    both. ``runuser`` records the user it was asked to become and then runs the
    command as the real (unprivileged) test user -- enough to prove the routing
    without needing privileges to test a privilege drop.
    """
    real_id = shutil.which("id") or "/usr/bin/id"
    (shims / "id").write_text(
        "#!/usr/bin/env bash\n"
        'if [[ "$1" == "-u" ]]; then echo 0; exit 0; fi\n'
        'if [[ "$1" == "-un" ]]; then echo root; exit 0; fi\n'
        f'exec {real_id} "$@"\n',
        encoding="utf-8",
    )
    (shims / "id").chmod(0o755)

    (shims / "runuser").write_text(
        "#!/usr/bin/env bash\n"
        f"printf 'runuser %s\\n' \"$*\" >> {witness}\n"
        "# Drop the `-u <user> --` prefix and run the rest as ourselves.\n"
        "shift 3\n"
        'exec "$@"\n',
        encoding="utf-8",
    )
    (shims / "runuser").chmod(0o755)

    (shims / "getent").write_text(
        "#!/usr/bin/env bash\n"
        f'if [[ "$2" == "{_FOREIGN}" ]]; then echo "{_FOREIGN}:x:1:1::/home/{_FOREIGN}:/bin/sh"; exit 0; fi\n'
        "exit 2\n",
        encoding="utf-8",
    )
    (shims / "getent").chmod(0o755)


def _with_path(shims: Path) -> dict[str, str]:
    """ONLY the PATH override -- never a full copy of ``os.environ``.

    An earlier revision returned the whole environment here, and ``_run_with``
    applies it *after* setting ``OMNI_HOME``. So the ambient ``OMNI_HOME`` won,
    and every test in this file silently reconciled the developer's real
    workspace instead of its tmp_path fixture -- passing for the wrong reason,
    and mutating a tree no test owns. Returning one key makes that impossible.
    """
    return {"PATH": f"{shims}:{os.environ.get('PATH', '/usr/bin:/bin')}"}


def _run_with(
    ws: Workspace, env_extra: dict[str, str], *args: str
) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env.pop("OMNI_HOME", None)
    env["OMNI_HOME"] = str(ws.root)
    env.update(env_extra)
    return subprocess.run(
        ["bash", str(ws.scripts / "reconcile-host.sh"), *args],
        capture_output=True,
        text=True,
        env=env,
        timeout=300,
        check=False,
    )


@pytest.fixture
def clone_ws(tmp_path: Path) -> Workspace:
    """A workspace with one real, manifest-governed clone.

    ``omnibase_core`` rather than an arbitrary name: ``present_clones`` is built
    from the shipped clone manifest, so a directory outside it is invisible to
    the script and the fetch path under test would never be reached.
    """
    ws = build_workspace(tmp_path)
    _make_clone(ws.root, "omnibase_core")
    _lock(ws)
    _stub(
        ws.scripts / "runtime_build" / "reconcile_deploy_clones.sh",
        ws.delegate_witness,
    )
    _stub(ws.scripts / "reconcile-workspace-venvs.sh", ws.delegate_witness)
    return ws


# --------------------------------------------------------------------------- #
# AC1 -- never write as the wrong user
# --------------------------------------------------------------------------- #
def test_a_foreign_owned_surface_is_refused_and_nothing_is_fetched(
    clone_ws: Workspace,
) -> None:
    """RED before the fix: the pre-fix script fetched as root and carried on.

    The refusal is the visible half. The assertion that matters more is the
    second one: no ``fetch`` reached the clone at all, because a fetch is
    exactly what deposited 1118 root-owned paths on `.201`.
    """
    shims = _shim_dir(clone_ws)
    git_witness = clone_ws.root.parent / "git-calls.log"
    _foreign_owner_stat(shims)
    _recording_git(shims, git_witness)

    result = _run_with(clone_ws, _with_path(shims))

    assert result.returncode == EXIT_INDETERMINATE, result.stdout + result.stderr
    assert _FOREIGN in result.stderr
    assert "cannot become that user" in result.stderr

    calls = git_witness.read_text(encoding="utf-8") if git_witness.exists() else ""
    assert "fetch" not in calls, (
        "the reconciler fetched into a clone it does not own; that write is the "
        f"OMN-17366 defect. git calls seen: {calls!r}"
    )
    assert not clone_ws.delegate_witness.exists(), (
        "the clone delegate ran despite the ownership refusal"
    )


def test_check_mode_is_also_refused_because_it_fetches(clone_ws: Workspace) -> None:
    """The deliberate divergence from the venv reconciler's check-mode exemption.

    ``reconcile-host.sh --check`` calls ``fetch_all`` to establish targets, and
    ``git fetch`` writes objects, refs and reflogs. So "check mode writes
    nothing" -- true for the venv surface -- is false here, and the exemption
    must not be copied across. A ``--check`` that deposits root-owned objects is
    the hazard itself.
    """
    shims = _shim_dir(clone_ws)
    git_witness = clone_ws.root.parent / "git-calls.log"
    _foreign_owner_stat(shims)
    _recording_git(shims, git_witness)

    result = _run_with(clone_ws, _with_path(shims), "--check")

    assert result.returncode == EXIT_INDETERMINATE, result.stdout + result.stderr
    calls = git_witness.read_text(encoding="utf-8") if git_witness.exists() else ""
    assert "fetch" not in calls, (
        "--check fetched into a foreign-owned clone; check mode is NOT read-only "
        "on this surface"
    )


def test_root_becomes_the_owner_instead_of_refusing(clone_ws: Workspace) -> None:
    """The `.201` case: root CAN become the operator, so it must -- not refuse.

    Refusing here would leave the host unreconciled forever, which is a worse
    outcome than the bug. The fix is a privilege drop, not a stop.
    """
    shims = _shim_dir(clone_ws)
    runuser_witness = clone_ws.root.parent / "runuser.log"
    _foreign_owner_stat(shims)
    _fake_root(shims, runuser_witness)

    result = _run_with(clone_ws, _with_path(shims))

    assert result.returncode != EXIT_INDETERMINATE, result.stdout + result.stderr
    assert f"writing as {_FOREIGN}" in result.stderr, (
        "the privilege drop must be announced; a silent one cannot be audited "
        "from a cron log"
    )
    recorded = runuser_witness.read_text(encoding="utf-8")
    assert f"-u {_FOREIGN}" in recorded, (
        f"the clone surface did not route through runuser. Saw: {recorded!r}"
    )


def test_owning_the_surface_keeps_the_guard_invisible(clone_ws: Workspace) -> None:
    """The developer-machine case must not regress.

    Owner == current user, so ``RUN_AS`` is empty and nothing shells through
    ``runuser``. The empty-array expansion this relies on is exactly the bash
    detail that breaks under ``set -u`` without a test noticing.
    """
    result = _run(clone_ws)

    assert "cannot become that user" not in result.stderr
    assert "writing as" not in result.stderr


def test_clones_owned_by_different_users_are_indeterminate(
    clone_ws: Workspace,
) -> None:
    """One delegate invocation cannot be two users at once.

    The clone delegate reconciles every clone in a single process, so a split
    ownership set has no correct answer -- running it as either owner writes
    into the other's tree as the wrong user, which is the defect. Refusing is
    the only honest response, and it fails closed.
    """
    shims = _shim_dir(clone_ws)
    real = shutil.which("stat") or "/usr/bin/stat"
    odd_one_out = clone_ws.root / "omnibase_core"
    shim = shims / "stat"
    shim.write_text(
        "#!/usr/bin/env bash\n"
        "# The registry root reads as the current user; one clone does not.\n"
        f'for a in "$@"; do case "$a" in {odd_one_out}*) echo {_FOREIGN}; exit 0 ;; esac; done\n'
        f'exec {real} "$@"\n',
        encoding="utf-8",
    )
    shim.chmod(0o755)

    result = _run_with(clone_ws, _with_path(shims))

    assert result.returncode == EXIT_INDETERMINATE, result.stdout + result.stderr
    assert "omnibase_core" in result.stderr
    assert _FOREIGN in result.stderr


# --------------------------------------------------------------------------- #
# The shared library is shared, not copied
# --------------------------------------------------------------------------- #
def test_both_reconcilers_source_the_one_privilege_library() -> None:
    """OMN-17366's central requirement, asserted structurally.

    The ticket is explicit: route the clone surface through the *same* guard
    rather than inventing a second one. Two implementations of a privilege rule
    drift, and nobody is watching the copy that drifts. This fails if either
    script grows its own ``as_owner``.
    """
    lib_name = _LIB.name
    for script in ("reconcile-host.sh", "reconcile-workspace-venvs.sh"):
        source = (_REPO_ROOT / "scripts" / script).read_text(encoding="utf-8")
        assert lib_name in source, f"{script} does not source {lib_name}"
        assert "as_owner() {" not in source, (
            f"{script} defines its own as_owner instead of using the shared "
            f"{lib_name} -- that is the second implementation OMN-17366 forbids"
        )


# --------------------------------------------------------------------------- #
# AC4 -- the gate is extended, not duplicated
# --------------------------------------------------------------------------- #
def _gate(repo_root: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["python3", str(_GATE), "--repo-root", str(repo_root)],
        capture_output=True,
        text=True,
        check=False,
    )


def test_the_gate_passes_on_the_real_repository() -> None:
    result = _gate(_REPO_ROOT)
    assert result.returncode == 0, result.stdout + result.stderr


def _fixture_repo(tmp_path: Path, host_body: str) -> Path:
    """A minimal repo the gate can scan, carrying a stand-in reconcile-host."""
    root = tmp_path / "repo"
    scripts = root / "scripts"
    scripts.mkdir(parents=True)
    shutil.copy2(
        _REPO_ROOT / "scripts" / "reconcile-workspace-venvs.sh",
        scripts / "reconcile-workspace-venvs.sh",
    )
    shutil.copy2(_LIB, scripts / _LIB.name)
    (scripts / "reconcile-host.sh").write_text(host_body, encoding="utf-8")
    return root


def test_the_gate_rejects_a_clone_fetch_that_skips_the_owner_helper(
    tmp_path: Path,
) -> None:
    """The exact pre-fix line, rejected.

    This is the invocation that deposited 1118 root-owned paths on `.201`.
    """
    root = _fixture_repo(
        tmp_path,
        "#!/usr/bin/env bash\n"
        'source "$SCRIPT_DIR/reconcile_privilege_lib.sh"\n'
        'git -C "$OMNI_HOME/$repo" fetch --quiet --prune origin "$BRANCH"\n',
    )

    result = _gate(root)

    assert result.returncode == 1
    assert "fetch" in result.stderr
    assert "as_owner" in result.stderr


def test_the_gate_rejects_a_clone_delegate_that_skips_the_owner_helper(
    tmp_path: Path,
) -> None:
    """Fetching as the owner while the delegate still runs as root fixes nothing.

    The delegate does its own fetch and checkout, so it is the larger of the two
    write paths. A gate that only covered the in-process fetch would pass the
    script while the damage continued.
    """
    root = _fixture_repo(
        tmp_path,
        "#!/usr/bin/env bash\n"
        'source "$SCRIPT_DIR/reconcile_privilege_lib.sh"\n'
        'as_owner git -C "$OMNI_HOME/$repo" fetch --prune origin "$BRANCH"\n'
        'env OMNI_HOME="$OMNI_HOME" bash "$CLONE_DELEGATE"\n',
    )

    result = _gate(root)

    assert result.returncode == 1
    assert "CLONE_DELEGATE" in result.stderr or "delegate" in result.stderr


def test_the_gate_does_not_flag_a_git_command_quoted_in_a_message(
    tmp_path: Path,
) -> None:
    """Refusals print the command to run by hand; documentation is not invocation.

    Every script in this family names the exact command in its error text, so a
    gate that matched on content alone would flag its own help output -- and a
    gate that cries wolf gets an allowlist bolted on, which is how enforcement
    dies.
    """
    root = _fixture_repo(
        tmp_path,
        "#!/usr/bin/env bash\n"
        'source "$SCRIPT_DIR/reconcile_privilege_lib.sh"\n'
        'as_owner git -C "$c" fetch --prune origin "$BRANCH"\n'
        'say "  run: git -C /data/omninode/omnibase_infra fetch origin dev"\n',
    )

    result = _gate(root)

    assert result.returncode == 0, result.stdout + result.stderr
