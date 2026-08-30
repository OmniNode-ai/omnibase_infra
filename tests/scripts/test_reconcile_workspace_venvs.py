# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Tests for ``scripts/reconcile-workspace-venvs.sh`` (OMN-17190).

Fully hermetic and offline. Every collaborator the script shells out to is
replaced by a shim on ``PATH`` or by an explicit override variable:

* ``uv``                            -> a shim recording its argv and returning a canned status
* ``install-node-skill-package.sh`` -> ``ONEX_RECONCILE_INSTALL_SCRIPT``
* the installed-omnimarket probe    -> a fake python that echoes a canned commit id
* the canonical omnimarket clone    -> a real local git repo (no network)

What these tests pin down is the *composition* rule, which is the whole reason
this script exists and the one thing no single existing tool gets right:

    the CLI venv has TWO governed layers, and reconciling only the first
    one BREAKS the second.

``omnimarket`` is deliberately absent from ``omnibase_infra``'s
``pyproject.toml`` and ``uv.lock`` (the layer graph is compat -> core -> spi ->
infra and omnimarket sits above infra), so a bare ``uv sync --frozen``
UNINSTALLS it and every ``onex skill`` / ``onex delegate`` dispatch then dies on
the OMN-14060 drift guard. ``--inexact`` is what makes the two layers coexist:
it applies every locked pin without removing what the lock does not mention.

Given that, the ORDER is chosen for a second reason and is asserted directly
here: PROVIDER FIRST, LOCK SECOND. The co-install carries a hardcoded
``COMPAT_PIN`` that downgrades the locked ``omnibase-compat`` (OMN-16262) --
reproduced live on 2026-08-30, where the downgrade broke the ``occ`` CLI
extension badly enough that the ``onex`` binary would not start at all. Ending
on the lock pass is what undoes it.

The provider co-install must pin to the LOCAL clone's ``HEAD``, never to
``origin/dev``: the guard compares against the local clone, so installing from
an unpulled remote tip leaves the venv *ahead* and the guard still refusing.
That is OMN-16366 (reversed drift), and it is asserted here directly.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT = _REPO_ROOT / "scripts" / "reconcile-workspace-venvs.sh"

_EXIT_OK = 0
_EXIT_DRIFT = 1
_EXIT_FAILED = 2
_EXIT_INDETERMINATE = 3

_SHA_LEN = 40


# --------------------------------------------------------------------------- #
# Fixture construction
# --------------------------------------------------------------------------- #
def _git(*args: str, cwd: Path) -> str:
    result = subprocess.run(
        ["git", *args], cwd=cwd, capture_output=True, text=True, check=True
    )
    return result.stdout.strip()


def _make_clone(root: Path, name: str) -> Path:
    """A real (local, network-free) git clone standing in for a canonical repo."""
    repo = root / name
    repo.mkdir(parents=True)
    _git("init", "--quiet", "-b", "dev", cwd=repo)
    _git("config", "user.email", "test@example.com", cwd=repo)
    _git("config", "user.name", "Test", cwd=repo)
    (repo / "f.txt").write_text("one", encoding="utf-8")
    _git("add", "f.txt", cwd=repo)
    _git("commit", "--quiet", "-m", "one", cwd=repo)
    return repo


def _advance(repo: Path, text: str) -> str:
    (repo / "f.txt").write_text(text, encoding="utf-8")
    _git("add", "f.txt", cwd=repo)
    _git("commit", "--quiet", "-m", text, cwd=repo)
    return _git("rev-parse", "HEAD", cwd=repo)


def _make_fake_venv(project: Path, installed_commit: str | None) -> Path:
    """A directory shaped like a venv whose python echoes a canned commit id.

    ``installed_commit`` of ``None`` models "omnimarket is not installed from
    git in this interpreter" -- the absent/PyPI case.
    """
    venv = project / ".venv"
    (venv / "bin").mkdir(parents=True, exist_ok=True)
    python = venv / "bin" / "python"
    emitted = installed_commit or ""
    python.write_text(
        "#!/usr/bin/env bash\n"
        "# Fake interpreter: the reconciler probes the installed omnimarket\n"
        "# commit by piping a here-doc program into it. Ignore the program and\n"
        "# echo the canned answer.\n"
        "cat >/dev/null 2>&1 || true\n"
        f"printf '%s\\n' '{emitted}'\n",
        encoding="utf-8",
    )
    python.chmod(0o755)
    return venv


def _make_uv_shim(bin_dir: Path, *, sync_exit: int = 0, check_exit: int = 0) -> Path:
    """A ``uv`` on PATH that logs argv and returns canned exit statuses.

    ``uv sync --frozen --check`` is the read-only probe (the reconciler adds
    ``--inexact`` for the CLI venv, so that the composed provider layer does not
    read as "extraneous"); ``uv sync --frozen [--inexact]`` is the mutation.
    Both are logged, so a test can assert that ``--check`` mode never invoked
    the mutating form and that the mutating form carried ``--inexact``.

    ``check_exit`` non-zero models a venv that does not satisfy its lock -- the
    only lock-conformance signal the reconciler trusts, since uv is the
    authority on that question and a self-computed stamp is not.
    """
    bin_dir.mkdir(parents=True, exist_ok=True)
    uv = bin_dir / "uv"
    uv.write_text(
        "#!/usr/bin/env bash\n"
        'printf "%s\\n" "$*" >> "$UV_SHIM_LOG"\n'
        'printf "uv %s\\n" "$*" >> "$ORDER_LOG"\n'
        'for a in "$@"; do\n'
        '  if [[ "$a" == "--check" ]]; then exit ' + str(check_exit) + "; fi\n"
        "done\n"
        "exit " + str(sync_exit) + "\n",
        encoding="utf-8",
    )
    uv.chmod(0o755)
    return uv


def _make_install_shim(path: Path, *, exit_code: int = 0) -> Path:
    """Stand-in for ``install-node-skill-package.sh``.

    Logs ``OMNIMARKET_REF`` so a test can prove which ref the reconciler pinned
    the provider layer to -- the OMN-16366 assertion.
    """
    path.write_text(
        "#!/usr/bin/env bash\n"
        'printf "%s\\n" "${OMNIMARKET_REF:-<unset>}" >> "$INSTALL_SHIM_LOG"\n'
        'printf "install %s\\n" "${OMNIMARKET_REF:-<unset>}" >> "$ORDER_LOG"\n'
        "exit " + str(exit_code) + "\n",
        encoding="utf-8",
    )
    path.chmod(0o755)
    return path


class _Workspace:
    """An ``$OMNI_HOME``-shaped tree with both governed venvs and every shim."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.omnimarket = _make_clone(root, "omnimarket")
        self.infra = root / "omnibase_infra"
        (self.infra / "scripts").mkdir(parents=True)
        (self.infra / "uv.lock").write_text("lock-v1\n", encoding="utf-8")
        self.omniclaude = root / "omniclaude"
        self.omniclaude.mkdir(parents=True)
        (self.omniclaude / "uv.lock").write_text("claude-lock-v1\n", encoding="utf-8")

        self.market_head = _git("rev-parse", "HEAD", cwd=self.omnimarket)
        self.infra_venv = _make_fake_venv(self.infra, self.market_head)
        _make_fake_venv(self.omniclaude, None)

        self.bin_dir = root / "shimbin"
        self.uv_log = root / "uv.log"
        self.install_log = root / "install.log"
        # One interleaved log across BOTH shims. Two separate logs can prove
        # that each collaborator ran; only a shared one can prove the ORDER,
        # and the order is the OMN-16262 fix.
        self.order_log = root / "order.log"
        self.install_script = self.infra / "scripts" / "install-node-skill-package.sh"
        _make_install_shim(self.install_script)
        _make_uv_shim(self.bin_dir)

    def set_installed_commit(self, commit: str | None) -> None:
        _make_fake_venv(self.infra, commit)

    def env(self) -> dict[str, str]:
        return {
            **os.environ,
            "OMNI_HOME": str(self.root),
            "PATH": f"{self.bin_dir}:{os.environ['PATH']}",
            "UV_SHIM_LOG": str(self.uv_log),
            "INSTALL_SHIM_LOG": str(self.install_log),
            "ORDER_LOG": str(self.order_log),
            "ONEX_RECONCILE_INSTALL_SCRIPT": str(self.install_script),
            # Keep the hook-venv surface deterministic: the plugin-data venv is
            # host state, not workspace state, and must not leak into a test.
            "CLAUDE_PLUGIN_DATA": str(self.root / "no-such-plugin-data"),
        }

    def run(self, *args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["bash", str(_SCRIPT), *args],
            capture_output=True,
            text=True,
            env=self.env(),
            check=False,
        )

    def uv_calls(self) -> list[str]:
        if not self.uv_log.exists():
            return []
        return [
            line
            for line in self.uv_log.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    def ordered_calls(self) -> list[tuple[str, str]]:
        """Every collaborator invocation, in the order it happened."""
        if not self.order_log.exists():
            return []
        out: list[tuple[str, str]] = []
        for line in self.order_log.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            kind, _, rest = line.partition(" ")
            out.append((kind, rest))
        return out

    def install_refs(self) -> list[str]:
        if not self.install_log.exists():
            return []
        return [
            line
            for line in self.install_log.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]


@pytest.fixture
def ws(tmp_path: Path) -> _Workspace:
    return _Workspace(tmp_path / "omni_home")


# --------------------------------------------------------------------------- #
# The script must exist and be executable at all
# --------------------------------------------------------------------------- #
def test_script_exists_and_is_executable() -> None:
    assert _SCRIPT.is_file(), f"missing reconciler: {_SCRIPT}"
    assert os.access(_SCRIPT, os.X_OK), f"reconciler not executable: {_SCRIPT}"


# --------------------------------------------------------------------------- #
# Fail-fast configuration (CLAUDE.md rule 8 -- no silent default)
# --------------------------------------------------------------------------- #
def test_unset_omni_home_is_indeterminate_and_names_the_variable(
    ws: _Workspace,
) -> None:
    env = ws.env()
    env.pop("OMNI_HOME")
    result = subprocess.run(
        ["bash", str(_SCRIPT), "--check"],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    assert result.returncode == _EXIT_INDETERMINATE
    assert "OMNI_HOME" in result.stdout + result.stderr


# --------------------------------------------------------------------------- #
# --check is read-only
# --------------------------------------------------------------------------- #
def test_check_mode_never_mutates(ws: _Workspace) -> None:
    """``--check`` may probe, but must never sync or co-install."""
    ws.set_installed_commit("0" * _SHA_LEN)  # force a DRIFT verdict
    result = ws.run("--check")

    assert result.returncode == _EXIT_DRIFT
    assert ws.install_refs() == [], "check mode ran the provider co-install"
    mutating = [c for c in ws.uv_calls() if "--check" not in c]
    assert mutating == [], f"check mode ran a mutating uv command: {mutating}"


def test_check_reports_drift_when_venv_is_behind_the_clone(ws: _Workspace) -> None:
    """The stale-venv case: the clone advanced, the venv did not."""
    ws.run()  # start from a reconciled venv
    new_head = _advance(ws.omnimarket, "two")
    assert new_head != ws.market_head

    result = ws.run("--check")
    assert result.returncode == _EXIT_DRIFT
    assert new_head[:12] in result.stdout


def test_check_is_clean_immediately_after_a_reconcile(ws: _Workspace) -> None:
    assert ws.run().returncode == _EXIT_OK
    assert ws.run("--check").returncode == _EXIT_OK


# --------------------------------------------------------------------------- #
# The composition rule: BOTH layers, in order
# --------------------------------------------------------------------------- #
def test_reconcile_applies_the_lock_without_removing_the_provider_layer(
    ws: _Workspace,
) -> None:
    """The CLI venv's lock pass MUST be ``--inexact``.

    Without it, uv removes every package the lock does not mention -- which is
    omnimarket and its eleven companions -- and the next dispatch dies on the
    OMN-14060 guard. This is the single assertion that keeps the two layers from
    destroying each other.
    """
    ws.set_installed_commit("0" * _SHA_LEN)
    result = ws.run()
    assert result.returncode == _EXIT_OK, result.stdout + result.stderr

    cli_syncs = [
        c
        for c in ws.uv_calls()
        if "sync" in c and "--check" not in c and "omnibase_infra" in c
    ]
    assert cli_syncs, "reconcile never applied the CLI venv's lock"
    for call in cli_syncs:
        assert "--frozen" in call, (
            "lock sync must be --frozen: re-resolving would silently move the "
            f"pins the lock exists to hold. Call: {call!r}"
        )
        assert "--inexact" in call, (
            "lock sync without --inexact UNINSTALLS omnimarket and bricks every "
            f"`onex` dispatch. Call: {call!r}"
        )


def test_provider_coinstall_runs_before_the_lock_pass(ws: _Workspace) -> None:
    """OMN-16262: the co-install downgrades a locked pin, so the lock pass ends the run.

    ``install-node-skill-package.sh`` carries a hardcoded
    ``COMPAT_PIN="omnibase-compat==0.5.5"`` and installs it ``--no-deps``,
    silently downgrading the locked 0.5.6. Reproduced live on this Mac
    2026-08-30 by this very reconciler: the downgrade removed
    ``omnibase_compat.contracts.pr_occ_stamp``, the ``occ`` CLI extension failed
    to load, and the ``onex`` binary would not start at all. Ending on the lock
    pass repairs it structurally instead of inheriting the bug.
    """
    ws.set_installed_commit("0" * _SHA_LEN)
    assert ws.run().returncode == _EXIT_OK

    order = ws.ordered_calls()
    provider_at = next(i for i, (kind, _) in enumerate(order) if kind == "install")
    lock_after = [
        i
        for i, (kind, call) in enumerate(order)
        if kind == "uv" and "--check" not in call and "omnibase_infra" in call
    ]
    assert lock_after, "no CLI lock pass ran at all"
    assert max(lock_after) > provider_at, (
        "the provider co-install ran last, so a pin it downgraded stays "
        f"downgraded. Order was: {order!r}"
    )


def test_provider_layer_is_pinned_to_local_clone_head_not_remote_tip(
    ws: _Workspace,
) -> None:
    """OMN-16366: pinning to origin/dev leaves reversed drift the guard still refuses."""
    head = _advance(ws.omnimarket, "two")
    ws.set_installed_commit("0" * _SHA_LEN)

    assert ws.run().returncode == _EXIT_OK
    assert ws.install_refs() == [head], (
        f"provider layer pinned to {ws.install_refs()!r}, expected the local "
        f"clone HEAD {head!r}"
    )


def test_reconcile_is_idempotent(ws: _Workspace) -> None:
    """A second consecutive run is a clean no-op, not a second install."""
    assert ws.run().returncode == _EXIT_OK
    first = len(ws.install_refs())

    assert ws.run().returncode == _EXIT_OK
    assert len(ws.install_refs()) == first, (
        "second reconcile re-ran the provider co-install despite nothing having "
        "moved -- not idempotent"
    )


def test_lock_nonconformance_alone_is_drift(ws: _Workspace) -> None:
    """The clone can be still while the venv falls behind its lock.

    uv is the authority on that question and the reconciler asks it, rather
    than stamping a hash of its own. A stamp records what a previous run
    BELIEVED; it cannot see a package mutated in place afterwards, which is
    exactly how the OMN-15620 cross-repo pollution went unnoticed.
    """
    assert ws.run().returncode == _EXIT_OK
    _make_uv_shim(ws.bin_dir, check_exit=1)
    assert ws.run("--check").returncode == _EXIT_DRIFT


# --------------------------------------------------------------------------- #
# Failure surfaces name the exact command
# --------------------------------------------------------------------------- #
def test_lock_sync_failure_refuses_and_names_the_exact_command(
    ws: _Workspace,
) -> None:
    _make_uv_shim(ws.bin_dir, sync_exit=1, check_exit=1)
    result = ws.run()
    combined = result.stdout + result.stderr

    assert result.returncode == _EXIT_FAILED
    assert "uv sync --frozen --inexact" in combined, (
        "a sync failure must print the exact command to re-run by hand, "
        f"--inexact included; got: {combined!r}"
    )


def test_provider_coinstall_failure_refuses_and_names_the_exact_command(
    ws: _Workspace,
) -> None:
    _make_install_shim(ws.install_script, exit_code=1)
    ws.set_installed_commit("0" * _SHA_LEN)
    result = ws.run()
    combined = result.stdout + result.stderr

    assert result.returncode == _EXIT_FAILED
    assert "install-node-skill-package.sh" in combined


def test_refusal_names_no_bypass_environment_variable(ws: _Workspace) -> None:
    """A sync failure is a real failure. There is no 'proceed anyway' switch here.

    The OMN-13930 override exists on the *guard* for the case where an operator
    knowingly accepts unverified results. A reconcile that cannot complete is a
    different thing: the venv is broken, and offering a bypass would just move
    the breakage to the next dispatch.
    """
    _make_uv_shim(ws.bin_dir, sync_exit=1, check_exit=1)
    result = ws.run()
    combined = result.stdout + result.stderr

    # Anchor on a real refusal first, so this can never pass vacuously against
    # an empty output (it did, while the script was still absent).
    assert result.returncode == _EXIT_FAILED
    assert "FAILED" in combined
    assert "ONEX_ALLOW" not in combined
    assert "=1 to" not in combined


# --------------------------------------------------------------------------- #
# The hook venv is a second surface, resolved live -- not from the stale doc path
# --------------------------------------------------------------------------- #
def test_hook_venv_is_reconciled_against_its_own_lock(ws: _Workspace) -> None:
    """`omniclaude/.venv` is the venv that actually executes hooks on this host.

    CLAUDE.md rule 11 and the memory record both name paths that do not exist
    here (`omniclaude/plugins/onex/lib/.venv`,
    `~/.claude/plugins/data/onex-omninode-tools/.venv`), so the reconciler must
    resolve the hook venv by probing rather than by trusting the documented path.
    """
    assert ws.run().returncode == _EXIT_OK
    synced_projects = [c for c in ws.uv_calls() if "--project" in c]
    assert any("omniclaude" in c for c in synced_projects), (
        f"the hook venv surface was never reconciled; uv calls were: {ws.uv_calls()!r}"
    )


def test_absent_hook_venv_is_skipped_not_failed(ws: _Workspace) -> None:
    """Never *create* a venv that isn't there -- that is repair-plugin-venv.sh's job."""
    import shutil

    shutil.rmtree(ws.omniclaude / ".venv")
    result = ws.run()
    assert result.returncode == _EXIT_OK, result.stdout + result.stderr
    assert "SKIP" in result.stdout.upper()


def test_clone_movement_moves_the_provider_layer_and_then_reapplies_the_lock(
    ws: _Workspace,
) -> None:
    """The common case: the clone advanced, the lock did not.

    Even here the lock pass must follow, because the co-install that just ran
    is the thing that can downgrade a locked pin (OMN-16262). It is cheap --
    `--inexact` against an already-conformant venv installs nothing -- and it is
    the only thing standing between a routine tick and a broken `onex` binary.
    """
    assert ws.run().returncode == _EXIT_OK

    head = _advance(ws.omnimarket, "two")
    ws.set_installed_commit(ws.market_head)  # venv still on the old commit

    assert ws.run().returncode == _EXIT_OK
    assert ws.install_refs()[-1] == head, "provider layer was not moved to the new HEAD"

    order = ws.ordered_calls()
    provider_at = max(i for i, (kind, _) in enumerate(order) if kind == "install")
    assert any(
        kind == "uv" and "--check" not in call and "omnibase_infra" in call
        for kind, call in order[provider_at:]
    ), (
        "the provider layer moved and no lock pass followed, so an OMN-16262 "
        f"downgrade would survive the tick. Order was: {order!r}"
    )
