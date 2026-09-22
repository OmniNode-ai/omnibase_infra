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
import shutil
import subprocess
from pathlib import Path

import pytest

from omnibase_core.validators.no_unguarded_git_subprocess import (
    scrub_git_location_env,
)

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
        ["git", *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=True,
        # Git exports GIT_DIR / GIT_WORK_TREE into every hook environment and
        # those OVERRIDE both `cwd=` and `git -C`. Under a pre-push hook an
        # unscrubbed fixture would init and commit into the REAL invoking
        # worktree instead of tmp_path (OMN-14891/OMN-18434).
        env=scrub_git_location_env(),
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


def _make_fake_venv(venv: Path, installed_commit: str | None) -> Path:
    """A directory shaped like a venv whose python echoes a canned commit id.

    ``installed_commit`` of ``None`` models "omnimarket is not installed from
    git in this interpreter" -- the absent/PyPI case.

    Takes the venv path directly rather than a project directory: after the
    OMN-17819 gate/dispatch split the two governed venvs no longer share a
    parent, and one of them is deliberately not inside any project.
    """
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


def _make_uv_shim(
    bin_dir: Path,
    *,
    sync_exit: int = 0,
    check_exit: int = 0,
    venv_relocatable: bool = True,
) -> Path:
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
        # `uv venv` is a DIFFERENT subcommand with a DIFFERENT calling
        # convention: the target directory is a positional argument, not
        # UV_PROJECT_ENVIRONMENT. The reconciler uses it for one reason -- only
        # `uv venv` accepts --relocatable, and an atomic rebuild is renamed into
        # place, which a venv with absolute console-script shebangs does not
        # survive. Measured against real uv 0.11.32: `uv sync` does not honour
        # UV_VENV_RELOCATABLE and writes absolute shebangs.
        'if [[ "${1:-}" == "venv" ]]; then\n'
        "  _reloc=0\n"
        '  _vbase=""\n'
        '  _prev=""\n'
        '  for a in "$@"; do\n'
        '    [[ "$a" == "--relocatable" ]] && _reloc=1\n'
        '    [[ "$_prev" == "--python" ]] && _vbase="$a"\n'
        '    _prev="$a"\n'
        "  done\n"
        '  _vdir="${!#}"\n'
        '  mkdir -p "$_vdir/bin"\n'
        # A staged venv has no omnimarket in it yet, which is what the provider
        # co-install is for. Echoing an empty commit models exactly that.
        '  printf \'#!/usr/bin/env bash\\ncat >/dev/null 2>&1 || true\\nprintf "%s\\\\n" ""\\n\' > "$_vdir/bin/python"\n'
        '  chmod 0755 "$_vdir/bin/python"\n'
        '  _d="$_vbase"\n'
        '  while [[ -L "$_d" ]]; do\n'
        '    _l="$(readlink "$_d")"\n'
        '    case "$_l" in /*) _d="$_l" ;; *) _d="${_d%/*}/$_l" ;; esac\n'
        "  done\n"
        '  { printf "home = %s\\n" "$(cd "${_d%/*}" && pwd -P)"\n'
        + ('    printf "relocatable = true\\n"\n' if venv_relocatable else "")
        + '  } > "$_vdir/pyvenv.cfg"\n'
        '  printf "venv %s\\n" "$*" >> "$UV_SHIM_LOG"\n'
        '  printf "uvvenv %s\\n" "$*" >> "$ORDER_LOG"\n'
        "  exit 0\n"
        "fi\n"
        # `uv sync` CREATES the environment when it is absent and RECREATES it
        # when --python names a different interpreter. The shim models that much
        # so a test can read the resulting pyvenv.cfg `home` line -- the same
        # line the reconciler itself reads, rather than a parallel assertion
        # about argv that would pass while the venv stayed wrong (OMN-17819).
        'if [[ -n "${UV_PROJECT_ENVIRONMENT:-}" ]]; then\n'
        '  _base=""\n'
        '  _prev=""\n'
        '  for a in "$@"; do\n'
        '    [[ "$_prev" == "--python" ]] && _base="$a"\n'
        '    _prev="$a"\n'
        "  done\n"
        '  if [[ -n "$_base" ]]; then\n'
        '    mkdir -p "${UV_PROJECT_ENVIRONMENT}/bin"\n'
        # Record the RESOLVED directory, as real uv does: it writes the
        # interpreter's own location (the Cellar/opt spelling), not the
        # directory of the path it was handed. A shim that recorded the flag's
        # dirname would model a uv that does not exist and would make a
        # correctly-rebuilt venv read as drift.
        '    _d="$_base"\n'
        '    while [[ -L "$_d" ]]; do\n'
        '      _l="$(readlink "$_d")"\n'
        '      case "$_l" in /*) _d="$_l" ;; *) _d="${_d%/*}/$_l" ;; esac\n'
        "    done\n"
        # PRESERVE an existing `relocatable = true`, which is what real uv
        # does: a sync into a relocatable venv leaves it relocatable and writes
        # self-locating shebangs for what it installs. A shim that dropped the
        # line would model a uv that does not exist, and would make the
        # pre-swap readback refuse every correctly staged rebuild.
        '    _keep=""\n'
        '    if [[ -f "${UV_PROJECT_ENVIRONMENT}/pyvenv.cfg" ]] && \\\n'
        '       grep -q "^relocatable = true$" "${UV_PROJECT_ENVIRONMENT}/pyvenv.cfg"; then\n'
        '      _keep="relocatable = true"\n'
        "    fi\n"
        '    { printf "home = %s\\n" "$(cd "${_d%/*}" && pwd -P)"\n'
        '      [[ -n "$_keep" ]] && printf "%s\\n" "$_keep"\n'
        '    } > "${UV_PROJECT_ENVIRONMENT}/pyvenv.cfg"\n'
        "  fi\n"
        "fi\n"
        # UV_PROJECT_ENVIRONMENT is how uv is told WHICH venv a project sync
        # targets, and after OMN-17819 that is the only thing distinguishing the
        # dispatch-venv sync from the gate-venv sync -- both carry the same
        # `--project <infra>`. Logging argv alone cannot tell them apart.
        'printf "env=%s %s\\n" "${UV_PROJECT_ENVIRONMENT:--}" "$*" >> "$UV_SHIM_LOG"\n'
        'printf "uv env=%s %s\\n" "${UV_PROJECT_ENVIRONMENT:--}" "$*" >> "$ORDER_LOG"\n'
        'for a in "$@"; do\n'
        '  if [[ "$a" == "--check" ]]; then exit ' + str(check_exit) + "; fi\n"
        "done\n"
        "exit " + str(sync_exit) + "\n",
        encoding="utf-8",
    )
    uv.chmod(0o755)
    return uv


def _make_content_verifier(infra: Path, *, exit_code: int = 0) -> Path:
    """Stub only the source-manifest check for delegate shell-order tests."""
    verifier = infra / "scripts" / "reconcile_verify_movement.py"
    verifier.write_text(
        "import sys\n"
        "if len(sys.argv) < 2 or sys.argv[1] != 'candidate-content-verify':\n"
        "    raise SystemExit(91)\n"
        f"raise SystemExit({exit_code})\n",
        encoding="utf-8",
    )
    return verifier


def _make_install_shim(path: Path, *, exit_code: int = 0) -> Path:
    """Stand-in for ``install-node-skill-package.sh``.

    Logs ``OMNIMARKET_REF`` so a test can prove which ref the reconciler pinned
    the provider layer to -- the OMN-16366 assertion.
    """
    path.write_text(
        "#!/usr/bin/env bash\n"
        'printf "%s\\n" "${OMNIMARKET_REF:-<unset>}" >> "$INSTALL_SHIM_LOG"\n'
        # argv carries `--execute <python>`: which interpreter the provider
        # layer was composed into is the whole OMN-17819 question.
        'printf "%s\\n" "$*" >> "$INSTALL_ARGV_LOG"\n'
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
        # The GATE venv: the clone's own project environment, lock-governed
        # only. It carries no omnimarket, which after OMN-17819 is the point.
        self.gate_venv = _make_fake_venv(self.infra / ".venv", None)
        # The DISPATCH venv: outside the clone, composed, and the one the
        # installed-omnimarket probe reads.
        self.dispatch_venv = root / ".onex-dispatch-venv"
        _make_fake_venv(self.dispatch_venv, self.market_head)
        # A fake "brew" interpreter, and a dispatch venv already built on it.
        # Both are fixture baseline, not assertion: every pre-existing test in
        # this file is about composition, and without them each one would read
        # as rule-11 interpreter drift on macOS and measure the wrong thing.
        self.brew_python = root / "fakebrew" / "bin" / "python3.13"
        self.brew_python.parent.mkdir(parents=True, exist_ok=True)
        self.brew_python.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
        self.brew_python.chmod(0o755)
        self.set_dispatch_interpreter(self.brew_python.parent)
        _make_fake_venv(self.omniclaude / ".venv", None)

        self.bin_dir = root / "shimbin"
        self.uv_log = root / "uv.log"
        self.install_log = root / "install.log"
        self.install_argv_log = root / "install-argv.log"
        # One interleaved log across BOTH shims. Two separate logs can prove
        # that each collaborator ran; only a shared one can prove the ORDER,
        # and the order is the OMN-16262 fix.
        self.order_log = root / "order.log"
        self.install_script = self.infra / "scripts" / "install-node-skill-package.sh"
        _make_install_shim(self.install_script)
        _make_uv_shim(self.bin_dir)

    def set_dispatch_interpreter(self, home: Path) -> None:
        """Write the dispatch venv's own ``pyvenv.cfg`` ``home`` line.

        The reconciler reads this file to decide whether the venv was built on
        the required interpreter, so a test moves the interpreter by writing it
        rather than by asserting on argv.
        """
        (self.dispatch_venv / "pyvenv.cfg").write_text(
            f"home = {home}\nimplementation = CPython\n", encoding="utf-8"
        )

    def set_installed_commit(self, commit: str | None) -> None:
        """Move the DISPATCH venv's installed omnimarket commit.

        The gate venv is never given one: an omnimarket in there is the
        OMN-15620 impurity this split exists to remove.
        """
        _make_fake_venv(self.dispatch_venv, commit)
        self.set_dispatch_interpreter(self.brew_python.parent)

    def env(self) -> dict[str, str]:
        return {
            **os.environ,
            "OMNI_HOME": str(self.root),
            "PATH": f"{self.bin_dir}:{os.environ['PATH']}",
            "UV_SHIM_LOG": str(self.uv_log),
            "INSTALL_SHIM_LOG": str(self.install_log),
            "INSTALL_ARGV_LOG": str(self.install_argv_log),
            "ORDER_LOG": str(self.order_log),
            "ONEX_RECONCILE_INSTALL_SCRIPT": str(self.install_script),
            # Keep the hook-venv surface deterministic: the plugin-data venv is
            # host state, not workspace state, and must not leak into a test.
            "CLAUDE_PLUGIN_DATA": str(self.root / "no-such-plugin-data"),
            # A DECOY, deliberately set (OMN-17819). uv reads
            # UV_PROJECT_ENVIRONMENT from the ambient environment, so any caller
            # that has one -- a CI runner, an activated venv, a `uv run` parent
            # -- silently redirects a sync that means "the project's own venv".
            # This repo's own CI exports it, and the gate pass targeted the
            # runner's venv while still exiting 0, leaving the canonical clone
            # unpurified. Every sync that must target a project default has to
            # clear it, and a fixture with a clean environment cannot see that.
            "UV_PROJECT_ENVIRONMENT": str(self.root / "decoy-ambient-venv"),
            # Rule 11's interpreter, named explicitly so the suite is hermetic:
            # probing the real /opt/homebrew would make every outcome depend on
            # what happens to be installed on the machine running the tests.
            "ONEX_DISPATCH_BASE_PYTHON": str(self.brew_python),
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

    def dispatch_syncs(self) -> list[str]:
        """Mutating uv syncs that targeted the DISPATCH venv."""
        return [
            c
            for c in self.uv_calls()
            if "sync" in c and "--check" not in c and f"env={self.dispatch_venv} " in c
        ]

    def gate_syncs(self) -> list[str]:
        """Mutating uv syncs that targeted the GATE venv (uv's project default)."""
        return [
            c
            for c in self.uv_calls()
            if "sync" in c
            and "--check" not in c
            and c.startswith("env=- ")
            and str(self.infra) in c
        ]

    def install_argv(self) -> list[str]:
        if not self.install_argv_log.exists():
            return []
        return [
            line
            for line in self.install_argv_log.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

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


def test_reconcile_accepts_omnimarket_worktree_at_the_declared_clone_root(
    ws: _Workspace,
) -> None:
    """A governed candidate may be a Git worktree, whose .git is a file."""
    primary = ws.root.parent / "omnimarket-primary"
    ws.omnimarket.rename(primary)
    subprocess.run(
        ["git", "worktree", "add", "--detach", str(ws.root / "omnimarket")],
        cwd=primary,
        check=True,
        env=scrub_git_location_env(),
    )
    ws.omnimarket = ws.root / "omnimarket"
    ws.market_head = _git("rev-parse", "HEAD", cwd=ws.omnimarket)
    ws.set_installed_commit(ws.market_head)

    result = ws.run()

    assert result.returncode == _EXIT_OK, result.stdout + result.stderr


def test_reconcile_refuses_a_non_repository_at_the_declared_clone_root(
    ws: _Workspace,
) -> None:
    shutil.rmtree(ws.omnimarket)
    ws.omnimarket.mkdir()
    (ws.omnimarket / ".git").write_text("not a git dir\n", encoding="utf-8")

    result = ws.run()

    assert result.returncode == _EXIT_INDETERMINATE
    assert "no canonical omnimarket clone" in result.stdout + result.stderr


def test_reconcile_refuses_a_subdirectory_of_an_omnimarket_repository(
    ws: _Workspace,
) -> None:
    parent_repo = ws.root / "omnimarket-parent"
    ws.omnimarket.rename(parent_repo)
    nested_clone = parent_repo / "nested-clone"
    nested_clone.mkdir()
    ws.omnimarket = ws.root / "omnimarket"
    ws.omnimarket.symlink_to(nested_clone, target_is_directory=True)

    result = ws.run()

    assert result.returncode == _EXIT_INDETERMINATE
    assert "no canonical omnimarket clone" in result.stdout + result.stderr


def test_content_candidate_force_composes_core_and_market_after_lock(
    ws: _Workspace,
) -> None:
    """Content mode explicitly replaces released package bytes in the target."""
    core_source = ws.root / "omnibase_core"
    market_source = ws.root / "omnimarket"
    for path in (core_source, market_source):
        path.mkdir(parents=True, exist_ok=True)
    stale_marker = ws.dispatch_venv / "lib" / "python3.13" / "site-packages"
    stale_marker.mkdir(parents=True)
    (stale_marker / "stale-core-release.marker").write_text("0.47.18\n")
    ws.set_installed_commit("0" * _SHA_LEN)
    _make_content_verifier(ws.infra)
    manifest = ws.root / "candidate-content.json"
    manifest.write_text("{}\n", encoding="utf-8")
    env = ws.env()
    env["ONEX_CANDIDATE_CONTENT_MANIFEST"] = str(manifest)
    env["ONEX_CANDIDATE_CONTENT_MANIFEST_SHA256"] = "a" * 64

    result = subprocess.run(
        ["bash", str(_SCRIPT)],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )

    assert result.returncode == _EXIT_OK, result.stdout + result.stderr
    ordered = ws.ordered_calls()
    install_index = next(
        index
        for index, (kind, call) in enumerate(ordered)
        if kind == "uv" and "pip install" in call and "--reinstall-package" in call
    )
    install_call = ordered[install_index][1]
    assert "--reinstall-package omnibase-infra" in install_call
    assert "--reinstall-package omnibase-core" in install_call
    assert "--reinstall-package omnimarket" in install_call
    assert "--no-deps" not in install_call
    assert f"--python {ws.dispatch_venv}/bin/python" in install_call
    assert str(ws.infra) in install_call
    assert str(core_source) in install_call
    assert str(market_source) in install_call
    assert install_index > next(
        index
        for index, (kind, call) in enumerate(ordered)
        if kind == "uv" and "sync --frozen --inexact" in call
    )
    assert any(
        kind == "uv" and "pip check" in call
        for kind, call in ordered[install_index + 1 :]
    )
    assert stale_marker.joinpath("stale-core-release.marker").read_text() == "0.47.18\n"


def test_content_candidate_refuses_stale_source_before_pair_install(
    ws: _Workspace,
) -> None:
    _make_content_verifier(ws.infra, exit_code=1)
    manifest = ws.root / "candidate-content.json"
    manifest.write_text("{}\n", encoding="utf-8")
    env = ws.env()
    env["ONEX_CANDIDATE_CONTENT_MANIFEST"] = str(manifest)
    env["ONEX_CANDIDATE_CONTENT_MANIFEST_SHA256"] = "a" * 64

    result = subprocess.run(
        ["bash", str(_SCRIPT)],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )

    assert result.returncode == _EXIT_FAILED
    assert "source no longer matches" in result.stdout + result.stderr
    assert not any("pip install" in call for call in ws.uv_calls())
    assert not any("pip check" in call for call in ws.uv_calls())


def test_ordinary_reconcile_does_not_force_install_candidate_sources(
    ws: _Workspace,
) -> None:
    result = ws.run()

    assert result.returncode == _EXIT_OK, result.stdout + result.stderr
    assert not any("pip install" in call for call in ws.uv_calls())
    assert not any("pip check" in call for call in ws.uv_calls())


# --------------------------------------------------------------------------- #
# The composition rule: BOTH layers, in order
# --------------------------------------------------------------------------- #
def test_reconcile_applies_the_lock_without_removing_the_provider_layer(
    ws: _Workspace,
) -> None:
    """The DISPATCH venv's lock pass MUST be ``--inexact``.

    Without it, uv removes every package the lock does not mention -- which is
    omnimarket and its eleven companions -- and the next dispatch dies on the
    OMN-14060 guard. This is the single assertion that keeps the two layers from
    destroying each other.

    Scoped to the dispatch venv since OMN-17819: the gate venv's pass carries
    the same ``--project <infra>`` and must NOT be ``--inexact``, so a filter
    that matched on the project path alone would now assert the opposite of the
    truth for one of the two.
    """
    ws.set_installed_commit("0" * _SHA_LEN)
    result = ws.run()
    assert result.returncode == _EXIT_OK, result.stdout + result.stderr

    dispatch_syncs = ws.dispatch_syncs()
    assert dispatch_syncs, (
        f"reconcile never applied the dispatch venv's lock; uv calls were: "
        f"{ws.uv_calls()!r}"
    )
    for call in dispatch_syncs:
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
        if kind == "uv" and "--check" not in call and f"env={ws.dispatch_venv} " in call
    ]
    assert lock_after, "no dispatch-venv lock pass ran at all"
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
        kind == "uv" and "--check" not in call and f"env={ws.dispatch_venv} " in call
        for kind, call in order[provider_at:]
    ), (
        "the provider layer moved and no lock pass followed, so an OMN-16262 "
        f"downgrade would survive the tick. Order was: {order!r}"
    )


# --------------------------------------------------------------------------- #
# OMN-17819: the gate venv and the dispatch venv are two directories
# --------------------------------------------------------------------------- #
# Until OMN-17819 the provider layer was composed into
# ``$OMNI_HOME/omnibase_infra/.venv`` -- which is also the canonical clone's
# PROJECT venv, the one ``uv run pytest`` executes in and the one the OMN-15620
# purity gate judges. One directory was being asked to satisfy two contradictory
# contracts: OMN-14060 refuses a dispatch when omnimarket is ABSENT, OMN-15620
# refuses a test session when it is PRESENT. The measured effect on this Mac was
# that every focused ``uv run pytest`` in the canonical clone was refused before
# collection, on a 600-second self-healing tick that reverted the refusal's own
# recommended repair.
#
# These assertions pin the split. Note what they do NOT assert: that the
# reconciler REMOVES omnimarket from wherever it finds it. Removal alone is the
# CLI-bricking path (it is exactly what the refusal message used to recommend);
# the fix is that the provider layer is ROUTED to a venv where it belongs.
def test_provider_layer_is_composed_into_the_dispatch_venv_not_the_gate_venv(
    ws: _Workspace,
) -> None:
    ws.set_installed_commit("0" * _SHA_LEN)
    assert ws.run().returncode == _EXIT_OK

    argv = ws.install_argv()
    assert argv, "the provider co-install never ran"
    for call in argv:
        assert str(ws.dispatch_venv) in call, (
            "the provider co-install did not target the dispatch venv "
            f"({ws.dispatch_venv}); argv was {call!r}"
        )
        assert str(ws.gate_venv) not in call, (
            "the provider co-install targeted the GATE venv. That is the "
            "OMN-17819 defect: an undeclared `onex.nodes` provider there makes "
            "the OMN-15620 purity gate refuse every `uv run pytest` in the "
            f"canonical clone. argv was {call!r}"
        )


def test_gate_venv_lock_pass_is_exact_never_inexact(ws: _Workspace) -> None:
    """``--inexact`` on the gate venv is what re-admits the pollution.

    ``--inexact`` exists so a COMPOSED layer can coexist with a lock. The gate
    venv has no composed layer by construction, so the flag there means only
    "leave undeclared distributions installed" -- which is the refusal
    condition, restored.
    """
    _make_uv_shim(ws.bin_dir, check_exit=1)  # nothing satisfies its lock
    assert ws.run().returncode == _EXIT_OK

    gate_syncs = ws.gate_syncs()
    assert gate_syncs, (
        f"the gate venv's lock was never applied; uv calls were: {ws.uv_calls()!r}"
    )
    for call in gate_syncs:
        assert "--inexact" not in call, (
            "the gate venv was synced --inexact, which leaves an undeclared "
            "`onex.nodes` provider installed and keeps the OMN-15620 gate "
            f"refusing. Call: {call!r}"
        )
        assert "--frozen" in call, f"gate sync must be --frozen. Call: {call!r}"


def test_dispatch_venv_is_reconciled_before_the_gate_venv_is_purified(
    ws: _Workspace,
) -> None:
    """Order across the two surfaces, not just within one.

    The gate pass is the step that takes omnimarket away from the clone's
    ``.venv``. Running it before a working dispatch venv exists would leave this
    host with no interpreter that can run ``onex`` at all.
    """
    _make_uv_shim(ws.bin_dir, check_exit=1)
    ws.set_installed_commit("0" * _SHA_LEN)
    assert ws.run().returncode == _EXIT_OK

    calls = ws.uv_calls()
    last_dispatch = max(i for i, c in enumerate(calls) if c in ws.dispatch_syncs())
    first_gate = min(i for i, c in enumerate(calls) if c in ws.gate_syncs())
    assert last_dispatch < first_gate, (
        "the gate venv was purified before the dispatch venv was finished, so "
        "there is a window with no runnable `onex` on the host. Calls were: "
        f"{calls!r}"
    )


def test_a_dispatch_failure_refuses_without_touching_the_gate_venv(
    ws: _Workspace,
) -> None:
    """A half-applied split is worse than the defect it replaces.

    If the dispatch venv cannot be built, purifying the gate venv anyway would
    remove the only omnimarket on the host and leave nothing able to dispatch.
    """
    _make_install_shim(ws.install_script, exit_code=1)
    ws.set_installed_commit("0" * _SHA_LEN)
    result = ws.run()

    assert result.returncode == _EXIT_FAILED
    assert ws.gate_syncs() == [], (
        "the gate venv was synced after the dispatch layer failed; uv calls "
        f"were: {ws.uv_calls()!r}"
    )


def test_installed_commit_is_read_from_the_dispatch_venv(ws: _Workspace) -> None:
    """The drift comparison must ask the interpreter a dispatch actually uses.

    Reading the gate venv would ask about a package that is now required to be
    absent from it, and every tick would read "not installed" and re-install
    forever.
    """
    stale = "0" * _SHA_LEN
    _make_fake_venv(ws.dispatch_venv, stale)
    _make_fake_venv(ws.infra / ".venv", ws.market_head)  # gate venv "agrees"

    result = ws.run("--check")
    assert result.returncode == _EXIT_DRIFT, (
        "the reconciler read the gate venv's commit and called the workspace "
        f"in sync: {result.stdout!r}"
    )
    assert stale[:12] in result.stdout


def test_dispatch_venv_is_not_inside_the_canonical_clone(ws: _Workspace) -> None:
    """A venv inside the clone is a venv some probe will find while asking
    about the clone. Placement is the property, not an implementation detail."""
    assert ws.dispatch_venv.parent == ws.root
    assert str(ws.infra) not in str(ws.dispatch_venv)


# --------------------------------------------------------------------------- #
# The dispatch venv is built on the brew interpreter (CLAUDE.md rule 11)
# --------------------------------------------------------------------------- #
# macOS grants Local Network access per binary path and signature, not per
# Python version. A uv-managed interpreter never surfaces the privacy dialog and
# its LAN connections fail silently with EHOSTUNREACH. The dispatch venv is what
# `scripts/onex` execs and `onex` talks to the lab host's services, so it has to
# be built on the brew binary at its literal resolved path.
#
# This is a pre-existing defect the OMN-17819 split made fixable, not one the
# split introduced: the venv it replaced was already uv-managed.
def test_dispatch_venv_is_built_on_the_required_interpreter(ws: _Workspace) -> None:
    """Asserted by reading the venv's own ``pyvenv.cfg``, not by matching argv.

    An argv assertion passes whenever the flag is present, including when uv
    ignored it and left the environment on the interpreter it already had. The
    file is what the reconciler itself reads.
    """
    import shutil

    shutil.rmtree(ws.dispatch_venv)
    assert ws.run().returncode == _EXIT_OK

    cfg = (ws.dispatch_venv / "pyvenv.cfg").read_text(encoding="utf-8")
    assert f"home = {ws.brew_python.parent}" in cfg, (
        "the dispatch venv was not built on the required interpreter; its "
        f"pyvenv.cfg reads: {cfg!r}"
    )


def test_an_existing_venv_on_the_wrong_interpreter_is_rebuilt(ws: _Workspace) -> None:
    """Drift, not a state to leave alone.

    Without this the requirement would hold only for hosts that built their
    dispatch venv after it landed, and every host that already had one would
    keep a silently LAN-blind CLI forever — which is the actual starting state
    on the machine this was written for.
    """
    uv_managed = ws.root / "uv-managed" / "cpython-3.12-macos-aarch64-none" / "bin"
    uv_managed.mkdir(parents=True)
    ws.set_dispatch_interpreter(uv_managed)

    result = ws.run()
    assert result.returncode == _EXIT_OK, result.stdout + result.stderr

    cfg = (ws.dispatch_venv / "pyvenv.cfg").read_text(encoding="utf-8")
    assert f"home = {ws.brew_python.parent}" in cfg, (
        f"the wrong interpreter survived the reconcile; pyvenv.cfg: {cfg!r}"
    )
    assert "interpreter drift" in result.stdout, (
        "the rebuild happened silently; a lane reading the output cannot tell "
        f"its CLI was relocated. Output: {result.stdout!r}"
    )


def test_check_mode_reports_interpreter_drift_instead_of_in_sync(
    ws: _Workspace,
) -> None:
    """`--check` is what the SessionStart line runs.

    A verdict that stayed silent about the interpreter would print "in sync"
    over a CLI whose LAN calls to the lab host fail silently — a probe naming
    something narrower than it measures, which is the OMN-17295 defect class.
    """
    uv_managed = ws.root / "uv-managed" / "bin"
    uv_managed.mkdir(parents=True)
    ws.set_dispatch_interpreter(uv_managed)

    result = ws.run("--check")
    assert result.returncode == _EXIT_DRIFT
    assert "wrong interpreter" in result.stdout
    assert ws.uv_calls() == [] or all("--check" in c for c in ws.uv_calls()), (
        "check mode mutated something while reporting interpreter drift"
    )


def test_a_missing_brew_interpreter_refuses_and_names_every_path_tried(
    ws: _Workspace,
) -> None:
    """An empty PATH probe was read as "uv is not installed" once already
    (OMN-17335). A refusal that does not name what it looked at repeats it."""
    env = ws.env()
    env["ONEX_DISPATCH_BASE_PYTHON"] = str(ws.root / "no-such-python")
    result = subprocess.run(
        ["bash", str(_SCRIPT)],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    combined = result.stdout + result.stderr
    # The named interpreter does not exist, so uv cannot build on it. Either the
    # reconciler refuses up front or uv fails — both must be a refusal, never a
    # silent fallback to whatever interpreter was already there.
    assert result.returncode in (_EXIT_FAILED, _EXIT_INDETERMINATE), combined
    assert ws.gate_syncs() == [], (
        "the gate venv was purified even though the dispatch venv could not be "
        "built on the required interpreter"
    )


# --------------------------------------------------------------------------- #
# The dispatch venv may never BE a clone's own venv
# --------------------------------------------------------------------------- #
def test_refuses_to_compose_into_the_canonical_clones_gate_venv(
    ws: _Workspace,
) -> None:
    """The override relocates the composed venv; it never re-collapses the two.

    Pointing `ONEX_DISPATCH_VENV` at the clone's `.venv` reconstructs the exact
    OMN-17819 defect by hand.
    """
    env = ws.env()
    env["ONEX_DISPATCH_VENV"] = str(ws.infra / ".venv")
    result = subprocess.run(
        ["bash", str(_SCRIPT)],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    combined = result.stdout + result.stderr

    assert result.returncode == _EXIT_INDETERMINATE, combined
    assert "GATE venv" in combined
    assert ws.install_argv() == [], (
        "the provider co-install ran against the gate venv anyway"
    )


def test_refuses_a_dispatch_venv_inside_any_other_clone(ws: _Workspace) -> None:
    """Not just the one clone this ticket was about.

    A composed layer is undeclared in whatever project owns that venv, so its
    own purity checks and test runs would be refused — the same defect, moved.
    """
    other = _make_clone(ws.root, "omniother")
    env = ws.env()
    env["ONEX_DISPATCH_VENV"] = str(other / ".venv")
    result = subprocess.run(
        ["bash", str(_SCRIPT)],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    combined = result.stdout + result.stderr

    assert result.returncode == _EXIT_INDETERMINATE, combined
    assert str(other) in combined


def test_refuses_a_dispatch_venv_inside_a_worktree(ws: _Workspace) -> None:
    """In a worktree ``.git`` is a FILE.

    A `-d` test would wave through exactly the per-ticket worktrees lanes spend
    all day inside, which is where a hand-set override is most likely to point.
    """
    fake_worktree = ws.root / "omni_worktrees" / "OMN-1" / "omnibase_infra"
    fake_worktree.mkdir(parents=True)
    (fake_worktree / ".git").write_text("gitdir: /elsewhere\n", encoding="utf-8")

    env = ws.env()
    env["ONEX_DISPATCH_VENV"] = str(fake_worktree / ".venv")
    result = subprocess.run(
        ["bash", str(_SCRIPT)],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    assert result.returncode == _EXIT_INDETERMINATE, result.stdout + result.stderr


def test_a_path_that_merely_ends_in_venv_is_not_refused(ws: _Workspace) -> None:
    """Positive control for the refusals above.

    Without it every one of them could pass against a guard that refused
    unconditionally, which would make the reconciler unusable rather than safe.
    """
    elsewhere = ws.root / "not-a-clone"
    elsewhere.mkdir()
    env = ws.env()
    env["ONEX_DISPATCH_VENV"] = str(elsewhere / ".venv")
    result = subprocess.run(
        ["bash", str(_SCRIPT)],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    assert result.returncode == _EXIT_OK, result.stdout + result.stderr


# --------------------------------------------------------------------------- #
# No machine-absolute path literal (CLAUDE.md rules 6 and 8)
# --------------------------------------------------------------------------- #
def test_the_reconciler_carries_no_machine_absolute_path_literal() -> None:
    """Every path resolves from ``$OMNI_HOME``, which is itself fail-fast.

    The brew interpreter candidates are the deliberate exception rule 11 names:
    launchd and cron run with a restricted PATH that cannot resolve
    ``$(brew --prefix)``, so those two must be literal. They are not
    machine-specific — no ``/Users`` or ``/Volumes`` component.
    """
    source = _SCRIPT.read_text(encoding="utf-8")
    offenders = [
        line
        for line in source.splitlines()
        if ("/Users/" in line or "/Volumes/" in line)
        and not line.lstrip().startswith("#")
    ]
    assert offenders == [], (
        f"machine-absolute path literal in the reconciler: {offenders!r}"
    )


def test_unset_omni_home_is_still_fail_fast_with_no_default(tmp_path: Path) -> None:
    """Pinned here rather than assumed: a silent default would reconcile some
    other checkout's venv and report success for a venv nobody is running."""
    env = {k: v for k, v in os.environ.items() if k != "OMNI_HOME"}
    result = subprocess.run(
        ["bash", str(_SCRIPT)],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    combined = result.stdout + result.stderr
    assert result.returncode == _EXIT_INDETERMINATE
    assert "OMNI_HOME" in combined


def test_the_interpreter_requirement_is_exercised_off_macos(ws: _Workspace) -> None:
    """The enforcement must be provable on the merge gate, not only on a Mac.

    CI runs on Linux, where the LAN-grant constraint does not exist and the
    reconciler correctly stands down. Without a seam that switches the
    requirement ON, every assertion above would be macOS-only and the rule-11
    enforcement would ship untested by the gate that guards it — rule 5's
    "opt-in verification never gets adopted", applied to a safety requirement.

    Both seams only ever ADD the requirement; see the no-bypass test below.
    """
    env = ws.env()
    env.pop("ONEX_DISPATCH_BASE_PYTHON")
    env["ONEX_DISPATCH_REQUIRE_BASE_PYTHON"] = "1"
    env["ONEX_DISPATCH_BASE_PYTHON_CANDIDATES"] = str(ws.brew_python)
    import shutil

    shutil.rmtree(ws.dispatch_venv)
    result = subprocess.run(
        ["bash", str(_SCRIPT)],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    assert result.returncode == _EXIT_OK, result.stdout + result.stderr
    cfg = (ws.dispatch_venv / "pyvenv.cfg").read_text(encoding="utf-8")
    assert f"home = {ws.brew_python.parent}" in cfg, (
        f"the requirement did not apply once switched on off-macOS; pyvenv.cfg: {cfg!r}"
    )


def test_no_acceptable_interpreter_refuses_and_names_every_path_tried(
    ws: _Workspace,
) -> None:
    """Hermetic on every host.

    Asserting this against the built-in candidates would pass on CI and fail on
    every developer Mac, which has brew installed — the same host-state coupling
    two fixtures in this suite already had to be repaired for.
    """
    missing = [str(ws.root / "no-python-a"), str(ws.root / "no-python-b")]
    env = ws.env()
    env.pop("ONEX_DISPATCH_BASE_PYTHON")
    env["ONEX_DISPATCH_REQUIRE_BASE_PYTHON"] = "1"
    env["ONEX_DISPATCH_BASE_PYTHON_CANDIDATES"] = ":".join(missing)
    result = subprocess.run(
        ["bash", str(_SCRIPT)],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    combined = result.stdout + result.stderr

    assert result.returncode == _EXIT_FAILED, combined
    for candidate in missing:
        assert candidate in combined, (
            f"the refusal did not name {candidate}; an unnamed candidate list is "
            "how 'not on PATH' got read as 'not installed' (OMN-17335). "
            f"Output: {combined!r}"
        )
    assert ws.gate_syncs() == [], "the gate venv was purified anyway"


def test_no_environment_value_turns_the_interpreter_requirement_off() -> None:
    """The two seams are one-way.

    A variable that switched a safety requirement off is the bypass rule 17
    forbids, and it would be reached for the first time by whoever is most
    inconvenienced by the gate. Asserted against the source's own predicate
    rather than by trying values, so a new escape hatch is a red test.
    """
    source = _SCRIPT.read_text(encoding="utf-8")
    start = source.index("dispatch_requires_base_python() {")
    body = source[start : source.index("\n}", start)]

    # Every branch inside the predicate must be a `return 0` (requirement ON)
    # or the platform test. Nothing may return non-zero on an env value.
    assert "return 1" not in body, (
        f"dispatch_requires_base_python can be switched OFF by an env value: {body!r}"
    )
    for off_ish in ('!= "1"', '== "0"', "ONEX_DISPATCH_SKIP", "ALLOW"):
        assert off_ish not in body, (
            f"an opt-out spelling appeared in the predicate: {off_ish!r}"
        )


def test_a_brew_style_interpreter_and_its_venv_compare_equal(
    ws: _Workspace,
) -> None:
    """The three spellings of one brew installation must read as the same one.

    A real host offers ``<prefix>/bin/python3.13`` (a symlink), records
    ``<prefix>/opt/python@3.13/bin`` in the venv it builds, and resolves both
    into ``<prefix>/Cellar/python@3.13/<version>/bin``. Two earlier attempts at
    this predicate compared directories and both refused a venv that was
    correct; see ``dispatch_interpreter_ok``'s own comment for the measurements.
    """
    prefix = ws.root / "brewprefix"
    cellar_bin = prefix / "Cellar" / "python@3.13" / "3.13.3" / "bin"
    cellar_bin.mkdir(parents=True)
    real_python = cellar_bin / "python3.13"
    real_python.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    real_python.chmod(0o755)

    (prefix / "bin").mkdir()
    (prefix / "bin" / "python3.13").symlink_to(real_python)
    (prefix / "opt").mkdir()
    (prefix / "opt" / "python@3.13").symlink_to(cellar_bin.parent)

    # The venv records the `opt` spelling, as a real brew-built venv does.
    ws.set_dispatch_interpreter(prefix / "opt" / "python@3.13" / "bin")

    env = ws.env()
    env["ONEX_DISPATCH_BASE_PYTHON"] = str(prefix / "bin" / "python3.13")
    result = subprocess.run(
        ["bash", str(_SCRIPT), "--check"],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    assert "wrong interpreter" not in result.stdout, (
        "three spellings of one installation compared unequal, so a correctly "
        f"built venv reads as drift: {result.stdout!r}"
    )


def test_an_interpreter_resolving_into_a_framework_bundle_compares_equal(
    ws: _Workspace,
) -> None:
    """The exact shape that broke the second attempt, pinned.

    Measured on a real host: brew's ``python3.13`` resolves through the Cellar
    and on into ``Frameworks/Python.framework/Versions/3.13/bin``, while the
    venv records the Cellar's own ``bin``. Fully resolving the file therefore
    compared two different real directories and refused a correct venv.
    """
    prefix = ws.root / "fw-prefix"
    cellar = prefix / "Cellar" / "python@3.13" / "3.13.3"
    framework_bin = (
        cellar / "Frameworks" / "Python.framework" / "Versions" / "3.13" / "bin"
    )
    framework_bin.mkdir(parents=True)
    real_python = framework_bin / "python3.13"
    real_python.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    real_python.chmod(0o755)

    (cellar / "bin").mkdir(parents=True)
    (cellar / "bin" / "python3.13").symlink_to(real_python)
    (prefix / "bin").mkdir()
    (prefix / "bin" / "python3.13").symlink_to(cellar / "bin" / "python3.13")

    ws.set_dispatch_interpreter(cellar / "bin")

    env = ws.env()
    env["ONEX_DISPATCH_BASE_PYTHON"] = str(prefix / "bin" / "python3.13")
    result = subprocess.run(
        ["bash", str(_SCRIPT), "--check"],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    assert "wrong interpreter" not in result.stdout, (
        "the framework-bundle shape still reads as drift; this is the exact "
        f"measurement the second attempt failed on: {result.stdout!r}"
    )


def test_a_genuinely_different_interpreter_still_compares_unequal(
    ws: _Workspace,
) -> None:
    """Positive control.

    Without it the two tests above would pass against a predicate loosened
    until it accepted anything — which would silently retire the rule-11 check
    rather than fix it.
    """
    other = ws.root / "some-other-python" / "bin"
    other.mkdir(parents=True)
    ws.set_dispatch_interpreter(other)

    result = subprocess.run(
        ["bash", str(_SCRIPT), "--check"],
        capture_output=True,
        text=True,
        env=ws.env(),
        check=False,
    )
    assert result.returncode == _EXIT_DRIFT
    assert "wrong interpreter" in result.stdout


def test_a_sibling_prefix_is_not_accepted_as_containment(ws: _Workspace) -> None:
    """Containment is on path SEGMENTS, not on string prefix.

    Without the trailing separator, an installation at ``/opt/homebrew-other``
    would satisfy a requirement rooted at ``/opt/homebrew``.
    """
    sibling = ws.root / "fakebrew-other" / "bin"
    sibling.mkdir(parents=True)
    ws.set_dispatch_interpreter(sibling)

    result = subprocess.run(
        ["bash", str(_SCRIPT), "--check"],
        capture_output=True,
        text=True,
        env=ws.env(),
        check=False,
    )
    assert result.returncode == _EXIT_DRIFT, (
        f"a sibling prefix was accepted as containment; output: {result.stdout!r}"
    )
    assert "wrong interpreter" in result.stdout


# --------------------------------------------------------------------------- #
# A rebuild is staged at a sibling and renamed in, never done in place
# --------------------------------------------------------------------------- #
# The dispatch venv is the one path every lane's `onex` execs. Rebuilding it in
# place makes it a half-built environment with no omnimarket in it for the
# length of the rebuild, and for the whole of a FAILED one -- measured on the
# live host on 2026-09-17, where a wrong interpreter predicate rebuilt and then
# refused its own readback on a ~600s tick and lanes failed against the gap.
_UV_STAGING_SUFFIX = ".rebuilding"


def _drift_the_interpreter(ws: _Workspace) -> Path:
    """Move the live dispatch venv onto an interpreter outside the brew root.

    Returns the staging path the reconciler is expected to build at. Written as
    a real directory, because ``real_dir`` resolves by ``cd``-ing and a path
    that does not exist would make the predicate refuse for the wrong reason.
    """
    elsewhere = ws.root / "uv-managed-store" / "cpython-3.12" / "bin"
    elsewhere.mkdir(parents=True, exist_ok=True)
    ws.set_dispatch_interpreter(elsewhere)
    return Path(str(ws.dispatch_venv) + _UV_STAGING_SUFFIX)


def _mutating_uv_targets(ws: _Workspace) -> list[str]:
    """The UV_PROJECT_ENVIRONMENT of every uv call that WRITES.

    ``--check`` calls are read-only probes and legitimately name the live venv;
    counting them would make "nothing wrote to the live venv" unassertable.
    """
    targets: list[str] = []
    for line in ws.uv_calls():
        if not line.startswith("env="):
            continue
        target, _, argv = line.partition(" ")
        if "--check" in argv.split():
            continue
        targets.append(target[len("env=") :])
    return targets


def test_a_rebuild_never_writes_to_the_live_dispatch_venv(ws: _Workspace) -> None:
    """Every mutating step of a rebuild targets the sibling, not the live venv.

    This is the property, and it is asserted over the collaborator log rather
    than over the script's prose: the lock pass and the provider co-install must
    both name the staging path, and nothing that writes may name the live one.
    """
    staging = _drift_the_interpreter(ws)

    result = ws.run()

    assert result.returncode == _EXIT_OK, result.stdout + result.stderr
    targets = _mutating_uv_targets(ws)
    assert str(staging) in targets, (
        f"the rebuild did not sync into the staging sibling: {targets!r}"
    )
    assert str(ws.dispatch_venv) not in targets, (
        "a rebuild wrote straight to the live dispatch venv, which is the "
        f"in-place rebuild this staging exists to end: {targets!r}"
    )
    argv = ws.install_argv_log.read_text(encoding="utf-8")
    assert str(staging / "bin" / "python") in argv, (
        f"the provider layer was composed into the wrong interpreter: {argv!r}"
    )
    assert str(ws.dispatch_venv / "bin" / "python") not in argv, (
        f"the provider co-install targeted the live venv during a rebuild: {argv!r}"
    )


def test_a_proven_rebuild_is_renamed_into_place(ws: _Workspace) -> None:
    """The swap actually happens, and leaves no staging directory behind."""
    staging = _drift_the_interpreter(ws)

    result = ws.run()

    assert result.returncode == _EXIT_OK, result.stdout + result.stderr
    assert not staging.exists(), (
        "the staging directory survived a successful swap, so a later run would "
        "treat live state as scrap"
    )
    cfg = (ws.dispatch_venv / "pyvenv.cfg").read_text(encoding="utf-8")
    assert str(ws.brew_python.parent) in cfg, (
        f"the live venv was not replaced by the rebuilt one: {cfg!r}"
    )
    assert "relocatable = true" in cfg, (
        "the venv that was renamed into place is not relocatable, so its "
        f"console scripts name a directory that no longer exists: {cfg!r}"
    )


def test_a_failed_rebuild_leaves_the_live_dispatch_venv_serving(
    ws: _Workspace,
) -> None:
    """The refusal path is the reason to stage at all.

    A rebuild that dies partway used to leave the live venv gutted, which is
    what broke `onex delegate` for every concurrent lane. Here the sync fails
    and the live venv must come through byte-for-byte.
    """
    _make_uv_shim(ws.bin_dir, sync_exit=1)
    staging = _drift_the_interpreter(ws)
    before = (ws.dispatch_venv / "bin" / "python").read_text(encoding="utf-8")

    result = ws.run()

    assert result.returncode == _EXIT_FAILED, result.stdout + result.stderr
    assert "UNTOUCHED" in result.stdout, (
        f"the refusal does not tell the reader the live venv survived: {result.stdout!r}"
    )
    after = (ws.dispatch_venv / "bin" / "python").read_text(encoding="utf-8")
    assert after == before, "a failed rebuild damaged the live dispatch venv"
    assert ws.market_head in after, (
        "the live venv lost the provider commit it was serving before the "
        "rebuild was attempted"
    )
    assert not staging.exists() or staging.is_dir(), (
        "the staged build is neither absent nor a directory left for diagnosis"
    )


def test_a_staged_venv_that_is_not_relocatable_is_refused(ws: _Workspace) -> None:
    """Positive control for the relocatability readback.

    Without it the swap would rename in a venv whose ~100 console scripts --
    `onex` among them -- carry an absolute shebang naming the staging path,
    which is about to stop existing. That is strictly worse than the in-place
    rebuild being replaced, so it must refuse rather than swap.
    """
    _make_uv_shim(ws.bin_dir, venv_relocatable=False)
    _drift_the_interpreter(ws)
    before = (ws.dispatch_venv / "bin" / "python").read_text(encoding="utf-8")

    result = ws.run()

    assert result.returncode == _EXIT_FAILED, result.stdout + result.stderr
    assert "not relocatable" in result.stdout, (
        f"the refusal does not name the reason: {result.stdout!r}"
    )
    after = (ws.dispatch_venv / "bin" / "python").read_text(encoding="utf-8")
    assert after == before, (
        "the live dispatch venv was replaced by a venv that cannot be moved"
    )


def test_an_additive_pass_is_not_staged(ws: _Workspace) -> None:
    """Control: only a REBUILD is staged.

    Staging every pass would double the disk cost and the wall clock of the
    common case -- the clone advanced, the interpreter did not -- for no gain,
    because an additive provider or lock pass never leaves the venv unusable.
    This proves the narrowing is real rather than incidental.
    """
    ws.set_installed_commit("0" * _SHA_LEN)

    result = ws.run()

    assert result.returncode == _EXIT_OK, result.stdout + result.stderr
    staging = Path(str(ws.dispatch_venv) + _UV_STAGING_SUFFIX)
    assert not staging.exists()
    assert str(ws.dispatch_venv) in _mutating_uv_targets(ws), (
        "an additive pass stopped writing to the live venv"
    )
    assert not any(line.startswith("venv ") for line in ws.uv_calls()), (
        "an additive pass created a venv, so every tick now pays for a rebuild"
    )
