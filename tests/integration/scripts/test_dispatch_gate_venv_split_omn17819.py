# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Integration smoke for the OMN-17819 dispatch/gate venv split."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from omnibase_core.validators.no_unguarded_git_subprocess import (
    scrub_git_location_env,
)

pytestmark = pytest.mark.integration

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SCRIPT = _REPO_ROOT / "scripts" / "reconcile-workspace-venvs.sh"
_SHA_LEN = 40


def _git(*args: str, cwd: Path) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=True,
        env=scrub_git_location_env(),
    )
    return result.stdout.strip()


def _make_clone(root: Path, name: str) -> Path:
    repo = root / name
    repo.mkdir(parents=True)
    _git("init", "--quiet", "-b", "dev", cwd=repo)
    _git("config", "user.email", "test@example.com", cwd=repo)
    _git("config", "user.name", "Test", cwd=repo)
    (repo / "f.txt").write_text("one", encoding="utf-8")
    _git("add", "f.txt", cwd=repo)
    _git("commit", "--quiet", "-m", "one", cwd=repo)
    return repo


def _make_fake_venv(venv: Path, installed_commit: str | None) -> None:
    (venv / "bin").mkdir(parents=True, exist_ok=True)
    emitted = installed_commit or ""
    python = venv / "bin" / "python"
    python.write_text(
        "#!/usr/bin/env bash\n"
        "cat >/dev/null 2>&1 || true\n"
        f"printf '%s\\n' '{emitted}'\n",
        encoding="utf-8",
    )
    python.chmod(0o755)


def _make_uv_shim(bin_dir: Path, *, check_exit: int) -> Path:
    bin_dir.mkdir(parents=True, exist_ok=True)
    uv = bin_dir / "uv"
    uv.write_text(
        "#!/usr/bin/env bash\n"
        'printf "env=%s %s\\n" "${UV_PROJECT_ENVIRONMENT:--}" "$*" >> "$UV_SHIM_LOG"\n'
        # `uv sync` creates the environment when it is absent and recreates it
        # when --python names a different interpreter, writing the pyvenv.cfg
        # the reconciler reads back (OMN-17819, CLAUDE.md rule 11). Without
        # this the shim models a uv that silently leaves the venv on the wrong
        # interpreter, and the reconciler correctly refuses.
        'if [[ -n "${UV_PROJECT_ENVIRONMENT:-}" ]]; then\n'
        '  _base=""; _prev=""\n'
        '  for a in "$@"; do\n'
        '    [[ "$_prev" == "--python" ]] && _base="$a"\n'
        '    _prev="$a"\n'
        "  done\n"
        '  if [[ -n "$_base" ]]; then\n'
        '    mkdir -p "${UV_PROJECT_ENVIRONMENT}/bin"\n'
        '    printf "home = %s\\n" "${_base%/*}" > "${UV_PROJECT_ENVIRONMENT}/pyvenv.cfg"\n'
        "  fi\n"
        "fi\n"
        'for arg in "$@"; do\n'
        '  if [[ "$arg" == "--check" ]]; then exit ' + str(check_exit) + "; fi\n"
        "done\n"
        "exit 0\n",
        encoding="utf-8",
    )
    uv.chmod(0o755)
    return uv


def _make_install_shim(path: Path) -> Path:
    path.write_text(
        '#!/usr/bin/env bash\nprintf "%s\\n" "$*" >> "$INSTALL_ARGV_LOG"\nexit 0\n',
        encoding="utf-8",
    )
    path.chmod(0o755)
    return path


def _read_lines(path: Path) -> list[str]:
    if not path.exists():
        return []
    return [line for line in path.read_text(encoding="utf-8").splitlines() if line]


def test_reconciler_routes_provider_to_dispatch_venv_and_purifies_gate_venv(
    tmp_path: Path,
) -> None:
    omni_home = tmp_path / "omni_home"
    omnimarket = _make_clone(omni_home, "omnimarket")
    market_head = _git("rev-parse", "HEAD", cwd=omnimarket)

    infra = omni_home / "omnibase_infra"
    (infra / "scripts").mkdir(parents=True)
    (infra / "uv.lock").write_text("lock-v1\n", encoding="utf-8")
    (omni_home / "omniclaude").mkdir(parents=True)
    ((omni_home / "omniclaude") / "uv.lock").write_text(
        "claude-lock-v1\n",
        encoding="utf-8",
    )

    gate_venv = infra / ".venv"
    dispatch_venv = omni_home / ".onex-dispatch-venv"
    _make_fake_venv(gate_venv, None)
    _make_fake_venv(dispatch_venv, "0" * _SHA_LEN)

    bin_dir = omni_home / "shimbin"
    uv_log = omni_home / "uv.log"
    install_argv_log = omni_home / "install-argv.log"
    install_script = infra / "scripts" / "install-node-skill-package.sh"
    _make_uv_shim(bin_dir, check_exit=1)
    _make_install_shim(install_script)

    env = {
        **os.environ,
        "OMNI_HOME": str(omni_home),
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "UV_SHIM_LOG": str(uv_log),
        "INSTALL_ARGV_LOG": str(install_argv_log),
        "ONEX_RECONCILE_INSTALL_SCRIPT": str(install_script),
        "CLAUDE_PLUGIN_DATA": str(omni_home / "no-such-plugin-data"),
    }
    result = subprocess.run(
        ["bash", str(_SCRIPT)],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )

    assert result.returncode == 0, result.stderr

    install_calls = _read_lines(install_argv_log)
    assert install_calls, "provider co-install did not run"
    assert all(str(dispatch_venv) in call for call in install_calls)
    assert all(str(gate_venv) not in call for call in install_calls)

    uv_calls = _read_lines(uv_log)
    dispatch_syncs = [
        call
        for call in uv_calls
        if "sync" in call and "--check" not in call and f"env={dispatch_venv} " in call
    ]
    gate_syncs = [
        call
        for call in uv_calls
        if "sync" in call
        and "--check" not in call
        and call.startswith("env=- ")
        and str(infra) in call
    ]

    assert dispatch_syncs, f"dispatch venv was not synced: {uv_calls!r}"
    assert gate_syncs, f"gate venv was not synced: {uv_calls!r}"
    assert all("--inexact" in call for call in dispatch_syncs)
    assert all("--inexact" not in call for call in gate_syncs)
    assert market_head[:12] in result.stdout
