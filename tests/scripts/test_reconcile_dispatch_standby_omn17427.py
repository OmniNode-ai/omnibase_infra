# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""A provider-only pass reuses the previous generation instead of re-cloning (OMN-17427).

Observed trigger, 2026-10-17: every advance of the omnimarket clone's HEAD made
the reconciler stage a copy of the WHOLE dispatch venv (about 31,000 files, all
new to the file-change tracker), rewrite one package in it, swap it in, and
delete the old generation (about 31,000 more events). The lock was in sync on
every one of those passes; only the omnimarket commit had moved. Measured on the
Mac: `.onex-dispatch-venv.rebuilding` held 13,518 new files at 08:51 ET.

The swap now keeps the generation it displaced as `.onex-dispatch-venv.previous`,
and the next pass stages from THAT, so a pass rewrites the package that changed
and nothing else. A previous generation that no longer matches the live venv's
package set is not trusted and the pass falls back to the whole-venv clone.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from omnibase_core.validators.no_unguarded_git_subprocess import (
    scrub_git_location_env,
)

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT = _REPO_ROOT / "scripts" / "reconcile-workspace-venvs.sh"
_OLD = "0" * 40
_SITE = Path("lib") / "python3.13" / "site-packages"


def _git(*args: str, cwd: Path) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=True,
        env=scrub_git_location_env(),
    ).stdout.strip()


def _commit(repo: Path, content: str) -> str:
    (repo / "f.txt").write_text(content, encoding="utf-8")
    _git("add", "f.txt", cwd=repo)
    _git("commit", "--quiet", "-m", content, cwd=repo)
    return _git("rev-parse", "HEAD", cwd=repo)


def _fake_python(venv: Path) -> None:
    (venv / "bin").mkdir(parents=True, exist_ok=True)
    python = venv / "bin" / "python"
    # `installed_market_commit` pipes a program in and reads the commit back.
    python.write_text(
        "#!/usr/bin/env bash\ncat >/dev/null 2>&1 || true\n"
        'cat "$(dirname "$0")/../.installed_commit" 2>/dev/null || echo\n',
        encoding="utf-8",
    )
    python.chmod(0o755)


def _make_venv(venv: Path, base_bin: Path, commit: str | None) -> None:
    _fake_python(venv)
    (venv / ".installed_commit").write_text(f"{commit or ''}\n", encoding="utf-8")
    (venv / "pyvenv.cfg").write_text(
        f"home = {base_bin}\nrelocatable = true\n", encoding="utf-8"
    )
    for dist in ("pydantic-2.0.dist-info", "omnimarket-0.1.dist-info"):
        (venv / _SITE / dist).mkdir(parents=True, exist_ok=True)


def _installed(venv: Path) -> str:
    return (venv / ".installed_commit").read_text(encoding="utf-8").strip()


class _Host:
    """A workspace with a lock-clean dispatch venv and a movable omnimarket HEAD."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.market = root / "omnimarket"
        self.market.mkdir(parents=True)
        _git("init", "--quiet", "-b", "dev", cwd=self.market)
        _git("config", "user.email", "t@example.com", cwd=self.market)
        _git("config", "user.name", "T", cwd=self.market)
        self.head = _commit(self.market, "one")

        self.infra = root / "omnibase_infra"
        (self.infra / "scripts").mkdir(parents=True)
        (self.infra / "uv.lock").write_text("lock-v1\n", encoding="utf-8")
        (root / "omniclaude").mkdir()
        (root / "omniclaude" / "uv.lock").write_text("c\n", encoding="utf-8")

        base_bin = root / "pybase" / "bin"
        base_bin.mkdir(parents=True)
        self.base_python = base_bin / "python3.13"
        self.base_python.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        self.base_python.chmod(0o755)

        self.gate = self.infra / ".venv"
        _fake_python(self.gate)
        self.dispatch = root / ".onex-dispatch-venv"
        self.previous = Path(f"{self.dispatch}.previous")
        _make_venv(self.dispatch, base_bin, _OLD)

        self.bin = root / "shimbin"
        self.bin.mkdir()
        self.cp_log = root / "cp.log"
        self.install_log = root / "install.log"
        self._shims()

    def _shims(self) -> None:
        uv = self.bin / "uv"
        # Lock-clean everywhere: `--check` exits 0, every write is a no-op. The
        # shim never touches pyvenv.cfg, so what the reconciler reads back is
        # what the fixture (or a clone of it) wrote.
        # `uv venv --relocatable --python <py> <dir>` is the one subcommand that
        # builds an environment (the interpreter-drift rebuild uses it).
        uv.write_text(
            "#!/usr/bin/env bash\n"
            'if [[ "$1" == "venv" ]]; then\n'
            '  for a in "$@"; do target="$a"; done\n'
            f'  mkdir -p "$target/bin" && printf "home = {self.base_python.parent}\\n'
            'relocatable = true\\n" > "$target/pyvenv.cfg"\n'
            '  printf "#!/usr/bin/env bash\\ncat >/dev/null\\nexit 0\\n" > "$target/bin/python"\n'
            '  chmod +x "$target/bin/python"\n'
            "fi\n"
            "exit 0\n",
            encoding="utf-8",
        )
        uv.chmod(0o755)
        cp = self.bin / "cp"
        cp.write_text(
            '#!/usr/bin/env bash\nprintf "%s\\n" "$*" >> "$CP_LOG"\nexec /bin/cp "$@"\n',
            encoding="utf-8",
        )
        cp.chmod(0o755)
        install = self.infra / "scripts" / "install-node-skill-package.sh"
        # The co-install records the commit it was asked for in the venv it was
        # pointed at, which is how the next pass's `installed_market_commit`
        # reads it back.
        install.write_text(
            '#!/usr/bin/env bash\nprintf "%s\\n" "$*" >> "$INSTALL_LOG"\n'
            'target="$2"; venv="$(dirname "$(dirname "$target")")"\n'
            'printf "%s\\n" "$OMNIMARKET_REF" > "$venv/.installed_commit"\n',
            encoding="utf-8",
        )
        install.chmod(0o755)
        self.install = install

    def advance_market(self, content: str) -> str:
        self.head = _commit(self.market, content)
        return self.head

    def clones(self) -> list[str]:
        if not self.cp_log.exists():
            return []
        return [
            line
            for line in self.cp_log.read_text(encoding="utf-8").splitlines()
            if str(self.dispatch) in line
        ]

    def repair(self) -> subprocess.CompletedProcess[str]:
        env = {
            **os.environ,
            "OMNIBASE_PATH": str(self.root),
            "PATH": f"{self.bin}:{os.environ['PATH']}",
            "CP_LOG": str(self.cp_log),
            "INSTALL_LOG": str(self.install_log),
            "ONEX_RECONCILE_INSTALL_SCRIPT": str(self.install),
            "ONEX_DISPATCH_BASE_PYTHON": str(self.base_python),
            "CLAUDE_PLUGIN_DATA": str(self.root / "no-such-plugin-data"),
        }
        return subprocess.run(
            ["bash", str(_SCRIPT)],
            capture_output=True,
            text=True,
            env=env,
            check=False,
        )


@pytest.fixture
def host(tmp_path: Path) -> _Host:
    return _Host(tmp_path / "workspace")


def test_second_provider_pass_does_not_reclone_the_venv(host: _Host) -> None:
    first = host.repair()
    assert first.returncode == 0, first.stdout + first.stderr
    assert _installed(host.dispatch) == host.head
    assert len(host.clones()) == 1, "the first pass has no previous generation"
    assert host.previous.is_dir(), "the displaced generation must be kept"
    assert _installed(host.previous) == _OLD

    moved = host.advance_market("two")
    second = host.repair()

    assert second.returncode == 0, second.stdout + second.stderr
    assert _installed(host.dispatch) == moved
    assert len(host.clones()) == 1, (
        "the second pass cloned the whole venv again instead of staging from "
        f"the previous generation: {host.clones()!r}"
    )
    assert host.previous.is_dir()
    assert _installed(host.previous) != moved, "previous must be the displaced one"


def test_diverged_previous_generation_falls_back_to_a_clone(host: _Host) -> None:
    assert host.repair().returncode == 0
    host.advance_market("two")
    # A package the live venv does not carry: reusing this generation would hand
    # the live venv a package set nobody asked for.
    (host.previous / _SITE / "stray-9.9.dist-info").mkdir()

    result = host.repair()

    assert result.returncode == 0, result.stdout + result.stderr
    assert len(host.clones()) == 2, "an untrusted previous generation was reused"
    assert not (host.dispatch / _SITE / "stray-9.9.dist-info").exists()


def test_rebuild_for_a_wrong_interpreter_does_not_keep_the_old_venv(
    host: _Host,
) -> None:
    (host.dispatch / "pyvenv.cfg").write_text(
        "home = /somewhere/else/bin\nrelocatable = true\n", encoding="utf-8"
    )

    result = host.repair()

    assert result.returncode == 0, result.stdout + result.stderr
    assert not host.previous.exists(), (
        "a venv built on the wrong interpreter is scrap, not a staging seed"
    )
