# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Cross-boundary equivalence regression for the release-identity gate (OMN-14471).

The legacy freestanding gate ``scripts/check_release_identity.py`` was refactored
into the canonical ``node_release_identity_compute`` COMPUTE node fronted by a thin
CLI shim at the SAME path. This test proves the refactor is behavior-preserving by
running the FROZEN legacy gate and the NEW shim over the SAME temp git repositories
across every gate branch and asserting IDENTICAL exit codes, stdout, and stderr.

A green that only exercised the new node in isolation would be insufficient (it
could pass while silently diverging from the behavior it replaces); this test binds
the two implementations to the same inputs and demands byte-for-byte equality — and
independently asserts each scenario's expected exit code so both sides going wrong
in the same way cannot pass.

Both gates run in-process via their real ``main(argv)`` (each drives the actual
collector I/O — ``git``/``pyproject.toml`` reads — against the temp repo, then the
real decision logic). Running in-process (rather than one subprocess per scenario)
pays the heavy ``omnibase_infra`` import once instead of ~14x; ``test_real_shim_..``
below additionally runs the REAL shim file as a subprocess to prove it executes as a
script. The frozen legacy gate lives at
``tests/scripts/fixtures/legacy_check_release_identity.py.txt`` (captured verbatim
from ``origin/dev`` at refactor time); the new shim is the file under test. Both
expose module-level ``_REPO_ROOT`` / ``_PYPROJECT`` that the in-process runner
repoints at each temp repo.
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from types import ModuleType

import pytest

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[2]
_NEW_SHIM = _REPO_ROOT / "scripts" / "check_release_identity.py"
_LEGACY_FIXTURE = (
    Path(__file__).resolve().parent
    / "fixtures"
    / "legacy_check_release_identity.py.txt"
)

# A clean env with PYTHONPATH stripped by default so in-process tests do not depend
# on an ambient shadow (reference_pythonpath_shadows_worktree_source).
_CLEAN_ENV = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}


def _load_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# NEW shim: loaded from the real path (pays the omnibase_infra import once).
_NEW = _load_module("_crid_new_shim", _NEW_SHIM)

# FROZEN legacy gate: materialize the .txt fixture as an importable .py and load it.
_LEGACY_DIR = Path(tempfile.mkdtemp(prefix="crid_legacy_"))
_LEGACY_PY = _LEGACY_DIR / "legacy_check_release_identity.py"
_LEGACY_PY.write_text(_LEGACY_FIXTURE.read_text())
_OLD = _load_module("_crid_legacy_gate", _LEGACY_PY)


def _git(root: Path, *args: str) -> str:
    proc = subprocess.run(
        ["git", *args], cwd=root, capture_output=True, text=True, check=True
    )
    return proc.stdout.strip()


def _pyproject(version: str | None) -> str:
    lines = ["[project]", 'name = "equivtest"']
    if version is not None:
        lines.append(f'version = "{version}"')
    return "\n".join(lines) + "\n"


def _write(root: Path, rel: str, content: str = "x\n") -> None:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def _build_repo(
    tmp_path: Path,
    *,
    version: str | None,
    tags: list[str],
    changed: list[str],
) -> tuple[Path, str]:
    """Create a temp git repo. Returns ``(repo_root, base_sha)``.

    ``base_sha`` is the initial commit the tags point at; changed files (if any)
    land in a second commit so a ``--base <base_sha>`` diff surfaces exactly them.
    """
    root = tmp_path
    (root / "scripts").mkdir()
    (root / "pyproject.toml").write_text(_pyproject(version))
    _write(root, "README.md", "base\n")

    _git(root, "init", "-q")
    _git(root, "config", "user.email", "t@example.com")
    _git(root, "config", "user.name", "equiv-test")
    _git(root, "config", "commit.gpgsign", "false")
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "base")
    base_sha = _git(root, "rev-parse", "HEAD")

    for tag in tags:
        _git(root, "tag", tag, base_sha)

    if changed:
        for rel in changed:
            _write(root, rel, "changed\n")
        _git(root, "add", "-A")
        _git(root, "commit", "-q", "-m", "change")

    return root, base_sha


def _run_in_process(
    mod: ModuleType, root: Path, argv: list[str]
) -> tuple[int, str, str]:
    """Run a gate's ``main(argv)`` against ``root`` by repointing its module globals.

    Couples to the gate's module-level ``_REPO_ROOT`` / ``_PYPROJECT`` (both gates
    expose them); this drives the real collector I/O against the temp repo.
    """
    saved_root, saved_pyproject = mod._REPO_ROOT, mod._PYPROJECT
    mod._REPO_ROOT = root
    mod._PYPROJECT = root / "pyproject.toml"
    out, err = io.StringIO(), io.StringIO()
    try:
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            rc = mod.main(argv)
    finally:
        mod._REPO_ROOT, mod._PYPROJECT = saved_root, saved_pyproject
    return rc, out.getvalue(), err.getvalue()


# Each scenario: (id, version, tags, changed, mode, explicit_files, expected_exit).
# mode: "base" (diff vs base_sha) | "strict" (no base/explicit) | "explicit" (--changed-file).
_SCENARIOS = [
    ("bumped_src_change", "2.0.0", ["v1.0.0"], ["src/foo.py"], "base", [], 0),
    ("not_bumped_src_change_fail", "1.0.0", ["v1.0.0"], ["src/foo.py"], "base", [], 1),
    ("exempt_docs_only", "1.0.0", ["v1.0.0"], ["docs/x.md"], "base", [], 0),
    ("config_error_no_version", None, ["v1.0.0"], ["src/foo.py"], "base", [], 2),
    (
        "config_error_malformed",
        "not-a-version",
        ["v1.0.0"],
        ["src/foo.py"],
        "base",
        [],
        2,
    ),
    ("no_published_tag", "1.0.0", [], ["src/foo.py"], "base", [], 0),
    ("bare_semver_tag_ahead", "2.0.0", ["1.5.0"], ["src/foo.py"], "base", [], 0),
    (
        "mixed_tags_pick_highest",
        "1.4.0",
        ["v1.0.0", "v1.3.0", "garbage", "v1.2.0"],
        ["src/foo.py"],
        "base",
        [],
        0,
    ),
    ("strict_no_base_fail", "1.0.0", ["v1.0.0"], [], "strict", [], 1),
    ("strict_no_base_ok", "2.0.0", ["v1.0.0"], [], "strict", [], 0),
    ("explicit_src_fail", "1.0.0", ["v1.0.0"], [], "explicit", ["src/foo.py"], 1),
    ("explicit_docs_ok", "1.0.0", ["v1.0.0"], [], "explicit", ["docs/x.md"], 0),
    (
        "explicit_mixed_src_present",
        "1.0.0",
        ["v1.0.0"],
        [],
        "explicit",
        ["docs/x.md", "src/deep/mod.py"],
        1,
    ),
]


def _args_for(mode: str, base_sha: str, explicit_files: list[str]) -> list[str]:
    if mode == "base":
        return ["--base", base_sha]
    if mode == "strict":
        return []
    if mode == "explicit":
        return [arg for f in explicit_files for arg in ("--changed-file", f)]
    raise AssertionError(f"unknown mode {mode!r}")  # pragma: no cover


@pytest.mark.parametrize("scenario", _SCENARIOS, ids=[s[0] for s in _SCENARIOS])
def test_new_shim_matches_legacy_gate(
    tmp_path: Path,
    scenario: tuple[str, str | None, list[str], list[str], str, list[str], int],
) -> None:
    """The new node/shim must be byte-for-byte identical to the legacy gate."""
    _id, version, tags, changed, mode, explicit_files, expected_exit = scenario
    root, base_sha = _build_repo(tmp_path, version=version, tags=tags, changed=changed)
    args = _args_for(mode, base_sha, explicit_files)

    old_rc, old_out, old_err = _run_in_process(_OLD, root, args)
    new_rc, new_out, new_err = _run_in_process(_NEW, root, args)

    # Byte-for-byte equivalence across the whole boundary.
    assert new_rc == old_rc, f"exit code differs: new={new_rc} old={old_rc}\n{new_err}"
    assert new_out == old_out, f"stdout differs:\nNEW:\n{new_out}\nOLD:\n{old_out}"
    assert new_err == old_err, f"stderr differs:\nNEW:\n{new_err}\nOLD:\n{old_err}"

    # Independently pin the expected exit code so both sides breaking the same way
    # cannot pass (prove-RED-against-exists-but-wrong discipline).
    assert new_rc == expected_exit, (
        f"scenario {_id}: expected exit {expected_exit}, got {new_rc}\n"
        f"stdout={new_out!r} stderr={new_err!r}"
    )


def test_real_shim_resists_pythonpath_shadow_from_editable_install(
    tmp_path: Path,
) -> None:
    """The real shim must resolve ``omnibase_infra`` from repo-local ``src/``, not
    a decoy earlier on ``PYTHONPATH`` (OMN-14504).

    Runs the REAL ``scripts/check_release_identity.py`` at its REAL repo path
    (never copied) via ``sys.executable`` — this test suite runs under this
    project's own ``uv``-managed venv, whose editable install already appended
    the literal ``_SRC_DIR`` string to ``sys.path`` (via a plain-path ``.pth``
    file processed at interpreter startup) before the shim's own bootstrap runs.
    That is the exact condition that makes a ``not in sys.path`` guard a no-op:
    if the bootstrap only inserted when NOT already present, it would never fire
    here, and an adversarial ``PYTHONPATH`` entry (which lands ahead of
    site-packages/``.pth`` entries) would win the import.

    A copied shim (as ``test_real_shim_subprocess_fail_emits_two_guidance_lines``
    above uses) cannot exercise this: a copy's self-located ``_SRC_DIR`` points
    under a src-less temp dir, so ``_SRC_DIR.is_dir()`` is False and the guard
    never reaches the ``sys.path`` check at all — a green there would be a
    RED-on-absence vacuous pass, not proof of shadow resistance.
    """
    decoy_root = tmp_path / "decoy_pythonpath"
    pkg_dir = decoy_root / "omnibase_infra" / "nodes" / "node_release_identity_compute"
    pkg_dir.mkdir(parents=True)
    (decoy_root / "omnibase_infra" / "__init__.py").write_text("")
    (decoy_root / "omnibase_infra" / "nodes" / "__init__.py").write_text("")
    (pkg_dir / "__init__.py").write_text(
        "class ModelReleaseIdentityRequest:\n"
        "    def __init__(self, **kw):\n"
        "        self.__dict__.update(kw)\n"
        "\n"
        "\n"
        "class HandlerReleaseIdentity:\n"
        "    def handle(self, request):\n"
        "        raise RuntimeError('CRID_DECOY_SENTINEL_9f3a')\n"
    )

    proc = subprocess.run(
        [sys.executable, str(_NEW_SHIM), "--changed-file", "docs/x.md"],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        env={**_CLEAN_ENV, "PYTHONPATH": str(decoy_root)},
        check=False,
    )

    assert "CRID_DECOY_SENTINEL_9f3a" not in proc.stderr, (
        "PYTHONPATH shadow won: the shim resolved the decoy omnibase_infra "
        "package instead of the real repo-local src/.\n"
        f"returncode={proc.returncode}\nstdout={proc.stdout!r}\nstderr={proc.stderr!r}"
    )
    # docs-only change is exempt regardless of this real repo's live version/tag
    # state (Step 4 of HandlerReleaseIdentity.handle short-circuits on no
    # packaged-source change) — deterministic exit 0 proves the REAL handler ran.
    assert proc.returncode == 0, (
        f"expected the real gate's exempt/docs-only decision (exit 0); got "
        f"{proc.returncode}.\nstdout={proc.stdout!r}\nstderr={proc.stderr!r}"
    )
    assert proc.stdout.startswith("OK: no packaged src/** change"), proc.stdout


def test_real_shim_subprocess_fail_emits_two_guidance_lines(tmp_path: Path) -> None:
    """End-to-end: the REAL shim file runs as a script and emits the FAIL guidance.

    Proves the committed ``scripts/check_release_identity.py`` executes standalone
    (shebang, imports resolve, repo-root detection) — not just its ``main`` in-process.
    """
    root, base_sha = _build_repo(
        tmp_path, version="1.0.0", tags=["v1.0.0"], changed=["src/foo.py"]
    )
    shim = root / "scripts" / "check_release_identity.py"
    shutil.copyfile(_NEW_SHIM, shim)

    proc = subprocess.run(
        [sys.executable, str(shim), "--base", base_sha],
        cwd=root,
        capture_output=True,
        text=True,
        env={**_CLEAN_ENV, "PYTHONPATH": str(_REPO_ROOT / "src")},
        check=False,
    )
    assert proc.returncode == 1
    assert proc.stdout == ""
    lines = proc.stderr.splitlines()
    assert len(lines) == 2, f"expected exactly two stderr lines, got: {proc.stderr!r}"
    assert lines[0].startswith("FAIL: packaged source changed but pyproject version")
    assert "1.0.0" in lines[0]
    assert lines[1].startswith("Merging code onto an already-published version")
    assert "e.g. 1.0.1" in lines[1]
