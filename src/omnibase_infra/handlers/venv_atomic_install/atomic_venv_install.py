# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Atomic in-place install of the packages a venv pass moves (OMN-22712).

The reconcile's additive pass used to clone the whole live dispatch venv (about
57,000 files), install into the clone, swap it in and delete the old tree: about
57,000 creates and 57,000 removes a pass, about 45 passes a day on a lab Mac,
which is what grows fseventsd without bound. This effect installs ONLY the
packages the plan moves into a side directory, exposes the venv plus those
packages to the caller's checks through a self-locating interpreter wrapper, and
puts each package directory in place with one exchange syscall and a journal that
restores every path on a failure after that.

It is stdlib-only because it runs under the interpreter of the venv it repairs,
before that venv is known to be healthy. The refusal of a downgrade or a removal
is the plan tool's (``scripts/venv_install_plan.py``), loaded by path so the two
cannot disagree.

Commands (exit codes: 0 ok, 1 not supported / failed, 2 usage, 3 refused, 4 uv
failed, 5 commit failed): ``supported``, ``exchange``, ``stage``, ``commit``,
``rollback``, ``finalize``. The command line is the node runtime's
(``nodes/node_venv_atomic_install/runtime_venv_atomic_install.py``); the typed
operation is ``HandlerVenvAtomicInstall``.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from typing import TextIO

EXIT_OK = 0
EXIT_USAGE = 2
EXIT_REFUSED = 3
EXIT_UV_ERROR = 4
EXIT_COMMIT_FAILED = 5

# Any failure of the commit must restore the venv: the file system, the probe
# of the live footprint, a malformed journal or plan file.
_RESTORE_ERRORS = (
    OSError,
    RuntimeError,
    ValueError,
    KeyError,
    subprocess.SubprocessError,
)

_PLAN_FILE = "plan.json"
_JOURNAL_FILE = "journal.json"
_GIT_REVISIONS_FILE = "git-revisions.txt"
_GIT_REVISION_REQUIREMENT = re.compile(r"git\+\S+?@([0-9a-fA-F]{7,40})(?=$|[\s#])")
_MIN_REVISION_FOLDER = 7
_STAGE_SITE = "site"
_STAGE_BIN = "bin"
_STAGE_BACKUP = "removed"
_WRAPPER = "python"
_SKIP_TOP = frozenset({".lock", "bin", "__pycache__"})
_PLAN_TOOL = Path(__file__).resolve().parents[4] / "scripts" / "venv_install_plan.py"
_EXEC_LINE = re.compile(
    r"""(\"\$\(dirname -- \"\$\(realpath -- \"\$0\"\)\"\)\")/'[^']*/([^'/]+)'"""
)
_OLD_FILES_PROBE = r"""
import json, sys
from importlib.metadata import PackageNotFoundError, distribution
out = {}
for name in sys.argv[1:]:
    try:
        dist = distribution(name)
    except PackageNotFoundError:
        continue
    tops, scripts = set(), set()
    for f in dist.files or []:
        parts = f.parts
        if parts and parts[0] == "..":
            if len(parts) > 1 and parts[-2] == "bin":
                scripts.add(parts[-1])
        elif parts:
            tops.add(parts[0])
    out[name] = {"tops": sorted(tops), "scripts": sorted(scripts)}
print(json.dumps(out))
"""


def _say(text: str = "", *, file: TextIO | None = None) -> None:
    """One line to stdout, or to ``file``; the tool's output is its interface."""
    (file or sys.stdout).write(text + "\n")


def _load_plan_tool() -> ModuleType:
    spec = importlib.util.spec_from_file_location("venv_install_plan", _PLAN_TOOL)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {_PLAN_TOOL}")
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("venv_install_plan", module)
    spec.loader.exec_module(module)
    return module


def _clean_env() -> dict[str, str]:
    return {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}


def exchange(a: Path, b: Path) -> bool:
    """Swap two paths in ONE syscall; False when the platform cannot."""
    libc = ctypes.CDLL(ctypes.util.find_library("c") or None, use_errno=True)
    fa, fb = os.fsencode(str(a)), os.fsencode(str(b))
    if sys.platform == "darwin":
        rc = libc.renamex_np(fa, fb, 2)  # RENAME_SWAP
    elif hasattr(libc, "renameat2"):
        rc = libc.renameat2(-100, fa, -100, fb, 2)  # AT_FDCWD, RENAME_EXCHANGE
    else:
        return False
    return rc == 0


def exchange_supported() -> bool:
    libc = ctypes.CDLL(ctypes.util.find_library("c") or None, use_errno=True)
    return sys.platform == "darwin" or hasattr(libc, "renameat2")


def _venv_paths(python_bin: str) -> tuple[Path, Path]:
    """Return (site-packages, bin dir) of the venv ``python_bin`` belongs to."""
    out = subprocess.run(
        [python_bin, "-c", "import sysconfig;print(sysconfig.get_paths()['purelib'])"],
        capture_output=True,
        text=True,
        check=False,
        env=_clean_env(),
    )
    if out.returncode != 0 or not out.stdout.strip():
        raise RuntimeError(f"cannot read site-packages from {python_bin}: {out.stderr}")
    return Path(out.stdout.strip()), Path(python_bin).parent


def _relocate_script(path: Path) -> None:
    """Make a staged console script self-locating, as the live ones are.

    ``uv pip install --target`` writes ``dirname/'<abs python>'``, which joins
    into a path that does not exist. Only the relocatable form is rewritten; an
    absolute ``#!`` line already names the live interpreter and is left alone.
    """
    text = path.read_text(encoding="utf-8")
    fixed = _EXEC_LINE.sub(r"\1/'\2'", text)
    if fixed != text:
        path.write_text(fixed, encoding="utf-8")


def _requirement_name(plan: ModuleType, requirement: str) -> str:
    match = re.match(r"\s*([A-Za-z0-9][A-Za-z0-9._-]*)", requirement)
    return plan.normalize_name(match.group(1)) if match else ""


def _plan_requirements(
    plan: ModuleType, changes: list, originals: list[str]
) -> list[str]:
    """One requirement per planned change.

    A requirement the caller named is used verbatim (it carries the VCS source
    or the specifier uv resolved from); a package only the resolver pulled in is
    pinned to the version the plan printed.
    """
    named = {_requirement_name(plan, r): r for r in originals}
    reqs: list[str] = []
    for change in changes:
        if change.after is None:
            continue
        if change.name in named:
            reqs.append(named[change.name])
        elif change.verdict is plan.PlanVerdict.SOURCE:
            reqs.append(f"{change.name} @ {change.after}")
        else:
            reqs.append(f"{change.name}=={change.after}")
    return reqs


def write_wrapper(python_bin: str, stage_dir: Path) -> Path:
    """An interpreter that sees the venv PLUS the staged packages.

    The checks that run before the exchange take this path as their interpreter:
    the venv is the live one, the staged packages are first on its path, and the
    answer describes the venv as it will be after the exchange. The interpreter
    writes no bytecode: a .pyc written into the stage costs a create and a rename
    per source file and records the stage path, which is gone after the exchange.
    """
    site = stage_dir / _STAGE_SITE
    site.mkdir(parents=True, exist_ok=True)
    wrapper = stage_dir / _WRAPPER
    wrapper.write_text(
        f"#!/bin/sh\nexec env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH='{site}' '{python_bin}' \"$@\"\n",
        encoding="utf-8",
    )
    wrapper.chmod(0o755)
    return wrapper


def record_git_revisions(stage_dir: Path, requirements: list[str]) -> list[str]:
    """Record the git shas the requirements name, for the finalize prune."""
    shas: list[str] = []
    for requirement in requirements:
        match = _GIT_REVISION_REQUIREMENT.search(requirement)
        if match is None:
            continue
        sha = match.group(1).lower()
        if sha not in shas:
            shas.append(sha)
    if shas:
        stage_dir.mkdir(parents=True, exist_ok=True)
        (stage_dir / _GIT_REVISIONS_FILE).write_text(
            "\n".join(shas) + "\n", encoding="utf-8"
        )
    return shas


def prune_git_checkouts(stage_dir: Path) -> list[Path]:
    """Remove the uv git checkouts the recorded revisions resolved to, and only those."""
    revisions_file = stage_dir / _GIT_REVISIONS_FILE
    if not revisions_file.is_file():
        return []
    try:
        shas = [
            line.strip().lower()
            for line in revisions_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except OSError as exc:
        _say(f"  cannot read the recorded git revisions: {exc}", file=sys.stderr)
        return []
    if not shas:
        return []
    try:
        out = subprocess.run(
            ["uv", "cache", "dir"],
            capture_output=True,
            text=True,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        _say(f"  cannot ask uv for its cache dir: {exc}", file=sys.stderr)
        return []
    if out.returncode != 0 or not out.stdout.strip():
        _say(f"  uv cache dir failed: {out.stderr.strip()}", file=sys.stderr)
        return []
    cache = Path(out.stdout.strip())
    removed: list[Path] = []
    for path in sorted(cache.glob("git-v*/checkouts/*/*")):
        if path.is_symlink() or not path.is_dir():
            continue
        if len(path.name) < _MIN_REVISION_FOLDER:
            continue
        if not any(sha.startswith(path.name) for sha in shas):
            continue
        try:
            shutil.rmtree(path)
            path.with_name(path.name + ".ok").unlink(missing_ok=True)
            removed.append(path)
            _say(f"  pruned uv git checkout {path}")
        except OSError as exc:
            _say(f"  cannot prune {path}: {exc}", file=sys.stderr)
    return removed


def stage(
    python_bin: str,
    stage_dir: Path,
    requirements: list[str],
    *,
    no_deps: bool,
    label: str,
) -> int:
    """Plan (refusing a downgrade or removal), then install only the moved packages."""
    plan = _load_plan_tool()
    write_wrapper(python_bin, stage_dir)
    record_git_revisions(stage_dir, requirements)
    _say(f"== {label}: staging only the moved packages (atomic in-place) ==")
    dry = plan._run_uv(python_bin, requirements, no_deps=no_deps, dry_run=True)
    combined = f"{dry.stdout}\n{dry.stderr}"
    if dry.returncode != 0:
        _say(combined.strip(), file=sys.stderr)
        _say(f"ERROR: uv could not resolve the plan for {label}.", file=sys.stderr)
        return EXIT_UV_ERROR
    changes = plan.parse_plan(combined)
    if not changes:
        _say(f"  no change — every requirement is already satisfied ({label}).")
        return EXIT_OK
    for change in changes:
        _say(change.render())
    refused = [c for c in changes if c.verdict in plan.REFUSING_VERDICTS]
    if refused:
        _say(
            f"REFUSED: {label} would move installed package(s) backwards.",
            file=sys.stderr,
        )
        for change in refused:
            _say(
                f"  {change.name}: {change.before} -> {change.after} ({change.verdict.value})",
                file=sys.stderr,
            )
        _say("  Nothing was changed (OMN-18675).", file=sys.stderr)
        return EXIT_REFUSED
    site = stage_dir / _STAGE_SITE
    argv = ["uv", "pip", "install", "--python", python_bin, "--no-deps", "--target"]
    real = subprocess.run(
        [*argv, str(site), *_plan_requirements(plan, changes, requirements)],
        capture_output=True,
        text=True,
        check=False,
    )
    if real.returncode != 0:
        _say(f"{real.stdout}\n{real.stderr}".strip(), file=sys.stderr)
        _say(
            f"ERROR: staging the install failed (exit {real.returncode}).",
            file=sys.stderr,
        )
        return EXIT_UV_ERROR
    stage_bin = stage_dir / _STAGE_BIN
    stage_bin.mkdir(exist_ok=True)
    if (site / "bin").is_dir():
        for script in sorted((site / "bin").iterdir()):
            _relocate_script(script)
            script.replace(stage_bin / script.name)
        (site / "bin").rmdir()
    names = {c.name for c in changes if c.after is not None}
    plan_file = stage_dir / _PLAN_FILE
    if plan_file.exists():
        names |= set(json.loads(plan_file.read_text(encoding="utf-8")))
    plan_file.write_text(json.dumps(sorted(names)), encoding="utf-8")
    _say(f"  staged {len(names)} package(s) in {stage_dir} (live venv untouched)")
    return EXIT_OK


def _live_footprint(
    python_bin: str, names: list[str]
) -> dict[str, dict[str, list[str]]]:
    out = subprocess.run(
        [python_bin, "-c", _OLD_FILES_PROBE, *names],
        capture_output=True,
        text=True,
        check=False,
        env=_clean_env(),
    )
    if out.returncode != 0:
        raise RuntimeError(f"cannot read the live footprint: {out.stderr}")
    return json.loads(out.stdout)


def _journal_write(stage_dir: Path, journal: list[dict[str, str]]) -> None:
    (stage_dir / _JOURNAL_FILE).write_text(json.dumps(journal), encoding="utf-8")


def _remove_aside(
    stage_dir: Path, journal: list[dict[str, str]], live: Path, key: str
) -> None:
    backup = stage_dir / _STAGE_BACKUP / key
    backup.parent.mkdir(parents=True, exist_ok=True)
    live.rename(backup)
    journal.append({"op": "removed", "live": str(live), "stage": str(backup)})
    _journal_write(stage_dir, journal)


def _put(
    stage_dir: Path, journal: list[dict[str, str]], src: Path, dst: Path, key: str
) -> None:
    """Move ``src`` (staged) to ``dst`` (live); the old ``dst`` lands in ``src``."""
    if dst.exists() or dst.is_symlink():
        if exchange(src, dst):
            journal.append({"op": "exchanged", "live": str(dst), "stage": str(src)})
        else:
            _remove_aside(stage_dir, journal, dst, key)
            src.rename(dst)
            journal.append({"op": "added", "live": str(dst), "stage": str(src)})
    else:
        src.rename(dst)
        journal.append({"op": "added", "live": str(dst), "stage": str(src)})
    _journal_write(stage_dir, journal)


def _undo(journal: list[dict[str, str]]) -> None:
    for entry in reversed(journal):
        live, stage_path = Path(entry["live"]), Path(entry["stage"])
        if entry["op"] == "exchanged":
            if not exchange(stage_path, live):
                raise RuntimeError(f"cannot exchange {live} back")
        elif entry["op"] == "added":
            live.rename(stage_path)
        elif entry["op"] == "removed":
            stage_path.rename(live)


def commit_stage(python_bin: str, stage_dir: Path) -> int:
    """Put every staged package directory in place, journalled and reversible."""
    plan_file = stage_dir / _PLAN_FILE
    if not plan_file.exists():
        _say("  nothing staged; nothing to commit.")
        return EXIT_OK
    names = json.loads(plan_file.read_text(encoding="utf-8"))
    site, bin_dir = _venv_paths(python_bin)
    stage_site, stage_bin = stage_dir / _STAGE_SITE, stage_dir / _STAGE_BIN
    journal: list[dict[str, str]] = []
    try:
        old = _live_footprint(python_bin, names)
        new_tops = sorted(
            e.name for e in stage_site.iterdir() if e.name not in _SKIP_TOP
        )
        new_scripts = (
            sorted(e.name for e in stage_bin.iterdir()) if stage_bin.is_dir() else []
        )
        # Package directories first, dist-info last: the import and the metadata
        # each change in one syscall; the pair is consistent again once the last
        # exchange returns, and a failure anywhere restores every one of them.
        for top in sorted(new_tops, key=lambda t: t.endswith(".dist-info")):
            _put(stage_dir, journal, stage_site / top, site / top, f"site/{top}")
        for script in new_scripts:
            _put(
                stage_dir,
                journal,
                stage_bin / script,
                bin_dir / script,
                f"bin/{script}",
            )
        for foot in old.values():
            for top in foot["tops"]:
                if top not in new_tops and (site / top).exists():
                    _remove_aside(stage_dir, journal, site / top, f"site/{top}")
            for script in foot["scripts"]:
                if script not in new_scripts and (bin_dir / script).exists():
                    _remove_aside(stage_dir, journal, bin_dir / script, f"bin/{script}")
    except _RESTORE_ERRORS as exc:
        _say(
            f"ERROR: commit failed ({type(exc).__name__}: {exc}); restoring.",
            file=sys.stderr,
        )
        try:
            _undo(journal)
        except _RESTORE_ERRORS as undo_exc:
            _say(f"ERROR: restore failed too: {undo_exc}", file=sys.stderr)
        return EXIT_COMMIT_FAILED
    _say(
        f"  committed {len(names)} package(s); {len(journal)} path(s) exchanged in place"
    )
    return EXIT_OK


def rollback_stage(stage_dir: Path) -> int:
    """Undo a committed stage from its journal."""
    journal_file = stage_dir / _JOURNAL_FILE
    if not journal_file.exists():
        _say("  no journal; nothing to roll back.")
        return EXIT_OK
    try:
        _undo(json.loads(journal_file.read_text(encoding="utf-8")))
    except _RESTORE_ERRORS as exc:
        _say(f"ERROR: rollback failed: {exc}", file=sys.stderr)
        return EXIT_COMMIT_FAILED
    journal_file.unlink()
    _say("  rolled back: the previous packages are back in the live venv")
    return EXIT_OK


def run_command(
    command: str,
    *,
    python_bin: str | None = None,
    stage_dir: Path | None = None,
    requirements: list[str] | None = None,
    no_deps: bool = False,
    label: str = "install",
    paths: list[str] | None = None,
) -> int:
    """Run one subcommand; the runtime CLI and the typed handler both end here."""
    if command == "supported":
        return EXIT_OK if exchange_supported() else 1
    if command == "exchange":
        if not paths or len(paths) != 2:
            return EXIT_USAGE
        return EXIT_OK if exchange(Path(paths[0]), Path(paths[1])) else 1
    if stage_dir is None:
        _say(f"ERROR: {command} needs a stage dir.", file=sys.stderr)
        return EXIT_USAGE
    if command == "finalize":
        prune_git_checkouts(stage_dir)
        shutil.rmtree(stage_dir, ignore_errors=True)
        return EXIT_OK
    if command == "rollback":
        return rollback_stage(stage_dir)
    if python_bin is None:
        _say(f"ERROR: {command} needs a python.", file=sys.stderr)
        return EXIT_USAGE
    if command == "commit":
        return commit_stage(python_bin, stage_dir)
    if command == "stage":
        return stage(
            python_bin,
            stage_dir,
            requirements or [],
            no_deps=no_deps,
            label=label,
        )
    return EXIT_USAGE
