# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""stage_workspace.sh must invoke check_sibling_lock_pins.py with an interpreter
that actually has pydantic installed, not the bare ``python3`` resolved off
PATH (OMN-15131).

Live failure (run https://github.com/OmniNode-ai/omnibase_infra/actions/runs/30170235015,
2026-07-25T18:45Z), one step after the OMN-15122 fix landed (step 3b contract
resolution succeeded -- "Copied 5 runtime contract YAMLs from omnibase_core"):

    Traceback (most recent call last):
      File ".../scripts/runtime_build/check_sibling_lock_pins.py", line 83, in <module>
        from pydantic import BaseModel, ConfigDict, Field
    ModuleNotFoundError: No module named 'pydantic'
    ERROR: sibling-pin preflight failed against .../omnimarket/uv.lock

Root cause, confirmed directly on the omninode-deploy-runner container (not
inferred): the bare ``python3`` resolved off PATH there is a system
interpreter with zero packages installed -- not even pydantic. The crash is
misreported downstream as a lock-drift condition (OMN-12977 wording); it is
not -- the check never ran far enough to compare a single pin.

Verified on the same container: both the repo's own
``.venv/bin/python`` (built by deploy-runtime.sh's own ``uv sync`` earlier in
the job) and ``uv run python`` have pydantic (2.13.4) importable. Only bare
``python3`` does not.

The fix mirrors the interpreter-resolution precedence deploy-runtime.sh's own
``check_sibling_lock_pins()`` bash function already uses for this exact
script (repo-venv python -> uv run -> bare python3, hard-failing only if none
resolve) instead of introducing a second, divergent resolution order.

These tests extract and execute the pure
``resolve_sibling_lock_pins_python()`` bash function from stage_workspace.sh
in isolation against a real (fake) filesystem -- proving RED against the
exact "bare python3 has no pydantic" condition, not merely asserting the
source text mentions the right strings.
"""

from __future__ import annotations

import re
import stat
import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]
STAGE_SCRIPT = REPO_ROOT / "scripts" / "runtime_build" / "stage_workspace.sh"


def _script_text() -> str:
    return STAGE_SCRIPT.read_text(encoding="utf-8")


def _extract_function(name: str) -> str:
    """Return the source text of a single top-level bash function ``name()``.

    Mirrors the extraction helper used for OMN-15122's
    ``resolve_core_contracts_dir()`` tests: the function under test is pure
    (no top-level script deps beyond ``PWD``/``uv``/``python3`` on PATH), so
    it can be executed in isolation for a real behavioral assertion without
    sourcing (and running the ``set -euo pipefail`` top-level body / RT-1
    checkout logic) of the whole script.
    """
    text = _script_text()
    match = re.search(
        rf"^{re.escape(name)}\s*\(\)\s*\{{.*?\n\}}",
        text,
        re.DOTALL | re.MULTILINE,
    )
    assert match is not None, (
        f"could not extract function {name}() from stage_workspace.sh"
    )
    return match.group(0)


def _write_executable(path: Path, body: str) -> None:
    path.write_text(body, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)


def _run_resolver(
    *, cwd: Path, stub_bin_dir: Path, have_venv_python: bool, have_uv: bool
) -> subprocess.CompletedProcess[str]:
    """Execute the extracted resolver in a harness with a deterministic PATH.

    ``have_venv_python`` controls whether ``<cwd>/.venv/bin/python`` exists
    and is executable (the repo-venv precedence branch). ``have_uv`` controls
    whether a stub ``uv`` is placed on PATH (the fallback branch). Neither
    stub actually imports pydantic -- these tests assert which interpreter
    *gets selected*, matching the live-verified fact (recorded above) that
    the venv-python and uv-run branches both resolve to environments with
    pydantic and the bare-python3 branch does not.
    """
    if have_venv_python:
        venv_bin = cwd / ".venv" / "bin"
        venv_bin.mkdir(parents=True, exist_ok=True)
        _write_executable(venv_bin / "python", "#!/usr/bin/env bash\nexit 0\n")

    path_entries = [str(stub_bin_dir)]
    if have_uv:
        _write_executable(stub_bin_dir / "uv", "#!/usr/bin/env bash\nexit 0\n")
    # Always provide a bare python3 stub so the "no interpreter available"
    # branch is exercised only when explicitly intended by a given test.
    _write_executable(stub_bin_dir / "python3", "#!/usr/bin/env bash\nexit 0\n")

    harness = "\n".join(
        [
            "set -uo pipefail",  # no -e: need the resolver's own stdout+exit code
            f'export PATH="{":".join(path_entries)}:/usr/bin:/bin"',
            f'cd "{cwd}"',
            _extract_function("resolve_sibling_lock_pins_python"),
            "resolve_sibling_lock_pins_python",
        ]
    )
    return subprocess.run(
        ["bash", "-c", harness],
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.mark.unit
def test_defines_resolve_sibling_lock_pins_python_function() -> None:
    text = _script_text()
    assert re.search(r"^resolve_sibling_lock_pins_python\s*\(\)", text, re.MULTILINE), (
        "stage_workspace.sh must define resolve_sibling_lock_pins_python()"
    )


@pytest.mark.unit
def test_check_sibling_lock_pins_no_longer_invoked_with_bare_python3_unconditionally() -> (
    None
):
    """Regression guard for the exact OMN-15131 line: the preflight call site
    must route through the resolver, not a bare ``python3 "${SCRIPT_DIR}/..."``
    literal.
    """
    text = _script_text()
    assert 'python3 "${SCRIPT_DIR}/check_sibling_lock_pins.py"' not in text, (
        "check_sibling_lock_pins.py must not be invoked with an unconditional bare python3"
    )
    assert "resolve_sibling_lock_pins_python" in text


@pytest.mark.unit
def test_resolver_prefers_repo_venv_python_when_present(tmp_path: Path) -> None:
    stub_bin_dir = tmp_path / "stubbin"
    stub_bin_dir.mkdir()
    cwd = tmp_path / "repo_root"
    cwd.mkdir()

    result = _run_resolver(
        cwd=cwd, stub_bin_dir=stub_bin_dir, have_venv_python=True, have_uv=True
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == str(cwd / ".venv" / "bin" / "python")


@pytest.mark.unit
def test_resolver_falls_back_to_uv_run_when_no_repo_venv(tmp_path: Path) -> None:
    stub_bin_dir = tmp_path / "stubbin"
    stub_bin_dir.mkdir()
    cwd = tmp_path / "repo_root"
    cwd.mkdir()

    result = _run_resolver(
        cwd=cwd, stub_bin_dir=stub_bin_dir, have_venv_python=False, have_uv=True
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "uv-run"


@pytest.mark.unit
def test_resolver_falls_back_to_bare_python3_only_as_last_resort(
    tmp_path: Path,
) -> None:
    stub_bin_dir = tmp_path / "stubbin"
    stub_bin_dir.mkdir()
    cwd = tmp_path / "repo_root"
    cwd.mkdir()

    result = _run_resolver(
        cwd=cwd, stub_bin_dir=stub_bin_dir, have_venv_python=False, have_uv=False
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "python3"


@pytest.mark.unit
def test_resolver_reproduces_live_failure_condition_directly() -> None:
    """This is the exact condition observed on the omninode-deploy-runner
    container (2026-07-25): no repo .venv staged yet at the point this
    preflight runs, but ``uv`` IS on PATH. Before the fix, the call site
    ignored both and always ran bare ``python3`` -- which on that container
    has no pydantic -- producing the live ModuleNotFoundError traceback.
    The resolver must select the uv-run branch here, not python3.
    """
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        stub_bin_dir = tmp_path / "stubbin"
        stub_bin_dir.mkdir()
        cwd = tmp_path / "repo_root"
        cwd.mkdir()

        result = _run_resolver(
            cwd=cwd, stub_bin_dir=stub_bin_dir, have_venv_python=False, have_uv=True
        )

        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() != "python3", (
            "resolver must not select bare python3 when uv is available -- "
            "this is the exact OMN-15131 live-failure condition"
        )
