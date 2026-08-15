# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Tests for the OMN-13419 single-canonical in-memory bus import gate.

Also covers the OMN-14988 path-normalization regression: BSD grep (macOS)
reports the search root ``src/`` as ``src//`` in every hit, and the original
normalization used the bash-version-dependent ``${file//\\/\\//\\/}`` form,
which retains a literal backslash under bash 3.2 (stock macOS
``/usr/bin/env bash``). Every allowlisted file was therefore reported as a
violation on macOS -- 8 false positives, 0 true positives.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[4]
SCRIPT = REPO_ROOT / "scripts" / "validation" / "check_no_infra_inmemory_import.sh"

# One allowlisted hit per allowlisted module, reproduced verbatim from the
# `.200` (macOS / BSD grep) reproduction recorded on OMN-14988. Note the
# ``src//`` doubled slash: that is what BSD grep emits, and what the gate must
# normalize before comparing against its own single-slash ALLOWLIST.
BSD_GREP_ALLOWLISTED_HITS = (
    "src//omnibase_infra/backends/auto_configure.py:38:"
    "    from omnibase_infra.event_bus.event_bus_inmemory import EventBusInmemory as _Cls",
    "src//omnibase_infra/runtime/runtime_host_process.py:91:"
    "from omnibase_infra.event_bus.event_bus_inmemory import EventBusInmemory",
    "src//omnibase_infra/runtime/util_wiring.py:148:"
    "from omnibase_infra.event_bus.event_bus_inmemory import EventBusInmemory",
    "src//omnibase_infra/runtime/service_kernel.py:104:"
    "from omnibase_infra.event_bus.event_bus_inmemory import EventBusInmemory",
    "src//omnibase_infra/runtime/transition_notification_publisher.py:752:"
    "    from omnibase_infra.event_bus.event_bus_inmemory import EventBusInmemory",
    "src//omnibase_infra/event_bus/__init__.py:44:"
    "from omnibase_infra.event_bus.event_bus_inmemory import EventBusInmemory",
    "src//omnibase_infra/event_bus/testing/adapter_protocol_event_publisher_inmemory.py:18:"
    "    from omnibase_infra.event_bus import EventBusInmemory",
    "src//omnibase_infra/event_bus/testing/adapter_protocol_event_publisher_inmemory.py:65:"
    "    from omnibase_infra.event_bus.event_bus_inmemory import (",
)


def _bash_interpreters() -> list[str]:
    """Every distinct bash on this machine, so bash 3.2 is covered where present.

    On macOS this yields both ``/bin/bash`` (3.2.57 -- the interpreter
    ``#!/usr/bin/env bash`` resolves to under the stock macOS PATH, and the one
    that exhibited OMN-14988) and any newer Homebrew bash. On Linux CI it
    yields the single 5.x bash.
    """
    candidates = [
        "/bin/bash",
        shutil.which("bash"),
        "/opt/homebrew/bin/bash",
        "/usr/local/bin/bash",
    ]
    seen: set[str] = set()
    found: list[str] = []
    for candidate in candidates:
        if not candidate:
            continue
        real = os.path.realpath(candidate)
        if real in seen or not os.access(real, os.X_OK):
            continue
        seen.add(real)
        found.append(candidate)
    return found


BASH_INTERPRETERS = _bash_interpreters()


def _install_script(tmp_path: Path) -> Path:
    script_path = tmp_path / "scripts" / "validation" / SCRIPT.name
    script_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(SCRIPT, script_path)
    return script_path


def _run_gate(
    tmp_path: Path,
    *,
    bash: str = "bash",
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    script_path = _install_script(tmp_path)

    return subprocess.run(
        [bash, str(script_path)],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=False,
        env=env,
    )


def _shell_quote(value: str) -> str:
    return "'" + value.replace("'", "'\\''") + "'"


def _install_bsd_grep_stub(tmp_path: Path, hits: tuple[str, ...]) -> dict[str, str]:
    """Shadow ``grep`` with a stub emitting BSD-style ``src//`` hits.

    Pins the doubled-slash input so the gate's normalization is exercised on
    every platform, not only on machines whose grep happens to be BSD grep.
    """
    bin_dir = tmp_path / "stub-bin"
    bin_dir.mkdir(parents=True, exist_ok=True)
    stub = bin_dir / "grep"
    body = "\n".join(f"printf '%s\\n' {_shell_quote(hit)}" for hit in hits)
    stub.write_text(f"#!/bin/sh\n{body}\nexit 0\n", encoding="utf-8")
    stub.chmod(0o755)

    env = dict(os.environ)
    env["PATH"] = f"{bin_dir}{os.pathsep}{env.get('PATH', '')}"
    return env


# --------------------------------------------------------------------------
# OMN-13419: the gate's original behavior (must not regress while fixing 14988)
# --------------------------------------------------------------------------


def test_gate_rejects_direct_module_import(tmp_path: Path) -> None:
    module = tmp_path / "src" / "omnibase_infra" / "nodes" / "bad_import.py"
    module.parent.mkdir(parents=True)
    module.write_text(
        "import omnibase_infra.event_bus.event_bus_inmemory as infra_bus\n",
        encoding="utf-8",
    )

    result = _run_gate(tmp_path)

    assert result.returncode == 1
    assert "Single-canonical in-memory bus" in result.stdout
    assert "bad_import.py" in result.stdout


def test_gate_allows_allowlisted_adapter_import(tmp_path: Path) -> None:
    adapter = (
        tmp_path / "src" / "omnibase_infra" / "event_bus" / "event_bus_inmemory.py"
    )
    adapter.parent.mkdir(parents=True)
    adapter.write_text(
        "import omnibase_infra.event_bus.event_bus_inmemory as infra_bus\n",
        encoding="utf-8",
    )

    result = _run_gate(tmp_path)

    assert result.returncode == 0
    assert "OK: no disallowed imports" in result.stdout


# --------------------------------------------------------------------------
# OMN-14988: doubled-slash path normalization
# --------------------------------------------------------------------------


@pytest.mark.parametrize("bash", BASH_INTERPRETERS)
@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        # The exact reported case.
        (
            "src//omnibase_infra/backends/auto_configure.py",
            "src/omnibase_infra/backends/auto_configure.py",
        ),
        # Already-normalized input is a no-op (GNU grep on Linux CI).
        (
            "src/omnibase_infra/backends/auto_configure.py",
            "src/omnibase_infra/backends/auto_configure.py",
        ),
        # Multiple and longer slash runs collapse fully, not partially.
        ("src//a//b.py", "src/a/b.py"),
        ("src///a.py", "src/a.py"),
        ("src////a.py", "src/a.py"),
    ],
)
def test_path_normalization_collapses_doubled_slashes(
    tmp_path: Path, bash: str, raw: str, expected: str
) -> None:
    script_path = _install_script(tmp_path)

    result = subprocess.run(
        [bash, str(script_path), "--print-normalized-path", raw],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == expected
    # The bash 3.2 failure mode was `src\/omnibase_infra/...`.
    assert "\\" not in result.stdout


@pytest.mark.parametrize("bash", BASH_INTERPRETERS)
def test_gate_exits_zero_on_bsd_grep_doubled_slash_hits(
    tmp_path: Path, bash: str
) -> None:
    """The OMN-14988 regression, driven through the real script end to end.

    With BSD-style ``src//`` hits pinned by the grep stub, the pre-fix script
    exits 1 with all 8 allowlisted files reported as violations under bash 3.2.
    The fixed script exits 0 under every bash.
    """
    env = _install_bsd_grep_stub(tmp_path, BSD_GREP_ALLOWLISTED_HITS)

    result = _run_gate(tmp_path, bash=bash, env=env)

    assert result.returncode == 0, (
        f"{bash} reported false positives on allowlisted, doubled-slash paths:\n"
        f"{result.stdout}{result.stderr}"
    )
    assert "OK: no disallowed imports" in result.stdout
    assert "violation(s)" not in result.stdout


@pytest.mark.parametrize("bash", BASH_INTERPRETERS)
def test_gate_still_flags_non_allowlisted_doubled_slash_hit(
    tmp_path: Path, bash: str
) -> None:
    """Normalization must not neuter the gate: a real violation still fails."""
    violating = (
        "src//omnibase_infra/nodes/node_bad/handler.py:12:"
        "from omnibase_infra.event_bus.event_bus_inmemory import EventBusInmemory"
    )
    env = _install_bsd_grep_stub(tmp_path, BSD_GREP_ALLOWLISTED_HITS + (violating,))

    result = _run_gate(tmp_path, bash=bash, env=env)

    assert result.returncode == 1
    assert "1 violation(s)" in result.stdout
    assert "node_bad/handler.py" in result.stdout


@pytest.mark.parametrize("bash", BASH_INTERPRETERS)
def test_gate_exits_zero_against_the_real_repo_tree(bash: str) -> None:
    """The reported symptom: RC=1 on a clean checkout with zero local changes."""
    result = subprocess.run(
        [bash, str(SCRIPT)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, (
        f"{bash}: gate is red on a clean tree:\n{result.stdout}{result.stderr}"
    )
    assert "OK: no disallowed imports" in result.stdout


def test_gate_does_not_use_the_bash_version_dependent_substitution() -> None:
    """Static ratchet: the fragile escaped-slash form must not come back.

    ``${file//\\/\\//\\/}`` silently changes meaning between bash 3.2 and 4.3+,
    so it must not be reintroduced as a "simplification". This assertion is RED
    against the pre-fix script on every platform, including Linux CI, where the
    behavioral tests above cannot reproduce the bash 3.2 failure.

    Comment lines are excluded: the script documents the fragile form verbatim
    so the trap stays readable at the call site.
    """
    fragile = "${file//" + "\\/\\//\\/" + "}"
    executable_lines = [
        line
        for line in SCRIPT.read_text(encoding="utf-8").splitlines()
        if not line.lstrip().startswith("#")
    ]

    offending = [line for line in executable_lines if fragile in line]
    assert not offending, (
        "OMN-14988: the bash-version-dependent slash substitution is back. "
        f"Use the variable-held pattern/replacement form (_normalize_path). {offending}"
    )
