# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Tests for the pre-commit interpreter-resolution gate (OMN-17219).

Recurrence guard for the defect class reported by an external collaborator: a
pre-commit hook whose ``entry:`` shells out to a bare ``python`` refuses every
``git commit`` on macOS outside an activated venv ("Executable `python` not
found"), because macOS ships ``python3`` only. Same class as OMN-16958
(omnimemory's ``validate-spdx-headers``).

These tests drive the runtime that backs the core precommit-interpreter-resolution
hook (``omnibase_core.nodes.node_precommit_interpreter_check_compute``) over a
temporary repository root holding a written ``.pre-commit-config.yaml``.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from omnibase_core.nodes.node_precommit_interpreter_check_compute.runtime_precommit_interpreter_check import (
    main as core_main,
)

REPO_ROOT = Path(__file__).parent.parent.parent
CORE_MODULE = (
    "omnibase_core.nodes.node_precommit_interpreter_check_compute"
    ".runtime_precommit_interpreter_check"
)

# The core rule refuses to report a pass when it scanned fewer than ten local hooks.
_FILLER_HOOKS = 10


def _hook(hook_id: str, entry: str) -> dict[str, str]:
    return {"id": hook_id, "name": hook_id, "entry": entry, "language": "system"}


def _write_config(root: Path, entries: list[str]) -> None:
    hooks = [
        _hook(f"filler-{i}", "uv run python scripts/ci/filler.py")
        for i in range(_FILLER_HOOKS)
    ]
    hooks.extend(_hook(f"case-{i}", entry) for i, entry in enumerate(entries))
    config = {"repos": [{"repo": "local", "hooks": hooks}]}
    (root / ".pre-commit-config.yaml").write_text(yaml.safe_dump(config))


def _run_gate(root: Path, capsys: pytest.CaptureFixture[str]) -> tuple[int, str, str]:
    rc = core_main(["--root", str(root)])
    captured = capsys.readouterr()
    return rc, captured.out, captured.err


def _violations(err: str) -> list[str]:
    return [line for line in err.splitlines() if line.startswith("  - ")]


def _check_entry(
    root: Path, capsys: pytest.CaptureFixture[str], entry: str
) -> tuple[int, str, str]:
    _write_config(root, [entry])
    return _run_gate(root, capsys)


def _check_script_line(
    root: Path, capsys: pytest.CaptureFixture[str], line: str
) -> tuple[int, str, str]:
    scripts = root / "scripts"
    scripts.mkdir()
    (scripts / "hook.sh").write_text(f"#!/usr/bin/env bash\n{line}\n")
    _write_config(root, ["bash scripts/hook.sh"])
    return _run_gate(root, capsys)


@pytest.mark.unit
@pytest.mark.parametrize(
    "entry",
    [
        "python scripts/ci/check_one_occ_producer.py",
        "python3 scripts/ci/check_one_occ_producer.py",
        "env PYTHONPATH=src python -m omnimarket.validators.x",
        "exec python scripts/ci/x.py",
        "bash -c 'python scripts/ci/x.py'",
    ],
)
def test_bare_interpreter_entry_is_rejected(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], entry: str
) -> None:
    rc, _, err = _check_entry(tmp_path, capsys, entry)
    assert rc == 1, f"expected a violation for {entry!r}"
    violations = _violations(err)
    assert len(violations) == 1, err
    assert violations[0].startswith("  - case-0: ")
    assert "bare" in violations[0]


@pytest.mark.unit
@pytest.mark.parametrize(
    "entry",
    [
        "uv run python scripts/ci/x.py",
        "uv run --frozen python scripts/check_dep_provenance.py",
        "uv run --no-project --with pyyaml python scripts/ci/x.py",
        "env PYTHONPATH=src uv run python -m omnimarket.validators.x",
        "bash -c 'uv run mypy src/omnimarket'",
        "/opt/homebrew/bin/python3 scripts/ci/x.py",
        '"$PYTHON" scripts/ci/x.py',
        "uv run pytest tests/ci/test_x.py -q",
        ".pre-commit-hooks/reject-deploy-gate-skip-token.sh",
    ],
)
def test_resolvable_interpreter_entry_is_accepted(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], entry: str
) -> None:
    rc, out, err = _check_entry(tmp_path, capsys, entry)
    assert rc == 0, err
    assert out.startswith("OK: ")


@pytest.mark.unit
def test_suppression_marker_is_honored(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    entry = "python scripts/ci/x.py  # precommit-interp-ok: illustrative"
    rc, _, err = _check_entry(tmp_path, capsys, entry)
    assert rc == 0, err


@pytest.mark.unit
@pytest.mark.parametrize(
    "entry",
    [
        "python scripts/ci/x.py  # precommit-interp-ok",
        "python scripts/ci/x.py  # precommit-interp-ok:",
    ],
)
def test_suppression_marker_requires_a_reason(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], entry: str
) -> None:
    rc, _, err = _check_entry(tmp_path, capsys, entry)
    assert rc == 1, f"bare suppression marker bypassed the gate: {entry!r}"
    assert _violations(err)


@pytest.mark.unit
def test_shell_script_bare_python_is_rejected(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    (scripts / "hook.sh").write_text(
        "#!/usr/bin/env bash\n"
        "# python foo.py  <- a comment, not an invocation\n"
        "command -v python3 >/dev/null && python3 -c 'import yaml'\n"
        "python scripts/ci/x.py\n"
    )
    _write_config(tmp_path, ["bash scripts/hook.sh"])
    rc, _, err = _run_gate(tmp_path, capsys)
    assert rc == 1, err
    violations = _violations(err)
    assert len(violations) == 1, err
    assert violations[0].startswith("  - scripts/hook.sh:4:")


@pytest.mark.unit
def test_live_config_has_no_bare_interpreter_hooks() -> None:
    """The repo's own committed config must satisfy the rule (RED before fix)."""
    result = subprocess.run(
        [sys.executable, "-m", CORE_MODULE, "--root", str(REPO_ROOT)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, (
        "committed .pre-commit-config.yaml has a bare-interpreter hook:\n"
        f"{result.stdout}\n{result.stderr}"
    )


@pytest.mark.unit
def test_gate_fails_closed_on_vacuous_scan(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A parse regression that scans nothing must fail, not silently pass."""
    (tmp_path / ".pre-commit-config.yaml").write_text("repos: []\n")
    rc, out, err = _run_gate(tmp_path, capsys)
    assert rc == 1, out
    assert "vacuous" in err


# --- CodeRabbit findings on PR #2223: bypass shapes that must NOT slip through


@pytest.mark.unit
@pytest.mark.parametrize(
    "entry",
    [
        # `uv run` covers the interpreter it wraps, but not a nested shell:
        # the option-value parser must not lose the `bash -c` body.
        "uv run --with pyyaml bash -c 'python scripts/ci/x.py'",
        # A command LIST: only the first command used to be scanned.
        "bash -c 'true && python scripts/ci/x.py'",
        "bash -c 'echo hi; python3 scripts/ci/x.py'",
        "bash -c 'uv run mypy src || python scripts/ci/x.py'",
        # Inline assignment / wrapper in front of the interpreter.
        "bash -c 'FOO=1 python scripts/ci/x.py'",
    ],
)
def test_bypass_shapes_are_rejected(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], entry: str
) -> None:
    rc, _, err = _check_entry(tmp_path, capsys, entry)
    assert rc == 1, f"bypass not caught: {entry!r}"
    assert _violations(err)


@pytest.mark.unit
@pytest.mark.parametrize(
    "entry",
    [
        # `--opt=value` form carries its own value; `python` here is still uv's.
        "uv run --with=pyyaml python scripts/ci/x.py",
        # A value-taking option must not leave its value read as the command
        # word (CodeRabbit finding): both of these are uv-resolved and fine.
        "uv run --with pyyaml python3 scripts/ci/x.py",
        "uv run --python 3.12 python scripts/ci/x.py",
        "bash -c 'uv run python a.py && uv run python b.py'",
    ],
)
def test_uv_run_variants_stay_accepted(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], entry: str
) -> None:
    rc, _, err = _check_entry(tmp_path, capsys, entry)
    assert rc == 0, err


@pytest.mark.unit
@pytest.mark.parametrize(
    "line",
    [
        "FOO=1 python script.py",
        "env FOO=1 python script.py",
        "if python -c 'import yaml'; then :; fi",
        "while python check.py; do :; done",
        "uv run true && python script.py",
        "out=$(python -c 'print(1)')",
        "exec python script.py",
    ],
)
def test_shell_command_positions_are_scanned(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], line: str
) -> None:
    rc, _, err = _check_script_line(tmp_path, capsys, line)
    assert rc == 1, f"bare python not caught in {line!r}"
    assert _violations(err)


@pytest.mark.unit
@pytest.mark.parametrize(
    "line",
    [
        "uv run python script.py",
        "command -v python3 >/dev/null && python3 -c 'import yaml'",
        'exec python3 "$GATE_WRAPPER" --repo-root "$REPO_ROOT"',
        "/opt/homebrew/bin/python3 script.py",
        '"$PYTHON" script.py',
    ],
)
def test_resolvable_shell_lines_stay_accepted(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], line: str
) -> None:
    rc, _, err = _check_script_line(tmp_path, capsys, line)
    assert rc == 0, f"false positive on {line!r}: {err}"
    assert _violations(err) == []


@pytest.mark.unit
def test_script_suppression_marker_requires_a_reason(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    rc, _, err = _check_script_line(
        tmp_path, capsys, "python x.py # precommit-interp-ok"
    )
    assert rc == 1, err
    assert _violations(err)
