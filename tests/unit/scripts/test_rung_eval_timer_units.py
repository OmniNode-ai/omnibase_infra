# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""RED-first systemd unit and missing-night detector contracts for OMN-19456.

Exercise the service's exact detector command with a fake uv so gaps, crashes,
and an unread empty table cannot be mistaken for a clean evaluation night.
"""

from __future__ import annotations

import configparser
import re
import shlex
import subprocess
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.unit

_REPO = Path(__file__).resolve().parents[3]
_RUNG_EVAL_DIR = _REPO / "deploy" / "rung-eval"
_SERVICE = _RUNG_EVAL_DIR / "onex-rung-eval.service"
_TIMER = _RUNG_EVAL_DIR / "onex-rung-eval.timer"
_ENV_EXAMPLE = _RUNG_EVAL_DIR / "rung-eval.env.example"
_MANIFEST = _REPO / "deploy" / "unit-drift-manifest.yaml"
_ENV_NAMES = {
    "RUNG_EVAL_LAB_201_BASE_URL",
    "RUNG_EVAL_LAB_201_MODEL",
    "RUNG_EVAL_LAB_202_BASE_URL",
    "RUNG_EVAL_LAB_202_MODEL",
    "RUNG_EVAL_PLANNER_200_BASE_URL",
    "RUNG_EVAL_PLANNER_200_MODEL",
    "RUNG_EVAL_CLOUD_OPENROUTER_BASE_URL",
    "RUNG_EVAL_CLOUD_OPENROUTER_MODEL",
    "RUNG_EVAL_CLOUD_OPENROUTER_API_KEY",
    "RUNG_EVAL_OMNIMARKET_DIR",
    "RUNG_EVAL_TABLE_DIR",
    "RUNG_EVAL_WINDOW_DAYS",
}
_MISSING_LINES = (
    "MISSING 2026-01-01 | lab-202 | classification | pr_classification_37",
    "MISSING 2026-01-01 | lab-201 | classification | pr_classification_38",
)


def _read_unit(path: Path) -> configparser.ConfigParser:
    """Read case-sensitive systemd keys without interpreting shell variables."""
    parser = configparser.ConfigParser(
        strict=False, interpolation=None, delimiters=("=",)
    )
    parser.optionxform = str
    parser.read_string(path.read_text(encoding="utf-8"))
    return parser


def _exec_start_lines() -> list[str]:
    """Join continuations by hand to preserve repeated ExecStart directives."""
    commands = []
    pending = ""
    for raw_line in _SERVICE.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith(("#", ";")):
            continue
        if line.endswith("\\"):
            pending += line[:-1] + " "
            continue
        line = pending + line
        pending = ""
        if line.startswith("ExecStart="):
            commands.append(line)
    assert not pending, "unterminated systemd continuation"
    return commands


def _detector_argv(tmp_path: Path) -> list[str]:
    """Decode systemd escapes, leaving environment expansion to bash."""
    commands = _exec_start_lines()
    assert len(commands) == 2, commands
    command = commands[1].removeprefix("ExecStart=")
    command = command.replace("$$", "$").replace("%%", "%")
    command = command.replace("%h", str(tmp_path / "home"))
    argv = shlex.split(command)
    assert argv[:2] == ["/bin/bash", "-c"], argv
    assert len(argv) == 3, argv
    return argv


def test_timer_schedule_and_install_target() -> None:
    unit = _read_unit(_TIMER)
    timer = unit["Timer"]
    assert "UTC" in timer["OnCalendar"]
    assert timer["Persistent"] == "true"
    assert timer["Unit"] == "onex-rung-eval.service"
    assert unit["Install"]["WantedBy"] == "timers.target"


def test_service_and_value_free_environment(tmp_path: Path) -> None:
    service = _read_unit(_SERVICE)["Service"]
    assert service["Type"] == "oneshot"
    environment_file = service["EnvironmentFile"]
    assert environment_file.startswith("-")
    assert environment_file.endswith("rung-eval.env")

    for path in (_SERVICE, _ENV_EXAMPLE):
        assert not re.search(r"https?://", path.read_text(encoding="utf-8")), path

    names = set()
    for raw_line in _ENV_EXAMPLE.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        # An untouched copy of the template must set nothing: an empty value is
        # set-but-empty, which a rung reads as an error row, not as unresolved.
        assert not line.startswith("RUNG_EVAL_"), f"uncommented assignment: {line}"
        if line.startswith("# RUNG_EVAL_"):
            name, separator, value = line.removeprefix("# ").partition("=")
            assert separator == "=", line
            if name.endswith(("_BASE_URL", "_MODEL", "_API_KEY")):
                assert value.strip() == "", f"committed environment value: {name}"
            names.add(name.strip())
    assert names >= _ENV_NAMES, f"missing environment names: {_ENV_NAMES - names}"

    commands = _exec_start_lines()
    assert len(commands) == 2, commands
    assert commands[0].startswith("ExecStart=-"), commands[0]
    run_argv = shlex.split(commands[0].removeprefix("ExecStart=-"))
    script_index = next(
        index
        for index, arg in enumerate(run_argv)
        if arg.endswith("scripts/ci/run_delegation_rung_eval.py")
    )
    assert run_argv[script_index + 1] == "run", run_argv
    assert "--out-dir" in run_argv

    detector_script = _detector_argv(tmp_path)[2]
    for required in (
        "check-missing",
        "--through",
        "--window-days",
        "missing_latest.txt",
    ):
        assert required in detector_script


@pytest.mark.parametrize("name", ["onex-rung-eval.service", "onex-rung-eval.timer"])
def test_units_are_in_drift_manifest(name: str) -> None:
    manifest = yaml.safe_load(_MANIFEST.read_text(encoding="utf-8"))
    entries = {entry["name"]: entry for entry in manifest["units"]}
    assert name in entries, f"missing drift manifest entry: {name}"
    entry = entries[name]
    assert entry["tracked"] == f"deploy/rung-eval/{name}"
    assert entry["installed"] == f"~/.config/systemd/user/{name}"
    assert entry["hosts"] == ["omnipc2"]


@pytest.mark.parametrize(
    ("fake_rc", "missing_lines", "has_nights", "expected_rc"),
    [
        pytest.param(0, (), True, 0, id="clean"),
        pytest.param(1, _MISSING_LINES, True, 1, id="missing"),
        pytest.param(2, (), True, 2, id="crash"),
        pytest.param(0, (), False, 3, id="empty-table"),
    ],
)
def test_detector_command_reports_night_state(
    tmp_path: Path,
    fake_rc: int,
    missing_lines: tuple[str, ...],
    has_nights: bool,
    expected_rc: int,
) -> None:
    argv = _detector_argv(tmp_path)
    home = tmp_path / "home"
    home.mkdir()
    omnimarket = tmp_path / "omnimarket"
    omnimarket.mkdir()
    table = tmp_path / "table"
    table.mkdir()
    today = datetime.now(UTC).date()
    if has_nights:
        for night in (today - timedelta(days=2), today):
            (table / f"rung_eval_{night.isoformat()}.jsonl").write_text(
                "{}\n" * 8, encoding="utf-8"
            )

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    fake_uv = bin_dir / "uv"
    fake_uv.write_text(
        "#!/bin/sh\n"
        'if [ -n "${FAKE_UV_LOG:-}" ]; then\n'
        '    printf "%s\\n" "$@" >> "$FAKE_UV_LOG"\n'
        "fi\n"
        'if [ -n "${FAKE_UV_STDOUT:-}" ]; then\n'
        '    cat "$FAKE_UV_STDOUT"\n'
        "fi\n"
        'exit "${FAKE_UV_RC:-0}"\n',
        encoding="utf-8",
    )
    fake_uv.chmod(0o755)
    stdout_file = tmp_path / "uv.stdout"
    stdout_file.write_text(
        "".join(f"{line}\n" for line in missing_lines), encoding="utf-8"
    )
    log_file = tmp_path / "uv.argv"
    env = {
        "PATH": f"{bin_dir}:/usr/bin:/bin",
        "RUNG_EVAL_OMNIMARKET_DIR": str(omnimarket),
        "RUNG_EVAL_TABLE_DIR": str(table),
        "RUNG_EVAL_WINDOW_DAYS": "14",
        "HOME": str(home),
        "FAKE_UV_STDOUT": str(stdout_file),
        "FAKE_UV_RC": str(fake_rc),
        "FAKE_UV_LOG": str(log_file),
    }
    result = subprocess.run(
        argv, env=env, capture_output=True, text=True, check=False, timeout=30
    )
    assert result.returncode == expected_rc, result.stderr
    report = (table / "missing_latest.txt").read_text(encoding="utf-8").splitlines()
    first_line = report[0]

    if has_nights:
        assert re.search(r"read=\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", first_line)
    if not has_nights or fake_rc == 2:
        assert "UNREAD" in first_line
        assert "missing=0" not in first_line
    elif fake_rc == 1:
        assert "missing=2" in first_line
        assert report[1:3] == list(missing_lines)
    else:
        assert first_line.startswith("RUNG-EVAL ")
        for field in ("missing=0", "rows=16", "window_days=3"):
            assert field in first_line
        uv_argv = log_file.read_text(encoding="utf-8").splitlines()
        assert uv_argv[0] == "run", uv_argv
        assert "python" in uv_argv
        assert "check-missing" in uv_argv
        assert uv_argv[uv_argv.index("--through") + 1] == today.isoformat()
        assert uv_argv[uv_argv.index("--window-days") + 1] == "3"
