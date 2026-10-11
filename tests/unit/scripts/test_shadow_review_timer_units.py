# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Systemd unit and watcher-state freshness contracts for OMN-20422.

Exercise the service's exact freshness gate so missing or stale watcher state
cannot be mistaken for a clean shadow-review tick.
"""

from __future__ import annotations

import configparser
import os
import re
import shlex
import subprocess
import time
from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.unit

_REPO = Path(__file__).resolve().parents[3]
_SHADOW_REVIEW_DIR = _REPO / "deploy" / "shadow-review"
_SERVICE = _SHADOW_REVIEW_DIR / "onex-shadow-review.service"
_TIMER = _SHADOW_REVIEW_DIR / "onex-shadow-review.timer"
_ENV_EXAMPLE = _SHADOW_REVIEW_DIR / "shadow-review.env.example"
_MANIFEST = _REPO / "deploy" / "unit-drift-manifest.yaml"
_ENV_NAMES = {
    "OMNI_HOME",
    "SHADOW_REVIEW_OMNIMARKET_DIR",
    "SHADOW_REVIEW_WATCHER_STATE",
    "SHADOW_REVIEW_HARNESS_SCRIPT",
    "SHADOW_REVIEW_WINDOW_START",
    "SHADOW_REVIEW_STORE",
    "SHADOW_REVIEW_MAX_STATE_AGE_S",
    "SHADOW_REVIEW_MAX_REVIEWS",
}


def _read_unit(path: Path) -> configparser.ConfigParser:
    """Read case-sensitive systemd keys without interpreting shell variables."""
    parser = configparser.ConfigParser(
        strict=False, interpolation=None, delimiters=("=",)
    )
    parser.optionxform = str
    parser.read_string(path.read_text(encoding="utf-8"))
    return parser


def _exec_start_lines(directive: str) -> list[str]:
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
        if line.startswith(f"{directive}="):
            commands.append(line)
    assert not pending, "unterminated systemd continuation"
    return commands


def test_timer_schedule_and_install_target() -> None:
    unit = _read_unit(_TIMER)
    timer = unit["Timer"]
    assert "UTC" in timer["OnCalendar"]
    assert "/15" in timer["OnCalendar"]
    assert timer["Persistent"] == "false"
    assert timer["Unit"] == "onex-shadow-review.service"
    assert unit["Install"]["WantedBy"] == "timers.target"


def test_service_runs_shadow_review_without_comments_or_urls() -> None:
    service = _read_unit(_SERVICE)["Service"]
    assert service["Type"] == "oneshot"
    environment_file = service["EnvironmentFile"]
    assert environment_file.endswith("shadow-review.env")
    assert not environment_file.startswith("-")

    commands = _exec_start_lines("ExecStart")
    assert len(commands) == 1, commands
    for required in (
        "node_shadow_review_effect",
        "--watcher-state",
        "--store",
        "-u PYTHONPATH",
    ):
        assert required in commands[0], commands[0]

    for path in (_SERVICE, _ENV_EXAMPLE):
        for line in path.read_text(encoding="utf-8").splitlines():
            assert "http://" not in line, (path, line)
            assert "https://" not in line, (path, line)

    service_text = _SERVICE.read_text(encoding="utf-8")
    for forbidden in ("gh pr comment", "gh api", "--comment"):
        assert forbidden not in service_text


def test_environment_example_has_exact_value_free_commented_assignments() -> None:
    names = set()
    for raw_line in _ENV_EXAMPLE.read_text(encoding="utf-8").splitlines():
        if not raw_line.strip():
            continue
        assert raw_line.startswith("#"), f"uncommented line: {raw_line}"
        assignment = re.match(r"# ([A-Z][A-Z0-9_]*)=(.*)$", raw_line)
        if assignment:
            name, value = assignment.groups()
            assert value == "", f"committed environment value: {name}"
            names.add(name)
    assert names == _ENV_NAMES, names


def test_freshness_gate_reports_missing_stale_and_fresh_state(tmp_path: Path) -> None:
    commands = _exec_start_lines("ExecStartPre")
    assert len(commands) == 2, commands
    command = commands[1].removeprefix("ExecStartPre=")
    command = command.replace("$$", "$").replace("%%", "%")
    argv = shlex.split(command)
    assert argv[:2] == ["/bin/bash", "-c"], argv
    assert len(argv) == 3, argv

    state = tmp_path / "watcher-state.json"
    env = {
        "PATH": os.environ["PATH"],
        "SHADOW_REVIEW_WATCHER_STATE": str(state),
        "SHADOW_REVIEW_MAX_STATE_AGE_S": "900",
    }
    missing = subprocess.run(
        argv, env=env, capture_output=True, text=True, check=False, timeout=30
    )
    assert missing.returncode == 3, missing.stderr
    assert "UNREAD" in missing.stdout

    state.write_text("{}\n", encoding="utf-8")
    old_mtime = time.time() - 2000
    os.utime(state, (old_mtime, old_mtime))
    stale = subprocess.run(
        argv, env=env, capture_output=True, text=True, check=False, timeout=30
    )
    assert stale.returncode == 3, stale.stderr
    assert "old" in stale.stdout

    os.utime(state, None)
    fresh = subprocess.run(
        argv, env=env, capture_output=True, text=True, check=False, timeout=30
    )
    assert fresh.returncode == 0, fresh.stderr


def test_units_are_in_drift_manifest() -> None:
    manifest = yaml.safe_load(_MANIFEST.read_text(encoding="utf-8"))
    entries = {entry["name"]: entry for entry in manifest["units"]}
    for name in ("onex-shadow-review.service", "onex-shadow-review.timer"):
        assert name in entries, f"missing drift manifest entry: {name}"
        entry = entries[name]
        assert entry["tracked"] == f"deploy/shadow-review/{name}"
        assert entry["installed"] == f"~/.config/systemd/user/{name}"
        assert entry["hosts"] == ["omnipc2"]
