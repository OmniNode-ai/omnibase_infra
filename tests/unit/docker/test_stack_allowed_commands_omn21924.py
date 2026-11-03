# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-21924 (one-runtime plan task T27): the image's allowed commands are data.

``config/validation/stack_shape.yaml`` lists the commands a runtime service may start
(``image_commands``) and classifies every console script of ``pyproject.toml``
(``console_scripts_allowed`` or ``developer_only``). The rule is omnibase_core's
``ValidatorStackShape``; this file proves the real data refuses a planted ``python -m`` service
and leaves no console script unclassified. The real compose files are judged by
``tests/unit/validation/test_checker_stack_shape.py``.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest
import yaml

from omnibase_core.validation.checker_stack_shape import RULE_IMAGE_COMMAND
from omnibase_core.validation.validator_stack_shape import ValidatorStackShape

pytestmark = pytest.mark.unit

_REPO = Path(__file__).resolve().parents[3]
_DATA = Path("config/validation/stack_shape.yaml")
_COMPOSE = Path("docker/docker-compose.planted.yml")


def _real() -> dict[str, object]:
    loaded = yaml.safe_load((_REPO / _DATA).read_text(encoding="utf-8"))
    assert isinstance(loaded, dict)
    return loaded


def _names(key: str) -> set[str]:
    entries = _real()[key]
    assert isinstance(entries, list)
    return {str(entry["name"]) for entry in entries}


def _judge_command(tmp_path: Path, command: object) -> list[tuple[str, str]]:
    """Judge a one-file stack whose runtime service starts ``command``, under the real lists."""
    real = _real()
    runtime = real["runtime"]
    assert isinstance(runtime, dict)
    service = runtime["services"][0]
    data = {
        "runtime": runtime,
        "dependencies": [],
        "one_shots": [],
        "lanes": {"planted": {"compose_files": [_COMPOSE.as_posix()]}},
        "baseline": [],
        "image_commands": real["image_commands"],
    }
    (tmp_path / _DATA.parent).mkdir(parents=True)
    (tmp_path / _DATA).write_text(yaml.safe_dump(data), encoding="utf-8")
    (tmp_path / _COMPOSE.parent).mkdir(parents=True)
    (tmp_path / _COMPOSE).write_text(
        yaml.safe_dump({"services": {service: {"image": "x", "command": command}}}),
        encoding="utf-8",
    )
    result = ValidatorStackShape(root=tmp_path, data_path=_DATA.as_posix()).validate(
        tmp_path / _DATA
    )
    return [(str(i.code), i.message) for i in result.issues]


def test_allowed_commands_refuse_a_planted_python_module_service(
    tmp_path: Path,
) -> None:
    runtime = _real()["runtime"]
    assert isinstance(runtime, dict)
    issues = _judge_command(tmp_path, ["python", "-m", "planted.module"])
    assert [code for code, _ in issues] == [RULE_IMAGE_COMMAND]
    assert runtime["services"][0] in issues[0][1]
    assert "python -m planted.module" in issues[0][1]


def test_allowed_commands_admit_every_declared_image_command(tmp_path: Path) -> None:
    commands = _real()["image_commands"]
    assert isinstance(commands, list) and commands
    for index, entry in enumerate(commands):
        root = tmp_path / str(index)
        root.mkdir()
        assert _judge_command(root, [entry["command"], "--flag"]) == []


def test_allowed_commands_leave_no_console_script_unclassified() -> None:
    pyproject = tomllib.loads((_REPO / "pyproject.toml").read_text(encoding="utf-8"))
    declared = set(pyproject["project"]["scripts"])
    allowed = _names("console_scripts_allowed")
    developer = _names("developer_only")
    assert declared - allowed - developer == set()
    assert allowed & developer == set()
