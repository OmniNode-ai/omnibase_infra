# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The dev deploy-agent units must declare the PATH that finds ``uv`` (OMN-19522).

The agent's idle self-update runs ``uv sync --project <agent dir>`` before it
re-execs (``executor.py``), and several build steps call ``uv run``, all as a
bare ``"uv"`` resolved through the process PATH. ``uv`` is installed per user in
``~/.local/bin``, which is not on a systemd user manager's default PATH.

Neither dev unit declared a PATH. On the ``.201`` host the agent worked only
because that user manager's environment had been given ``~/.local/bin`` at some
point (``systemctl --user show-environment``, read 2026-09-25T13:36Z), an
undeclared host fact that a reboot drops. The ``.202`` user manager never had
it, so the dev-202 agent's self-update failed at 2026-09-25T13:34:21Z with
``[Errno 2] No such file or directory: 'uv'`` and it stayed on stale code.

So each dev unit declares PATH itself, first entry ``%h/.local/bin`` (systemd
expands ``%h`` to the unit user's home), and protects the name so the operator
env store the launcher sources cannot replace it.
"""

import re
from pathlib import Path

import pytest

_DEPLOY_DIR = Path(__file__).resolve().parents[2] / "deploy"
_AGENT_SRC = Path(__file__).resolve().parents[2] / "deploy_agent"
_DEV_UNITS = (
    _DEPLOY_DIR / "deploy-agent-dev.service",
    _DEPLOY_DIR / "deploy-agent-dev-202.service",
)

_ENVIRONMENT_RE = re.compile(r'^Environment="?([A-Za-z_][A-Za-z0-9_]*)=(.*?)"?$')


def _env(unit: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in unit.read_text().splitlines():
        if not line or line.lstrip().startswith("#"):
            continue
        match = _ENVIRONMENT_RE.match(line)
        if match:
            values[match.group(1)] = match.group(2)
    return values


def test_the_agent_really_invokes_a_bare_uv() -> None:
    """Positive control: the requirement below exists only because of this."""
    executor = (_AGENT_SRC / "executor.py").read_text()
    assert '["uv", "sync", "--project", agent_dir]' in executor


@pytest.mark.unit
@pytest.mark.parametrize("unit", _DEV_UNITS, ids=lambda p: p.name)
def test_dev_unit_declares_a_path_that_starts_with_the_user_uv_dir(unit: Path) -> None:
    env = _env(unit)
    assert "PATH" in env, f"{unit.name} must declare Environment=PATH="
    entries = env["PATH"].split(":")
    assert entries[0] == "%h/.local/bin", entries
    for system_dir in ("/usr/local/bin", "/usr/bin", "/bin"):
        assert system_dir in entries, (system_dir, entries)


@pytest.mark.unit
@pytest.mark.parametrize("unit", _DEV_UNITS, ids=lambda p: p.name)
def test_dev_unit_protects_path_from_the_env_store(unit: Path) -> None:
    protected = set(_env(unit)["DEPLOY_AGENT_ENV_PROTECTED"].split())
    assert "PATH" in protected
