# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Adoption gate of the deploy agent against real state files and real contract settings (OMN-26494).

Only the subprocess runner is replaced; the settings come from the node contract on disk and the
adoption record is written to and read from a real directory.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest

pytestmark = pytest.mark.integration

REPO_ROOT = Path(__file__).resolve().parents[3]
MODULE_FILE = (
    REPO_ROOT
    / "src/omnibase_infra/nodes/node_dev_roll_events_effect"
    / "runtime_dev_agent_adoption.py"
)
OLD = "a" * 40
NEW = "b" * 40


def _load() -> ModuleType:
    name = "omn26494_adoption_integration"
    spec = importlib.util.spec_from_file_location(name, MODULE_FILE)
    assert spec is not None and spec.loader is not None
    module = sys.modules[name] = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _runner(self_check_code: int) -> object:
    def run(argv: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        code = self_check_code if argv and argv[0] == sys.executable else 0
        return subprocess.CompletedProcess(argv, code, "", "")

    return run


def test_adoption_records_ref_after_passing_self_check(tmp_path: Path) -> None:
    adoption = _load()
    synced: list[str] = []
    agent_dir = str(REPO_ROOT / "scripts" / "deploy-agent")

    assert adoption.adopts(agent_dir, tmp_path, OLD, NEW, _runner(0), synced.append)

    settings = adoption.settings_at(REPO_ROOT)
    state = adoption.read_state(tmp_path, settings)
    assert state is not None
    assert (state.current_ref, state.previous_ref) == (NEW, OLD)
    assert synced == [agent_dir]


def test_adoption_refuses_ref_after_failing_self_check(tmp_path: Path) -> None:
    adoption = _load()
    agent_dir = str(REPO_ROOT / "scripts" / "deploy-agent")

    assert not adoption.adopts(
        agent_dir, tmp_path, OLD, NEW, _runner(1), lambda _d: None
    )

    settings = adoption.settings_at(REPO_ROOT)
    assert adoption.refused_reason(tmp_path, settings, NEW) == "self-check-exit-1"
