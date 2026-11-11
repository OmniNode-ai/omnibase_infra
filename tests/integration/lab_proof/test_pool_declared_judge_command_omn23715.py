# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The pool driver runs the judge command named by its overlay file, over a real subprocess (OMN-23715).

``declared_changed_path`` reads ``lab_proof_judge_command`` from the pool overlay on disk, expands the
environment, runs the command with the clone and probe phase text as JSON on stdin and parses the JSON
it prints. The stand-in command here is a real Python process, so the argv split, the environment
expansion, the stdin and stdout boundary and the overlay read are the ones production uses; only the
rule behind the command (omnimarket node_lab_proof_judge_compute) is not run, because that node is not
a dependency of this repository. Its own tests hold the rule.
"""

from __future__ import annotations

import importlib.util
import json
import shlex
import shutil
import sys
from pathlib import Path
from typing import Any

import pytest

from omnibase_infra.models.prepr_pool.model_prepr_pool_overlay import (
    ModelPreprPoolOverlay,
)

pytestmark = pytest.mark.integration

REPO_ROOT = Path(__file__).resolve().parents[3]
POOL_PY = REPO_ROOT / "scripts" / "runtime_build" / "prepr_runtime_pool.py"
FIXTURE_ROOT = (
    REPO_ROOT / "tests" / "unit" / "scripts" / "fixtures" / "pool_config_root"
)
OVERLAY_REL = Path("config/onex/overlays/prepr_runtime_pool/overlay.yaml")
ECHO = (
    "import json, sys; request = json.load(sys.stdin); "
    "print(json.dumps({'verdict': 'PASS', 'ok': True, "
    "'evidence': 'echo ' + ','.join(sorted(request)), 'notes': []}))"
)


def _pool() -> Any:
    spec = importlib.util.spec_from_file_location(
        "prepr_runtime_pool_omn23715", POOL_PY
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["prepr_runtime_pool_omn23715"] = module
    spec.loader.exec_module(module)
    return module


def _config_root(tmp_path: Path, command: str | None) -> Path:
    root = tmp_path / "config_root"
    shutil.copytree(FIXTURE_ROOT, root)
    if command is not None:
        with (root / OVERLAY_REL).open("a", encoding="utf-8") as overlay:
            overlay.write(f"lab_proof_judge_command: {json.dumps(command)}\n")
    ModelPreprPoolOverlay.read.cache_clear()
    return root


def test_the_overlay_command_is_run_with_the_phase_text_on_stdin(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("JUDGE_TEST_PYTHON", sys.executable)
    command = f"${{JUDGE_TEST_PYTHON}} -c {shlex.quote(ECHO)}"
    monkeypatch.setenv(
        "ONEX_WORKSPACE_CONFIG_ROOT", str(_config_root(tmp_path, command))
    )
    answer = _pool().declared_changed_path({"clone": "c", "probe": "p", "tests": "t"})
    assert answer == {
        "verdict": "PASS",
        "ok": True,
        "evidence": "echo clone,probe",
        "notes": [],
    }


def test_an_overlay_without_the_key_runs_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ONEX_WORKSPACE_CONFIG_ROOT", str(_config_root(tmp_path, None)))
    assert _pool().declared_changed_path({"clone": "c", "probe": "p"}) is None


def test_a_command_that_exits_non_zero_is_no_answer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    command = f"{sys.executable} -c {shlex.quote('import sys; sys.exit(3)')}"
    monkeypatch.setenv(
        "ONEX_WORKSPACE_CONFIG_ROOT", str(_config_root(tmp_path, command))
    )
    assert _pool().declared_changed_path({"clone": "c", "probe": "p"}) is None
