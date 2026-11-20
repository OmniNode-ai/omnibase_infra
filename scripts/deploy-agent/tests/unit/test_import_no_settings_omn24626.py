# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Importing the deploy agent reads no deploy-source setting (OMN-24626).

The refresh scripts import ``deploy_agent.executor`` before any deploy-source
setting exists. The settings have no default, so they are resolved at first use
through ``deploy_source()`` and an unset one raises there, naming itself. The
subprocess tests scrub the three settings, so the unit conftest cannot mask it.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest
from deploy_agent import executor as executor_mod

from omnibase_infra.handlers.handler_deploy_source_clone import DeploySourceCloneError

pytestmark = pytest.mark.unit

AGENT_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = AGENT_ROOT.parents[1]
SETTINGS = (
    "DEPLOY_AGENT_REPO_DIR",
    "DEPLOY_AGENT_CANONICAL_CLONE",
    "DEPLOY_AGENT_LAB_OVERLAY_SOURCE",
    "DEPLOY_AGENT_TRACKING_REF",
)
DEV_REFRESH = REPO_ROOT / "scripts/runtime_build/refresh_dev_lane.sh"
STABILITY_ENV_MODULE = (
    REPO_ROOT
    / "src/omnibase_infra/nodes/node_setup_local_provision_effect"
    / "runtime_llm_endpoint_env.py"
)


def _run_unset(code: str) -> subprocess.CompletedProcess[str]:
    env = {k: v for k, v in os.environ.items() if k not in SETTINGS}
    env["PYTHONPATH"] = os.pathsep.join([str(AGENT_ROOT), str(REPO_ROOT / "src")])
    return subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        cwd=AGENT_ROOT.parent,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )


def _assert_clean(result: subprocess.CompletedProcess[str]) -> None:
    assert result.returncode == 0, result.stderr
    assert "DeploySourceCloneError" not in result.stderr


@pytest.mark.parametrize("module", ["deploy_agent.executor", "deploy_agent.agent"])
def test_import_no_settings_module_imports_without_deploy_source(module: str) -> None:
    _assert_clean(_run_unset(f"import {module}"))


def _import_lines(path: Path) -> str:
    """The ``deploy_agent`` import statements a refresh path executes, verbatim."""
    lines = re.findall(
        r"^\s*(from deploy_agent[\w.]* import \w+)\s*(?:#.*)?$", path.read_text(), re.M
    )
    assert lines, f"{path} imports nothing from deploy_agent; the probe is vacuous"
    return "\n".join(lines)


def test_import_no_settings_dev_refresh_script_import_line() -> None:
    imports = _import_lines(DEV_REFRESH)
    assert "from deploy_agent import executor" in imports
    _assert_clean(
        _run_unset(imports + "\nexecutor.load_dev_boot_guard(executor._LLM_ENV_MODULE)")
    )


def test_import_no_settings_stability_refresh_env_module_import_line() -> None:
    imports = _import_lines(STABILITY_ENV_MODULE)
    assert "from deploy_agent import executor" in imports
    _assert_clean(_run_unset(imports))


def test_import_no_settings_accessor_names_the_missing_setting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in SETTINGS:
        monkeypatch.delenv(name, raising=False)
    executor_mod.deploy_source.cache_clear()
    try:
        with pytest.raises(DeploySourceCloneError, match="DEPLOY_AGENT_REPO_DIR"):
            executor_mod.deploy_repo_dir()
    finally:
        executor_mod.deploy_source.cache_clear()
