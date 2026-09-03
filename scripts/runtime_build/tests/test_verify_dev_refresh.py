# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Unit tests for scripts/runtime_build/verify_dev_refresh.py [OMN-14889]."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock

_SCRIPT_DIR = Path(__file__).resolve().parents[1]
# The verifier imports its sibling ``health_payload`` module the way it does
# when executed as a script (``sys.path[0]`` is the script's own directory).
# ``spec_from_file_location`` does not set that up, so the test must.
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "verify_dev_refresh.py"
_spec = importlib.util.spec_from_file_location("verify_dev_refresh", _SCRIPT_PATH)
assert _spec is not None and _spec.loader is not None
_mod = importlib.util.module_from_spec(_spec)
sys.modules["verify_dev_refresh"] = _mod
_spec.loader.exec_module(_mod)

check_cluster_health = _mod.check_cluster_health


def _completed(
    returncode: int = 0, stdout: str = "", stderr: str = ""
) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(
        args=["cmd"], returncode=returncode, stdout=stdout, stderr=stderr
    )


def test_cluster_health_uses_configured_broker_address() -> None:
    runner = MagicMock(return_value=_completed(stdout="Healthy:             true\n"))

    ok, _detail = check_cluster_health(
        "redpanda-container",
        broker_address="custom-redpanda:19092",
        runner=runner,
    )

    assert ok is True
    command = runner.call_args.args[0]
    assert "brokers=custom-redpanda:19092" in command
