# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Ordered in-process environment isolation regressions (OMN-15572)."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[3]
_VERIFICATION_VICTIMS = (
    "tests/unit/verification/test_cli.py::TestCLIMain::test_single_contract_pass",
    "tests/unit/verification/test_cli.py::TestCLIMain::test_all_contracts",
    "tests/unit/verification/test_cli.py::TestCLIMain::test_output_path_writes_file",
)
_ENVIRONMENT_POLLUTERS = (
    pytest.param(
        "tests/unit/cli/test_cli_delegate.py::TestPayloadScratch::"
        "test_payload_written_under_state_root_tmp_not_slash_tmp",
        id="delegate-receipt",
    ),
    pytest.param(
        "tests/unit/cli/test_cli_node_receipt.py::TestReceiptModeSuccess::"
        "test_stdout_is_exactly_one_validated_skill_result",
        id="node-receipt",
    ),
    pytest.param(
        "tests/unit/scripts/test_monitor_logs.py::TestSanitizeLogText::"
        "test_strips_ansi_color_codes",
        id="monitor-script-import",
    ),
)

# A synthetic unreachable DSN would make the verification probes QUARANTINE,
# which those victims correctly accept. The subprocess-only plugin instead
# returns deterministic empty rows when (and only when) the environment-loaded
# URL escaped its polluter, making the same production-sensitive boundary fail.
_DETERMINISTIC_DB_PROBE_PLUGIN = """
import json
import os

import pytest

_EXPECTED_NODE_IDS = json.loads(os.environ["OMN15572_EXPECTED_NODE_IDS"])
_SEEN_NODE_IDS = []

def pytest_runtest_setup(item):
    assert "PYTEST_XDIST_WORKER" not in os.environ, "ordered proof ran under xdist"
    expected_node_id = _EXPECTED_NODE_IDS[len(_SEEN_NODE_IDS)]
    assert item.nodeid == expected_node_id, (
        f"ordered proof expected {expected_node_id}, got {item.nodeid}"
    )
    _SEEN_NODE_IDS.append(item.nodeid)

    if "tests/unit/verification/test_cli.py::TestCLIMain::" not in item.nodeid:
        return

    from omnibase_infra.verification import cli as verification_cli

    def _make_runtime_db_query_fn():
        if "OMNIBASE_INFRA_DB_URL" not in os.environ:
            return None
        return lambda _sql: []

    verification_cli._make_runtime_db_query_fn = _make_runtime_db_query_fn


def pytest_sessionfinish(session, exitstatus):
    del exitstatus
    if _SEEN_NODE_IDS != _EXPECTED_NODE_IDS:
        session.exitstatus = pytest.ExitCode.TESTS_FAILED
"""


def _run_ordered_regression(
    polluter: str,
    tmp_path: Path,
) -> subprocess.CompletedProcess[str]:
    controlled_dsn = "postgresql://unit:unit@127.0.0.1:1/unit"
    env_file = tmp_path / "controlled.env"
    env_file.write_text(
        f"OMNIBASE_INFRA_DB_URL={controlled_dsn}\n",
        encoding="utf-8",
    )
    home_dir = tmp_path / "home"
    home_env_dir = home_dir / ".omnibase"
    home_env_dir.mkdir(parents=True)
    (home_env_dir / ".env").write_text(
        f"OMNIBASE_INFRA_DB_URL={controlled_dsn}\n",
        encoding="utf-8",
    )
    plugin_file = tmp_path / "receipt_isolation_probe_plugin.py"
    plugin_file.write_text(_DETERMINISTIC_DB_PROBE_PLUGIN, encoding="utf-8")
    subprocess_env = dict(os.environ)
    for key in tuple(subprocess_env):
        if key.startswith("PYTEST_"):
            subprocess_env.pop(key)
    for key in ("PYTHONPATH", "PYTHONHASHSEED"):
        subprocess_env.pop(key, None)
    subprocess_env["HOME"] = str(home_dir)
    subprocess_env["OMNIBASE_ENV_FILE"] = str(env_file)
    subprocess_env["OMN15572_EXPECTED_NODE_IDS"] = json.dumps(
        [polluter, *_VERIFICATION_VICTIMS]
    )
    subprocess_env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    subprocess_env.pop("OMNIBASE_INFRA_DB_URL", None)
    subprocess_env["PYTHONPATH"] = str(tmp_path)

    return subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "-o",
            "addopts=",
            "-p",
            "no:xdist",
            "-p",
            "no:xdist.plugin",
            "-p",
            "no:randomly",
            "-p",
            "no:random_order",
            "-p",
            "receipt_isolation_probe_plugin",
            polluter,
            *_VERIFICATION_VICTIMS,
            "--tb=short",
        ],
        cwd=_REPO_ROOT,
        env=subprocess_env,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )


def _assert_ordered_regression_passed(
    result: subprocess.CompletedProcess[str],
    polluter: str,
) -> None:
    assert result.returncode == 0, (
        f"ordered polluter/victim regression failed for {polluter}\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )


@pytest.mark.parametrize("polluter", _ENVIRONMENT_POLLUTERS)
def test_environment_loading_does_not_escape_owning_boundary(
    polluter: str,
    tmp_path: Path,
) -> None:
    """Each real env-loading boundary must leave verification victims hermetic."""
    result = _run_ordered_regression(polluter, tmp_path)
    _assert_ordered_regression_passed(result, polluter)


def test_ordered_regression_rejects_hostile_parent_pytest_controls(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Inherited xdist/plugin controls cannot split polluter from victims."""
    polluter = (
        "tests/unit/scripts/test_monitor_logs.py::TestSanitizeLogText::"
        "test_strips_ansi_color_codes"
    )
    monkeypatch.setenv("PYTEST_ADDOPTS", "-n 2 --dist=loadscope")
    monkeypatch.setenv("PYTEST_PLUGINS", "xdist.plugin")
    monkeypatch.setenv("PYTHONPATH", "/hostile/ambient/pythonpath")

    result = _run_ordered_regression(polluter, tmp_path)

    _assert_ordered_regression_passed(result, polluter)
