# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Ordered receipt-mode environment isolation regressions (OMN-15572)."""

from __future__ import annotations

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
_RECEIPT_POLLUTERS = (
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
)

# A synthetic unreachable DSN would make the verification probes QUARANTINE,
# which those victims correctly accept. The subprocess-only plugin instead
# returns deterministic empty rows when (and only when) the receipt-loaded URL
# escaped its polluter, making the same production-sensitive boundary fail.
_DETERMINISTIC_DB_PROBE_PLUGIN = """
import os


def pytest_runtest_setup(item):
    if "tests/unit/verification/test_cli.py::TestCLIMain::" not in item.nodeid:
        return

    from omnibase_infra.verification import cli as verification_cli

    def _make_runtime_db_query_fn():
        if "OMNIBASE_INFRA_DB_URL" not in os.environ:
            return None
        return lambda _sql: []

    verification_cli._make_runtime_db_query_fn = _make_runtime_db_query_fn
"""


@pytest.mark.parametrize("polluter", _RECEIPT_POLLUTERS)
def test_receipt_environment_does_not_escape_cli_test_package(
    polluter: str,
    tmp_path: Path,
) -> None:
    """Each real receipt boundary must leave all verification victims hermetic."""
    env_file = tmp_path / "controlled.env"
    env_file.write_text(
        "OMNIBASE_INFRA_DB_URL=postgresql://unit:unit@127.0.0.1:1/unit\n",
        encoding="utf-8",
    )
    plugin_file = tmp_path / "receipt_isolation_probe_plugin.py"
    plugin_file.write_text(_DETERMINISTIC_DB_PROBE_PLUGIN, encoding="utf-8")
    subprocess_env = dict(os.environ)
    subprocess_env["OMNIBASE_ENV_FILE"] = str(env_file)
    subprocess_env.pop("OMNIBASE_INFRA_DB_URL", None)
    pythonpath_entries = [str(tmp_path)]
    if existing_pythonpath := subprocess_env.get("PYTHONPATH"):
        pythonpath_entries.append(existing_pythonpath)
    subprocess_env["PYTHONPATH"] = os.pathsep.join(pythonpath_entries)

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
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

    assert result.returncode == 0, (
        f"ordered polluter/victim regression failed for {polluter}\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )
