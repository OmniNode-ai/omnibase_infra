# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Install by repository, branch and sha through the real command line (OMN-26764).

The command line runs as its own process against a real state directory: it
registers with no pull request flag, the list command reads the record back,
and an operator release removes it by the id derived from repository and sha.
No docker, git remote or shared daemon is touched.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration

_HEAD = "8020dbdcc9a4961cdc5684d14c63b3bc024eef30"
_TARGET = "/app/.venv/lib/python3.12/site-packages/omnimarket/x/handler.py"
_MODULE = "omnibase_infra.handlers.handler_dev_lane_branch_install"


def _cli(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", _MODULE, *args],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )


def test_dev_lane_branch_install_no_pr_registers_lists_and_releases(
    tmp_path: Path,
) -> None:
    state = tmp_path / "state"
    src = tmp_path / "handler.py"
    src.write_bytes(b"# fix\n")

    registered = _cli(
        "register",
        "--state-dir", str(state),
        "--lane", "dev",
        "--repo", "omnimarket",
        "--branch", "omn-26764-fix",
        "--head", _HEAD,
        "--ticket", "OMN-26764",
        "--registered-by", "lane-x",
        "--service", "omninode-runtime",
        "--expires-in-hours", "24",
        "--file", f"{_TARGET}={src}",
    )  # fmt: skip
    assert registered.returncode == 0, registered.stderr

    listed = _cli("list", "--state-dir", str(state))
    assert listed.returncode == 0, listed.stderr
    assert _HEAD in listed.stdout
    assert f"omnimarket-{_HEAD[:12]}" in listed.stdout
    json.loads(listed.stdout)

    released = _cli(
        "release",
        "--state-dir", str(state),
        "--id", f"omnimarket-{_HEAD[:12]}",
        "--reason", "proof done",
    )  # fmt: skip
    assert released.returncode == 0, released.stderr
    assert released.stdout.startswith("RELEASED ")

    after = _cli("list", "--state-dir", str(state))
    assert _HEAD not in after.stdout
