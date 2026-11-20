# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The branch install command line, run as a real process (OMN-26102)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.integration

_HEAD = "9142221309c2ce47e4e78d3645c9f1d250850806"
_NODE = Path(__file__).resolve().parents[3] / (
    "src/omnibase_infra/nodes/node_venv_branch_install"
)


def test_the_command_refuses_a_merged_pull_request_and_touches_no_state(
    tmp_path: Path,
) -> None:
    merged = tmp_path / "merged.json"
    merged.write_text(
        json.dumps(
            {
                "facts": {
                    "repo": "omnimarket",
                    "number": 6393,
                    "state": "MERGED",
                    "head_sha": _HEAD,
                }
            }
        ),
        encoding="utf-8",
    )
    done = subprocess.run(
        [
            sys.executable,
            "-m",
            "omnibase_infra.nodes.node_venv_branch_install.runtime_venv_branch_install",
            "install",
            "--host",
            "h051",
            "--venv",
            str(tmp_path / "venv"),
            "--state-dir",
            str(tmp_path / "state"),
            "--pr",
            "6393",
            "--unit",
            "onex-floor-fill-tick.service",
            "--pr-state",
            str(merged),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert done.returncode == 2
    assert not (tmp_path / "state").exists()


def test_the_contract_on_disk_declares_the_operation_the_handler_serves() -> None:
    contract = yaml.safe_load((_NODE / "contract.yaml").read_text(encoding="utf-8"))
    assert contract["node_type"] == "EFFECT_GENERIC"
    assert contract["handler_routing"]["handlers"][0]["operation"] == (
        "venv.branch_install"
    )
