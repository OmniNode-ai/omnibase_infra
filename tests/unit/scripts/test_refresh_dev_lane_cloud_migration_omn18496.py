# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""A failed cloud migration one-shot does not end a dev-lane refresh green (OMN-18496).

``onex-runtime-deploy`` deploys and verifies the runtime family when the dev
lane's omninode_cloud migration one-shot fails, stops ``onex-api`` and exits
``CLOUD_MIGRATION_FAILED_EXIT_CODE``. ``onex-api`` is not one of the refresh's
health-gated core services, so the health gate passes. The refresh must still
name the failure: result ``FAILED_CLOUD_MIGRATION``, the deploy's exit status in
the receipt, exit 1 (refresh failed, lane healthy) and no rollback.

The result block is extracted from the real script and run under the script's
own shell options with ``log``/``err`` replaced; no docker call is made.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_REPO = Path(__file__).resolve().parents[3]
_SCRIPT = _REPO / "scripts" / "runtime_build" / "refresh_dev_lane.sh"
_HANDLER = _REPO / "src" / "omnibase_infra" / "handlers" / "handler_runtime_deploy.sh"
_BLOCK_START = "# --- cloud migration one-shot (OMN-18496) "
_BLOCK_END = "# --- close with the lane census (OMN-20876) "


def _readonly(text: str, name: str) -> str:
    match = re.search(rf"^readonly {name}=.*$", text, re.M)
    assert match is not None, name
    return match[0]


def _cloud_block(text: str) -> str:
    start = text.index(_BLOCK_START)
    return text[start : text.index(_BLOCK_END, start)]


def _run_block(result: str, deploy_exit: int) -> subprocess.CompletedProcess[str]:
    text = _SCRIPT.read_text()
    shell = "\n".join(
        [
            "set -euo pipefail",
            'log() { echo "LOG:$*" >&2; }',
            'err() { echo "ERR:$*" >&2; }',
            _readonly(text, "CLOUD_MIGRATION_FAILED_DEPLOY_EXIT"),
            'ONEX_API_SERVICE="onex-api"',
            f'RESULT="{result}"',
            f"DEPLOY_EXIT={deploy_exit}",
            _cloud_block(text),
            'echo "RESULT=${RESULT}"',
        ]
    )
    return subprocess.run(
        ["bash", "-c", shell],
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )


def test_cloud_failure_replaces_a_green_refresh_result() -> None:
    proc = _run_block("SUCCESS", 3)
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "RESULT=FAILED_CLOUD_MIGRATION"
    assert "ERR:" in proc.stderr
    assert "onex-api" in proc.stderr
    assert "exited 3" in proc.stderr


@pytest.mark.parametrize(
    ("result", "deploy_exit"),
    [
        ("SUCCESS", 0),
        ("SUCCESS", 1),
        ("FAILED_ROLLED_BACK", 3),
        ("FAILED", 3),
    ],
)
def test_other_outcomes_are_left_as_they_were(result: str, deploy_exit: int) -> None:
    proc = _run_block(result, deploy_exit)
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == f"RESULT={result}"
    assert "ERR:" not in proc.stderr


def test_deploy_exit_code_matches_the_handler_constant() -> None:
    refresh = _readonly(_SCRIPT.read_text(), "CLOUD_MIGRATION_FAILED_DEPLOY_EXIT")
    handler = _readonly(_HANDLER.read_text(), "CLOUD_MIGRATION_FAILED_EXIT_CODE")
    assert refresh.split("=", 1)[1] == handler.split("=", 1)[1] == "3"


def test_cloud_check_runs_after_the_health_gate_and_before_the_census() -> None:
    text = _SCRIPT.read_text()
    health = text.index('log "=== Health-gate ==="')
    cloud = text.index(_BLOCK_START, health)
    census = text.index("lane_census_close ||", cloud)
    assert health < cloud < census


def test_receipt_records_the_deploy_exit_and_the_result_exits_one() -> None:
    text = _SCRIPT.read_text()
    assert '--argjson deploy_exit "${DEPLOY_EXIT}"' in text
    assert "deploy_exit: $deploy_exit" in text
    assert re.search(
        r"^\s*FAILED_ROLLED_BACK\|FAILED_BUILD_PROVENANCE\|FAILED_CLOUD_MIGRATION\) exit 1 ;;$",
        text,
        re.M,
    )
    assert "FAILED_CLOUD_MIGRATION" in text[: text.index("set -euo pipefail")]
