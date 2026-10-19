# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-18496: a refresh whose deploy exited with the cloud-degraded code reports
FAILED_CLOUD_MIGRATION, exit 1, and never runs the rollback.
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "scripts/runtime_build/refresh_dev_lane.sh"
DEPLOY_SCRIPT = ROOT / "src/omnibase_infra/handlers/handler_runtime_deploy.sh"


@pytest.mark.parametrize(
    ("result", "deploy_exit", "expected"),
    [
        ("SUCCESS", 3, "FAILED_CLOUD_MIGRATION"),
        ("SUCCESS", 0, "SUCCESS"),
        ("SUCCESS", 1, "SUCCESS"),
        ("FAILED", 3, "FAILED"),
        ("FAILED_ROLLED_BACK", 3, "FAILED_ROLLED_BACK"),
    ],
)
def test_apply_cloud_migration_deploy_exit(
    result: str, deploy_exit: int, expected: str
) -> None:
    match = re.search(
        r"^apply_cloud_migration_deploy_exit\(\) \{.*?^\}",
        SCRIPT.read_text(),
        re.M | re.S,
    )
    assert match is not None
    shell = "\n".join(
        [
            "set -euo pipefail",
            'err() { echo "ERR:$*" >&2; }',
            "DEPLOY_EXIT_CLOUD_MIGRATION_DEGRADED=3",
            f"RESULT={result}; DEPLOY_EXIT={deploy_exit}",
            match[0],
            "apply_cloud_migration_deploy_exit",
            'echo "RESULT=${RESULT}"',
        ]
    )
    env = {key: value for key, value in os.environ.items() if key in {"PATH", "HOME"}}
    proc = subprocess.run(
        ["bash", "-c", shell],
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
        env=env,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert f"RESULT={expected}\n" in proc.stdout
    if expected == "FAILED_CLOUD_MIGRATION":
        assert "omninode_cloud" in proc.stderr
    else:
        assert proc.stderr == ""


def test_refresh_applies_cloud_gate_after_health_gate_before_census() -> None:
    source = SCRIPT.read_text()
    health = source.index('log "=== Health-gate ==="')
    apply_gate = source.index("\napply_cloud_migration_deploy_exit\n", health)
    census = source.index("lane_census_close ||", health)
    assert health < apply_gate < census
    final_case = source[source.rindex('case "${RESULT}" in') :]
    assert re.search(r"FAILED_CLOUD_MIGRATION[^)]*\)\s*exit 1", final_case)


def test_degraded_exit_code_matches_deploy() -> None:
    pattern = r"^readonly DEPLOY_EXIT_CLOUD_MIGRATION_DEGRADED=(\d+)$"
    refresh = re.search(pattern, SCRIPT.read_text(), re.M)
    deploy = re.search(pattern, DEPLOY_SCRIPT.read_text(), re.M)
    assert refresh is not None
    assert deploy is not None
    assert refresh[1] == deploy[1]
