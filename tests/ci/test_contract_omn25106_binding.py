# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The OMN-25106 contract binds AC2 to a reading it can fail on (OMN-25169).

AC2 is the dev runtime answering ``/ready`` 200 after a redeploy, with one boot
line in its log. A curl that passes on any code under 400, or a check that
reads no log line, cannot show that criterion's falsifier, and unit tests that
never touch h201 cannot either.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]
CONTRACT = REPO_ROOT / "contracts" / "OMN-25106.yaml"
KERNEL = REPO_ROOT / "src" / "omnibase_infra" / "runtime" / "service_kernel.py"
BOOT_LINE = "ONEX runtime started successfully"
AGENT_TEST = "scripts/deploy-agent/tests/unit/test_wedged_restore_omn25106.py"


def _contract() -> dict[str, Any]:
    loaded = yaml.safe_load(CONTRACT.read_text())
    assert isinstance(loaded, dict)
    return loaded


def _checks_bound_to(ac: str) -> list[str]:
    return [
        check["check_value"]
        for entry in _contract()["dod_evidence"]
        if ac in entry["binds_ac"]
        for check in entry["checks"]
    ]


def test_ac2_is_bound_only_to_readings_not_to_unit_tests() -> None:
    checks = _checks_bound_to("AC2")

    assert checks, "AC2 is bound to no check"
    assert not [c for c in checks if "pytest" in c]


def test_the_ready_check_fails_on_any_code_but_200() -> None:
    ready = [c for c in _checks_bound_to("AC2") if "/ready" in c]

    assert len(ready) == 1
    # ``curl -sf`` passes on any status under 400; the code must be compared.
    assert "-f" not in ready[0].split("curl", 1)[1].split()[0]
    assert "%{http_code}" in ready[0]
    assert "= 200" in ready[0]


def test_a_check_reads_the_runtime_boot_line() -> None:
    reads = [c for c in _checks_bound_to("AC2") if "docker logs" in c]

    assert len(reads) == 1
    assert BOOT_LINE in reads[0]
    assert "grep -c" in reads[0]


def test_the_boot_line_the_check_greps_is_still_logged_by_the_kernel() -> None:
    """Renaming the log line turns this red instead of leaving a check that can never pass."""
    assert BOOT_LINE in KERNEL.read_text()


def test_the_agent_restore_test_is_in_the_evidence_commands() -> None:
    commands = [r["command"] for r in _contract()["evidence_requirements"]]

    agent = [c for c in commands if AGENT_TEST in c]
    assert len(agent) == 1
    assert "--project scripts/deploy-agent" in agent[0]
