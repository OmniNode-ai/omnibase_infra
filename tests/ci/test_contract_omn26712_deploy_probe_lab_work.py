# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-26712: the deploy probe of ``contracts/OMN-26712.yaml`` runs as a lab work item.

The AC3 deploy probe needs a Docker daemon, which a hosted runner does not have. It
therefore must (a) run only in the local Done gate, (b) bind no acceptance criterion
(AC3 stays bound by the hosted migration test) and (c) reach the lab through
``onex lab-work run`` on the dev bus lane at the checked-out sha, never as a raw
``docker`` command on the machine that evaluates the contract.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

pytestmark = pytest.mark.unit

_CONTRACT = Path(__file__).resolve().parents[2] / "contracts" / "OMN-26712.yaml"
_PROBE_ID = "dod-omn26712-ac3-deploy-probe"
_HOSTED_AC3_ID = "dod-omn26712-ac3-migration-idempotent"


def _items() -> dict[str, dict[str, Any]]:
    document = yaml.safe_load(_CONTRACT.read_text(encoding="utf-8"))
    return {item["id"]: item for item in document["dod_evidence"]}


def test_probe_is_local_done_gate_and_binds_no_criterion() -> None:
    probe = _items()[_PROBE_ID]
    assert probe.get("execution_scope") == "local_done_gate"
    assert not probe.get("binds_ac")


def test_probe_goes_through_lab_work_on_the_dev_lane_at_head() -> None:
    (check,) = _items()[_PROBE_ID]["checks"]
    command = str(check["check_value"])
    assert "onex lab-work run" in command
    assert "--bus-lane dev" in command
    assert "git rev-parse HEAD" in command
    assert "docker info" not in command


def test_probe_needs_docker_host() -> None:
    """Placement without ``--need docker`` can pick a host with no Docker daemon (h101)."""
    (check,) = _items()[_PROBE_ID]["checks"]
    command = str(check["check_value"])
    assert "--need docker" in command


def test_ac3_stays_bound_by_the_hosted_migration_test() -> None:
    hosted = _items()[_HOSTED_AC3_ID]
    assert hosted["binds_ac"] == ["AC3"]
    assert hosted.get("execution_scope") in (None, "hosted")
