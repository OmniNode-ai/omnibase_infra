# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-25942: the ticket contract binds the h201 install proof to a criterion.

The install criterion (a dev deploy job reads success with verification success
on the installed revision) had only an unbound item carrying a placeholder job
id. The contract must bind a criterion to a real job-record check.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.unit

CONTRACT = Path(__file__).resolve().parents[2] / "contracts" / "OMN-25942.yaml"


def _load() -> dict:
    return yaml.safe_load(CONTRACT.read_text())


def test_install_criterion_is_bound_by_a_hosted_item_and_reads_a_job_record() -> None:
    contract = _load()
    ac_ids = {ac["id"] for req in contract["requirements"] for ac in req["acceptance"]}
    assert "AC7" in ac_ids
    own = Path(__file__).name
    hosted = [
        item
        for item in contract["dod_evidence"]
        if "AC7" in item.get("binds_ac", []) and own in str(item["checks"])
    ]
    assert hosted, "no hosted dod_evidence item binds AC7 to this test"
    for item in hosted:
        assert item.get("execution_scope", "default") != "local_done_gate"
    # The live h201 job record cannot be read on a hosted runner, so its item
    # stays an unbound local Done gate (the hosted gate refuses a bound one).
    job_record = [
        item for item in contract["dod_evidence"] if "jobs-dev" in str(item["checks"])
    ]
    assert job_record, "no dod_evidence item reads a jobs-dev job record"
    for item in job_record:
        assert item.get("execution_scope") == "local_done_gate"
        assert not item.get("binds_ac")
        assert re.search(r"' [0-9a-f]{8}$", item["checks"][0]["check_value"])


def test_job_record_check_carries_no_placeholder() -> None:
    text = CONTRACT.read_text()
    assert "DEV_JOB_ID_PREFIX" not in text
