# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-25942: the ticket contract binds the h201 install proof to a criterion.

The install criterion (a dev deploy job reads success with verification success
on the installed revision) had only an unbound item carrying a placeholder job
id. The contract must bind a criterion to a real job-record check.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.unit

CONTRACT = Path(__file__).resolve().parents[2] / "contracts" / "OMN-25942.yaml"


def _load() -> dict:
    return yaml.safe_load(CONTRACT.read_text())


def test_install_criterion_is_bound_to_a_job_record_check() -> None:
    contract = _load()
    ac_ids = {ac["id"] for req in contract["requirements"] for ac in req["acceptance"]}
    bound = [
        item
        for item in contract["dod_evidence"]
        if "jobs-dev" in str(item["checks"]) and item.get("binds_ac")
    ]
    assert bound, "no bound dod_evidence item reads a jobs-dev job record"
    for item in bound:
        assert set(item["binds_ac"]) <= ac_ids


def test_job_record_check_carries_no_placeholder() -> None:
    text = CONTRACT.read_text()
    assert "DEV_JOB_ID_PREFIX" not in text
