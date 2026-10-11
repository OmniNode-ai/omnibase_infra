# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The lab-overlay parity job is an enforced context, under the name GitHub reports (OMN-20934).

``dev`` requires exactly one context, ``CI Summary``, so a verification job is
only enforced when ``EXPECTED_EXTERNAL_CONTEXTS`` names it exactly as GitHub
reports it. A misspelt name is not a weaker gate: the poller waits for a
check-run that never arrives and fails the whole branch at its deadline.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "lane-census-parity.yml"
ROUTING_POLICY = REPO_ROOT / "config" / "runner_routing_policy.yaml"

sys.path.insert(0, str(REPO_ROOT / "scripts" / "ci"))
import ci_summary_gate as gate


def _job_name() -> str:
    workflow = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    (job,) = workflow["jobs"].values()
    name: str = job["name"]
    return name


@pytest.mark.live_contact(
    "tests/ci/fixtures/lab_overlay_parity_check_run_omn20934.json"
)
def test_the_recorded_check_run_name_is_the_registered_context(
    recorded_response: dict[str, object],
) -> None:
    response = recorded_response["response"]
    assert isinstance(response, dict)
    reported = response["check_run"]["name"]
    assert reported == _job_name()
    assert reported in gate.EXPECTED_EXTERNAL_CONTEXTS
    # control: the name as a person would misremember it is not registered, so
    # the membership above is what separates an enforced context from an absent one
    assert reported.lower() not in gate.EXPECTED_EXTERNAL_CONTEXTS


def test_the_alerter_policy_declares_the_registered_context() -> None:
    policy = yaml.safe_load(ROUTING_POLICY.read_text(encoding="utf-8"))
    declared = policy["route"]["nonrequired_check_alert"]["umbrella_enforced_contexts"][
        "omnibase_infra"
    ]
    assert _job_name() in declared
