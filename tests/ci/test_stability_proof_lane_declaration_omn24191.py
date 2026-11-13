# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-24191: the declared stability lane satisfies exactly the proofs it can.

The three stability proofs (C15, C16, C28) grade the lane the repository
variable ``LAB_PROOF_LANES_JSON`` declares. The deployment overlay's own
test and omnimarket's overlay model test hold the same shape (documentation-range
addresses here; the real addresses live only in the deployment overlay), so a change
to the shape of the value fails the test of the repository that changed.

This file spells the declared value as a golden string and holds it against
the resolver and the three workflows:

* the golden value parses under the resolver into one stability lane;
* each workflow's resolver step requires exactly the fields the golden value
  declares (a workflow that adds a required field fails here, so the
  declaration is revisited);
* the declared lane satisfies exactly the proofs it can satisfy today (C28);
  C15 and C16 need ``gateway_url``, which the stability lane does not declare
  until the runtime serves the tenant route (one-runtime plan T18);
* a lane that declares ``gateway_url`` satisfies all three (control);
* a workflow requiring an undeclared field is refused (control);
* a loopback or host-alias URL is refused by the resolver (control).
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
import yaml

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]
ACTION_DIR = REPO_ROOT / ".github" / "actions" / "resolve-lab-lane"
WORKFLOWS = REPO_ROOT / ".github" / "workflows"

GOLDEN_VALUE = (
    '[{"name":"lane-a","lane_role":"stability",'
    '"compose_project":"omnibase-infra-lane-a",'
    '"bus_lane":"lane-a",'
    '"docker_runs_on":["self-hosted","label-a"],'
    '"ingress_url":"http://192.0.2.10:18085",'
    '"projection_url":"http://192.0.2.10:13002",'
    '"deploy_agent_url":"http://192.0.2.10:8098"}]'
)

PROOF_REQUIRES: dict[str, str] = {
    "chain-canary.yml": "ingress_url gateway_url bus_lane",
    "chain-canary-c16-receipt-identity.yml": "gateway_url bus_lane",
    "chain-canary-c28-consumer-flow.yml": (
        "projection_url docker_runs_on compose_project"
    ),
}

# C15 and C16 need gateway_url, which the stability lane does not declare
# until the runtime serves the tenant route (one-runtime plan T18).
PROOFS_SATISFIED_TODAY = {"chain-canary-c28-consumer-flow.yml"}

STABILITY_MATCH = "lane_role=stability"


def _load_resolver() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "resolve_lab_lane_omn24191", ACTION_DIR / "resolve_lab_lane.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


resolver = _load_resolver()


def _resolver_steps(workflow: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        step
        for job in workflow["jobs"].values()
        for step in job.get("steps", [])
        if step.get("uses") == "./.github/actions/resolve-lab-lane"
    ]


def _lanes() -> list[dict[str, Any]]:
    return resolver.parse_lanes(GOLDEN_VALUE, "LAB_PROOF_LANES_JSON")


def test_golden_value_parses_under_the_resolver() -> None:
    lanes = _lanes()
    assert len(lanes) == 1
    lane = lanes[0]
    assert lane["name"] == "lane-a"
    assert lane["lane_role"] == "stability"


@pytest.mark.parametrize("workflow", sorted(PROOF_REQUIRES))
def test_each_workflow_requires_the_spelled_fields(workflow: str) -> None:
    loaded = yaml.safe_load((WORKFLOWS / workflow).read_text(encoding="utf-8"))
    steps = _resolver_steps(loaded)
    assert steps, f"{workflow}: no lane resolver step"
    for step in steps:
        given = step["with"]
        assert given["require"] == PROOF_REQUIRES[workflow], (
            f"{workflow}: the resolver step requires {given['require']!r}, "
            f"not the spelled {PROOF_REQUIRES[workflow]!r}"
        )
        assert STABILITY_MATCH in given.get("match", "").split(), (
            f"{workflow}: a lane that is not a stability lane is eligible"
        )


def test_the_declared_lane_satisfies_exactly_the_proofs_satisfied_today() -> None:
    lanes = _lanes()
    satisfied: set[str] = set()
    tried_by_workflow: dict[str, str] = {}
    for workflow, required in PROOF_REQUIRES.items():
        lane, tried = resolver.choose_lane(
            lanes, required.split(), lambda url: None, {"lane_role": "stability"}
        )
        tried_by_workflow[workflow] = " | ".join(tried)
        if lane is not None:
            satisfied.add(workflow)
    assert satisfied == PROOFS_SATISFIED_TODAY, (
        f"the declared lane satisfies {sorted(satisfied)}, not "
        f"{sorted(PROOFS_SATISFIED_TODAY)}"
    )
    for workflow in PROOF_REQUIRES:
        if workflow not in PROOFS_SATISFIED_TODAY:
            assert "does not declare gateway_url" in tried_by_workflow[workflow], (
                f"{workflow}: the refusal does not name the missing field"
            )


def test_a_lane_with_gateway_url_satisfies_all_three() -> None:
    lanes = _lanes()
    lanes[0]["gateway_url"] = "http://192.0.2.10:18090"
    for workflow, required in PROOF_REQUIRES.items():
        lane, _tried = resolver.choose_lane(
            lanes, required.split(), lambda url: None, {"lane_role": "stability"}
        )
        assert lane is not None, (
            f"{workflow}: a lane that declares gateway_url still does not "
            f"satisfy the proof; PROOFS_SATISFIED_TODAY must be updated"
        )


def test_a_workflow_requiring_an_undeclared_field_is_refused() -> None:
    lanes = _lanes()
    require = [
        "projection_url",
        "docker_runs_on",
        "compose_project",
        "deploy_wait_field",
    ]
    lane, tried = resolver.choose_lane(
        lanes, require, lambda url: None, {"lane_role": "stability"}
    )
    assert lane is None
    assert any("deploy_wait_field" in line for line in tried), (
        "the refusal does not name the missing field"
    )


@pytest.mark.parametrize(
    "bad_host",
    ["127.0.0.1:13002", "host.docker.internal:13002"],
)
def test_a_loopback_or_host_alias_value_is_refused(bad_host: str) -> None:
    bad_value = GOLDEN_VALUE.replace("192.0.2.10:13002", bad_host)
    assert bad_value != GOLDEN_VALUE, "the substitution did not take"
    with pytest.raises(resolver.OverlayError):
        resolver.parse_lanes(bad_value, "LAB_PROOF_LANES_JSON")
