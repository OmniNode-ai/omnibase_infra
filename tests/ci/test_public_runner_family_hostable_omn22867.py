# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""OMN-22867: workflows that read no lab service run on a hosted runner.

CI runners are for private repositories only (RULING 2026-10-16T07:07:09Z). Each workflow named
here selects the literal GitHub-hosted label in every job and is listed in the routing policy's
hosted_runner_allowlist with a reason.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS = REPO_ROOT / ".github" / "workflows"
POLICY = REPO_ROOT / "config" / "runner_routing_policy.yaml"
HOSTED_RUNNER = "ubuntu-latest"
FAMILY = (
    "artifact-reconciliation-webhook.yml",
    "dependency-cascade.yml",
    "fresh-deploy-fitness.yml",
    "pr-merged-event.yml",
    "prod-promotion-lineage.yml",
    "release-drift-monitor.yml",
    "runner-routing-audit.yml",
    "nightly-tests.yml",
)


@pytest.mark.parametrize("name", FAMILY)
def test_every_job_selects_a_hosted_label(name: str) -> None:
    data = yaml.safe_load((WORKFLOWS / name).read_text(encoding="utf-8"))
    selected = {
        job_name: job["runs-on"]
        for job_name, job in data["jobs"].items()
        if "runs-on" in job
    }
    assert selected, f"{name} declares no runs-on"
    assert all(value == HOSTED_RUNNER for value in selected.values()), selected


def test_every_family_file_is_in_the_hosted_allowlist() -> None:
    policy = yaml.safe_load(POLICY.read_text(encoding="utf-8"))
    entries = {
        item["path"]: item.get("reason", "")
        for item in policy["hosted_runner_allowlist"]
    }
    missing = [n for n in FAMILY if f".github/workflows/{n}" not in entries]
    assert not missing, missing
    blank = [n for n in FAMILY if not entries[f".github/workflows/{n}"].strip()]
    assert not blank, blank
