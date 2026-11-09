# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""OMN-22867: the scheduled, dispatched and reusable workflows of this public repository run on a hosted runner.

CI runners are for private repositories only (RULING 2026-10-16T07:07:09Z). Each workflow named
here holds jobs that touch no lab service, so every job selects the literal GitHub-hosted label
and the file is listed in the routing policy's hosted_runner_allowlist with a reason.
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
    "alerter-injected-red-canary.yml",
    "attest-source-hash.yml",
    "branch-protection-audit.yml",
    "dep-provenance-lineage-live-canary.yml",
    "dependabot-runner-image-lock-refresh.yml",
    "in-progress-probe-hygiene.yml",
    "linear-demotion-guard-armed.yml",
    "linear-done-guard-armed.yml",
    "lockfile-cve-autobump.yml",
    "merge-queue-bypass-audit.yml",
    "node-migration-sync.yml",
    "omnimarket-contract-pin-refresh.yml",
    "plugin-pin-cascade.yml",
    "release-python-dry-run.yml",
    "release-python-reusable.yml",
    "sync-revert-watchdog.yml",
    "validate-validator-requirements.yml",
    "validator-runtime-profiles.yml",
)


@pytest.mark.parametrize("name", FAMILY)
def test_every_job_selects_the_hosted_label(name: str) -> None:
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
