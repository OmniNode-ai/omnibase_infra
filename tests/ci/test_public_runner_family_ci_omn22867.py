# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""OMN-22867: every job of ci.yml in this public repository runs on a hosted runner.

CI runners are for private repositories only (RULING 2026-10-16T07:07:09Z). The core
public_runner_placement validator refused the computed runs-on of 56 jobs in ci.yml; each now
selects the literal GitHub-hosted label, and the file stays in the routing policy's
hosted_runner_allowlist.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]
CI = REPO_ROOT / ".github" / "workflows" / "ci.yml"
POLICY = REPO_ROOT / "config" / "runner_routing_policy.yaml"
HOSTED_RUNNER = "ubuntu-latest"
HOSTED_LABELS = frozenset({"ubuntu-latest", "ubuntu-24.04"})
MIGRATED_JOBS = (
    "ai-slop-check",
    "application-database-domain-enforcement",
    "arch-invariants",
    "bus-acl-boundary",
    "canonical-file-shape",
    "canonical-handler-shape-gate",
    "ci-cascade-reason-graph",
    "ci-evidence-policy",
    "ci-summary",
    "compliance",
    "compose-required-env-coverage",
    "contract-path-preflight",
    "contract-sync-gate",
    "customer-path-boundary",
    "delegation-seam-gate",
    "demo-loop-gate",
    "dep-provenance-lineage-gate",
    "detect-changes",
    "effect-assertion-gate",
    "event-chain-gate",
    "exposed-identifier-gate",
    "fingerprint-check",
    "infra-node-handler-ownership",
    "integration-guard",
    "kafka-boundary-compat",
    "lab-probe-windows",
    "laptop-profile-paths",
    "lint",
    "lockfile-cve-scan",
    "lockfile-registry-allowlist",
    "migration-conflict-check",
    "migration-freeze",
    "migration-integration",
    "migration-required-check",
    "no-noncanonical-lifecycle-classes",
    "no-plugin-daemon-classes-gate",
    "node-boundary-imports",
    "node-migration-declaration-check",
    "onex-validation",
    "pin-reachability",
    "plugin-env-service-completeness",
    "reconciler-movement-proof",
    "runtime-profiles-validator",
    "schema-handshake",
    "service-integration-suites",
    "shape-gate-independence",
    "skip-count-ratchet",
    "subscribe-wiring-health",
    "test-failure-ratchet",
    "tests-gate",
    "topic-drift-check",
    "topic-enum-drift",
    "topic-naming-lint",
    "trivyignore-expiry-check",
    "validate-validator-requirements",
    "version-pin-check",
)


def _jobs() -> dict[str, dict[str, object]]:
    data = yaml.safe_load(CI.read_text(encoding="utf-8"))
    return data["jobs"]


@pytest.mark.parametrize("job", MIGRATED_JOBS)
def test_migrated_job_selects_the_hosted_label(job: str) -> None:
    assert _jobs()[job]["runs-on"] == HOSTED_RUNNER


def test_no_job_selects_a_computed_or_lab_runner() -> None:
    selected = {
        name: job["runs-on"] for name, job in _jobs().items() if "runs-on" in job
    }
    assert selected, "ci.yml declares no runs-on"
    bad = {
        name: value for name, value in selected.items() if value not in HOSTED_LABELS
    }
    assert not bad, bad


def test_ci_yml_stays_in_the_hosted_allowlist() -> None:
    policy = yaml.safe_load(POLICY.read_text(encoding="utf-8"))
    entries = {
        item["path"]: item.get("reason", "")
        for item in policy["hosted_runner_allowlist"]
    }
    assert entries.get(".github/workflows/ci.yml", "").strip()
