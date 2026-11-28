# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""OMN-22867: the lab-free jobs of this public repository run on a hosted runner.

CI runners are for private repositories only (RULING 2026-10-16T07:07:09Z). Each job named
here reads GitHub, a registry or the checked-out tree and reaches no lab service, so it
selects the literal GitHub-hosted label. Its workflow file is listed in the routing policy's
hosted_runner_allowlist with a reason. Other jobs in the same files are not named and keep
their own placement.
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
JOBS = (
    ("auto-merge.yml", "auto-merge"),
    ("auto-tag-on-merge.yml", "auto-tag"),
    ("deliver-dev-candidate-to-staging.yml", "dispatch-to-staging"),
    ("docker-build.yml", "build-summary"),
    ("docker-build.yml", "dockerfile-pin-check"),
    ("docker-build.yml", "domain-adapter-proof"),
    ("docker-build.yml", "image-analysis"),
    ("docker-build.yml", "security-scan"),
    ("hostile-reviewer.yml", "hostile-review-gate"),
    ("hostile-reviewer.yml", "hostile-review-thread-gate"),
    ("lane-census-parity.yml", "lab-overlay-tests"),
    ("lane-census-staleness.yml", "lane-census-staleness"),
    ("nightly-integration.yml", "notify-on-failure"),
    ("omni-standards-compliance.yml", "ci-naming-convention"),
    ("omni-standards-compliance.yml", "handler-contract-compliance"),
    ("omni-standards-compliance.yml", "omni-standards-gate"),
    ("omni-standards-compliance.yml", "type-safety"),
    ("omni-standards-compliance.yml", "type-union-check"),
    ("probe-placement-drift-reusable.yml", "drift"),
    ("product-readiness-shadow.yml", "lint-shadow"),
    ("product-readiness-shadow.yml", "reason-graph"),
    ("product-readiness-shadow.yml", "tests-shadow"),
    ("product-readiness-shadow.yml", "typecheck-shadow"),
    ("release.yml", "post-release-checks"),
    ("release.yml", "post-release-dev-bump"),
    ("release.yml", "release"),
    ("release.yml", "sync-main"),
    ("release.yml", "trigger-runtime-rebuild"),
    ("release-train-nightly.yml", "cut"),
    ("release-train-nightly.yml", "decide"),
    ("runner-image-build-smoke.yml", "runner-image-multiarch-smoke"),
    ("runtime-rebuild-trigger.yml", "rerun-refused-deliveries"),
    ("runtime-rebuild-trigger-reusable.yml", "deliver-sibling-candidate"),
    ("sibling-lock-refresh.yml", "refresh"),
)
FILES = tuple(dict.fromkeys(name for name, _ in JOBS))


@pytest.mark.parametrize(("name", "job"), JOBS)
def test_job_selects_the_hosted_label(name: str, job: str) -> None:
    data = yaml.safe_load((WORKFLOWS / name).read_text(encoding="utf-8"))
    assert data["jobs"][job]["runs-on"] == HOSTED_RUNNER


def test_every_file_is_in_the_hosted_allowlist_with_a_reason() -> None:
    policy = yaml.safe_load(POLICY.read_text(encoding="utf-8"))
    entries = {
        item["path"]: item.get("reason", "")
        for item in policy["hosted_runner_allowlist"]
    }
    missing = [n for n in FILES if f".github/workflows/{n}" not in entries]
    assert not missing, missing
    blank = [n for n in FILES if not entries[f".github/workflows/{n}"].strip()]
    assert not blank, blank
