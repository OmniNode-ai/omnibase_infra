# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Integration coverage for the runtime candidate build runner policy."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.integration

REPO_ROOT = Path(__file__).parents[2]
WORKFLOW_PATH = ".github/workflows/build-workspace-candidate-runtime.yml"


def test_candidate_runtime_build_uses_allowlisted_clean_cloud_egress() -> None:
    """The candidate and routing policy must agree on the hosted ECR lane."""
    workflow = yaml.safe_load((REPO_ROOT / WORKFLOW_PATH).read_text(encoding="utf-8"))
    policy = yaml.safe_load(
        (REPO_ROOT / "config/runner_routing_policy.yaml").read_text(encoding="utf-8")
    )

    assert workflow["jobs"]["build-workspace-candidate"]["runs-on"] == ("ubuntu-latest")
    allowlisted_paths = {entry["path"] for entry in policy["hosted_runner_allowlist"]}
    assert WORKFLOW_PATH in allowlisted_paths
