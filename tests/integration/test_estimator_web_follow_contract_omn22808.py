# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-22808: the shipped provision contract declares a followable estimator web source."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from omnibase_infra.nodes.node_setup_local_provision_effect.runtime_cost_estimator_build import (
    ModelCostEstimatorWebFollow,
)

pytestmark = pytest.mark.integration

CONTRACT = (
    Path(__file__).resolve().parents[2]
    / "src/omnibase_infra/nodes/node_setup_local_provision_effect/contract.yaml"
)


def test_shipped_contract_follow_block_builds_the_follow_model() -> None:
    declaration = yaml.safe_load(CONTRACT.read_text(encoding="utf-8"))["config"][
        "cost_estimator_build"
    ]
    build = declaration["build"]
    follow = ModelCostEstimatorWebFollow(
        repository="web-front-end",
        repository_url="https://example.invalid/web-front-end.git",
        branch=str(build["branch"]),
        container=str(declaration["follow"]["container"]),
        source_label="ai.omninode.image.source-repo",
        revision_label="org.opencontainers.image.revision",
        read_timeout_seconds=int(build["read_timeout_seconds"]),
    )
    assert follow.container == "omninode-cost-estimator-web"
    assert follow.branch == "dev"
