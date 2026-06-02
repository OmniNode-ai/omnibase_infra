# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Lane/digest contract tests for deploy-agent command/result models (OMN-12572).

The deploy-agent boundary must carry ``runtime_lane`` and ``image_digest`` so
lane and digest truth can be enforced at the effect boundary. Digest is the
authority; production must pin a stability-proven digest rather than rebuild
from a ref.
"""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

import pytest
from deploy_agent.events import (
    EnumRuntimeLane,
    ModelRebuildCompleted,
    ModelRebuildRequested,
    Phase,
    PhaseStatus,
    Scope,
)
from pydantic import ValidationError

pytestmark = pytest.mark.unit


def test_runtime_lane_enum_members() -> None:
    assert EnumRuntimeLane.DEV.value == "dev"
    assert EnumRuntimeLane.STABILITY_TEST.value == "stability-test"
    assert EnumRuntimeLane.PROD.value == "prod"


def test_rebuild_requested_carries_lane_ref_and_digest() -> None:
    cmd = ModelRebuildRequested(
        correlation_id=uuid4(),
        requested_by="ci",
        scope=Scope.RUNTIME,
        runtime_lane=EnumRuntimeLane.STABILITY_TEST,
        image_ref="ghcr.io/omninode-ai/runtime:0.37.0",
        image_digest="sha256:" + "a" * 64,
    )
    assert cmd.runtime_lane is EnumRuntimeLane.STABILITY_TEST
    assert cmd.image_ref == "ghcr.io/omninode-ai/runtime:0.37.0"
    assert cmd.image_digest == "sha256:" + "a" * 64


def test_rebuild_requested_requires_runtime_lane() -> None:
    with pytest.raises(ValidationError, match="runtime_lane"):
        ModelRebuildRequested(
            correlation_id=uuid4(),
            requested_by="ci",
            scope=Scope.RUNTIME,
            # runtime_lane omitted — must be rejected, no silent default
        )


def test_prod_rebuild_requested_requires_image_digest() -> None:
    """Prod deploys the exact digest proven in stability — a prod request
    without a digest is a contract violation (digest is the authority)."""
    with pytest.raises(ValidationError, match="image_digest"):
        ModelRebuildRequested(
            correlation_id=uuid4(),
            requested_by="ci",
            scope=Scope.RUNTIME,
            runtime_lane=EnumRuntimeLane.PROD,
            image_ref="ghcr.io/omninode-ai/runtime:0.37.0",
            # image_digest omitted for a prod request — must be rejected
        )


def test_dev_and_stability_rebuild_requested_allow_absent_digest() -> None:
    """Dev/stability-test build from a ref, so the digest may be resolved by the
    build rather than pinned up front."""
    for lane in (EnumRuntimeLane.DEV, EnumRuntimeLane.STABILITY_TEST):
        cmd = ModelRebuildRequested(
            correlation_id=uuid4(),
            requested_by="ci",
            scope=Scope.RUNTIME,
            runtime_lane=lane,
        )
        assert cmd.runtime_lane is lane
        assert cmd.image_digest is None


def test_rebuild_completed_carries_lane_and_digest() -> None:
    now = datetime.now(UTC)
    completed = ModelRebuildCompleted(
        correlation_id=uuid4(),
        requested_git_ref="origin/main",
        git_sha="abc123",
        started_at=now,
        completed_at=now,
        duration_seconds=1.0,
        scope=Scope.RUNTIME,
        runtime_lane=EnumRuntimeLane.PROD,
        image_ref="ghcr.io/omninode-ai/runtime:0.37.0",
        image_digest="sha256:" + "b" * 64,
        phase_results={Phase.RUNTIME: PhaseStatus.SUCCESS},
    )
    assert completed.runtime_lane is EnumRuntimeLane.PROD
    assert completed.image_digest == "sha256:" + "b" * 64
    assert completed.status == "success"


def test_rebuild_completed_requires_runtime_lane() -> None:
    now = datetime.now(UTC)
    with pytest.raises(ValidationError, match="runtime_lane"):
        ModelRebuildCompleted(
            correlation_id=uuid4(),
            requested_git_ref="origin/main",
            git_sha="abc123",
            started_at=now,
            completed_at=now,
            duration_seconds=1.0,
            scope=Scope.RUNTIME,
            # runtime_lane omitted
            phase_results={Phase.RUNTIME: PhaseStatus.SUCCESS},
        )
