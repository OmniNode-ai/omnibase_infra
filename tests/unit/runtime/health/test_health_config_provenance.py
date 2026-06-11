# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Tests for the config-provenance health check (OMN-12958)."""

from __future__ import annotations

from omnibase_infra.runtime.config_provenance import ModelConfigProvenance
from omnibase_infra.runtime.health.health_config_provenance import (
    check_config_provenance_health,
)


def _provenance(
    *, deployed_sha: str | None, source_sha: str | None
) -> ModelConfigProvenance:
    return ModelConfigProvenance(
        config_name="bifrost_delegation",
        deployed_path="/app/data/delegation/bifrost_delegation.yaml",
        deployed_sha256=deployed_sha,
        source_path="/app/src/omnimarket/configs/bifrost_delegation.yaml",
        source_sha256=source_sha,
    )


def test_in_sync_is_healthy() -> None:
    health = check_config_provenance_health(
        _provenance(deployed_sha="abc", source_sha="abc")
    )
    assert health.status == "healthy"
    assert health.details is not None
    assert health.details["has_drifted"] is False


def test_drift_is_degraded() -> None:
    health = check_config_provenance_health(
        _provenance(deployed_sha="abc", source_sha="def")
    )
    assert health.status == "degraded"
    assert health.error is not None
    assert "re-seed required" in health.error
    assert health.details is not None
    assert health.details["has_drifted"] is True


def test_absent_deployed_is_unhealthy() -> None:
    health = check_config_provenance_health(
        _provenance(deployed_sha=None, source_sha="abc")
    )
    assert health.status == "unhealthy"
    assert health.error is not None
    assert "deployed config absent" in health.error


def test_absent_source_is_degraded() -> None:
    health = check_config_provenance_health(
        _provenance(deployed_sha="abc", source_sha=None)
    )
    assert health.status == "degraded"
    assert health.error is not None
    assert "cannot prove config provenance" in health.error
