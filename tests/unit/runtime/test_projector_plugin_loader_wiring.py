# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Unit tests verifying ProjectorShell is wired into ProjectorPluginLoader (OMN-4484).

OMN-1169 implemented ProjectorShell. This test verifies that ProjectorPluginLoader._create_projector
instantiates ProjectorShell (not ProjectorShellPlaceholder) when a database pool is provided.

Tests call _create_projector() directly to avoid I/O, since the wiring logic lives entirely
in that method.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from omnibase_infra.runtime.projector_plugin_loader import (
    ProjectorPluginLoader,
    ProjectorShellPlaceholder,
)
from omnibase_infra.runtime.projector_shell import ProjectorShell


def _make_minimal_contract() -> object:
    """Build a minimal ModelProjectorContract for testing."""
    from omnibase_core.models.projectors import ModelProjectorContract

    return ModelProjectorContract.model_validate(
        {
            "projector_kind": "materialized_view",
            "projector_id": "test-projector-wiring",
            "name": "Wiring Test Projector",
            "version": "1.0.0",
            "aggregate_type": "TestAggregate",
            "consumed_events": ["test.created.v1"],
            "projection_schema": {
                "table": "test_wiring_projections",
                "primary_key": "id",
                "columns": [
                    {"name": "id", "type": "UUID", "source": "event.payload.id"}
                ],
            },
            "behavior": {"mode": "upsert"},
        }
    )


@pytest.mark.unit
def test_loader_produces_projector_shell_when_pool_provided() -> None:
    """After OMN-1169, loader must instantiate ProjectorShell when pool is provided."""
    mock_pool = MagicMock()
    loader = ProjectorPluginLoader(pool=mock_pool)
    contract = _make_minimal_contract()

    projector = loader._create_projector(contract)  # type: ignore[arg-type]

    assert isinstance(projector, ProjectorShell), (
        f"Expected ProjectorShell, got {type(projector).__name__}. "
        "ProjectorShellPlaceholder indicates pool wiring is broken."
    )
    assert not projector.is_placeholder
    assert callable(projector.project)


@pytest.mark.unit
def test_loader_produces_placeholder_when_no_pool() -> None:
    """Loader falls back to ProjectorShellPlaceholder when no pool is provided.

    ProjectorShellPlaceholder is the correct fallback for discovery-only scenarios
    where database access is not needed.
    """
    loader = ProjectorPluginLoader(pool=None)
    contract = _make_minimal_contract()

    projector = loader._create_projector(contract)  # type: ignore[arg-type]

    assert isinstance(projector, ProjectorShellPlaceholder), (
        f"Expected ProjectorShellPlaceholder without pool, got {type(projector).__name__}."
    )
    assert projector.is_placeholder
