# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2026 OmniNode Team
"""Tests for runtime-grounded consumer group ID derivation [OMN-7040]."""

from __future__ import annotations

import pytest

from omnibase_infra.verification.probes.probe_subscription import (
    _derive_consumer_group_id,
)


@pytest.mark.unit
def test_derive_consumer_group_id_matches_canonical_format() -> None:
    """Derived group ID must follow {env}.{service}.{node}.{purpose}.{version} format."""
    from omnibase_infra.models import ModelNodeIdentity

    identity = ModelNodeIdentity(
        env="local",
        service="runtime_config",
        node_name="runtime_config",
        version="1.0.0",
    )
    group_id, grounding = _derive_consumer_group_id(
        "node_registration_orchestrator", identity=identity
    )
    assert group_id == "local.runtime_config.runtime_config.consume.1.0.0"
    assert grounding == "EXACT"


@pytest.mark.unit
def test_derive_consumer_group_id_with_identity_has_five_segments() -> None:
    """With identity, derived group ID must have at minimum 5 dot-separated segments."""
    from omnibase_infra.models import ModelNodeIdentity

    identity = ModelNodeIdentity(
        env="dev",
        service="omnibase_infra",
        node_name="test_node",
        version="v1",
    )
    group_id, grounding = _derive_consumer_group_id("test_node", identity=identity)
    parts = group_id.split(".")
    assert len(parts) >= 5, f"Group ID '{group_id}' has fewer than 5 segments"
    assert grounding == "EXACT"


@pytest.mark.unit
def test_derive_consumer_group_id_without_identity_does_not_crash() -> None:
    """Without identity, derivation should fall back gracefully."""
    group_id, grounding = _derive_consumer_group_id("node_registration_orchestrator")
    assert isinstance(group_id, str)
    assert len(group_id) > 0
    assert grounding in {"DISCOVERED", "FABRICATED"}


@pytest.mark.unit
def test_derive_consumer_group_id_without_identity_uses_unknown_markers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without identity, fabricated ID uses 'unknown' env/service markers."""
    from omnibase_infra.verification.probes import probe_subscription

    monkeypatch.setattr(
        probe_subscription,
        "_discover_identity_via_rpk",
        lambda _contract_name: None,
    )
    group_id, grounding = _derive_consumer_group_id("node_registration_orchestrator")
    assert "unknown" in group_id
    assert grounding == "FABRICATED"
