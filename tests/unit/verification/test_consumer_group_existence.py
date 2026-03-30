# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2026 OmniNode Team
"""Tests for consumer group existence validation utility [OMN-7040]."""

from __future__ import annotations

import pytest

from omnibase_infra.verification.probes.probe_subscription import (
    _derive_consumer_group_id,
)


@pytest.mark.unit
def test_derived_group_follows_canonical_format() -> None:
    """Derived group ID must follow {env}.{service}.{node}.{purpose}.{version} format."""
    from omnibase_infra.models import ModelNodeIdentity

    identity = ModelNodeIdentity(
        env="local",
        service="runtime_config",
        node_name="runtime_config",
        version="1.0.0",
    )
    group_id = _derive_consumer_group_id(
        "node_registration_orchestrator", identity=identity
    )
    assert group_id == "local.runtime_config.runtime_config.consume.1.0.0"


@pytest.mark.unit
def test_derived_group_without_identity_does_not_crash() -> None:
    """Without identity, derivation should fall back gracefully."""
    group_id = _derive_consumer_group_id("node_registration_orchestrator")
    assert isinstance(group_id, str)
    assert len(group_id) > 0
