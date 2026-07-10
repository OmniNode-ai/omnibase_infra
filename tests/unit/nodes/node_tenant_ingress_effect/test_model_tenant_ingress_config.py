# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for ModelTenantIngressConfig validation."""

from __future__ import annotations

import pytest

from omnibase_infra.nodes.node_tenant_ingress_effect.models.model_tenant_ingress_config import (
    ModelTenantIngressConfig,
)

CANONICAL_TOPIC = "onex.cmd.omnimarket.delegate-skill.v1"


def test_accepts_valid_config() -> None:
    config = ModelTenantIngressConfig(
        tenants=("acme", "beta"),
        canonical_topic=CANONICAL_TOPIC,
    )
    assert config.tenants == ("acme", "beta")


def test_rejects_reserved_slug() -> None:
    with pytest.raises(ValueError, match="reserved"):
        ModelTenantIngressConfig(tenants=("system",), canonical_topic=CANONICAL_TOPIC)


def test_rejects_duplicate_tenants() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        ModelTenantIngressConfig(
            tenants=("acme", "acme"), canonical_topic=CANONICAL_TOPIC
        )


def test_rejects_malformed_slug() -> None:
    with pytest.raises(ValueError, match="DNS-compatible"):
        ModelTenantIngressConfig(
            tenants=("Not_Valid!",), canonical_topic=CANONICAL_TOPIC
        )


def test_rejects_tenant_prefixed_canonical_topic() -> None:
    with pytest.raises(ValueError, match="bare"):
        ModelTenantIngressConfig(
            tenants=("acme",), canonical_topic=f"tenant-acme.{CANONICAL_TOPIC}"
        )


def test_rejects_empty_canonical_topic() -> None:
    with pytest.raises(ValueError, match="not be empty"):
        ModelTenantIngressConfig(tenants=("acme",), canonical_topic="")


def test_config_is_frozen() -> None:
    config = ModelTenantIngressConfig(
        tenants=("acme",), canonical_topic=CANONICAL_TOPIC
    )
    with pytest.raises(ValueError):
        config.canonical_topic = "onex.cmd.other.v1"  # type: ignore[misc]
