# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Unit tests for the canonical verified-tenant payload stamp (OMN-14367).

Pins the single shape both tenant-stamp producers (the auto-wiring
``tenant_scoped_ingress`` stamp and the gateway forwarder's consume_inbound)
must emit so they cannot diverge again: ``tenant_id`` = the verified slug, no
separate ``tenant_slug`` key, verified value always wins.
"""

from __future__ import annotations

from types import MappingProxyType

from omnibase_infra.shared.tenant_stamp import stamp_verified_tenant_slug


def test_stamps_slug_into_tenant_id() -> None:
    result = stamp_verified_tenant_slug({"prompt": "hi"}, "acme")
    assert result == {"prompt": "hi", "tenant_id": "acme"}


def test_overwrites_client_supplied_tenant_id() -> None:
    """The verified slug always wins over a self-reported tenant_id."""
    result = stamp_verified_tenant_slug(
        {"prompt": "hi", "tenant_id": "attacker-tenant"}, "acme"
    )
    assert result["tenant_id"] == "acme"


def test_emits_no_separate_tenant_slug_key() -> None:
    """Canonical shape carries tenant_id only -- the consumer forbids extras."""
    result = stamp_verified_tenant_slug({"prompt": "hi"}, "acme")
    assert "tenant_slug" not in result


def test_preserves_other_payload_keys() -> None:
    result = stamp_verified_tenant_slug({"prompt": "hi", "task_type": "test"}, "beta")
    assert result["prompt"] == "hi"
    assert result["task_type"] == "test"
    assert result["tenant_id"] == "beta"


def test_returns_new_dict_without_mutating_input() -> None:
    original = {"prompt": "hi"}
    result = stamp_verified_tenant_slug(original, "acme")
    assert result is not original
    assert "tenant_id" not in original


def test_accepts_any_mapping_not_only_dict() -> None:
    """Input is a Mapping; a read-only mapping must still yield a plain dict."""
    result = stamp_verified_tenant_slug(MappingProxyType({"prompt": "hi"}), "acme")
    assert isinstance(result, dict)
    assert result == {"prompt": "hi", "tenant_id": "acme"}
