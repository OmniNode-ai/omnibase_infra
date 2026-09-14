# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Overlay-only construction tests for the local claim bootstrap interface."""

from __future__ import annotations

import pytest

from omnibase_infra.runtime.action_authorization_claim import (
    ActionAuthorizationClaimOverlayError,
    ModelActionAuthorizationClaimOverlay,
    build_local_claim_interface,
)


class _Port:
    async def claim(self, request: object) -> object:
        raise AssertionError("not exercised")


def _overlay() -> dict[str, object]:
    return {
        "postgres_connection_ref": "overlay://postgres/rsd-action-claim",
        "local_tool_ref": "overlay://tools/rsd-bootstrap",
        "unix_socket_path": "/private/tmp/rsd-action-claim.sock",
        "restricted_principal": "rsd_action_authorization_claim",
        "socket_owner_uid": 501,
        "authorized_unix_uid": 501,
    }


@pytest.mark.unit
def test_bootstrap_accepts_only_an_explicit_typed_overlay() -> None:
    seen: list[ModelActionAuthorizationClaimOverlay] = []

    def factory(overlay: ModelActionAuthorizationClaimOverlay) -> _Port:
        seen.append(overlay)
        return _Port()

    interface = build_local_claim_interface(_overlay(), adapter_factory=factory)

    assert seen[0].postgres_connection_ref == "overlay://postgres/rsd-action-claim"
    assert interface is not None


@pytest.mark.unit
def test_bootstrap_rejects_missing_or_malformed_overlay_references() -> None:
    malformed = _overlay()
    malformed["postgres_connection_ref"] = "postgres://fallback"

    with pytest.raises(ActionAuthorizationClaimOverlayError):
        build_local_claim_interface(malformed, adapter_factory=lambda _: _Port())

    missing = _overlay()
    del missing["local_tool_ref"]
    with pytest.raises(ActionAuthorizationClaimOverlayError):
        build_local_claim_interface(missing, adapter_factory=lambda _: _Port())
