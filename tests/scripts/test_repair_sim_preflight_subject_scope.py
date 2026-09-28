# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Unit tests for the narrowly scoped disposable Keycloak repair."""

from __future__ import annotations

from typing import Any

import pytest

from scripts.runtime_build.bootstrap_sim_preflight_auth_omn19726 import BASIC_SCOPE
from scripts.runtime_build.repair_sim_preflight_subject_scope import ADMIN_BASE, repair


class FakeResponse:
    def __init__(self, body: Any) -> None:
        self.body = body

    def raise_for_status(self) -> None:
        return None

    def json(self) -> Any:
        return self.body


class FakeSession:
    def __init__(self) -> None:
        self.client = {
            "id": "client-id",
            "clientId": "omnidash",
            "description": "Disposable OmniDash replay validation client (OMN-19726).",
            "directAccessGrantsEnabled": False,
            "serviceAccountsEnabled": False,
            "standardFlowEnabled": True,
        }
        self.scopes: list[dict[str, Any]] = []
        self.defaults: list[dict[str, Any]] = []
        self.created = 0
        self.attached = 0

    def get(self, url: str, **kwargs: Any) -> FakeResponse:
        if url.endswith("/clients"):
            assert kwargs["params"] == {"clientId": "omnidash"}
            return FakeResponse([self.client])
        if url.endswith("/client-scopes"):
            assert kwargs["params"] == {"search": "basic"}
            return FakeResponse(self.scopes)
        if url.endswith("/client-scopes/basic-id"):
            return FakeResponse(self.scopes[0])
        if url.endswith("/default-client-scopes"):
            return FakeResponse(self.defaults)
        raise AssertionError(f"unexpected GET path: {url}")

    def post(self, url: str, **kwargs: Any) -> FakeResponse:
        assert url.endswith("/client-scopes")
        assert kwargs["json"] == BASIC_SCOPE
        self.created += 1
        self.scopes.append({**BASIC_SCOPE, "id": "basic-id"})
        return FakeResponse({})

    def put(self, url: str, **kwargs: Any) -> FakeResponse:
        assert url.endswith("/default-client-scopes/basic-id")
        self.attached += 1
        self.defaults.append({"id": "basic-id", "name": "basic"})
        return FakeResponse({})


def test_repair_creates_and_attaches_native_subject_scope_idempotently() -> None:
    session = FakeSession()
    base = ADMIN_BASE

    first = repair(session, base)
    second = repair(session, base)

    assert first == {"subject_scope_verified": True, "subject_scope_created": True}
    assert second == {"subject_scope_verified": True, "subject_scope_created": False}
    assert session.created == 1
    assert session.attached == 1


def test_repair_refuses_non_disposable_client_before_mutation() -> None:
    session = FakeSession()
    session.client["description"] = "ordinary client"

    with pytest.raises(ValueError, match="approved disposable"):
        repair(session, ADMIN_BASE)

    assert session.created == 0
    assert session.attached == 0


def test_repair_refuses_conflicting_basic_scope_before_attachment() -> None:
    session = FakeSession()
    session.scopes = [
        {
            **BASIC_SCOPE,
            "id": "basic-id",
            "protocolMappers": [{"name": "fake", "protocolMapper": "fake"}],
        }
    ]

    with pytest.raises(ValueError, match="conflicts"):
        repair(session, ADMIN_BASE)

    assert session.created == 0
    assert session.attached == 0
