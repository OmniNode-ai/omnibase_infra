# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""The dashboard's own Keycloak client exists in the roster (OMN-26768).

WHAT WAS MEASURED. On 2026-11-28 a sign-in on https://dev.dash.omninode.ai
ended on Keycloak's "Client not found" page. The dev realm is seeded from this
roster and carries every client in it: an unauthenticated authorization request
answered 302 for ``omnidash-spa`` and for the other eight, and 400 with "Client
not found" for ``omnidash`` alone. The omnidash Deployment asks for
``omnidash`` -- ``KEYCLOAK_CLIENT_ID`` is that literal in the deployment
repository's omnidash Deployment -- and takes a client secret from
omnidash-credentials, so it is a confidential client and ``omnidash-spa``,
which is ``publicClient: true`` and holds no secret, is not the one it asks
for. The roster was missing the client the application names.

WHY CONFIDENTIAL AND WHY NO OVERRIDE. The Deployment reads
``KEYCLOAK_CLIENT_SECRET`` from omnidash-credentials, which every environment
fills from its own ``omninode/<env>/keycloak`` slot, so the client holds a
secret and ``secretEnv`` is what pushes it. The roster carries every
environment's hosts in one list by design (see the header of
desired-clients.json), so the dev callback sits in the base ``redirectUris``
beside the production one and no per-environment override is needed; the
staging realm narrows its browser clients through the overrides OMN-23298
declared, and this client is not one of them.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import pytest

_SCRIPT_PATH = Path(__file__).parent.parent / "seed-keycloak-clients.py"
_ROSTER_PATH = (
    Path(__file__).resolve().parents[2] / "docker" / "keycloak" / "desired-clients.json"
)

#: The client id the omnidash Deployment sets as KEYCLOAK_CLIENT_ID.
DASHBOARD_CLIENT = "omnidash"

#: The environment variable the reconciler reads the secret from. It is the key
#: the dev slot omninode/dev/keycloak already carries, read 2026-11-28.
SECRET_ENV = "OMNIDASH_CLIENT_SECRET"

#: The callback the dashboard returns to, named on OMN-26768 AC23. It is
#: covered by a wildcard redirect entry rather than listed in full, which is
#: how every other browser client in this roster declares its hosts.
DEV_CALLBACK = "https://dev.dash.omninode.ai/auth/callback"


def _mod() -> types.ModuleType:
    if "seed_keycloak_clients" in sys.modules:
        return sys.modules["seed_keycloak_clients"]
    spec = importlib.util.spec_from_file_location("seed_keycloak_clients", _SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["seed_keycloak_clients"] = module
    spec.loader.exec_module(module)
    return module


def _roster_clients() -> list[dict[str, Any]]:
    clients = json.loads(_ROSTER_PATH.read_text(encoding="utf-8"))["clients"]
    assert isinstance(clients, list)
    return clients


def _entry(client_id: str) -> dict[str, Any]:
    by_id = {c["clientId"]: c for c in _roster_clients()}
    assert client_id in by_id, (
        f"{client_id!r} is absent from the roster. The dev realm is seeded "
        f"from this file and answered Client not found for it on 2026-11-28."
    )
    return by_id[client_id]


def _resolved(environment: str) -> dict[str, dict[str, Any]]:
    return {
        c["clientId"]: c
        for c in _mod().apply_environment_overrides(_roster_clients(), environment)
    }


def _covers(patterns: list[str], url: str) -> bool:
    """Does any wildcard redirect entry admit this exact URL?"""
    for pattern in patterns:
        if pattern.endswith("/*") and url.startswith(pattern[:-1]):
            return True
        if pattern == url:
            return True
    return False


@pytest.mark.unit
class TestTheDashboardClientIsDeclared:
    """AC23: the roster declares the client the dashboard actually asks for."""

    def test_the_roster_declares_the_dashboard_client(self) -> None:
        assert _entry(DASHBOARD_CLIENT)["clientId"] == DASHBOARD_CLIENT

    def test_it_is_confidential_and_the_roster_pushes_its_secret(self) -> None:
        spec = _entry(DASHBOARD_CLIENT)
        assert spec["publicClient"] is False, (
            "the Deployment takes a client secret, so a public client is the "
            "wrong shape; omnidash-spa is the public one and is not what the "
            "Deployment asks for"
        )
        assert spec.get("bearerOnly") is not True, (
            "bearerOnly:true makes Keycloak clear the secret on every "
            "reconcile update (OMN-16504)"
        )
        assert spec["secretEnv"] == SECRET_ENV

    def test_it_is_a_browser_login_client_with_pkce(self) -> None:
        spec = _entry(DASHBOARD_CLIENT)
        assert spec["standardFlowEnabled"] is True
        assert spec["serviceAccountsEnabled"] is False
        assert spec["attributes"]["pkce.code.challenge.method"] == "S256"

    def test_the_dev_callback_is_admitted(self) -> None:
        spec = _entry(DASHBOARD_CLIENT)
        assert _covers(spec["redirectUris"], DEV_CALLBACK), (
            f"no redirect entry admits {DEV_CALLBACK}: {spec['redirectUris']}"
        )
        origin = f"{urlparse(DEV_CALLBACK).scheme}://{urlparse(DEV_CALLBACK).netloc}"
        assert origin in spec["webOrigins"], (
            f"{origin} is not a web origin, so the browser's token request "
            f"is refused by CORS even once the redirect is accepted"
        )

    def test_the_token_carries_the_tenant_claims_and_the_api_audience(self) -> None:
        mappers = {m["name"]: m for m in _entry(DASHBOARD_CLIENT)["protocolMappers"]}
        for claim in ("tenant_id", "tenant_slug", "principal_id"):
            assert claim in mappers, f"{claim} claim mapper missing"
            assert mappers[claim]["config"]["access.token.claim"] == "true"
        audience = mappers["onex-api-audience"]["config"]
        assert audience["included.client.audience"] == "onex-api"


@pytest.mark.unit
class TestTheDashboardClientResolvesTheSameEverywhere:
    """One shape for every realm: this client declares no exception."""

    def test_it_declares_no_per_environment_override(self) -> None:
        assert "environmentOverrides" not in _entry(DASHBOARD_CLIENT), (
            "the roster carries every environment's hosts in one list; an "
            "override here would need adding to "
            "CLIENTS_ALLOWED_AN_ENVIRONMENT_OVERRIDE deliberately"
        )

    @pytest.mark.parametrize("environment", ["dev", "staging", "prod"])
    def test_every_environment_resolves_the_base_shape(self, environment: str) -> None:
        spec = _resolved(environment)[DASHBOARD_CLIENT]
        base = {
            k: v
            for k, v in _entry(DASHBOARD_CLIENT).items()
            if k != "environmentOverrides"
        }
        assert spec == base
        assert spec["secretEnv"] == SECRET_ENV, (
            "the reconciler reads this variable unconditionally and exits 1 "
            "when it is unset, so every realm the Job runs against needs the "
            "value in its own credentials Secret"
        )
