# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Desired-state invariants for the ``onex-verify`` verification client (OMN-16421).

Every live probe of the authenticated ``api.omninode.ai`` round trip stalled the
same way: the only clients that carry ``tenant_id``/``tenant_slug`` claims are
browser-login clients (``omniweb``), and every machine (``client_credentials``)
client is deliberately un-tenanted -- so an agent session could prove the auth
gate *rejects* (401) but never that it *admits* (200). ``onex-verify`` is the
dedicated, least-privilege, non-prod verification client that closes that gap:
a service-account client whose token carries tenant claims mapped from
attributes on its own service-account user, bound to a designated verification
tenant.

Invariants pinned here, because each one has a tempting wrong "fix":

1. The client exists, is confidential, and has a service account -- otherwise
   Keycloak refuses the ``client_credentials`` grant (``unauthorized_client``).
2. It has **zero browser surface** (no standard flow, no direct access grants,
   no redirect URIs) -- it must never become a login client.
3. It holds **zero realm roles** -- unlike ``onex-admin`` / ``omniweb-signup-admin``
   it has no Keycloak admin-API capability at all. Verification reads its own
   identity and its own tenant through the gateway; it administers nothing.
   Granting it ``realm-management:*`` "to make a probe easier" is the
   OMN-14088 outage class.
4. It declares **no** ``secretEnv`` -- deliberately. ``_resolve_client_secret``
   in ``scripts/seed-keycloak-clients.py`` returns ``None`` for a spec without
   ``secretEnv``, so the reconciler creates the client, lets Keycloak generate
   its secret, and never manages or overwrites it afterwards. The secret's
   reference home is the ``onex-dev`` k8s Secret named in the runbook
   (``onex-verify-client-credentials``), populated read-use-discard from
   Keycloak at rotation time. Declaring a ``secretEnv`` here would instead make
   the seed Job ``_die()`` until a new env indirection is wired and seeded --
   an out-of-band dependency a verification-only credential must not have.
5. Its token carries ``tenant_id`` and ``tenant_slug`` **access-token** claims
   via user-attribute mappers, plus the ``onex-api`` audience -- exactly what
   ``auth_oidc.py`` (claim extraction) and the gateway's audience validation
   require for ``GET /v1/whoami`` / ``GET /v1/tenants`` to return 200.

Related Tickets:
    - OMN-16421: authenticated api.omninode.ai round trip unprovable from an
      agent session (this client is the fix)
    - OMN-16456: the sibling least-privilege-client precedent (signup-admin)
    - OMN-14088: outage class -- never over-privilege a convenience client
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
DESIRED_CLIENTS = REPO_ROOT / "docker" / "keycloak" / "desired-clients.json"

#: The dedicated verification-only client an agent session authenticates as.
VERIFY_CLIENT_ID = "onex-verify"
#: Claims the gateway's OIDC layer resolves tenant identity from
#: (``auth_oidc.py`` defaults: KEYCLOAK_TENANT_ID_CLAIM / KEYCLOAK_TENANT_SLUG_CLAIM).
EXPECTED_TENANT_CLAIMS = {"tenant_id", "tenant_slug"}
#: The audience the onex-api gateway validates access tokens against.
EXPECTED_AUDIENCE = "onex-api"


def _clients() -> dict[str, dict[str, Any]]:
    payload = json.loads(DESIRED_CLIENTS.read_text(encoding="utf-8"))
    return {c["clientId"]: c for c in payload["clients"]}


def _verify_spec() -> dict[str, Any]:
    clients = _clients()
    assert VERIFY_CLIENT_ID in clients, (
        f"{VERIFY_CLIENT_ID!r} missing from {DESIRED_CLIENTS}. Without it there "
        "is no non-interactive, least-privilege path to prove the "
        "authenticated api.omninode.ai round trip (OMN-16421)."
    )
    return clients[VERIFY_CLIENT_ID]


@pytest.mark.unit
class TestOnexVerifyClientDesiredState:
    """The verification client must be able to grant, and nothing more."""

    def test_verify_client_is_declared(self) -> None:
        assert VERIFY_CLIENT_ID in _clients()

    def test_verify_client_has_a_service_account(self) -> None:
        spec = _verify_spec()
        assert spec.get("serviceAccountsEnabled") is True, (
            "onex-verify needs serviceAccountsEnabled:true or Keycloak refuses "
            "its client_credentials grant (unauthorized_client)"
        )
        assert spec.get("publicClient") is False, (
            "onex-verify must stay confidential -- its grant authenticates "
            "with a client_secret"
        )

    def test_verify_client_has_no_browser_login_surface(self) -> None:
        spec = _verify_spec()
        assert spec.get("standardFlowEnabled") is False
        assert spec.get("directAccessGrantsEnabled") is False
        assert "redirectUris" not in spec, (
            "a verification-only machine client has no browser redirect "
            "surface; adding one would turn it into a login client"
        )

    def test_verify_client_holds_zero_realm_roles(self) -> None:
        spec = _verify_spec()
        assert "realmRoles" not in spec, (
            "onex-verify must hold NO realm-management roles -- it reads its "
            "own identity and tenant through the gateway and administers "
            "nothing. Granting roles here is the OMN-14088 outage class."
        )

    def test_verify_client_declares_no_secret_env(self) -> None:
        spec = _verify_spec()
        assert "secretEnv" not in spec, (
            "onex-verify deliberately declares no secretEnv: the reconciler "
            "must leave the Keycloak-generated secret unmanaged (see module "
            "docstring, invariant 4). Wiring a secretEnv makes the seed Job "
            "fail-closed on an env var nothing needs to inject."
        )

    def test_verify_client_maps_tenant_claims_into_the_access_token(self) -> None:
        spec = _verify_spec()
        mappers = {m["name"]: m for m in spec.get("protocolMappers", [])}
        for claim in sorted(EXPECTED_TENANT_CLAIMS):
            assert claim in mappers, (
                f"onex-verify must map the {claim!r} user attribute into its "
                "tokens -- without it GET /v1/whoami 401s on the tenant check "
                "even for a validly signed token (the exact OMN-16421 symptom)"
            )
            mapper = mappers[claim]
            assert mapper["protocolMapper"] == "oidc-usermodel-attribute-mapper"
            config = mapper["config"]
            assert config["user.attribute"] == claim
            assert config["claim.name"] == claim
            assert config["access.token.claim"] == "true", (
                f"{claim!r} must land in the ACCESS token -- the gateway "
                "resolves tenant identity from the access token, not the ID "
                "token"
            )

    def test_verify_client_carries_the_onex_api_audience(self) -> None:
        spec = _verify_spec()
        audience_mappers = [
            m
            for m in spec.get("protocolMappers", [])
            if m["protocolMapper"] == "oidc-audience-mapper"
            and m["config"].get("included.client.audience") == EXPECTED_AUDIENCE
            and m["config"].get("access.token.claim") == "true"
        ]
        assert audience_mappers, (
            f"onex-verify's access token must carry aud={EXPECTED_AUDIENCE!r} "
            "or the gateway's JWT validation rejects it before any route runs"
        )
