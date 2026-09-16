# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""A client the runtime introspects with must never be declared bearer-only (OMN-16504).

``onex-api`` ended a reconcile with an empty Keycloak client secret and every
gateway-attach introspection call started failing 401. Root cause, diagnosed
live and recorded on OMN-16504 (comment ``2ac3deca``): ``onex-api`` was the
realm's only ``bearerOnly: true`` confidential client, and Keycloak clears a
bearer-only client's secret on every admin-API update regardless of which
fields the request actually changes (comment ``1b5041f9`` reproduced this
behaviourally -- a representation PUT that omitted ``secret`` entirely still
left it empty). A client that must authenticate to the introspection endpoint
with a client secret is a *confidential* client by definition; declaring it
``bearerOnly`` is the defect, not a Keycloak quirk to route around.

This module treats the invariant as roster-wide rather than pinning it to one
clientId, because the failure mode is "any client this repo later wires as an
introspection caller / secret holder must not be bearer-only" -- not
"onex-api specifically." A client is classified as an introspection caller /
secret holder here if it is confidential (``publicClient`` absent or
``false``) and is not the two legitimately-public browser clients
(``onex-customer``, ``omnidash-spa``). ``bearerOnly: true`` on such a client
is what this test forbids.

Related:
    - OMN-16504: onex-api ends a reconcile with no client secret; introspection 401
    - omnibase_infra scripts/seed-keycloak-clients.py: the reconciler this
      roster feeds, including OMN-16504's confidential-secret-empty guard and
      its ``_SECRET_CLEARING_FLAGS`` handling of ``bearerOnly``/``publicClient``
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
DESIRED_CLIENTS = REPO_ROOT / "docker" / "keycloak" / "desired-clients.json"

#: The client this ticket's incident was about.
INTROSPECTION_CLIENT_ID = "onex-api"

#: Public (browser, PKCE) clients are exempt: a public client cannot hold a
#: secret at all, so "confidential and not bearer-only" does not apply to it.
#: This is an exact allowlist, not a pattern -- a new public client must be
#: added here deliberately, the same way test_keycloak_realm_single_source.py
#: (omninode_infra) pins an exact ``files`` list rather than a membership check.
KNOWN_PUBLIC_CLIENT_IDS = frozenset({"onex-customer", "omnidash-spa"})


def _clients() -> list[dict[str, Any]]:
    payload = json.loads(DESIRED_CLIENTS.read_text(encoding="utf-8"))
    clients = payload["clients"]
    assert isinstance(clients, list)
    return clients


def _by_id(client_id: str) -> dict[str, Any]:
    clients = {c["clientId"]: c for c in _clients()}
    assert client_id in clients, f"{client_id!r} missing from {DESIRED_CLIENTS}"
    return clients[client_id]


@pytest.mark.unit
class TestIntrospectionCallersAreNotBearerOnly:
    """A confidential client cannot authenticate if it is also bearer-only."""

    def test_onex_api_is_not_bearer_only(self) -> None:
        spec = _by_id(INTROSPECTION_CLIENT_ID)
        assert spec.get("bearerOnly") is not True, (
            "onex-api authenticates to the introspection endpoint with a "
            "client secret -- bearerOnly:true makes Keycloak clear that "
            "secret on every reconcile update (OMN-16504) and a bearer-only "
            "client can never hold one going forward"
        )

    def test_onex_api_stays_confidential_with_a_service_account(self) -> None:
        spec = _by_id(INTROSPECTION_CLIENT_ID)
        assert spec.get("publicClient") is False, (
            "onex-api must stay confidential; a public client cannot hold "
            "the secret introspection needs"
        )
        assert spec.get("serviceAccountsEnabled") is True, (
            "the live onex-api client already has serviceAccountsEnabled:true "
            "(OMN-16504 comment 1b5041f9); the roster must declare what is "
            "actually live rather than silently diverging from it"
        )

    def test_no_confidential_client_is_declared_bearer_only(self) -> None:
        """Roster-wide: a future confidential client must not repeat this shape."""
        offenders = [
            client["clientId"]
            for client in _clients()
            if client["clientId"] not in KNOWN_PUBLIC_CLIENT_IDS
            and client.get("publicClient", False) is False
            and client.get("bearerOnly") is True
        ]
        assert offenders == [], (
            f"confidential client(s) declared bearerOnly:true: {offenders}. "
            "A bearer-only client cannot hold a client secret in this "
            "Keycloak version (OMN-16504) -- if a client genuinely never "
            "authenticates with a secret it should be publicClient:true "
            "instead, not confidential-and-bearer-only."
        )
