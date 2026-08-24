# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Desired-state invariants for the Keycloak signup-admin client (OMN-16456).

``omniweb``'s ``lib/keycloak-admin.ts`` mints a Keycloak **admin API** token via
``client_credentials`` and then calls ``POST admin/realms/<realm>/users`` +
``PUT .../send-verify-email``. That is the only code path that creates a Keycloak
user for a self-service signup.

Until OMN-16456 it authenticated as ``KEYCLOAK_CLIENT_ID`` -- deployed as the
literal ``"omniweb"``, i.e. the **browser login client**, which this very file
declares ``serviceAccountsEnabled: false`` with no ``realmRoles``. Keycloak
refuses a ``client_credentials`` grant for such a client outright
(``unauthorized_client``), so every waitlist-OFF signup 500s at the token
exchange. Waitlist mode being ON is the only reason that has not surfaced.

Two invariants are pinned here because a comment would not have stopped either
from regressing:

1. A dedicated, least-privilege ``omniweb-signup-admin`` client exists with a
   service account and exactly the one realm-management role it needs.
2. The browser ``omniweb`` client never acquires a service account or any
   ``realmRoles``. Granting realm-management to the browser client is the
   tempting one-line "fix" and is the OMN-14088 outage class -- the client whose
   secret is handed to a browser-facing login flow must not be able to
   administer the realm.

Related Tickets:
    - OMN-16456: waitlist-OFF signup structurally broken (browser client used as
      admin credential)
    - OMN-14088: omniweb browser-client credential handling
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
DESIRED_CLIENTS = REPO_ROOT / "docker" / "keycloak" / "desired-clients.json"

#: The dedicated service client the omniweb signup backend authenticates as.
SIGNUP_ADMIN_CLIENT_ID = "omniweb-signup-admin"
#: The browser login client. Never an admin credential.
BROWSER_CLIENT_ID = "omniweb"
#: Env-var indirection the reconciler resolves the client secret through. The
#: value never lives in git -- see ``_resolve_client_secret`` in
#: ``scripts/seed-keycloak-clients.py``.
SIGNUP_ADMIN_SECRET_ENV = "KEYCLOAK_SIGNUP_CLIENT_SECRET"
#: Least privilege: user create + send-verify-email + the exact-email lookup the
#: resend path uses. Keycloak's ``UserPermissions.canQuery()`` is satisfied by
#: ``canView()``, which ``manage-users`` already grants, so ``view-users`` would
#: be redundant breadth.
EXPECTED_REALM_ROLES = ["realm-management:manage-users"]


def _clients() -> dict[str, dict[str, Any]]:
    payload = json.loads(DESIRED_CLIENTS.read_text(encoding="utf-8"))
    return {c["clientId"]: c for c in payload["clients"]}


@pytest.mark.unit
class TestSignupAdminClientDesiredState:
    """The signup backend must have a credential that can actually grant."""

    def test_signup_admin_client_is_declared(self) -> None:
        assert SIGNUP_ADMIN_CLIENT_ID in _clients(), (
            f"{SIGNUP_ADMIN_CLIENT_ID!r} missing from {DESIRED_CLIENTS}. Without "
            "it the omniweb signup backend has no client that can mint an "
            "admin-API token, and waitlist-OFF signup cannot work."
        )

    def test_signup_admin_client_has_a_service_account(self) -> None:
        spec = _clients()[SIGNUP_ADMIN_CLIENT_ID]
        assert spec.get("serviceAccountsEnabled") is True, (
            "the signup-admin client needs serviceAccountsEnabled:true or "
            "Keycloak refuses its client_credentials grant (unauthorized_client)"
        )
        assert spec.get("publicClient") is False, (
            "the signup-admin client must stay confidential -- its grant "
            "authenticates with a client_secret"
        )

    def test_signup_admin_client_holds_exactly_the_manage_users_role(self) -> None:
        spec = _clients()[SIGNUP_ADMIN_CLIENT_ID]
        assert spec.get("realmRoles") == EXPECTED_REALM_ROLES, (
            "the signup-admin client must hold exactly "
            f"{EXPECTED_REALM_ROLES} -- no more (least privilege), no less "
            "(POST admin/realms/<realm>/users 403s without manage-users)"
        )

    def test_signup_admin_client_resolves_its_secret_by_env_indirection(self) -> None:
        spec = _clients()[SIGNUP_ADMIN_CLIENT_ID]
        assert spec.get("secretEnv") == SIGNUP_ADMIN_SECRET_ENV, (
            "the client secret must be a secretEnv indirection resolved at "
            "reconcile time, never a literal in this file"
        )

    def test_signup_admin_client_has_no_browser_login_surface(self) -> None:
        spec = _clients()[SIGNUP_ADMIN_CLIENT_ID]
        assert spec.get("standardFlowEnabled") is False
        assert spec.get("directAccessGrantsEnabled") is False
        assert "redirectUris" not in spec, (
            "a machine-to-machine client has no browser redirect surface; the "
            "verify-email link deliberately points at the browser client instead"
        )


@pytest.mark.unit
class TestBrowserClientStaysUnprivileged:
    """OMN-14088 outage class: never make the browser client an admin."""

    def test_browser_client_has_no_service_account(self) -> None:
        spec = _clients()[BROWSER_CLIENT_ID]
        assert spec.get("serviceAccountsEnabled") is False, (
            f"{BROWSER_CLIENT_ID!r} is the browser login client. Enabling a "
            "service account on it to make the signup admin call work is the "
            f"wrong fix -- use {SIGNUP_ADMIN_CLIENT_ID!r} instead."
        )

    def test_browser_client_holds_no_realm_management_roles(self) -> None:
        spec = _clients()[BROWSER_CLIENT_ID]
        assert "realmRoles" not in spec, (
            f"{BROWSER_CLIENT_ID!r} must never hold realm-management roles "
            "(OMN-14088 outage class)"
        )

    def test_no_secret_literals_leaked_into_desired_state(self) -> None:
        """Every confidential client indirects through secretEnv, never a value."""
        for client_id, spec in _clients().items():
            assert "secret" not in spec, (
                f"{client_id!r} declares a literal 'secret' key. Client secrets "
                "resolve through secretEnv at reconcile time and must never be "
                "committed."
            )
