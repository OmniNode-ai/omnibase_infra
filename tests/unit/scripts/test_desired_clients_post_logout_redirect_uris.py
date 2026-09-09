# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Desired-state invariants for RP-initiated logout on the browser-login client (OMN-18080).

The app's sign-out button ended only next-auth's own session cookie. The
Keycloak SSO session survived, so the next authorization request auto-resolved
to the still-signed-in user and Keycloak answered with its "already signed in"
page -- the operator had no working way to sign out at all, and therefore no way
to sign in as a second tenant.

The app-side half of the fix (omniweb) redirects to the realm's
``end_session_endpoint`` with ``id_token_hint`` and ``post_logout_redirect_uri``.
Keycloak honours that redirect only when the URI is REGISTERED on the client,
and this file is where it is registered: the reconciler
(``scripts/seed-keycloak-clients.py``) is the only sanctioned writer of the live
realm, and its input is ``docker/keycloak/desired-clients.json`` -- mirrored
byte-for-byte into ``omninode_infra`` ``k8s/onex-dev/jobs/desired-clients.json``
by the ``desired-clients.json parity`` gate.

Invariants pinned here, because each has a tempting wrong "fix":

1. ``omniweb`` declares ``post.logout.redirect.uris``. Absent, Keycloak falls
   back to the client's ``redirectUris`` -- which happens to cover the app host
   today, so the logout would work by ACCIDENT and silently break the moment
   ``redirectUris`` is narrowed. An accident is not a registration.

2. The declaration is an explicit list, never ``+`` or a bare ``*``. ``+`` means
   "reuse redirectUris", which is the same accident spelled deliberately; a
   wildcard makes the post-logout redirect an open redirect that Keycloak will
   happily follow after destroying the session.

3. Every declared post-logout URI's ORIGIN also appears among ``redirectUris``.
   This is the check that catches a transposed host -- the exact defect found
   live in the hand-applied ``omnidash`` client declaration, whose
   ``post.logout.redirect.uris`` names a host that does not exist and that its
   own ``redirectUris`` do not contain. A post-logout URI on a host the client
   never logs in from can only ever be dead config.

4. Every declared URI ends with ``/signed-out``: omniweb's ``SIGNED_OUT_PATH``
   (``lib/federated-logout.ts``), the page Keycloak lands the browser on after
   it has destroyed the session. Keycloak matches post-logout URIs exactly, so
   renaming that route on one side alone silently breaks the redirect.

5. ``pkce.code.challenge.method`` survives alongside it. ``_reconcile_client``
   REPLACES the whole ``attributes`` map on drift (``update_payload[field] =
   spec[field]``) rather than merging it, so a declaration that adds the logout
   attribute and drops PKCE would disable PKCE on the live client on the next
   reconcile.

Related Tickets:
    - OMN-18080: sign-out never ended the Keycloak session (this registration)
    - OMN-16209: the sibling failure class -- an unregistered/bind-address URI
      is refused outright by Keycloak, so the public origin is not optional
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
DESIRED_CLIENTS = REPO_ROOT / "docker" / "keycloak" / "desired-clients.json"

#: The browser-login client the app authenticates users with
#: (``KEYCLOAK_CLIENT_ID`` on the omniweb Deployment).
LOGIN_CLIENT_ID = "omniweb"
#: Keycloak stores the registered post-logout URIs as one ``##``-delimited
#: string in this client attribute.
POST_LOGOUT_ATTRIBUTE = "post.logout.redirect.uris"
POST_LOGOUT_SEPARATOR = "##"
#: ``SIGNED_OUT_PATH`` in omniweb ``lib/federated-logout.ts``.
SIGNED_OUT_PATH = "/signed-out"


def _clients() -> dict[str, dict[str, Any]]:
    payload = json.loads(DESIRED_CLIENTS.read_text(encoding="utf-8"))
    return {c["clientId"]: c for c in payload["clients"]}


def _login_client() -> dict[str, Any]:
    clients = _clients()
    assert LOGIN_CLIENT_ID in clients, (
        f"{LOGIN_CLIENT_ID!r} missing from {DESIRED_CLIENTS}; it is the browser "
        "login client the app authenticates users with."
    )
    return clients[LOGIN_CLIENT_ID]


def _post_logout_uris() -> list[str]:
    attributes = _login_client().get("attributes") or {}
    raw = attributes.get(POST_LOGOUT_ATTRIBUTE)
    assert isinstance(raw, str) and raw.strip(), (
        f"{LOGIN_CLIENT_ID!r} declares no {POST_LOGOUT_ATTRIBUTE!r}. Without it "
        "Keycloak's post-logout redirect works only by falling back to "
        "redirectUris -- an accident, not a registration (OMN-18080)."
    )
    return [uri for uri in raw.split(POST_LOGOUT_SEPARATOR) if uri]


@pytest.mark.unit
class TestLoginClientPostLogoutRedirectUris:
    """RP-initiated logout must be registered, explicit, and reachable."""

    def test_post_logout_redirect_uris_are_declared(self) -> None:
        assert _post_logout_uris()

    def test_declaration_is_explicit_never_a_wildcard_or_plus(self) -> None:
        for uri in _post_logout_uris():
            assert uri not in {"+", "*"}, (
                f"{uri!r} is not a registration: '+' reuses redirectUris and '*' "
                "makes the post-logout redirect an open redirect that Keycloak "
                "follows after destroying the session."
            )
            assert "*" not in uri, f"wildcard in post-logout URI {uri!r}"

    def test_every_post_logout_origin_is_a_redirect_uri_origin(self) -> None:
        """A post-logout host the client never logs in from is dead config."""
        redirect_origins = {
            urlparse(uri).scheme + "://" + urlparse(uri).netloc
            for uri in _login_client().get("redirectUris", [])
        }
        for uri in _post_logout_uris():
            parsed = urlparse(uri)
            origin = f"{parsed.scheme}://{parsed.netloc}"
            assert origin in redirect_origins, (
                f"post-logout URI {uri!r} is on origin {origin!r}, which is not "
                f"among the client's redirectUris origins {sorted(redirect_origins)}. "
                "A transposed host here is invisible until someone signs out."
            )

    def test_every_post_logout_uri_targets_the_signed_out_route(self) -> None:
        for uri in _post_logout_uris():
            assert urlparse(uri).path == SIGNED_OUT_PATH, (
                f"post-logout URI {uri!r} does not target {SIGNED_OUT_PATH!r}. "
                "Keycloak matches these exactly; renaming the route on one side "
                "alone silently breaks the redirect."
            )

    def test_the_public_app_origins_are_both_registered(self) -> None:
        origins = {
            f"{urlparse(uri).scheme}://{urlparse(uri).netloc}"
            for uri in _post_logout_uris()
        }
        for required in ("https://app.omninode.ai", "https://dev.app.omninode.ai"):
            assert required in origins, (
                f"{required} has no registered post-logout URI; sign-out on that "
                "lane would leave the Keycloak session alive (OMN-18080)."
            )

    def test_pkce_survives_the_attributes_replacement(self) -> None:
        """``_reconcile_client`` replaces the attributes map wholesale, not merges."""
        attributes = _login_client().get("attributes") or {}
        assert attributes.get("pkce.code.challenge.method") == "S256", (
            "the reconciler PUTs the declared attributes map in place of the "
            "live one, so dropping PKCE here disables PKCE on the live client."
        )
