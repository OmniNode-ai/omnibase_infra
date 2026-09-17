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
        """The BASE entry, which is the dev-system shape. See the class below
        for the declared production override (OMN-18582)."""
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
        """Roster-wide: a future confidential client must not repeat this shape.

        Reads BASE entries only. A declared per-environment override is a
        different thing and is governed by TestTheDeclaredEnvironmentException
        below, which is stricter about it than this check could be.
        """
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


#: OMN-18582. The clients allowed to declare a per-environment override at all.
#: An exact allowlist, like KNOWN_PUBLIC_CLIENT_IDS above and for the same
#: reason: a second exception must be a deliberate, reviewed edit to THIS list
#: rather than a field someone adds to a roster entry and nobody notices.
CLIENTS_ALLOWED_AN_ENVIRONMENT_OVERRIDE = frozenset({"onex-api"})

#: Every field an override block may carry. Two change the client and two
#: document it. Anything else is refused rather than silently ignored --
#: a typo'd key would otherwise read as "no override" and ship the base shape
#: to production, which is the outcome the override exists to prevent.
_ALLOWED_OVERRIDE_KEYS = frozenset(
    {"bearerOnly", "serviceAccountsEnabled", "consumerSecretEnv", "reason", "ticket"}
)


@pytest.mark.unit
class TestTheDeclaredEnvironmentException:
    """The one exception the operator ruled for, and the shape a second needs.

    Operator ruling 2026-09-17T17:33:52Z, verbatim "yeah we'll take your
    recommendation" (resolve by timestamp; the ledger rolled that day):
    ``onex-api`` is confidential with a consumer on dev-system and bearer-only
    on production, because that is what each realm was MEASURED to be. These
    tests exist so the exception cannot quietly become a pattern.
    """

    def test_only_allowlisted_clients_declare_an_override(self) -> None:
        carriers = {
            client["clientId"]
            for client in _clients()
            if "environmentOverrides" in client
        }
        assert carriers <= CLIENTS_ALLOWED_AN_ENVIRONMENT_OVERRIDE, (
            f"undeclared per-environment exception(s): "
            f"{sorted(carriers - CLIENTS_ALLOWED_AN_ENVIRONMENT_OVERRIDE)}. "
            "One realm shape for one roster is the default; a second exception "
            "is a decision, so add the clientId to "
            "CLIENTS_ALLOWED_AN_ENVIRONMENT_OVERRIDE deliberately rather than "
            "letting a new field into the roster unreviewed."
        )

    def test_every_override_carries_a_reason_and_a_ticket(self) -> None:
        for client in _clients():
            for environment, block in (
                client.get("environmentOverrides") or {}
            ).items():
                where = f"{client['clientId']}.{environment}"
                assert (
                    isinstance(block.get("reason"), str) and block["reason"].strip()
                ), f"{where} deviates from the roster with no stated reason"
                assert isinstance(block.get("ticket"), str) and block["ticket"], (
                    f"{where} deviates from the roster with no owning ticket"
                )

    def test_no_override_carries_an_unrecognised_key(self) -> None:
        for client in _clients():
            for environment, block in (
                client.get("environmentOverrides") or {}
            ).items():
                unknown = sorted(set(block) - _ALLOWED_OVERRIDE_KEYS)
                assert unknown == [], (
                    f"{client['clientId']}.{environment} declares {unknown}, which "
                    "the resolver does not act on; a typo'd key reads as no "
                    "override and ships the base shape to that realm"
                )

    def test_the_onex_api_exception_is_exactly_the_ruled_shape(self) -> None:
        override = _by_id(INTROSPECTION_CLIENT_ID)["environmentOverrides"]["prod"]
        assert override["bearerOnly"] is True
        assert override["serviceAccountsEnabled"] is False
        # Explicit null removes the key. An OMITTED consumerSecretEnv would
        # inherit the base, and the run would then refuse on production with
        # consumer_secret_absent -- which is the exact failure this ruling
        # closes, reintroduced by an omission.
        assert "consumerSecretEnv" in override
        assert override["consumerSecretEnv"] is None
        assert override["ticket"] == "OMN-18582"

    def test_no_override_makes_a_client_confidential_with_a_secret_it_cannot_hold(
        self,
    ) -> None:
        """The OMN-16504 invariant, re-stated for resolved shapes.

        The base check above reads base entries. This one reads what each
        environment actually RESOLVES to, so an override cannot reintroduce the
        combination the base check forbids: bearer-only while still naming a
        secret source is the shape whose secret Keycloak clears on every
        update.
        """
        for client in _clients():
            for environment, block in (
                client.get("environmentOverrides") or {}
            ).items():
                resolved = {
                    **client,
                    **{k: v for k, v in block.items() if k not in {"reason", "ticket"}},
                }
                resolved = {k: v for k, v in resolved.items() if v is not None}
                if resolved.get("bearerOnly") is not True:
                    continue
                named = resolved.get("secretEnv") or resolved.get("consumerSecretEnv")
                assert not named, (
                    f"{client['clientId']} resolves to bearerOnly:true in "
                    f"{environment!r} while still naming {named!r}; Keycloak "
                    "clears a bearer-only client's secret on every admin-API "
                    "update, so that consumer would start getting HTTP 401 "
                    "(OMN-16504)"
                )
