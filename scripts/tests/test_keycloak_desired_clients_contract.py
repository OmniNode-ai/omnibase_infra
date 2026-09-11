#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Contract tests for the canonical Keycloak client configuration."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, cast

import pytest

_CONFIG_PATH = Path(__file__).parents[2] / "docker/keycloak/desired-clients.json"
_CUSTOMER_HOSTS = frozenset(
    {
        "https://omninode.ai",
        "https://app.omninode.ai",
        "https://dev.omninode.ai",
        "https://dev.app.omninode.ai",
    }
)
_CUSTOMER_DEFAULT_SCOPES = ["basic", "web-origins", "roles", "profile", "email"]


def _client(config: dict[str, Any], client_id: str) -> dict[str, Any]:
    matches = [
        client for client in config["clients"] if client["clientId"] == client_id
    ]
    assert len(matches) == 1
    return cast("dict[str, Any]", matches[0])


def test_omniweb_web_origins_exactly_cover_managed_ingress_hosts() -> None:
    config = json.loads(_CONFIG_PATH.read_text())
    omniweb = _client(config, "omniweb")

    assert "https://dev.app.omninode.ai/*" in omniweb["redirectUris"]
    assert omniweb["webOrigins"] == [
        "https://app.omninode.ai",
        "https://omninode.ai",
        "https://www.omninode.ai",
        "https://dev.app.omninode.ai",
        "https://dev.omninode.ai",
        "http://localhost:3000",
        "http://localhost:8080",
    ]


def test_omniweb_user_identity_claim_contract() -> None:
    config = json.loads(_CONFIG_PATH.read_text())
    omniweb = _client(config, "omniweb")
    mappers = {mapper["name"]: mapper for mapper in omniweb["protocolMappers"]}

    principal = mappers["principal_id"]
    assert principal["protocolMapper"] == "oidc-usermodel-attribute-mapper"
    assert principal["config"]["user.attribute"] == "principal_id"
    assert principal["config"]["claim.name"] == "principal_id"
    assert principal["config"]["id.token.claim"] == "true"
    assert principal["config"]["access.token.claim"] == "true"
    assert principal["config"]["userinfo.token.claim"] == "true"

    assert "gateway-attach-audience" not in mappers
    assert all(
        mapper.get("config", {}).get("included.custom.audience") != "gateway-attach"
        for mapper in mappers.values()
    )


def test_omnidash_spa_mints_onex_api_audience() -> None:
    """OMN-17512 (finding F10 on OMN-17281): omnidash-spa minted `aud: account`
    (Keycloak's built-in default) with no onex-api audience at all, so
    POST /v1/auth/provision 401'd on a valid, freshly minted token.

    Ruling (OMN-17281 comment 10db4a5a): the fix is a narrow audience mapper
    on the SPA client, not widening the accepted audience in auth_oidc.py.
    This asserts the mapper exists and mints exactly `onex-api` as an access
    token audience.
    """
    config = json.loads(_CONFIG_PATH.read_text())
    omnidash_spa = _client(config, "omnidash-spa")

    assert omnidash_spa["publicClient"] is True
    assert omnidash_spa["directAccessGrantsEnabled"] is True

    mappers = {mapper["name"]: mapper for mapper in omnidash_spa["protocolMappers"]}
    audience_mapper = mappers["onex-api-audience"]
    assert audience_mapper["protocolMapper"] == "oidc-audience-mapper"
    assert audience_mapper["config"]["included.client.audience"] == "onex-api"
    assert audience_mapper["config"]["access.token.claim"] == "true"


def test_onex_customer_is_a_dedicated_public_customer_client() -> None:
    """OMN-17527: the customer sign-on path gets its own client.

    Operator ruling (OMN-17527, 2026-09-02): the documented customer
    registration path moves to a dedicated customer client and stops
    borrowing `omnidash-spa`, which is an internal dashboard SPA client.

    The replacement must be public and PKCE-only. Resource-owner password
    credentials are not part of the browser sign-on path.
    """
    config = json.loads(_CONFIG_PATH.read_text())
    customer = _client(config, "onex-customer")

    assert customer["publicClient"] is True
    assert customer["standardFlowEnabled"] is True
    assert customer["directAccessGrantsEnabled"] is False
    assert customer["serviceAccountsEnabled"] is False
    assert customer["attributes"]["pkce.code.challenge.method"] == "S256"


def test_onex_customer_admits_the_customer_app_host_and_not_the_dashboard() -> None:
    """OMN-17527 acceptance 3 (source finding S1 on OMN-17275).

    The documented registration URL returned 400 `Invalid parameter:
    redirect_uri` because `omnidash-spa` admits only the internal dashboard
    host. The customer client must admit the host a customer actually lands
    on -- and must not acquire the dashboard host, which would recreate the
    coupling this ticket exists to remove.
    """
    config = json.loads(_CONFIG_PATH.read_text())
    customer = _client(config, "onex-customer")

    _assert_onex_customer_exact_redirect_scope_contract(customer)

    for host in (
        "https://app.omninode.ai",
        "https://dev.app.omninode.ai",
        "https://omninode.ai",
        "https://dev.omninode.ai",
    ):
        assert host in customer["redirectUris"]
        assert host in customer["webOrigins"]

    dashboard_hosts = ("dash.omninode.ai", "dev.dash.omninode.ai")
    assert all(
        host not in uri for uri in customer["redirectUris"] for host in dashboard_hosts
    )
    assert all(
        host not in origin
        for origin in customer["webOrigins"]
        for host in dashboard_hosts
    )


def _assert_onex_customer_exact_redirect_scope_contract(
    customer: dict[str, Any],
) -> None:
    """Pin the public client's complete browser and default-scope contract."""
    assert set(customer["redirectUris"]) == _CUSTOMER_HOSTS
    assert set(customer["webOrigins"]) == _CUSTOMER_HOSTS
    # These are Keycloak's standard browser scopes only: basic identity,
    # origin handling, roles, profile, and email. Group/composite scopes would
    # widen the claims a public customer client receives.
    assert customer["defaultClientScopes"] == _CUSTOMER_DEFAULT_SCOPES


@pytest.mark.parametrize("field", ["redirectUris", "webOrigins"])
def test_onex_customer_exact_hosts_reject_extra_uri(field: str) -> None:
    """OMN-17849: an extra path/origin must fail the exact-set assertion."""
    config = json.loads(_CONFIG_PATH.read_text())
    customer = copy.deepcopy(_client(config, "onex-customer"))
    customer[field].append("https://omninode.ai/anything")

    with pytest.raises(AssertionError):
        _assert_onex_customer_exact_redirect_scope_contract(customer)


def test_onex_customer_exact_scopes_reject_extra_scope() -> None:
    """OMN-17849: an unapproved composite scope must fail the exact assertion."""
    config = json.loads(_CONFIG_PATH.read_text())
    customer = copy.deepcopy(_client(config, "onex-customer"))
    customer["defaultClientScopes"].append("groups")

    with pytest.raises(AssertionError):
        _assert_onex_customer_exact_redirect_scope_contract(customer)


def test_onex_customer_redirects_do_not_use_wildcard_paths() -> None:
    """OMN-17527: public customer redirects must stay exact-host scoped."""
    config = json.loads(_CONFIG_PATH.read_text())
    customer = _client(config, "onex-customer")

    assert all("*" not in uri for uri in customer["redirectUris"])
    assert all(uri.startswith("https://") for uri in customer["redirectUris"])


def test_onex_customer_mints_onex_api_audience_and_tenant_claims() -> None:
    """OMN-17527 acceptance 4 (source finding F10 on OMN-17281).

    `aud` is minted by client configuration, not requested by the caller, so
    the customer client needs its own `oidc-audience-mapper` for `onex-api`
    or `POST /v1/auth/provision` 401s on a valid token. The three tenant
    identity mappers match the `omniweb` client -- once a customer is
    provisioned, their token has to carry the tenant the API resolves them
    against.
    """
    config = json.loads(_CONFIG_PATH.read_text())
    customer = _client(config, "onex-customer")
    mappers = {mapper["name"]: mapper for mapper in customer["protocolMappers"]}

    audience = mappers["onex-api-audience"]
    assert audience["protocolMapper"] == "oidc-audience-mapper"
    assert audience["config"]["included.client.audience"] == "onex-api"
    assert audience["config"]["access.token.claim"] == "true"
    assert audience["config"]["id.token.claim"] == "false"

    for claim in ("tenant_id", "tenant_slug", "principal_id"):
        mapper = mappers[claim]
        assert mapper["protocolMapper"] == "oidc-usermodel-attribute-mapper"
        assert mapper["config"]["user.attribute"] == claim
        assert mapper["config"]["claim.name"] == claim
        assert mapper["config"]["access.token.claim"] == "true"
        assert mapper["config"]["id.token.claim"] == "false"
        assert mapper["config"]["userinfo.token.claim"] == "true"


def test_omnidash_spa_is_untouched_by_the_customer_client_work() -> None:
    """OMN-17527 acceptance 6, asserted rather than assumed.

    A change that "fixes" the customer path by further mutating the internal
    dashboard client has not done what the ticket asks. This pins the SPA
    client's own host set and its OMN-17512 audience mapper against exactly
    that drift.
    """
    config = json.loads(_CONFIG_PATH.read_text())
    omnidash_spa = _client(config, "omnidash-spa")

    assert omnidash_spa["redirectUris"] == [
        "https://dash.omninode.ai/*",
        "https://dev.dash.omninode.ai/*",
        "http://localhost:3000/*",
        "http://localhost:8080/*",
    ]
    assert omnidash_spa["webOrigins"] == [
        "https://dash.omninode.ai",
        "https://dev.dash.omninode.ai",
        "http://localhost:3000",
        "http://localhost:8080",
    ]
    assert [mapper["name"] for mapper in omnidash_spa["protocolMappers"]] == [
        "onex-api-audience"
    ]


def test_customer_client_carries_no_client_secret_indirection() -> None:
    """A public client has no secret, so it must not declare `secretEnv`.

    `seed-keycloak-clients.py`'s `_resolve_client_secret()` reads every
    declared `secretEnv` unconditionally and `_die()`s when the named
    variable is unset. Declaring one here would add a 15th required key to
    the `keycloak-smtp-credentials` Secret and hard-fail the reconcile for
    every client after this one.
    """
    config = json.loads(_CONFIG_PATH.read_text())
    customer = _client(config, "onex-customer")

    assert "secretEnv" not in customer


def test_onex_customer_admits_no_loopback_redirect_or_origin() -> None:
    """OMN-17527: the customer client carries no localhost surface.

    Raised as a MAJOR hostile-reviewer finding on omnibase_infra#3159. Every
    other browser client in this file (`omniweb`, `omnidash-spa`) admits
    `localhost:3000`/`localhost:8080` because a developer runs those SPAs
    locally. No customer ever does -- there is no documented customer flow
    that lands on a loopback host -- so a loopback entry here is an
    unjustified widening, and one that matters even now that
    `directAccessGrantsEnabled` is off: `webOrigins` grants CORS to the
    realm's token endpoint, so every listed origin is a browser context
    allowed to run the code exchange against it.

    Asserted in the negative on purpose. The natural next edit is someone
    copying the host list off `omniweb`, which would silently reintroduce it.
    """
    config = json.loads(_CONFIG_PATH.read_text())
    customer = _client(config, "onex-customer")

    for value in (*customer["redirectUris"], *customer["webOrigins"]):
        assert "localhost" not in value
        assert "127.0.0.1" not in value
        assert "[::1]" not in value


def test_all_clients_have_full_scope_allowed_explicitly_false() -> None:
    """Every client in desired-clients.json must explicitly set fullScopeAllowed to false.

    Keycloak defaults fullScopeAllowed to true when the field is absent, which puts every
    realm role into the token regardless of defaultClientScopes. Setting it explicitly to
    false makes the `roles` client scope meaningful: only roles reachable via configured
    scope mappings appear in tokens, preventing privilege over-issuance.

    Asserted in the positive -- the value must be the boolean False, not merely absent.
    """
    config = json.loads(_CONFIG_PATH.read_text())
    violations = [
        c["clientId"]
        for c in config["clients"]
        if c.get("fullScopeAllowed") is not False
    ]
    assert not violations, (
        f"These clients are missing fullScopeAllowed=false: {violations}"
    )


def test_role_bearing_service_clients_have_matching_scope_mappings() -> None:
    """Service clients with realmRoles must also declare clientScopeMappings.

    realmRoles assigns roles to the service account user (user-level). With
    fullScopeAllowed=false, Keycloak's scope filter drops those roles from
    client_credentials tokens unless the same roles are also in the client's
    scope (clientScopeMappings). Both lists must cover the same roles.
    """
    config = json.loads(_CONFIG_PATH.read_text())
    for client in config["clients"]:
        realm_roles = client.get("realmRoles", [])
        if not realm_roles:
            continue
        scope_mappings = client.get("clientScopeMappings", [])
        missing = set(realm_roles) - set(scope_mappings)
        assert not missing, (
            f"Client '{client['clientId']}' has realmRoles {sorted(missing)} not "
            f"covered by clientScopeMappings -- those roles will be dropped from "
            f"client_credentials tokens when fullScopeAllowed=false"
        )


def test_onex_admin_declares_the_client_management_roles_its_code_path_calls() -> None:
    """OMN-18170: ``onex-admin`` must declare manage-clients AND view-clients.

        ``onex-api``'s per-tenant credential chain
        (``_provision_tenant_credentials_p0b`` -> ``ensure_tenant_keycloak_client``)
        does four things against the Keycloak Admin API with this one credential:
        a client LOOKUP (``GET /clients?clientId=``), a client CREATE
        (``POST /clients``), a fail-closed re-READ, and a client-secret read. The
        create needs ``manage-clients``; the reads need ``view-clients``.

    Both are declared rather than relying on one implying the other.
        Observed on the ``omninode`` realm on 2026-09-11, ``manage-clients`` was
        not composite (it expanded to ``[]``) while ``view-clients`` expanded to
        ``['query-clients']`` -- so manage-clients alone did not confer the read,
        and the lookup returned an EMPTY LIST (HTTP 200, not 403), silently
        reporting "no such client" for a client that exists and turning the
        idempotent-reuse branch into a duplicate create.

        That composite structure is realm state an administrator can change, so it
        is recorded here as a dated observation, not as a standing fact. This
        assertion does not depend on it: declaring both roles is correct whether or
        not one is ever made composite of the other, and declaring the narrower one
        alongside the broader one costs nothing.

        Why this test and not only the generic coverage test above: that one
        checks realmRoles and clientScopeMappings agree with EACH OTHER, so a
        client declaring neither half of a required role passes it. That is the
        exact shape of the defect this pins. Between 2026-08-15 and 2026-09-11
        ``manage-clients`` was assigned to the service-account user out of band
        but was in neither list here, so the seeder never added the matching
        scope mapping, Keycloak's scope filter stripped the role from every
        client_credentials token, and ``POST /v1/tenants/bootstrap`` answered
        503 on every call -- zero tenants could be minted on onex-dev.
    """
    config = json.loads(_CONFIG_PATH.read_text())
    onex_admin = _client(config, "onex-admin")

    # The client-management half, needed by ensure_tenant_keycloak_client.
    required = {
        "realm-management:manage-clients",
        "realm-management:view-clients",
        # The user-management half, which predates this and must SURVIVE any
        # future edit to the lists above. Asserted rather than assumed: a
        # subset check on the new roles alone would pass an edit that REPLACED
        # these two, and losing manage-users breaks /v1/auth/provision -- the
        # one tenant path that kept working throughout the OMN-18170 outage.
        "realm-management:manage-users",
        "realm-management:view-users",
    }
    for field in ("realmRoles", "clientScopeMappings"):
        missing = required - set(onex_admin.get(field, []))
        assert not missing, (
            f"onex-admin is missing {sorted(missing)} from '{field}'. Without "
            f"the client-management pair in BOTH lists, POST "
            f"/v1/tenants/bootstrap 503s on every call; without the "
            f"user-management pair, /v1/auth/provision breaks too (OMN-18170)."
        )
