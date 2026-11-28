# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""The dashboard server's confidential client is in the roster (OMN-23491).

The dashboard deployment signs in with client id ``omnidash`` and a client
secret, and its post-deploy verify expects that id, but the roster only had
the public ``omnidash-spa``. Staging sign-in failed with "Client not found".

The entry is confidential, standard flow only, and asks for no PKCE: the
dashboard signs in through keycloak-connect, which sends no code_challenge, so
a required S256 would refuse every sign-in. It carries the same tenant claims
as the browser login client, because the dashboard reads ``tenant_id`` from
the token.

On staging and prod the roster pushes the secret from ``OMNIDASH_CLIENT_SECRET``,
which the seed job reads from the same keycloak slot the dashboard reads, so
the realm and the app hold one value and the confidential secret guard
(OMN-16504) never sees an empty secret. Dev pushes it the same way since
OMN-26768: it declared no ``secretEnv`` while the dev seed job had no such
variable, and omninode_infra#2199 added one.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

_SCRIPT_PATH = Path(__file__).parent.parent / "seed-keycloak-clients.py"
_ROSTER_PATH = (
    Path(__file__).resolve().parents[2] / "docker" / "keycloak" / "desired-clients.json"
)

CLIENT_ID = "omnidash"
SECRET_ENV = "OMNIDASH_CLIENT_SECRET"
BASE_HOSTS = (
    "https://dash.omninode.ai",
    "https://dev.dash.omninode.ai",
    "http://localhost:3000",
    "http://localhost:8080",
)
STAGING_HOST = "https://staging.dash.omninode.ai"
TENANT_CLAIMS = {"tenant_id", "tenant_slug", "principal_id"}


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


def _entry() -> dict[str, Any]:
    matches = [c for c in _roster_clients() if c["clientId"] == CLIENT_ID]
    assert len(matches) == 1, f"expected one {CLIENT_ID} entry, found {len(matches)}"
    return matches[0]


def _resolved(environment: str) -> dict[str, Any]:
    resolved = _mod().apply_environment_overrides(_roster_clients(), environment)
    return next(c for c in resolved if c["clientId"] == CLIENT_ID)


def _logout_uris(spec: dict[str, Any]) -> list[str]:
    return spec.get("attributes", {}).get("post.logout.redirect.uris", "").split("##")


@pytest.mark.unit
class TestTheClientShape:
    def test_it_is_a_confidential_standard_flow_client(self) -> None:
        entry = _entry()
        assert entry["publicClient"] is False
        assert entry["standardFlowEnabled"] is True
        assert entry["directAccessGrantsEnabled"] is False
        assert entry["serviceAccountsEnabled"] is False
        assert entry["fullScopeAllowed"] is False
        assert "bearerOnly" not in entry

    def test_no_pkce_is_required_in_any_realm(self) -> None:
        """keycloak-connect sends no code_challenge, so S256 would refuse sign-in."""
        for environment in ("dev", "staging", "prod"):
            attributes = _resolved(environment).get("attributes", {})
            assert "pkce.code.challenge.method" not in attributes, environment

    def test_the_base_hosts_and_sign_out_uris_match(self) -> None:
        entry = _entry()
        assert entry["redirectUris"] == [f"{host}/*" for host in BASE_HOSTS]
        assert entry["webOrigins"] == list(BASE_HOSTS)
        assert _logout_uris(entry) == [f"{host}/*" for host in BASE_HOSTS]

    def test_the_token_carries_the_tenant_claims(self) -> None:
        mappers = {m["name"]: m for m in _entry()["protocolMappers"]}
        assert TENANT_CLAIMS.issubset(mappers)
        for claim in TENANT_CLAIMS:
            config = mappers[claim]["config"]
            assert config["claim.name"] == claim
            assert config["access.token.claim"] == "true"
        audience = mappers["onex-api-audience"]["config"]
        assert audience["included.client.audience"] == "onex-api"


@pytest.mark.unit
class TestEachRealmGetsItsShape:
    def test_staging_pushes_the_secret_and_serves_only_the_staging_host(self) -> None:
        spec = _resolved("staging")
        assert spec["secretEnv"] == SECRET_ENV
        assert spec["redirectUris"] == [f"{STAGING_HOST}/*"]
        assert spec["webOrigins"] == [STAGING_HOST]
        assert _logout_uris(spec) == [f"{STAGING_HOST}/*"]

    def test_prod_pushes_the_secret_on_the_base_hosts(self) -> None:
        spec = _resolved("prod")
        assert spec["secretEnv"] == SECRET_ENV
        assert spec["redirectUris"] == _entry()["redirectUris"]

    def test_dev_pushes_the_secret_like_the_other_realms(self) -> None:
        """OMN-26768. This asserted the opposite until 2026-11-28: dev
        declared no secretEnv because the dev seed job had no such variable.
        omninode_infra#2199 added it to the job and to the dev reconcile
        credentials, so the premise is gone, and leaving dev out is now the
        defect: a confidential client with no declared secret gets one minted
        by Keycloak that is never written back to the omnidash-credentials
        Secret the dashboard reads, so sign-in fails invalid_client while the
        empty-secret guard (OMN-16504) sees a non-empty value."""
        spec = _resolved("dev")
        assert spec["secretEnv"] == SECRET_ENV
        assert "consumerSecretEnv" not in spec


@pytest.mark.unit
class TestTheSeederPushesTheSecret:
    def test_the_secret_is_read_from_the_named_variable(self) -> None:
        spec = _resolved("staging")
        with patch.dict("os.environ", {SECRET_ENV: "value-from-the-slot"}):
            assert _mod()._resolve_client_secret(spec) == "value-from-the-slot"

    def test_an_unset_secret_fails_closed_on_staging_and_prod(self) -> None:
        for environment in ("staging", "prod"):
            spec = _resolved(environment)
            with patch.dict("os.environ", {}, clear=True):
                with pytest.raises(SystemExit):
                    _mod()._resolve_client_secret(spec)

    def test_dev_reads_the_secret_and_fails_closed_without_it(self) -> None:
        spec = _resolved("dev")
        with patch.dict("os.environ", {SECRET_ENV: "value-from-the-dev-slot"}):
            assert _mod()._resolve_client_secret(spec) == "value-from-the-dev-slot"
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(SystemExit):
                _mod()._resolve_client_secret(spec)
