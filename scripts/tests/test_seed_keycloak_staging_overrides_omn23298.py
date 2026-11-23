# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""The staging realm gets the staging hosts from the roster (OMN-23298).

The staging seed Job reconciles this roster on every staging sync. The
browser clients' redirect URIs and web origins are reconciled to exactly the
roster list, so the staging hosts that were added by hand with kcadm were
wiped on the next sync. They are declared here instead, as a ``staging``
override on omniweb, onex-customer and omnidash-spa, and the staging Job
names ``KC_ENVIRONMENT=staging``.

An override replaces a whole field (``apply_environment_overrides``), and the
seeder reconciles ``attributes`` to the roster map, removing undeclared keys
that Keycloak does not manage. So each staging list is the full staging set,
and omniweb's staging ``attributes`` repeats the PKCE method next to the
staging sign-out URI.
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

STAGING_SHAPE = {
    "omniweb": {
        "redirectUris": ["https://staging.app.omninode.ai/*"],
        "webOrigins": [
            "https://staging.app.omninode.ai",
            "https://staging.omninode.ai",
        ],
        "post.logout.redirect.uris": "https://staging.app.omninode.ai/signed-out",
    },
    "onex-customer": {
        "redirectUris": [
            "https://staging.omninode.ai",
            "https://staging.app.omninode.ai",
        ],
        "webOrigins": [
            "https://staging.omninode.ai",
            "https://staging.app.omninode.ai",
        ],
    },
    "omnidash-spa": {
        "redirectUris": ["https://staging.dash.omninode.ai/*"],
        "webOrigins": ["https://staging.dash.omninode.ai"],
    },
}
STAGING_HOSTS = (
    "https://staging.omninode.ai",
    "https://staging.app.omninode.ai",
    "https://staging.dash.omninode.ai",
)


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


def _resolved(environment: str) -> dict[str, dict[str, Any]]:
    return {
        c["clientId"]: c
        for c in _mod().apply_environment_overrides(_roster_clients(), environment)
    }


def _base(client_id: str) -> dict[str, Any]:
    entry = next(c for c in _roster_clients() if c["clientId"] == client_id)
    return {k: v for k, v in entry.items() if k != "environmentOverrides"}


@pytest.mark.unit
class TestStagingGetsTheStagingHosts:
    def test_staging_resolves_each_browser_client_to_the_staging_hosts(self) -> None:
        resolved = _resolved("staging")
        for client_id, shape in STAGING_SHAPE.items():
            spec = resolved[client_id]
            assert spec["redirectUris"] == shape["redirectUris"], client_id
            assert spec["webOrigins"] == shape["webOrigins"], client_id
        logout = resolved["omniweb"]["attributes"]["post.logout.redirect.uris"]
        assert logout == STAGING_SHAPE["omniweb"]["post.logout.redirect.uris"]

    def test_staging_accepts_no_prod_dev_or_localhost_host(self) -> None:
        resolved = _resolved("staging")
        for client_id in STAGING_SHAPE:
            spec = resolved[client_id]
            uris = [*spec["redirectUris"], *spec["webOrigins"]]
            logout = spec.get("attributes", {}).get("post.logout.redirect.uris")
            if logout:
                uris.extend(logout.split("##"))
            for uri in uris:
                assert uri.startswith(STAGING_HOSTS), (client_id, uri)

    def test_an_attributes_override_keeps_every_base_key(self) -> None:
        """The seeder removes an undeclared attribute, so an override that
        dropped a base key would remove it from the realm: here, PKCE."""
        checked = 0
        for client in _roster_clients():
            for environment, block in client.get("environmentOverrides", {}).items():
                if "attributes" not in block:
                    continue
                checked += 1
                base = client.get("attributes", {})
                assert set(base) <= set(block["attributes"]), (
                    client["clientId"],
                    environment,
                )
        assert checked, "no attributes override found"
        omniweb = _resolved("staging")["omniweb"]["attributes"]
        assert omniweb["pkce.code.challenge.method"] == "S256"

    def test_only_the_named_fields_change_on_staging(self) -> None:
        resolved = _resolved("staging")
        for client_id in STAGING_SHAPE:
            base = _base(client_id)
            spec = resolved[client_id]
            changed = {k for k in set(base) | set(spec) if base.get(k) != spec.get(k)}
            allowed = {"redirectUris", "webOrigins", "attributes"}
            assert changed <= allowed, (client_id, changed)

    def test_onex_api_keeps_its_base_shape_on_staging(self) -> None:
        """Its only override is prod, so staging moving from dev to staging
        keeps the confidential shape the staging runtime-effects needs."""
        assert _resolved("staging")["onex-api"] == _resolved("dev")["onex-api"]

    def test_dev_and_prod_are_unchanged_for_the_browser_clients(self) -> None:
        for environment in ("dev", "prod"):
            resolved = _resolved(environment)
            for client_id in STAGING_SHAPE:
                assert resolved[client_id] == _base(client_id), (client_id, environment)


@pytest.mark.unit
class TestTheSeederTakesStaging:
    def test_staging_is_accepted_as_an_environment_name(self) -> None:
        argv = ["seed", "--environment", "staging"]
        with patch.object(sys, "argv", argv):
            args = _mod()._parse_args()
        assert args.environment == "staging"
