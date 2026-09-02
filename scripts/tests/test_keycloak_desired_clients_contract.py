#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Contract tests for the canonical Keycloak client configuration."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

_CONFIG_PATH = Path(__file__).parents[2] / "docker/keycloak/desired-clients.json"


def _client(config: dict[str, Any], client_id: str) -> dict[str, Any]:
    matches = [
        client for client in config["clients"] if client["clientId"] == client_id
    ]
    assert len(matches) == 1
    return matches[0]


def test_omniweb_allows_the_managed_staging_callback() -> None:
    config = json.loads(_CONFIG_PATH.read_text())
    omniweb = _client(config, "omniweb")

    assert "https://dev.app.omninode.ai/*" in omniweb["redirectUris"]
    assert "https://dev.app.omninode.ai" in omniweb["webOrigins"]


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
