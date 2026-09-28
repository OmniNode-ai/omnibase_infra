# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Idempotently add Keycloak's native subject scope to the disposable client."""

from __future__ import annotations

import argparse
import json
import logging
import os
import stat
from pathlib import Path
from typing import Any

from scripts.runtime_build.bootstrap_sim_preflight_auth_omn19726 import (
    BASIC_SCOPE,
    ISSUER,
    SUBJECT_MAPPER,
)

DISPOSABLE_DESCRIPTION = "Disposable OmniDash replay validation client (OMN-19726)."
ADMIN_BASE = ISSUER.removesuffix("/realms/omninode") + "/admin"


def _scope_matches(scope: dict[str, Any]) -> bool:
    mappers = scope.get("protocolMappers")
    return (
        scope.get("name") == BASIC_SCOPE["name"]
        and scope.get("protocol") == BASIC_SCOPE["protocol"]
        and scope.get("attributes") == BASIC_SCOPE["attributes"]
        and isinstance(mappers, list)
        and len(mappers) == 1
        and {key: value for key, value in mappers[0].items() if key != "id"}
        == SUBJECT_MAPPER
    )


def repair(session: Any, base: str) -> dict[str, bool]:
    """Create and attach only the native subject scope; refuse any ambiguity."""
    if base != ADMIN_BASE:
        raise ValueError("only the disposable OMN-19726 realm is supported")
    realm_url = base + "/realms/omninode"
    response = session.get(
        realm_url + "/clients",
        params={"clientId": "omnidash"},
        timeout=15,
        allow_redirects=False,
    )
    response.raise_for_status()
    clients = response.json()
    if (
        not isinstance(clients, list)
        or len(clients) != 1
        or clients[0].get("clientId") != "omnidash"
        or clients[0].get("description") != DISPOSABLE_DESCRIPTION
        or clients[0].get("directAccessGrantsEnabled") is not False
        or clients[0].get("serviceAccountsEnabled") is not False
        or clients[0].get("standardFlowEnabled") is not True
    ):
        raise ValueError("live client is not the approved disposable browser client")
    client_id = clients[0].get("id")
    if not isinstance(client_id, str) or not client_id:
        raise ValueError("disposable client identifier is missing")

    scopes_url = realm_url + "/client-scopes"
    response = session.get(
        scopes_url,
        params={"search": "basic"},
        timeout=15,
        allow_redirects=False,
    )
    response.raise_for_status()
    scopes = response.json()
    if not isinstance(scopes, list):
        raise ValueError("Keycloak client-scope inventory is malformed")
    basics = [scope for scope in scopes if scope.get("name") == "basic"]
    if len(basics) > 1:
        raise ValueError("multiple basic client scopes exist")
    created = not basics
    if created:
        response = session.post(
            scopes_url,
            json=BASIC_SCOPE,
            timeout=15,
            allow_redirects=False,
        )
        response.raise_for_status()
        response = session.get(
            scopes_url,
            params={"search": "basic"},
            timeout=15,
            allow_redirects=False,
        )
        response.raise_for_status()
        scopes = response.json()
        basics = [scope for scope in scopes if scope.get("name") == "basic"]
    if len(basics) != 1:
        raise ValueError("basic subject client scope was not created uniquely")
    scope_id = basics[0].get("id")
    if not isinstance(scope_id, str) or not scope_id:
        raise ValueError("basic client scope identifier is missing")
    response = session.get(
        scopes_url + "/" + scope_id, timeout=15, allow_redirects=False
    )
    response.raise_for_status()
    if not _scope_matches(response.json()):
        raise ValueError(
            "existing basic client scope conflicts with native subject scope"
        )

    defaults_url = realm_url + "/clients/" + client_id + "/default-client-scopes"
    response = session.get(defaults_url, timeout=15, allow_redirects=False)
    response.raise_for_status()
    defaults = response.json()
    if not isinstance(defaults, list):
        raise ValueError("client default scopes are malformed")
    named = [scope for scope in defaults if scope.get("name") == "basic"]
    if named and (len(named) != 1 or named[0].get("id") != scope_id):
        raise ValueError("client has a conflicting basic default scope")
    attached = bool(named)
    if not attached:
        response = session.put(
            defaults_url + "/" + scope_id, timeout=15, allow_redirects=False
        )
        response.raise_for_status()
    response = session.get(defaults_url, timeout=15, allow_redirects=False)
    response.raise_for_status()
    verified = response.json()
    if not isinstance(verified, list) or not any(
        item.get("id") == scope_id and item.get("name") == "basic" for item in verified
    ):
        raise ValueError("basic subject scope was not attached as a default")
    return {"subject_scope_verified": True, "subject_scope_created": created}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--private-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        private_dir = args.private_dir
        if (
            not private_dir.is_absolute()
            or private_dir.is_symlink()
            or stat.S_IMODE(private_dir.stat().st_mode) != 0o700
        ):
            raise ValueError("private directory required")
        credentials = private_dir / "credentials.env"
        if (
            credentials.is_symlink()
            or not credentials.is_file()
            or stat.S_IMODE(credentials.stat().st_mode) != 0o600
        ):
            raise ValueError("private credentials required")
        env = dict(line.split("=", 1) for line in credentials.read_text().splitlines())
        # Only the approved disposable realm may be repaired.
        issuer = os.environ.get("KEYCLOAK_ISSUER_URL")  # url-authority-ok: sim guard
        if issuer != ISSUER:
            raise ValueError("only the disposable realm is supported")
        import requests

        logging.disable(logging.CRITICAL)
        session = requests.Session()
        session.trust_env = False
        response = session.post(
            "http://auth.localhost:28080/realms/master/protocol/openid-connect/token",
            data={
                "grant_type": "password",
                "client_id": "admin-cli",
                "username": env["SIM_PREFLIGHT_KEYCLOAK_ADMIN_USERNAME"],
                "password": env["SIM_PREFLIGHT_KEYCLOAK_ADMIN_PASSWORD"],
            },
            timeout=15,
            allow_redirects=False,
        )
        response.raise_for_status()
        session.headers["Authorization"] = "Bearer " + response.json()["access_token"]
        result = repair(session, ADMIN_BASE)
    except Exception:  # noqa: BLE001 -- never expose credential-bearing HTTP errors
        print(json.dumps({"subject_scope_repair_failed": True}))
        return 65
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
