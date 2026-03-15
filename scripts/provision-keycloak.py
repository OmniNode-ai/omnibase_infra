#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2025 OmniNode Team
"""Keycloak provisioning script for ONEX Infrastructure.

Ensures the keycloak database exists, waits for Keycloak readiness, provisions
the ``onex-admin`` and ``onex-service`` clients, and writes credentials to
``~/.omnibase/.env``.

Security rule: secrets, tokens, and passwords are NEVER logged.  Log messages
for secret fields use ``(SET, N chars)`` or ``(NOT SET)`` format only.

Usage:
    uv run python scripts/provision-keycloak.py \\
      --kc-url http://localhost:28080 \\
      --realm omninode \\
      --admin-username admin \\
      --admin-password "${KEYCLOAK_ADMIN_PASSWORD:-keycloak-dev-password}" \\
      --env-file ~/.omnibase/.env \\
      [--postgres-port 5436] \\
      [--skip-infisical] \\
      [--dry-run]

Note on --kc-url:
    Must be the Keycloak root base URL with NO path prefix.
    ``http://localhost:28080`` is correct.
    ``http://localhost:28080/auth/`` would break all admin API paths.

Note on KEYCLOAK_ADMIN_URL written to env:
    Always ``http://keycloak:8080`` (internal Docker DNS).  The ``--kc-url``
    arg is the external URL used only by this script during bootstrap.

.. versionadded:: 0.12.0
    Added for Keycloak local dev integration (OMN-3362).
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("provision-keycloak")

# Add scripts dir to path so _infisical_util can be imported from any cwd.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _infisical_util import update_env_file

# ---------------------------------------------------------------------------
# Logging helpers — never log secret values
# ---------------------------------------------------------------------------


def _secret_repr(value: str | None) -> str:
    """Return a safe log representation for a secret value."""
    if not value:
        return "(NOT SET)"
    return f"(SET, {len(value)} chars)"


def log(msg: str) -> None:
    logger.info(msg)


# ---------------------------------------------------------------------------
# Step 4.1 — Ensure keycloak database exists
# ---------------------------------------------------------------------------


def ensure_keycloak_db(postgres_password: str, postgres_port: int = 5436) -> None:
    """Create the ``keycloak`` database in postgres if it does not exist.

    Path A (preferred): TCP connection via psycopg2.
    Path B (fallback):  ``docker exec`` + ``psql`` if psycopg2 is unavailable.

    Raises:
        RuntimeError: If postgres is unreachable via both paths.
    """
    log(f"Ensuring keycloak database exists (postgres localhost:{postgres_port})")
    try:
        import psycopg2  # type: ignore[import-untyped]

        conn = psycopg2.connect(
            host="localhost",
            port=postgres_port,
            user="postgres",
            password=postgres_password,
            dbname="postgres",
            connect_timeout=10,
        )
        conn.autocommit = True
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT 1 FROM pg_database WHERE datname = 'keycloak'")
                if cur.fetchone():
                    log("keycloak database already exists")
                else:
                    cur.execute("CREATE DATABASE keycloak")
                    log("Created keycloak database")
        finally:
            conn.close()
        return
    except ImportError:
        log("psycopg2 not available — falling back to docker exec")
    except Exception as exc:
        raise RuntimeError(
            f"Cannot connect to postgres at localhost:{postgres_port}. "
            f"Is omnibase_infra running?  Error: {exc}"
        ) from exc

    # Fallback: docker exec with the postgres container
    try:
        result = subprocess.run(
            [
                "docker",
                "compose",
                "-f",
                str(
                    Path(__file__).resolve().parent.parent
                    / "docker"
                    / "docker-compose.infra.yml"
                ),
                "ps",
                "-q",
                "postgres",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        container_id = result.stdout.strip()
        if not container_id:
            raise RuntimeError(
                "postgres container not found via docker compose ps.  "
                "Start omnibase_infra first."
            )
        check = subprocess.run(
            [
                "docker",
                "exec",
                container_id,
                "psql",
                "-U",
                "postgres",
                "-tAc",
                "SELECT 1 FROM pg_database WHERE datname='keycloak'",
            ],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
        if check.stdout.strip() == "1":
            log("keycloak database already exists (via docker exec)")
        else:
            subprocess.run(
                [
                    "docker",
                    "exec",
                    container_id,
                    "psql",
                    "-U",
                    "postgres",
                    "-c",
                    "CREATE DATABASE keycloak",
                ],
                check=True,
                timeout=15,
            )
            log("Created keycloak database (via docker exec)")
    except RuntimeError:
        raise
    except Exception as exc:
        raise RuntimeError(
            f"Failed to ensure keycloak DB via docker exec: {exc}"
        ) from exc


# ---------------------------------------------------------------------------
# Step 4.2 — Wait for Keycloak readiness
# ---------------------------------------------------------------------------


def wait_for_keycloak(kc_url: str, timeout_s: int = 120) -> None:
    """Poll Keycloak until the master realm endpoint responds or timeout expires.

    Keycloak 26 dev mode (``start-dev``) does not expose ``/health/ready``
    by default — the MicroProfile Health extension is not loaded in that
    configuration.  We fall back to probing ``/realms/master``, which returns
    200 with a JSON realm descriptor as soon as Keycloak is fully initialised.

    Args:
        kc_url: External base URL (e.g. ``http://localhost:28080``).
        timeout_s: Maximum seconds to wait.

    Raises:
        TimeoutError: If Keycloak is not ready within ``timeout_s`` seconds.
    """
    import httpx

    probe_url = f"{kc_url}/realms/master"
    log(f"Waiting for Keycloak readiness at {probe_url} (timeout {timeout_s}s)")
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            r = httpx.get(probe_url, timeout=5)
            if r.status_code == 200 and r.json().get("realm") == "master":
                log("Keycloak is ready")
                return
        except Exception:  # noqa: BLE001 — re-raises as typed error
            pass
        time.sleep(5)
    raise TimeoutError(
        f"Keycloak not ready after {timeout_s}s at {kc_url}.  "
        "Check: docker compose --profile auth logs keycloak"
    )


# ---------------------------------------------------------------------------
# Step 4.2b — Patch master realm SSL to NONE for local dev
# ---------------------------------------------------------------------------


def patch_master_realm_ssl_none(
    postgres_password: str, postgres_port: int = 5436
) -> None:
    """Set ``ssl_required='NONE'`` on the master realm and omninode realm.

    Keycloak 26 initialises the master realm with ``ssl_required='EXTERNAL'``
    by default, which blocks plain-HTTP password-grant token requests even
    when the server is running in ``start-dev`` mode.  We patch both realms
    directly in the keycloak database so that local dev works without TLS.

    This is idempotent — safe to call on every provision run.

    .. warning::
        This is a local-dev-only convenience.  In production environments,
        Keycloak should run behind a TLS-terminating reverse proxy with
        ``ssl_required`` left at its default.

    Args:
        postgres_password: Password for the postgres superuser.
        postgres_port: External postgres port (default 5436).
    """
    log("Patching master and omninode realm ssl_required to NONE (local dev)")
    try:
        import psycopg2  # type: ignore[import-untyped]

        conn = psycopg2.connect(
            host="localhost",
            port=postgres_port,
            user="postgres",
            password=postgres_password,
            dbname="keycloak",
            connect_timeout=10,
        )
        conn.autocommit = True
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE realm SET ssl_required = 'NONE'"
                    " WHERE name IN ('master', 'omninode')"
                    " AND ssl_required != 'NONE'"
                )
                if cur.rowcount:
                    log(f"  ssl_required patched to NONE on {cur.rowcount} realm(s)")
                else:
                    log("  ssl_required already NONE on master + omninode realms")
        finally:
            conn.close()
    except ImportError:
        log(
            "psycopg2 not available — skipping ssl_required patch (may fail on token grant)"
        )
    except Exception as exc:  # noqa: BLE001 — boundary: catch-all for resilience
        log(f"Warning: could not patch ssl_required: {exc} — continuing anyway")


# ---------------------------------------------------------------------------
# Step 4.3 — Acquire bootstrap admin token (password grant, admin-cli)
# ---------------------------------------------------------------------------


def get_admin_token(kc_url: str, username: str, password: str) -> str:
    """Obtain a bootstrap admin token via password grant on the master realm.

    ``admin-cli`` is a public client — no client_secret is required.
    The returned token is short-lived and used only within this script.

    Security: ``password`` is consumed transiently and never logged.

    Returns:
        The access token string.
    """
    import httpx

    log(f"Acquiring admin token for user '{username}' on master realm")
    r = httpx.post(
        f"{kc_url}/realms/master/protocol/openid-connect/token",
        data={
            "client_id": "admin-cli",
            "grant_type": "password",
            "username": username,
            "password": password,
        },
        timeout=10,
    )
    r.raise_for_status()
    token: str = r.json()["access_token"]
    log(f"Admin token acquired {_secret_repr(token)}")
    return token


# ---------------------------------------------------------------------------
# Step 4.4 — Verify realm exists (with retry for import timing)
# ---------------------------------------------------------------------------


def verify_realm(kc_url: str, realm: str, token: str, timeout_s: int = 30) -> None:
    """Verify that the ``realm`` exists in Keycloak.

    File import (``--import-realm``) is the canonical import mechanism.
    This function retries briefly to handle import timing races.

    Raises:
        RuntimeError: If the realm is not found within ``timeout_s`` seconds,
            with an actionable hint about the mount path.
    """
    import httpx

    log(f"Verifying realm '{realm}' exists (timeout {timeout_s}s)")
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        r = httpx.get(
            f"{kc_url}/admin/realms/{realm}",
            headers={"Authorization": f"Bearer {token}"},
            timeout=5,
        )
        if r.status_code == 200:
            log(f"Realm '{realm}' confirmed")
            return
        time.sleep(3)
    raise RuntimeError(
        f"Realm '{realm}' not found after {timeout_s}s.  "
        "Check: ./docker/keycloak/omninode-realm.json is mounted at "
        "/opt/keycloak/data/import/ inside the keycloak container and is valid JSON."
    )


# ---------------------------------------------------------------------------
# Step 4.5 — Provision onex-admin client in master realm
# ---------------------------------------------------------------------------


def provision_onex_admin_client(kc_url: str, token: str) -> str:
    """Create the ``onex-admin`` confidential client in the master realm.

    Assigns ``manage-users`` and ``view-users`` roles from the
    ``omninode-realm`` client (KC 26+) or ``realm-management`` client (KC ≤25).
    Rotates the client secret.

    Used by onex-api's ``provision.py`` to PATCH user attributes
    (tenant_id, tenant_slug) via the KC Admin REST API.

    Returns:
        The rotated client secret (never logged verbatim).

    Raises:
        RuntimeError: If neither ``omninode-realm`` nor ``realm-management`` client is found.
    """
    import httpx

    log("Provisioning onex-admin client in master realm")
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    base = f"{kc_url}/admin/realms/master"

    # Create client if missing
    clients_resp = httpx.get(
        f"{base}/clients?clientId=onex-admin", headers=headers, timeout=10
    )
    clients_resp.raise_for_status()
    if not clients_resp.json():
        log("Creating onex-admin client")
        httpx.post(
            f"{base}/clients",
            headers=headers,
            json={
                "clientId": "onex-admin",
                "enabled": True,
                "publicClient": False,
                "serviceAccountsEnabled": True,
                "standardFlowEnabled": False,
                "directAccessGrantsEnabled": False,
                "protocol": "openid-connect",
            },
            timeout=10,
        ).raise_for_status()
        clients_resp = httpx.get(
            f"{base}/clients?clientId=onex-admin", headers=headers, timeout=10
        )
        clients_resp.raise_for_status()
    else:
        log("onex-admin client already exists — updating roles and rotating secret")

    client_uuid = clients_resp.json()[0]["id"]

    # Get service account user
    sa_resp = httpx.get(
        f"{base}/clients/{client_uuid}/service-account-user",
        headers=headers,
        timeout=10,
    )
    sa_resp.raise_for_status()
    sa_id = sa_resp.json()["id"]

    # Find the cross-realm management client for the target realm.
    #
    # Keycloak 25 and earlier: master realm has a single ``realm-management``
    # client that provides management roles for all realms.
    #
    # Keycloak 26+: master realm has per-realm management clients named
    # ``<realm-name>-realm`` (e.g. ``omninode-realm``) instead of the shared
    # ``realm-management`` client.
    #
    # We probe both names in order so the script works across KC versions.
    realm_mgmt_client_id: str | None = None
    for candidate in ("omninode-realm", "realm-management"):
        mgmt_resp = httpx.get(
            f"{base}/clients?clientId={candidate}", headers=headers, timeout=10
        )
        mgmt_resp.raise_for_status()
        if mgmt_resp.json():
            realm_mgmt_client_id = candidate
            break

    if realm_mgmt_client_id is None:
        raise RuntimeError(
            "Neither 'omninode-realm' nor 'realm-management' client found in "
            "master realm.  This indicates a non-standard Keycloak installation."
        )
    mgmt_id = mgmt_resp.json()[0]["id"]

    # Assign manage-users + view-users (sufficient for attribute patching)
    roles_resp = httpx.get(
        f"{base}/clients/{mgmt_id}/roles", headers=headers, timeout=10
    )
    roles_resp.raise_for_status()
    needed = [
        r for r in roles_resp.json() if r["name"] in ("manage-users", "view-users")
    ]
    httpx.post(
        f"{base}/users/{sa_id}/role-mappings/clients/{mgmt_id}",
        headers=headers,
        json=needed,
        timeout=10,
    ).raise_for_status()
    log("Assigned manage-users + view-users to onex-admin service account")

    # Rotate and return secret
    secret_resp = httpx.post(
        f"{base}/clients/{client_uuid}/client-secret", headers=headers, timeout=10
    )
    secret_resp.raise_for_status()
    secret: str = secret_resp.json()["value"]
    log(f"onex-admin client secret rotated {_secret_repr(secret)}")
    return secret


# ---------------------------------------------------------------------------
# Step 4.6 — Verify onex-admin can call omninode realm admin API
# ---------------------------------------------------------------------------


def verify_onex_admin_access(kc_url: str, realm: str, client_secret: str) -> None:
    """Mint a client_credentials token for onex-admin and hit the admin API.

    Fails fast on 403 to catch role assignment failures before runtime.

    Security: ``client_secret`` is never logged verbatim.
    """
    import httpx

    log("Verifying onex-admin access to omninode realm admin API")
    token_resp = httpx.post(
        f"{kc_url}/realms/master/protocol/openid-connect/token",
        data={
            "client_id": "onex-admin",
            "client_secret": client_secret,
            "grant_type": "client_credentials",
        },
        timeout=10,
    )
    token_resp.raise_for_status()
    access_token = token_resp.json()["access_token"]

    test_resp = httpx.get(
        f"{kc_url}/admin/realms/{realm}/users?max=1",
        headers={"Authorization": f"Bearer {access_token}"},
        timeout=10,
    )
    if test_resp.status_code == 403:
        raise RuntimeError(
            "onex-admin has a valid token but received 403 on admin endpoint.  "
            "Role assignment may have failed.  "
            "Check service account roles in Keycloak admin UI: "
            f"{kc_url}/admin/master/console/#/master/clients"
        )
    test_resp.raise_for_status()
    log("onex-admin verified: can access omninode realm admin API")


# ---------------------------------------------------------------------------
# Step 4.7 — Provision onex-service client in omninode realm
# ---------------------------------------------------------------------------


def provision_onex_service_client(kc_url: str, realm: str, token: str) -> str:
    """Create the ``onex-service`` confidential client in the omninode realm.

    Used by omniweb middleware to obtain a client_credentials Bearer token
    before calling ``/v1/auth/provision`` on onex-api.

    onex-api validates incoming tokens by checking ``azp == "onex-service"``.

    Returns:
        The rotated client secret (never logged verbatim).
    """
    import httpx

    log(f"Provisioning onex-service client in '{realm}' realm")
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    base = f"{kc_url}/admin/realms/{realm}"

    clients_resp = httpx.get(
        f"{base}/clients?clientId=onex-service", headers=headers, timeout=10
    )
    clients_resp.raise_for_status()
    if not clients_resp.json():
        log("Creating onex-service client")
        httpx.post(
            f"{base}/clients",
            headers=headers,
            json={
                "clientId": "onex-service",
                "enabled": True,
                "publicClient": False,
                "serviceAccountsEnabled": True,
                "standardFlowEnabled": False,
                "directAccessGrantsEnabled": False,
                "protocol": "openid-connect",
            },
            timeout=10,
        ).raise_for_status()
        clients_resp = httpx.get(
            f"{base}/clients?clientId=onex-service", headers=headers, timeout=10
        )
        clients_resp.raise_for_status()
    else:
        log("onex-service client already exists — rotating secret")

    client_uuid = clients_resp.json()[0]["id"]
    secret_resp = httpx.post(
        f"{base}/clients/{client_uuid}/client-secret", headers=headers, timeout=10
    )
    secret_resp.raise_for_status()
    secret: str = secret_resp.json()["value"]
    log(f"onex-service client secret rotated {_secret_repr(secret)}")
    return secret


# ---------------------------------------------------------------------------
# Step 4.8 — Write credentials to ~/.omnibase/.env
# ---------------------------------------------------------------------------


def write_env_credentials(
    env_file_path: Path,
    realm: str,
    onex_admin_secret: str,
    onex_service_secret: str,
    dry_run: bool = False,
) -> None:
    """Write the 7 KEYCLOAK_*/ONEX_SERVICE_* vars to the env file.

    ``KEYCLOAK_ADMIN_URL`` is always the INTERNAL Docker DNS address
    (``http://keycloak:8080``), regardless of the ``--kc-url`` argument.
    The ``--kc-url`` arg is the external URL used only by this script and
    must NEVER be written to the env file.

    ``KEYCLOAK_ADMIN_PASSWORD`` is NOT written — it is consumed transiently
    during bootstrap and must not persist in env files.

    In ``--dry-run`` mode, prints the vars to stdout without writing.

    Security: secret values are logged as ``(SET, N chars)`` only.
    """
    vars_to_write = {
        "KEYCLOAK_ADMIN_URL": "http://keycloak:8080",  # INTERNAL — never localhost
        "KEYCLOAK_REALM": realm,
        "KEYCLOAK_ADMIN_CLIENT_ID": "onex-admin",
        "KEYCLOAK_ADMIN_CLIENT_SECRET": onex_admin_secret,
        "KEYCLOAK_ISSUER": f"http://localhost:28080/realms/{realm}",
        "ONEX_SERVICE_CLIENT_ID": "onex-service",
        "ONEX_SERVICE_CLIENT_SECRET": onex_service_secret,
    }

    safe_log = {
        k: _secret_repr(v) if "SECRET" in k else v for k, v in vars_to_write.items()
    }

    if dry_run:
        log("[DRY RUN] Would write the following vars to env file:")
        for k, v in safe_log.items():
            log(f"  {k}={v}")
        return

    log(f"Writing credentials to {env_file_path}")
    for k, v in safe_log.items():
        log(f"  {k}={v}")
    update_env_file(env_file_path, vars_to_write)
    log("Credentials written successfully")


# ---------------------------------------------------------------------------
# Step 4.9 — Seed into Infisical
# ---------------------------------------------------------------------------


def seed_infisical(
    realm: str,
    onex_admin_secret: str,
    onex_service_secret: str,
    dry_run: bool = False,
) -> None:
    """Seed Keycloak config and secrets into Infisical if INFISICAL_ADDR is set.

    Paths:
    - ``/shared/auth/``: non-sensitive config (URLs, realm, client IDs)
    - ``/services/onex-api/auth/``: secrets (client secrets)

    Skips silently if ``INFISICAL_ADDR`` is not set.
    Security: secrets are logged as ``(SET, N chars)`` only.
    """
    import os

    infisical_addr = os.environ.get("INFISICAL_ADDR", "").strip()
    if not infisical_addr:
        log("INFISICAL_ADDR not set — skipping Infisical seed")
        return

    log(f"Seeding Keycloak config into Infisical at {infisical_addr}")

    shared_vars = {
        "KEYCLOAK_ADMIN_URL": "http://keycloak:8080",
        "KEYCLOAK_REALM": realm,
        "KEYCLOAK_ISSUER": f"http://localhost:28080/realms/{realm}",
        "KEYCLOAK_ADMIN_CLIENT_ID": "onex-admin",
        "ONEX_SERVICE_CLIENT_ID": "onex-service",
    }
    secret_vars = {
        "KEYCLOAK_ADMIN_CLIENT_SECRET": onex_admin_secret,
        "ONEX_SERVICE_CLIENT_SECRET": onex_service_secret,
    }

    if dry_run:
        log("[DRY RUN] Would seed /shared/auth/ with:")
        for k, v in shared_vars.items():
            log(f"  {k}={v}")
        log("[DRY RUN] Would seed /services/onex-api/auth/ with:")
        for k, v in secret_vars.items():
            log(f"  {k}={_secret_repr(v)}")
        return

    try:
        import httpx

        # Retrieve auth token from env
        client_id = os.environ.get("INFISICAL_CLIENT_ID", "")
        client_secret = os.environ.get("INFISICAL_CLIENT_SECRET", "")
        project_id = os.environ.get("INFISICAL_PROJECT_ID", "")

        if not all([client_id, client_secret, project_id]):
            log(
                "INFISICAL_CLIENT_ID / INFISICAL_CLIENT_SECRET / INFISICAL_PROJECT_ID "
                "not set — skipping Infisical seed.  "
                "Run provision-infisical.py first."
            )
            return

        # Obtain machine identity token
        token_resp = httpx.post(
            f"{infisical_addr}/api/v1/auth/universal-auth/login",
            json={"clientId": client_id, "clientSecret": client_secret},
            timeout=10,
        )
        token_resp.raise_for_status()
        access_token = token_resp.json()["accessToken"]

        headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json",
        }

        def _upsert_secret(path: str, key: str, value: str, env: str = "dev") -> None:
            # Try create first; if 409 (already exists), update
            resp = httpx.post(
                f"{infisical_addr}/api/v3/secrets/raw/{key}",
                headers=headers,
                json={
                    "workspaceId": project_id,
                    "environment": env,
                    "secretPath": path,
                    "secretValue": value,
                },
                timeout=10,
            )
            if resp.status_code == 409:
                # Already exists — update
                httpx.patch(
                    f"{infisical_addr}/api/v3/secrets/raw/{key}",
                    headers=headers,
                    json={
                        "workspaceId": project_id,
                        "environment": env,
                        "secretPath": path,
                        "secretValue": value,
                    },
                    timeout=10,
                ).raise_for_status()
            else:
                resp.raise_for_status()

        for k, v in shared_vars.items():
            _upsert_secret("/shared/auth", k, v)
            log(f"  Seeded /shared/auth/{k}={v}")

        for k, v in secret_vars.items():
            _upsert_secret("/services/onex-api/auth", k, v)
            log(f"  Seeded /services/onex-api/auth/{k}={_secret_repr(v)}")

        log("Infisical seed complete")

    except Exception as exc:  # noqa: BLE001 — boundary: catch-all for resilience
        log(f"WARNING: Infisical seed failed (non-fatal): {exc}")
        log("You can re-run with INFISICAL_ADDR set once Infisical is provisioned.")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--kc-url",
        default="http://localhost:28080",
        help=(
            "External base URL for Keycloak (no path prefix). "
            "Default: http://localhost:28080.  "
            "Do NOT use http://localhost:28080/auth/ — it breaks admin API paths."
        ),
    )
    p.add_argument(
        "--realm",
        default="omninode",
        help="Keycloak realm to provision clients in.  Default: omninode",
    )
    p.add_argument(
        "--admin-username",
        default="admin",
        help="Keycloak bootstrap admin username.  Default: admin",
    )
    p.add_argument(
        "--admin-password",
        required=True,
        help=(
            "Keycloak bootstrap admin password.  "
            "NEVER written to env file — consumed transiently."
        ),
    )
    p.add_argument(
        "--env-file",
        type=Path,
        default=Path.home() / ".omnibase" / ".env",
        help="Path to the .env file.  Default: ~/.omnibase/.env",
    )
    p.add_argument(
        "--postgres-port",
        type=int,
        default=5436,
        help="External postgres port.  Default: 5436",
    )
    p.add_argument(
        "--skip-infisical",
        action="store_true",
        help="Skip Infisical seeding even if INFISICAL_ADDR is set.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Print all actions and vars without writing to env file "
            "or creating/modifying Keycloak resources."
        ),
    )
    return p


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    import os

    postgres_password = os.environ.get("POSTGRES_PASSWORD", "")
    if not postgres_password:
        logger.error(
            "POSTGRES_PASSWORD not set.  "
            "Run: source ~/.omnibase/.env before executing this script."
        )
        return 1

    log("=== provision-keycloak.py starting ===")
    log(f"  kc-url:         {args.kc_url}")
    log(f"  realm:          {args.realm}")
    log(f"  admin-username: {args.admin_username}")
    log(f"  admin-password: {_secret_repr(args.admin_password)}")
    log(f"  env-file:       {args.env_file}")
    log(f"  postgres-port:  {args.postgres_port}")
    log(f"  skip-infisical: {args.skip_infisical}")
    log(f"  dry-run:        {args.dry_run}")

    try:
        # Step 4.1
        if args.dry_run:
            log("[DRY RUN] Would ensure keycloak DB exists in postgres")
        else:
            ensure_keycloak_db(postgres_password, args.postgres_port)

        # Step 4.2
        if args.dry_run:
            log(f"[DRY RUN] Would wait for Keycloak readiness at {args.kc_url}")
        else:
            wait_for_keycloak(args.kc_url)

        # Step 4.2b — patch master realm SSL to NONE so plain-HTTP token grants work
        if args.dry_run:
            log("[DRY RUN] Would patch master/omninode realm ssl_required to NONE")
        else:
            patch_master_realm_ssl_none(postgres_password, args.postgres_port)

        # Step 4.3
        if args.dry_run:
            log(f"[DRY RUN] Would acquire admin token for user '{args.admin_username}'")
            # Use a placeholder token for dry-run path
            admin_token = "dry-run-placeholder-token"
        else:
            admin_token = get_admin_token(
                args.kc_url, args.admin_username, args.admin_password
            )

        # Step 4.4
        if args.dry_run:
            log(f"[DRY RUN] Would verify realm '{args.realm}' exists")
        else:
            verify_realm(args.kc_url, args.realm, admin_token)

        # Step 4.5
        if args.dry_run:
            log("[DRY RUN] Would provision onex-admin client in master realm")
            onex_admin_secret = "dry-run-admin-secret"
        else:
            onex_admin_secret = provision_onex_admin_client(args.kc_url, admin_token)

        # Step 4.6
        if args.dry_run:
            log("[DRY RUN] Would verify onex-admin access to omninode realm admin API")
        else:
            verify_onex_admin_access(args.kc_url, args.realm, onex_admin_secret)

        # Step 4.7
        if args.dry_run:
            log(
                f"[DRY RUN] Would provision onex-service client in '{args.realm}' realm"
            )
            onex_service_secret = "dry-run-service-secret"
        else:
            onex_service_secret = provision_onex_service_client(
                args.kc_url, args.realm, admin_token
            )

        # Step 4.8
        write_env_credentials(
            args.env_file,
            args.realm,
            onex_admin_secret,
            onex_service_secret,
            dry_run=args.dry_run,
        )

        # Step 4.9
        if not args.skip_infisical:
            seed_infisical(
                args.realm,
                onex_admin_secret,
                onex_service_secret,
                dry_run=args.dry_run,
            )
        else:
            log("--skip-infisical set — skipping Infisical seed")

        log("=== provision-keycloak.py complete ===")
        return 0

    except KeyboardInterrupt:
        log("Interrupted")
        return 130
    except Exception as exc:
        logger.exception("provision-keycloak.py failed: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
