#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Fully automated Infisical provisioning for ONEX Infrastructure.

Replaces the manual steps previously documented in setup-infisical-identity.sh.
Idempotent: safe to re-run; skips steps already completed.

Steps performed:
    1. Bootstrap admin user + org (via /api/v1/admin/bootstrap)
    2. Create project (workspace) in Infisical
    3. Create machine identity with Universal Auth
    4. Generate client credentials (client_id + client_secret)
    5. Add identity to project as admin
    6. Write INFISICAL_* vars to .env

Usage:
    uv run python scripts/provision-infisical.py
    uv run python scripts/provision-infisical.py --addr http://localhost:8880
    uv run python scripts/provision-infisical.py --org omninode --project omninode-infra
    uv run python scripts/provision-infisical.py --dry-run

.. versionadded:: 0.10.0
    Replaces setup-infisical-identity.sh stub (OMN-2287).
"""

from __future__ import annotations

import argparse
import logging
import os
import secrets
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("provision-infisical")

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_ENV_FILE = Path.home() / ".omnibase" / ".env"
_IDENTITY_FILE = _PROJECT_ROOT / ".infisical-identity"
_ADMIN_TOKEN_FILE = _PROJECT_ROOT / ".infisical-admin-token"

# Shared utility — avoids duplicating the parser in every Infisical script.
# Ensure the scripts dir is on the path so the import resolves from any cwd.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _infisical_util import _parse_env_file


def _write_env_vars(env_path: Path, updates: dict[str, str]) -> None:
    """Write or update key=value pairs in a .env file.

    Adds keys that don't exist (appended at the end of the file).
    Updates existing uncommented keys in-place when their current value is
    empty.  Commented-out keys (lines beginning with ``#``) are left
    untouched; if the corresponding key is in ``updates`` it will be
    appended as a new uncommented entry at the end of the file.
    Does not overwrite keys that are already set with a non-empty value.
    """
    lines: list[str] = []
    if env_path.is_file():
        lines = env_path.read_text(encoding="utf-8").splitlines()

    existing: dict[str, int] = {}  # key -> line index
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        if "=" in stripped:
            key = stripped.partition("=")[0].strip()
            # Strip "export " prefix so "export KEY=val" indexes as "KEY",
            # matching the bare key names in the `updates` dict.
            key = key.removeprefix("export ").strip()
            existing[key] = i

    # Update existing keys or append new ones
    appended: list[str] = []
    for key, value in updates.items():
        if key in existing:
            current_line = lines[existing[key]]
            current_val = current_line.partition("=")[2].strip()
            current_val = current_val.strip("'\"")
            if current_val:
                logger.info("  %s already set, skipping", key)
                continue
            # Preserve the `export ` prefix if the original line had it, so
            # that files using `export KEY=val` syntax for shell sourcing
            # compatibility do not silently lose that prefix on update.
            had_export = current_line.lstrip().startswith("export ")
            prefix = "export " if had_export else ""
            lines[existing[key]] = f"{prefix}{key}={value}"
            logger.info("  Updated %s", key)
        else:
            appended.append(f"{key}={value}")
            logger.info("  Added %s", key)

    if appended:
        if lines and lines[-1].strip():
            lines.append("")
        lines.append(
            "# --- Infisical client credentials (provisioned automatically) ---"
        )
        lines.extend(appended)

    content = "\n".join(lines) + "\n"
    # NOTE: the default env_path (~/.omnibase/.env) is outside the repo, so
    # gitignore is irrelevant here.  The .tmp file is replaced atomically via
    # os.rename (POSIX), so the exposure window is negligible regardless of
    # where the file lives.
    env_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = env_path.with_name(env_path.name + ".tmp")
    # Create the tmp file with 0o600 permissions from the start so there is
    # never a window where the file containing secrets is world-readable.
    fd = os.open(str(tmp), os.O_CREAT | os.O_WRONLY | os.O_TRUNC, 0o600)
    try:
        os.write(fd, content.encode("utf-8"))
    finally:
        os.close(fd)
    try:
        tmp.replace(env_path)  # atomic on POSIX
    except Exception:
        tmp.unlink(missing_ok=True)
        raise


def _bootstrap(client: object, addr: str, email: str, password: str, org: str) -> dict:  # type: ignore[type-arg]
    """Call /api/v1/admin/bootstrap to create admin user + org + initial identity."""
    import httpx

    resp = client.post(  # type: ignore[attr-defined]
        f"{addr}/api/v1/admin/bootstrap",
        json={"email": email, "password": password, "organization": org},
    )
    if resp.status_code == 400 and "already" in resp.text.lower():
        logger.info("Instance already bootstrapped")
        return {}
    resp.raise_for_status()
    return resp.json()


def _create_workspace(client: object, addr: str, token: str, project_name: str) -> dict:  # type: ignore[type-arg]
    """Create a project (workspace) and return its details."""
    resp = client.post(  # type: ignore[attr-defined]
        f"{addr}/api/v2/workspace",
        headers={"Authorization": f"Bearer {token}"},
        json={"projectName": project_name, "shouldCreateDefaultEnvs": True},
    )
    if resp.status_code in (400, 409) and (
        "already" in resp.text.lower() or "exist" in resp.text.lower()
    ):
        logger.info("Project '%s' already exists, fetching ID", project_name)
        # List workspaces to find existing one
        list_resp = client.get(  # type: ignore[attr-defined]
            f"{addr}/api/v3/workspaces",
            headers={"Authorization": f"Bearer {token}"},
        )
        list_resp.raise_for_status()
        workspaces = list_resp.json().get("workspaces", [])
        for ws in workspaces:
            if ws.get("name") == project_name or ws.get("slug") == project_name:
                return ws
        msg = f"Could not find existing workspace '{project_name}'"
        raise RuntimeError(msg)
    resp.raise_for_status()
    data = resp.json()
    return data.get("project", data.get("workspace", data))


def _create_identity(
    client: object,
    addr: str,
    token: str,
    name: str,
    org_id: str,  # type: ignore[type-arg]
) -> dict:  # type: ignore[type-arg]
    """Create a machine identity in the org."""
    resp = client.post(  # type: ignore[attr-defined]
        f"{addr}/api/v1/identities",
        headers={"Authorization": f"Bearer {token}"},
        json={"name": name, "organizationId": org_id, "role": "no-access"},
    )
    if resp.status_code in (400, 409) and (
        "already" in resp.text.lower() or "exist" in resp.text.lower()
    ):
        logger.info("Identity '%s' already exists, searching for it", name)
        search_resp = client.get(  # type: ignore[attr-defined]
            f"{addr}/api/v1/identities/search",
            headers={"Authorization": f"Bearer {token}"},
            params={"name": name, "organizationId": org_id},
        )
        search_resp.raise_for_status()
        identities = search_resp.json().get("identities", [])
        for identity in identities:
            if identity.get("name") == name:
                return identity
        msg = f"Could not find existing identity '{name}'"
        raise RuntimeError(msg)
    resp.raise_for_status()
    return resp.json().get("identity", resp.json())


def _configure_universal_auth(
    client: object,
    addr: str,
    token: str,
    identity_id: str,  # type: ignore[type-arg]
) -> None:
    """Configure Universal Auth on an identity."""
    # Default trusted-IP list covers loopback and the standard dev subnet.
    # Override by setting INFISICAL_TRUSTED_IPS to a comma-separated list
    # of CIDR blocks, e.g. "127.0.0.1/32,10.0.0.0/8".
    _trusted_ips_env = os.environ.get("INFISICAL_TRUSTED_IPS")
    if _trusted_ips_env:
        _trusted_ip_list = [
            {"ipAddress": cidr.strip()}
            for cidr in _trusted_ips_env.split(",")
            if cidr.strip()
        ]
    else:
        _trusted_ip_list = [
            {"ipAddress": "127.0.0.1/32"},
            {"ipAddress": "192.168.86.0/24"},
        ]
    resp = client.post(  # type: ignore[attr-defined]
        f"{addr}/api/v1/auth/universal-auth/identities/{identity_id}",
        headers={"Authorization": f"Bearer {token}"},
        json={
            "accessTokenTTL": 86400,  # 24h
            "accessTokenMaxTTL": 2592000,  # 30d
            "accessTokenNumUsesLimit": 0,  # unlimited
            "clientSecretTrustedIps": _trusted_ip_list,
            "accessTokenTrustedIps": _trusted_ip_list,
        },
    )
    if resp.status_code in (400, 409) and "already" in resp.text.lower():
        logger.info("Universal Auth already configured for identity %s", identity_id)
        return
    resp.raise_for_status()


def _create_client_secret(
    client: object,
    addr: str,
    token: str,
    identity_id: str,  # type: ignore[type-arg]
) -> dict:  # type: ignore[type-arg]
    """Create a client secret for Universal Auth and return {clientId, clientSecret}."""
    resp = client.post(  # type: ignore[attr-defined]
        f"{addr}/api/v1/auth/universal-auth/identities/{identity_id}/client-secrets",
        headers={"Authorization": f"Bearer {token}"},
        json={"description": "onex-runtime auto-provisioned"},
    )
    resp.raise_for_status()
    data = resp.json()
    # The client_id is the identity's clientId from universal auth config
    # Fetch it from the UA config
    ua_resp = client.get(  # type: ignore[attr-defined]
        f"{addr}/api/v1/auth/universal-auth/identities/{identity_id}",
        headers={"Authorization": f"Bearer {token}"},
    )
    ua_resp.raise_for_status()
    ua_data = ua_resp.json().get("identityUniversalAuth", ua_resp.json())
    return {
        "clientId": ua_data.get("clientId", ""),
        "clientSecret": data.get(
            "clientSecret", data.get("data", {}).get("clientSecret", "")
        ),
    }


def _create_infisical_folders(
    client: object,  # type: ignore[type-arg]
    addr: str,
    token: str,
    project_id: str,
    environments: tuple[str, ...] = ("dev", "staging", "prod"),
    transport_folders: tuple[str, ...] = (
        "consul",
        "db",
        "http",
        "mcp",
        "graph",
        "env",
        "kafka",
        "vault",
        "qdrant",
    ),
) -> None:
    """Create the /shared/<transport> folder structure in every environment.

    Infisical requires folders to exist before secrets can be written into them.
    Folders that already exist are silently skipped (the API returns the existing one).
    """
    for env in environments:
        # Create /shared root
        resp = client.post(  # type: ignore[attr-defined]
            f"{addr}/api/v1/folders",
            headers={"Authorization": f"Bearer {token}"},
            json={
                "workspaceId": project_id,
                "environment": env,
                "name": "shared",
                "path": "/",
            },
        )
        if resp.status_code not in (200, 201, 400, 409):
            resp.raise_for_status()
        for folder in transport_folders:
            resp = client.post(  # type: ignore[attr-defined]
                f"{addr}/api/v1/folders",
                headers={"Authorization": f"Bearer {token}"},
                json={
                    "workspaceId": project_id,
                    "environment": env,
                    "name": folder,
                    "path": "/shared",
                },
            )
            if resp.status_code not in (200, 201, 400, 409):
                resp.raise_for_status()
    logger.info("Folder structure created in environments: %s", list(environments))


def _add_identity_to_project(
    client: object,  # type: ignore[type-arg]
    addr: str,
    token: str,
    project_id: str,
    identity_id: str,
    role: str = "viewer",
) -> None:
    """Add a machine identity to a project with the given role."""
    resp = client.post(  # type: ignore[attr-defined]
        f"{addr}/api/v2/workspace/{project_id}/identity-memberships/{identity_id}",
        headers={"Authorization": f"Bearer {token}"},
        json={"role": role},
    )
    if resp.status_code in (400, 409) and (
        "already" in resp.text.lower() or "exist" in resp.text.lower()
    ):
        logger.info(
            "Identity %s already a member of project %s", identity_id, project_id
        )
        return
    resp.raise_for_status()


def main() -> int:
    """Run full Infisical provisioning."""
    parser = argparse.ArgumentParser(
        description="Automated Infisical provisioning for ONEX Infrastructure",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--addr",
        default=os.environ.get("INFISICAL_ADDR", "http://localhost:8880"),
        help="Infisical server address (default: http://localhost:8880)",
    )
    parser.add_argument(
        "--org",
        default="omninode",
        help="Organization name to create (default: omninode)",
    )
    parser.add_argument(
        "--project",
        default="omninode-infra",
        help="Project name to create (default: omninode-infra)",
    )
    parser.add_argument(
        "--identity-name",
        default="onex-runtime",
        help="Machine identity name (default: onex-runtime)",
    )
    parser.add_argument(
        "--admin-email",
        default="admin@omninode.local",
        help="Admin user email (default: admin@omninode.local)",
    )
    parser.add_argument(
        "--admin-password",
        default=None,
        help=(
            "Admin user password; if not set, reads INFISICAL_ADMIN_PASSWORD "
            "env var, else auto-generates"
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would happen without executing",
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        default=_ENV_FILE,
        help=f"Path to .env file for writing Infisical credentials (default: {_ENV_FILE})",
    )
    args = parser.parse_args()

    if not args.addr.startswith(("http://", "https://")):
        print(
            f"error: --addr must start with 'http://' or 'https://', got: {args.addr!r}",
            file=sys.stderr,
        )
        return 1

    try:
        import httpx
    except ImportError:
        logger.exception("httpx is required: uv add httpx")
        return 1

    # Check if already provisioned — look in both the .env file and the shell
    # environment so that sourcing ~/.omnibase/.env in ~/.zshrc is honoured.
    existing_env = _parse_env_file(args.env_file)
    _provision_keys = (
        "INFISICAL_CLIENT_ID",
        "INFISICAL_CLIENT_SECRET",
        "INFISICAL_PROJECT_ID",
    )
    if all(existing_env.get(k) or os.environ.get(k) for k in _provision_keys):
        logger.info(
            "Already provisioned: INFISICAL_CLIENT_ID/SECRET/PROJECT_ID are set "
            "(in %s or shell environment)",
            args.env_file,
        )
        logger.info("To re-provision, remove those keys from .env first.")
        logger.warning(
            "INFISICAL_ADDR in env file may differ from --addr %s — verify before use",
            args.addr,
        )
        return 0

    # Resolve admin password: priority: env var (highest) > CLI arg > auto-generate (lowest).
    # Avoid CLI arg as the primary source because plaintext flags are visible
    # in `ps aux` and /proc/PID/cmdline.
    admin_password = (
        os.environ.get("INFISICAL_ADMIN_PASSWORD")
        or args.admin_password
        or secrets.token_urlsafe(24)
    )

    if args.dry_run:
        logger.info("[DRY RUN] Would provision Infisical:")
        logger.info("  Org: %s", args.org)
        logger.info("  Project: %s", args.project)
        logger.info("  Identity: %s", args.identity_name)
        logger.info("  Admin email: %s", args.admin_email)
        logger.info("  .env file: %s", args.env_file)
        return 0

    # Check connectivity and readiness.
    # /api/status can return HTTP 200 before migrations are complete, so we
    # also inspect the response body for a "status": "ok" field to confirm
    # the server is fully ready (not just reachable).
    # NOTE: This check is intentionally placed after the --dry-run guard so
    # that dry-run mode does not require a running Infisical instance.
    try:
        with httpx.Client(timeout=10) as probe:
            resp = probe.get(f"{args.addr}/api/status")
            resp.raise_for_status()
            try:
                body = resp.json()
                if body.get("status") != "ok":
                    logger.error(
                        "Infisical at %s returned HTTP 200 but status field is %r "
                        "(expected 'ok'). The server may still be initialising.",
                        args.addr,
                        body.get("status"),
                    )
                    return 1
            except (ValueError, KeyError):
                # Cannot parse JSON — treat as not ready rather than assuming OK.
                # This is a benign startup condition (server returning HTML while
                # booting) so we log at WARNING level to avoid alarming tracebacks.
                logger.warning(
                    "Infisical at %s returned a non-JSON /api/status response. "
                    "The server may still be initialising.",
                    args.addr,
                )
                return 1
        logger.info("Infisical is reachable and ready at %s", args.addr)
    except Exception:
        logger.exception("Cannot reach Infisical at %s", args.addr)
        logger.info(
            "Start it with: docker compose -p omnibase-infra-runtime"
            " -f docker/docker-compose.infra.yml --profile secrets up -d infisical"
        )
        return 1

    try:
        with httpx.Client(timeout=30) as client:
            # Step 1: Bootstrap admin user + org
            logger.info("Step 1: Bootstrapping admin user and organization...")
            bootstrap_data = _bootstrap(
                client, args.addr, args.admin_email, admin_password, args.org
            )

            if not bootstrap_data:
                # Already bootstrapped — check for saved admin token
                if _ADMIN_TOKEN_FILE.is_file():
                    with _ADMIN_TOKEN_FILE.open() as f:
                        admin_token = f.readline().strip()
                    if not admin_token:
                        logger.error(
                            "Admin token file at %s exists but is empty or has a blank "
                            "first line. Delete the file and re-run to re-provision.",
                            _ADMIN_TOKEN_FILE,
                        )
                        return 1
                    logger.info(
                        "Instance already bootstrapped, using saved admin token"
                    )
                    # Get org id from existing workspaces list
                    orgs_resp = client.get(
                        f"{args.addr}/api/v1/organization",
                        headers={"Authorization": f"Bearer {admin_token}"},
                    )
                    orgs_resp.raise_for_status()
                    orgs = orgs_resp.json().get("organizations", [])
                    if not orgs:
                        logger.error(
                            "Instance is already bootstrapped but the org list returned "
                            "by /api/v1/organization is empty. The admin token at %s may "
                            "be stale or belong to a different instance. Delete the "
                            "Infisical DB state and re-run to start fresh.",
                            _ADMIN_TOKEN_FILE,
                        )
                        return 1
                    org_id = orgs[0]["id"]
                else:
                    logger.error(
                        "Instance already bootstrapped but no admin token found at %s. "
                        "To recover: wipe the Infisical DB state (drop the infisical_db "
                        "volume) and re-run this script to start fresh.",
                        _ADMIN_TOKEN_FILE,
                    )
                    return 1
            else:
                admin_token = (
                    bootstrap_data.get("identity", {})
                    .get("credentials", {})
                    .get("token", "")
                )
                if not admin_token:
                    logger.error(
                        "Bootstrap response missing identity.credentials.token"
                    )
                    return 1
                org_id = bootstrap_data.get("organization", {}).get("id", "")
                if not org_id:
                    logger.error("Bootstrap response missing organization.id")
                    return 1
                # Persist admin token and (if auto-generated) password for subsequent runs.
                # Password is written ONLY to the file — never to log output.
                token_file_lines = [admin_token]
                _password_was_auto_generated = (
                    not args.admin_password
                    and not os.environ.get("INFISICAL_ADMIN_PASSWORD")
                )
                if _password_was_auto_generated:
                    # SENSITIVE: this line stores a high-privilege admin credential.
                    # Do NOT share this file, copy it to other machines, or include
                    # it in backups. The file is chmod 0o600 (owner read/write only).
                    token_file_lines.append(f"admin_password={admin_password}")
                _admin_token_tmp = _ADMIN_TOKEN_FILE.with_name(
                    _ADMIN_TOKEN_FILE.name + ".tmp"
                )
                # Create with 0o600 from the start — no world-readable window.
                _atmp_fd = os.open(
                    str(_admin_token_tmp),
                    os.O_CREAT | os.O_WRONLY | os.O_TRUNC,
                    0o600,
                )
                try:
                    os.write(
                        _atmp_fd,
                        ("\n".join(token_file_lines) + "\n").encode("utf-8"),
                    )
                finally:
                    os.close(_atmp_fd)
                try:
                    _admin_token_tmp.replace(_ADMIN_TOKEN_FILE)  # atomic on POSIX
                except Exception:
                    _admin_token_tmp.unlink(missing_ok=True)
                    raise
                logger.info("Admin token saved to %s", _ADMIN_TOKEN_FILE)
                if _password_was_auto_generated:
                    logger.info(
                        "Generated admin password written to %s (not logged here)",
                        _ADMIN_TOKEN_FILE,
                    )

            logger.info(
                "Bootstrapped: org=%s (%s)",
                bootstrap_data.get("organization", {}).get("name", "(existing)"),
                org_id,
            )

            # Step 2: Create project
            logger.info("Step 2: Creating project '%s'...", args.project)
            workspace = _create_workspace(client, args.addr, admin_token, args.project)
            project_id = workspace.get("id") or workspace.get("_id", "")
            if not project_id:
                logger.error(
                    "Could not extract project ID from workspace response: %s",
                    workspace,
                )
                return 1
            logger.info("Project created: %s (%s)", workspace.get("name"), project_id)

            # Step 3: Create machine identity
            logger.info("Step 3: Creating machine identity '%s'...", args.identity_name)
            identity = _create_identity(
                client, args.addr, admin_token, args.identity_name, org_id
            )
            identity_id = identity.get("id") or identity.get("_id", "")
            if not identity_id:
                logger.error(
                    "Could not extract identity ID from identity response: %s", identity
                )
                return 1
            logger.info("Identity created: %s (%s)", identity.get("name"), identity_id)

            # Step 4: Configure Universal Auth
            logger.info("Step 4: Configuring Universal Auth...")
            _configure_universal_auth(client, args.addr, admin_token, identity_id)

            # Step 5: Create client credentials
            logger.info("Step 5: Creating client credentials...")
            credentials = _create_client_secret(
                client, args.addr, admin_token, identity_id
            )
            client_id = credentials["clientId"]
            client_secret = credentials["clientSecret"]
            if not client_id or not client_secret:
                logger.error(
                    "Failed to get client credentials (client_id=%s)",
                    client_id or "(empty)",
                )
                return 1
            logger.info("Client credentials created (client_id=%s)", client_id)

            # Step 6: Create folder structure in all environments
            logger.info("Step 6: Creating /shared/<transport> folder structure...")
            _create_infisical_folders(client, args.addr, admin_token, project_id)

            # Step 7: Add identity to project as admin (required for seed write access)
            logger.info("Step 7: Adding identity to project as admin...")
            _add_identity_to_project(
                client, args.addr, admin_token, project_id, identity_id, role="admin"
            )

            # Step 8: Write to .env
            logger.info("Step 8: Writing credentials to %s...", args.env_file)
            updates = {
                "INFISICAL_ADDR": args.addr,
                "INFISICAL_CLIENT_ID": client_id,
                "INFISICAL_CLIENT_SECRET": client_secret,
                "INFISICAL_PROJECT_ID": project_id,
            }
            _write_env_vars(args.env_file, updates)

            # Write identity marker file (metadata-only, no secrets).
            # Use the same atomic write-chmod-rename pattern as _ADMIN_TOKEN_FILE
            # to avoid a brief world-readable window between write and chmod.
            _identity_tmp = _IDENTITY_FILE.with_name(_IDENTITY_FILE.name + ".tmp")
            try:
                _identity_content = (
                    f"# Infisical Machine Identity\n"
                    f"# Provisioned automatically by provision-infisical.py\n"
                    f"#\n"
                    f"# Org: {args.org} ({org_id})\n"
                    f"# Project: {args.project} ({project_id})\n"
                    f"# Identity: {args.identity_name} ({identity_id})\n"
                    f"# Admin email: {args.admin_email}\n"
                    f"#\n"
                    f"# Client credentials written to .env\n"
                )
                # Create with 0o600 from the start — no world-readable window.
                _itmp_fd = os.open(
                    str(_identity_tmp),
                    os.O_CREAT | os.O_WRONLY | os.O_TRUNC,
                    0o600,
                )
                try:
                    os.write(_itmp_fd, _identity_content.encode("utf-8"))
                finally:
                    os.close(_itmp_fd)
                _identity_tmp.replace(_IDENTITY_FILE)  # atomic on POSIX
            except Exception:
                if _identity_tmp.exists():
                    _identity_tmp.unlink(missing_ok=True)
                raise
    except httpx.HTTPStatusError as exc:
        logger.exception(
            "HTTP error during provisioning: %s %s returned status %d. "
            "Check that Infisical is running and accessible at %s.",
            exc.request.method,
            exc.request.url,
            exc.response.status_code,
            args.addr,
        )
        return 1
    except httpx.RequestError as exc:
        logger.exception(
            "Network error during provisioning while connecting to %s: %s. "
            "Check that Infisical is reachable at %s.",
            exc.request.url,
            exc,
            args.addr,
        )
        return 1

    logger.info("")
    logger.info("Provisioning complete!")
    logger.info("  INFISICAL_ADDR=%s", args.addr)
    logger.info("  INFISICAL_CLIENT_ID=%s", client_id)
    logger.info("  INFISICAL_CLIENT_SECRET=<written to .env>")
    logger.info("  INFISICAL_PROJECT_ID=%s", project_id)
    logger.info("")
    logger.info("Next: seed secrets with:")
    logger.info(
        "  uv run python scripts/seed-infisical.py "
        "--contracts-dir src/omnibase_infra/nodes "
        "--import-env docs/env-example-full.txt "
        "--set-values --create-missing-keys --execute"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
