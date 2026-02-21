#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Register repos and seed shared platform secrets into Infisical.

This script is the central onboarding tool for the OmniNode platform's
Infisical-based secret management system. It has two subcommands:

    seed-shared
        Populate /shared/<transport>/ paths in Infisical with all
        platform-wide credentials (postgres host/user, kafka, consul,
        LLM endpoints, API keys, etc.). Run once after provisioning.

    onboard-repo
        Create /services/<repo>/ folder structure and seed repo-specific
        secrets (e.g. POSTGRES_DATABASE). Run once per new downstream repo.

Both subcommands are dry-run by default. Pass --execute to write.

The end state for ~/.omnibase/.env (5 bootstrap lines only):
    POSTGRES_PASSWORD=...
    INFISICAL_ADDR=http://localhost:8880
    INFISICAL_CLIENT_ID=...
    INFISICAL_CLIENT_SECRET=...
    INFISICAL_PROJECT_ID=...

Everything else lives in Infisical.

Usage:
    # Populate /shared/ paths from the platform env file (dry-run)
    uv run python scripts/register-repo.py seed-shared \\
        --env-file ~/.omnibase/.env

    # Apply the shared seed
    uv run python scripts/register-repo.py seed-shared \\
        --env-file ~/.omnibase/.env --execute

    # Onboard a downstream repo (dry-run)
    uv run python scripts/register-repo.py onboard-repo \\
        --repo omniclaude \\
        --env-file /Volumes/PRO-G40/Code/omniclaude/.env

    # Apply the onboarding
    uv run python scripts/register-repo.py onboard-repo \\
        --repo omniclaude \\
        --env-file /Volumes/PRO-G40/Code/omniclaude/.env --execute

.. versionadded:: 0.10.0
    Created as part of OMN-2287 Infisical migration.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from uuid import UUID

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("register-repo")

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_ADMIN_TOKEN_FILE = _PROJECT_ROOT / ".infisical-admin-token"
_BOOTSTRAP_ENV = Path.home() / ".omnibase" / ".env"

sys.path.insert(0, str(_PROJECT_ROOT / "src"))

# Shared utility — avoids duplicating the parser in every Infisical script.
# Insert the scripts dir so the import resolves when run from any cwd.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _infisical_util import _parse_env_file

# ---------------------------------------------------------------------------
# Keys that are NEVER seeded into Infisical (circular bootstrap dependency).
# These must come from the environment / .env file directly.
# ---------------------------------------------------------------------------
BOOTSTRAP_KEYS = frozenset(
    {
        "POSTGRES_PASSWORD",
        "INFISICAL_ADDR",
        "INFISICAL_CLIENT_ID",
        "INFISICAL_CLIENT_SECRET",
        "INFISICAL_PROJECT_ID",
        "INFISICAL_ENCRYPTION_KEY",
        "INFISICAL_AUTH_SECRET",
    }
)

# ---------------------------------------------------------------------------
# Platform-wide shared secrets and their Infisical paths.
# Keys not in any node contract are declared here explicitly.
#
# TODO (Task 6): This hardcoded dict is a transitional placeholder. It should
# be migrated to config/shared_key_registry.yaml once that registry file is
# implemented as part of the OMN-2287 plan. The script should load this
# mapping from YAML rather than maintaining it inline here. Until that file
# exists, keep this dict in sync with the platform's actual secret layout.
# ---------------------------------------------------------------------------
SHARED_PLATFORM_SECRETS: dict[str, list[str]] = {
    "/shared/db/": [
        "POSTGRES_HOST",
        "POSTGRES_PORT",
        "POSTGRES_USER",
        "POSTGRES_DSN",
        "POSTGRES_POOL_MIN",
        "POSTGRES_POOL_MAX",
        "POSTGRES_TIMEOUT_MS",
    ],
    "/shared/kafka/": [
        "KAFKA_BOOTSTRAP_SERVERS",
        "KAFKA_HOST_SERVERS",
        "KAFKA_REQUEST_TIMEOUT_MS",
    ],
    "/shared/consul/": [
        "CONSUL_HOST",
        "CONSUL_PORT",
        "CONSUL_SCHEME",
        "CONSUL_ACL_TOKEN",
        "CONSUL_ENABLED",
    ],
    "/shared/vault/": [
        "VAULT_ADDR",
        "VAULT_TOKEN",
    ],
    "/shared/llm/": [
        "REMOTE_SERVER_IP",
        "LLM_CODER_URL",
        "LLM_CODER_FAST_URL",
        "LLM_EMBEDDING_URL",
        "LLM_DEEPSEEK_R1_URL",
        "EMBEDDING_MODEL_URL",
        "VLLM_SERVICE_URL",
        "VLLM_DEEPSEEK_URL",
        "VLLM_LLAMA_URL",
        "ONEX_TREE_SERVICE_URL",
        "METADATA_STAMPING_SERVICE_URL",
    ],
    "/shared/auth/": [
        "OPENAI_API_KEY",
        "GOOGLE_API_KEY",
        "GEMINI_API_KEY",
        "Z_AI_API_KEY",
        "Z_AI_API_URL",
        "SERVICE_AUTH_TOKEN",
        "GH_PAT",
    ],
    "/shared/valkey/": [
        "VALKEY_PASSWORD",
    ],
    "/shared/env/": [
        "SLACK_WEBHOOK_URL",
        "SLACK_BOT_TOKEN",
        "SLACK_CHANNEL_ID",
    ],
}

# Per-repo folders to create under /services/<repo>/
REPO_TRANSPORT_FOLDERS = ("db", "kafka", "env")

# Per-repo keys to seed (sourced from repo .env)
REPO_SECRET_KEYS = [
    "POSTGRES_DATABASE",
    "POSTGRES_DSN",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_infisical_adapter() -> tuple[object, object]:
    """Load and initialise the Infisical adapter using env credentials.

    Returns (adapter, sanitize_fn).
    """
    from pydantic import SecretStr

    from omnibase_infra.adapters._internal.adapter_infisical import AdapterInfisical
    from omnibase_infra.adapters.models.model_infisical_config import (
        ModelInfisicalAdapterConfig,
    )
    from omnibase_infra.utils.util_error_sanitization import sanitize_error_message

    infisical_addr = os.environ.get("INFISICAL_ADDR", "http://localhost:8880")
    client_id = os.environ.get("INFISICAL_CLIENT_ID", "")
    client_secret = os.environ.get("INFISICAL_CLIENT_SECRET", "")
    project_id = os.environ.get("INFISICAL_PROJECT_ID", "")

    if not all([client_id, client_secret, project_id]):
        logger.error(
            "Missing Infisical credentials. "
            "Ensure INFISICAL_CLIENT_ID, INFISICAL_CLIENT_SECRET, "
            "INFISICAL_PROJECT_ID are set (via ~/.omnibase/.env or shell env)."
        )
        raise SystemExit(1)

    try:
        project_uuid = UUID(project_id)
    except ValueError:
        raise SystemExit(
            f"ERROR: INFISICAL_PROJECT_ID is not a valid UUID: {project_id!r}\n"
            "Check the INFISICAL_PROJECT_ID value in ~/.omnibase/.env or your shell environment.\n"
            "The expected format is: xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx"
        ) from None
    config = ModelInfisicalAdapterConfig(
        host=infisical_addr,
        client_id=SecretStr(client_id),
        client_secret=SecretStr(client_secret),
        project_id=project_uuid,
    )
    adapter = AdapterInfisical(config)
    adapter.initialize()
    return adapter, sanitize_error_message


def _create_folders_via_admin(
    addr: str,
    token: str,
    project_id: str,
    path_prefix: str,
    folder_names: list[str],
    environments: tuple[str, ...] = ("dev", "staging", "prod"),
) -> None:
    """Create folder structure in Infisical using admin token (httpx)."""
    import httpx

    headers = {"Authorization": f"Bearer {token}"}
    with httpx.Client(timeout=30) as client:
        for env in environments:
            # Ensure parent exists first
            parts = path_prefix.strip("/").split("/")
            current = "/"
            for part in parts:
                if not part:
                    continue
                part_resp = client.post(
                    f"{addr}/api/v1/folders",
                    headers=headers,
                    json={
                        "workspaceId": project_id,
                        "environment": env,
                        "name": part,
                        "path": current,
                    },
                )
                if part_resp.status_code not in (200, 201, 400, 409):
                    part_resp.raise_for_status()
                current = (
                    f"{current}{part}/" if current == "/" else f"{current}/{part}/"
                )

            for folder in folder_names:
                resp = client.post(
                    f"{addr}/api/v1/folders",
                    headers=headers,
                    json={
                        "workspaceId": project_id,
                        "environment": env,
                        "name": folder,
                        "path": path_prefix.rstrip("/") or "/",
                    },
                )
                if resp.status_code not in (200, 201, 400, 409):
                    resp.raise_for_status()
    logger.info(
        "Folders created: %s/[%s] in %s",
        path_prefix,
        ", ".join(folder_names),
        list(environments),
    )


def _upsert_secret(
    adapter: object,
    key: str,
    value: str,
    folder: str,
    *,
    overwrite: bool,
    sanitize: object,
) -> str:
    """Create or update a secret. Returns 'created', 'updated', or 'skipped'."""
    existing = None
    try:
        existing = adapter.get_secret(secret_name=key, secret_path=folder)  # type: ignore[attr-defined]
    except Exception:
        pass  # key doesn't exist yet

    if existing is not None:
        if not overwrite:
            return "skipped"
        adapter.update_secret(  # type: ignore[attr-defined]
            secret_name=key,
            secret_path=folder,
            secret_value=value,
        )
        return "updated"

    adapter.create_secret(  # type: ignore[attr-defined]
        secret_name=key,
        secret_path=folder,
        secret_value=value,
    )
    return "created"


# ---------------------------------------------------------------------------
# seed-shared subcommand
# ---------------------------------------------------------------------------


def cmd_seed_shared(args: argparse.Namespace) -> int:
    """Populate /shared/ paths from the platform .env file."""
    env_path = Path(args.env_file).expanduser()
    env_values = _parse_env_file(env_path)

    if not env_values:
        logger.error("No values found in %s", env_path)
        return 1

    # Build the work list, skipping bootstrap keys
    plan: list[tuple[str, str, str]] = []  # (folder, key, value)
    missing_value: list[tuple[str, str]] = []  # (folder, key) with no value

    for folder, keys in SHARED_PLATFORM_SECRETS.items():
        for key in keys:
            if key in BOOTSTRAP_KEYS:
                continue
            value = env_values.get(key, "")
            if value:
                plan.append((folder, key, value))
            else:
                missing_value.append((folder, key))

    print(f"\n=== seed-shared (env: {env_path}) ===")
    print(f"  {len(plan)} keys with values to write")
    print(f"  {len(missing_value)} keys with no value (will create empty slots)")

    if missing_value:
        print("\n  Keys with no value (empty placeholders):")
        for folder, key in missing_value:
            print(f"    {folder}{key}")

    print("\n  Keys to seed:")
    for folder, key, value in sorted(plan):
        if len(value) > 4:
            display = value[:4] + "..."
        elif value:
            display = "***"
        else:
            display = "(empty)"
        print(f"    {folder}{key} = {display}")

    if not args.execute:
        print("\n[dry-run] Pass --execute to write to Infisical.")
        return 0

    print("\nWriting to Infisical...")
    try:
        adapter, sanitize = _load_infisical_adapter()
    except SystemExit as e:
        return e.code or 1

    counts = {"created": 0, "updated": 0, "skipped": 0, "error": 0}

    try:
        for folder, key, value in plan:
            try:
                outcome = _upsert_secret(
                    adapter,
                    key,
                    value,
                    folder,
                    overwrite=args.overwrite,
                    sanitize=sanitize,
                )
                counts[outcome] += 1
                logger.info("  [%s] %s%s", outcome.upper(), folder, key)
            except Exception as exc:
                counts["error"] += 1
                logger.warning("  [ERROR] %s%s: %s", folder, key, sanitize(exc))  # type: ignore[operator]

        # Also create empty placeholders for keys with no value
        for folder, key in missing_value:
            try:
                outcome = _upsert_secret(
                    adapter,
                    key,
                    "",
                    folder,
                    overwrite=False,
                    sanitize=sanitize,
                )
                counts[outcome] += 1
            except Exception as exc:
                counts["error"] += 1
                logger.warning(
                    "  [ERROR placeholder] %s%s: %s", folder, key, sanitize(exc)
                )  # type: ignore[operator]
    finally:
        adapter.shutdown()  # type: ignore[attr-defined]

    print(
        f"\nDone: {counts['created']} created, {counts['updated']} updated, "
        f"{counts['skipped']} skipped, {counts['error']} errors"
    )
    return 1 if counts["error"] else 0


# ---------------------------------------------------------------------------
# onboard-repo subcommand
# ---------------------------------------------------------------------------


def cmd_onboard_repo(args: argparse.Namespace) -> int:
    """Create /services/<repo>/ folder structure and seed repo-specific secrets."""
    import re

    repo_name = args.repo
    # Reject names that could be used for path traversal or produce invalid
    # Infisical paths.  Allow only alphanumeric characters, hyphens, and
    # underscores (e.g. "omniclaude", "omni-bridge", "my_repo").
    if not re.fullmatch(r"[A-Za-z0-9_-]+", repo_name):
        raise SystemExit(
            f"ERROR: Invalid repo name '{repo_name}'. "
            "Only alphanumeric characters, hyphens (-), and underscores (_) are allowed. "
            "Slashes, dots, and other path characters are not permitted."
        )

    env_path = Path(args.env_file).expanduser()
    if not env_path.is_file():
        raise SystemExit(
            f"ERROR: env file not found: {env_path}\n"
            "Provide a valid path via --env-file."
        )
    env_values = _parse_env_file(env_path)

    infisical_addr = os.environ.get("INFISICAL_ADDR", "http://localhost:8880")
    project_id = os.environ.get("INFISICAL_PROJECT_ID", "")
    if not project_id:
        raise SystemExit(
            "ERROR: INFISICAL_PROJECT_ID is not set. "
            "Set it in your environment or ~/.omnibase/.env before running onboard-repo. "
            "You can find the project ID after running scripts/provision-infisical.py."
        )
    path_prefix = f"/services/{repo_name}"

    # Identify repo-specific secrets to seed
    plan: list[tuple[str, str, str]] = []
    for key in REPO_SECRET_KEYS:
        value = env_values.get(key, "")
        plan.append((f"{path_prefix}/db/", key, value))

    # Any extra keys in the env file that are NOT in shared and NOT bootstrap
    shared_keys_flat = {k for keys in SHARED_PLATFORM_SECRETS.values() for k in keys}
    extra: list[tuple[str, str, str]] = []
    for key, value in env_values.items():
        if key in BOOTSTRAP_KEYS:
            continue
        if key in shared_keys_flat:
            continue
        if any(key == pk for _, pk, _ in plan):
            continue
        extra.append((f"{path_prefix}/env/", key, value))

    print(f"\n=== onboard-repo: {repo_name} ===")
    print(f"  Infisical path: {path_prefix}/")
    print(f"  Env file: {env_path}")
    print("\n  Repo-specific keys:")
    for folder, key, value in plan:
        if value and len(value) > 4:
            display = value[:4] + "..."
        elif value:
            display = "***"
        else:
            display = "(empty)"
        print(f"    {folder}{key} = {display}")

    if extra:
        print(f"\n  Additional repo-only keys ({len(extra)}):")
        for folder, key, value in extra:
            if value and len(value) > 4:
                display = value[:4] + "..."
            elif value:
                display = "***"
            else:
                display = "(empty)"
            print(f"    {folder}{key} = {display}")

    if not args.execute:
        print("\n[dry-run] Pass --execute to create folders and write secrets.")
        return 0

    # Need admin token to create folders
    if not _ADMIN_TOKEN_FILE.is_file():
        logger.error(
            "Admin token not found at %s. Run scripts/provision-infisical.py first.",
            _ADMIN_TOKEN_FILE,
        )
        return 1

    with _ADMIN_TOKEN_FILE.open() as f:
        admin_token = f.readline().strip()

    print(f"\nCreating folder structure at {path_prefix}/...")
    _create_folders_via_admin(
        infisical_addr,
        admin_token,
        project_id,
        "/services/",
        [repo_name],
    )
    _create_folders_via_admin(
        infisical_addr,
        admin_token,
        project_id,
        f"{path_prefix}/",
        list(REPO_TRANSPORT_FOLDERS),
    )

    print("Seeding repo-specific secrets...")
    try:
        adapter, sanitize = _load_infisical_adapter()
    except SystemExit as e:
        return e.code or 1

    all_secrets = plan + extra
    counts = {"created": 0, "updated": 0, "skipped": 0, "error": 0}

    try:
        for folder, key, value in all_secrets:
            try:
                outcome = _upsert_secret(
                    adapter,
                    key,
                    value,
                    folder,
                    overwrite=args.overwrite,
                    sanitize=sanitize,
                )
                counts[outcome] += 1
                logger.info("  [%s] %s%s", outcome.upper(), folder, key)
            except Exception as exc:
                counts["error"] += 1
                logger.warning("  [ERROR] %s%s: %s", folder, key, sanitize(exc))  # type: ignore[operator]
    finally:
        adapter.shutdown()  # type: ignore[attr-defined]

    print(
        f"\nDone: {counts['created']} created, {counts['updated']} updated, "
        f"{counts['skipped']} skipped, {counts['error']} errors"
    )

    print(f"\nRepo '{repo_name}' is onboarded.")
    print("Its .env only needs:")
    print(f"  POSTGRES_DATABASE={repo_name.replace('-', '_')}")
    print("  (Infisical creds come from ~/.omnibase/.env via shell env)")
    return 1 if counts["error"] else 0


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # seed-shared
    p_shared = sub.add_parser(
        "seed-shared",
        help="Populate /shared/ paths in Infisical from platform .env",
    )
    p_shared.add_argument(
        "--env-file",
        default=str(_BOOTSTRAP_ENV),
        help=f"Path to platform .env (default: {_BOOTSTRAP_ENV})",
    )
    p_shared.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing Infisical values (default: skip existing)",
    )
    p_shared.add_argument(
        "--execute",
        action="store_true",
        help="Write to Infisical (default: dry-run)",
    )

    # onboard-repo
    p_repo = sub.add_parser(
        "onboard-repo",
        help="Create /services/<repo>/ folders and seed repo-specific secrets",
    )
    p_repo.add_argument("--repo", required=True, help="Repo name (e.g. omniclaude)")
    p_repo.add_argument(
        "--env-file",
        required=True,
        help="Path to the repo's .env file",
    )
    p_repo.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing Infisical values (default: skip existing)",
    )
    p_repo.add_argument(
        "--execute",
        action="store_true",
        help="Write to Infisical (default: dry-run)",
    )

    args = parser.parse_args()

    if args.command == "seed-shared":
        return cmd_seed_shared(args)
    if args.command == "onboard-repo":
        return cmd_onboard_repo(args)

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
