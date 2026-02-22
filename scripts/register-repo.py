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

import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("register-repo")

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_REGISTRY_PATH = _PROJECT_ROOT / "config" / "shared_key_registry.yaml"
_ADMIN_TOKEN_FILE = _PROJECT_ROOT / ".infisical-admin-token"
_BOOTSTRAP_ENV = Path.home() / ".omnibase" / ".env"

sys.path.insert(0, str(_PROJECT_ROOT / "src"))

# Shared utility — avoids duplicating the parser in every Infisical script.
# Insert the scripts dir so the import resolves when run from any cwd.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _infisical_util import _parse_env_file

# ---------------------------------------------------------------------------
# Registry loaders — load key lists from config/shared_key_registry.yaml.
# ---------------------------------------------------------------------------


def _read_registry_data() -> dict[str, object]:
    """Open and parse config/shared_key_registry.yaml.

    Returns the raw parsed dict from the YAML file.  Command functions
    (``cmd_seed_shared``, ``cmd_onboard_repo``) call this once and pass the
    result to ``_load_registry``, ``_bootstrap_keys``, and
    ``_identity_defaults`` via their ``data`` parameter so the file is only
    read a single time per invocation.
    """
    registry_path = _REGISTRY_PATH
    if not registry_path.exists():
        raise FileNotFoundError(f"Registry not found: {registry_path}")
    with open(registry_path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if data is None or not isinstance(data, dict):
        raise ValueError(
            f"Registry file is empty or not a YAML mapping: {registry_path}"
        )
    return data  # type: ignore[return-value]  # yaml.safe_load returns Any; runtime isinstance guard above ensures dict


def _load_registry(
    data: dict[str, object] | None = None,
) -> dict[str, list[str]]:
    """Load shared platform secrets from config/shared_key_registry.yaml.

    Returns a mapping of ``{infisical_folder_path: [key, ...]}`` identical in
    shape to the former ``SHARED_PLATFORM_SECRETS`` dict.

    Args:
        data: Pre-loaded registry dict from :func:`_read_registry_data`.  When
            provided the file is not re-read; when omitted the file is read
            once inside this function.
    """
    if data is None:
        data = _read_registry_data()
    shared = data.get("shared")
    if shared is None:
        raise ValueError(f"Registry missing 'shared' section: {_REGISTRY_PATH}")
    if not isinstance(shared, dict):
        raise ValueError(
            f"Expected 'shared' in {_REGISTRY_PATH} to be a mapping, "
            f"got {type(shared).__name__!r}. Check that 'shared:' is not null or a list."
        )
    for folder, keys in shared.items():
        if not isinstance(keys, list):
            raise ValueError(
                f"Expected 'shared.{folder}' in {_REGISTRY_PATH} to be a list, "
                f"got {type(keys).__name__!r}."
            )
        if not keys:
            raise ValueError(
                f"Folder '{folder}' has an empty key list in registry — "
                "this is likely an authoring error"
            )
        if not all(isinstance(k, str) for k in keys):
            raise ValueError(
                f"[ERROR] registry 'shared.{folder}' must be a list of strings in {_REGISTRY_PATH}"
            )
    return shared


def _bootstrap_keys(
    data: dict[str, object] | None = None,
) -> frozenset[str]:
    """Load bootstrap-only keys from registry.

    These keys must never be written to Infisical (circular bootstrap
    dependency — Infisical needs them to start).

    Args:
        data: Pre-loaded registry dict from :func:`_read_registry_data`.  When
            provided the file is not re-read; when omitted the file is read
            once inside this function.
    """
    if data is None:
        data = _read_registry_data()
    if "bootstrap_only" not in data:
        raise ValueError(f"Registry missing 'bootstrap_only' section: {_REGISTRY_PATH}")
    keys = data["bootstrap_only"]
    if not isinstance(keys, list):
        raise ValueError(
            f"[ERROR] registry 'bootstrap_only' must be a list in {_REGISTRY_PATH}"
        )
    if not all(isinstance(k, str) for k in keys):
        raise ValueError(
            f"[ERROR] registry 'bootstrap_only' entries must be strings in {_REGISTRY_PATH}"
        )
    return frozenset(keys)


def _identity_defaults(
    data: dict[str, object] | None = None,
) -> frozenset[str]:
    """Load identity-default keys from registry.

    These keys are baked into each repo's Settings class as ``default=`` and
    must NOT be seeded into Infisical.

    Args:
        data: Pre-loaded registry dict from :func:`_read_registry_data`.  When
            provided the file is not re-read; when omitted the file is read
            once inside this function.
    """
    if data is None:
        data = _read_registry_data()
    if "identity_defaults" not in data:
        raise ValueError(
            f"Registry missing 'identity_defaults' section: {_REGISTRY_PATH}"
        )
    keys = data["identity_defaults"]
    if not isinstance(keys, list):
        raise ValueError(
            f"[ERROR] registry 'identity_defaults' must be a list in {_REGISTRY_PATH}"
        )
    if not all(isinstance(k, str) for k in keys):
        raise ValueError(
            f"[ERROR] registry 'identity_defaults' entries must be strings in {_REGISTRY_PATH}"
        )
    return frozenset(keys)


# Per-repo folders to create under /services/<repo>/
REPO_TRANSPORT_FOLDERS = ("db", "kafka", "env")

# Per-repo keys to seed (sourced from repo .env).
# NOTE: POSTGRES_DATABASE is intentionally excluded — it is an identity_default
# (hardcoded as a Settings class default per repo) and must NOT be seeded into
# Infisical.
REPO_SECRET_KEYS = [
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
                current = f"{current}{part}/"

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
    except Exception as _get_exc:
        # Only suppress the error when the secret genuinely does not exist yet.
        # Re-raise for any exception that indicates a connection problem, auth
        # failure, or other infrastructure error — those must not be silently
        # swallowed, because the subsequent create_secret call will also fail
        # and the root cause will be lost.
        #
        # The Infisical adapter wraps all get_secret failures (including 404s)
        # as SecretResolutionError.  We distinguish "not found" from "real
        # error" by inspecting the exception message and cause chain.
        #
        # WORKAROUND: The Infisical Python SDK does not expose typed error
        # codes (e.g. an HTTP status attribute or a structured exception
        # subclass) that would let us cleanly identify a 404 "secret not
        # found" response without parsing message strings.  The string
        # patterns below ("not found", "404", "does not exist") are
        # therefore a best-effort heuristic that may break if the SDK
        # changes its error message wording in a future release.
        #
        # TODO: Replace this heuristic with proper error code inspection
        # once the Infisical SDK exposes typed status codes or a dedicated
        # SecretNotFoundError subclass.  Track against the SDK changelog
        # (https://github.com/Infisical/infisical-python) and remove the
        # string-matching blocks when a stable typed API is available.
        err_msg = str(_get_exc).lower()
        is_not_found = (
            "not found" in err_msg
            or "404" in err_msg
            or "does not exist" in err_msg
            or "secret not found" in err_msg
        )
        if not is_not_found:
            # Check the cause chain — the raw SDK exception may carry a more
            # informative status code or message.
            cause = getattr(_get_exc, "__cause__", None)
            if cause is not None:
                cause_msg = str(cause).lower()
                is_not_found = (
                    "not found" in cause_msg
                    or "404" in cause_msg
                    or "does not exist" in cause_msg
                )
        if not is_not_found:
            raise  # Re-raise: connection/auth/unexpected error, not a missing key

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
    if not env_path.is_file():
        print(f"ERROR: Env file not found: {env_path}", file=sys.stderr)
        raise SystemExit(1)
    env_values = _parse_env_file(env_path)

    if not env_values:
        logger.error("No values found in %s", env_path)
        return 1

    # Build the work list, skipping bootstrap keys
    plan: list[tuple[str, str, str]] = []  # (folder, key, value)
    missing_value: list[tuple[str, str]] = []  # (folder, key) with no value

    registry_data = _read_registry_data()
    shared_secrets = _load_registry(registry_data)
    bootstrap = _bootstrap_keys(registry_data)
    identity = _identity_defaults(registry_data)
    for folder, keys in shared_secrets.items():
        for key in keys:
            if key in bootstrap:
                continue
            if key in identity:
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
        display = "***" if value else "(empty)"
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
    if not infisical_addr or not infisical_addr.startswith(("http://", "https://")):
        print(
            f"ERROR: INFISICAL_ADDR is not a valid URL: {infisical_addr!r}\n"
            "It must start with http:// or https:// (e.g. http://localhost:8880).",
            file=sys.stderr,
        )
        raise SystemExit(1)
    project_id = os.environ.get("INFISICAL_PROJECT_ID", "")
    if not project_id:
        raise SystemExit(
            "ERROR: INFISICAL_PROJECT_ID is not set. "
            "Set it in your environment or ~/.omnibase/.env before running onboard-repo. "
            "You can find the project ID after running scripts/provision-infisical.py."
        )
    path_prefix = f"/services/{repo_name}"

    # Identify repo-specific secrets to seed.
    # Keys missing from the env file (e.g. POSTGRES_DSN, which is assembled and
    # written by the runtime after the DB is reachable) are intentionally seeded
    # as empty strings — they reserve the Infisical slot so the runtime can
    # update_secret without a prior create step.
    plan: list[tuple[str, str, str]] = []
    for key in REPO_SECRET_KEYS:
        value = env_values.get(key, "")
        plan.append((f"{path_prefix}/db/", key, value))

    # Any extra keys in the env file that are NOT in shared, NOT bootstrap,
    # and NOT an identity default (per-repo value baked into Settings.default=).
    registry_data = _read_registry_data()
    shared_keys_flat = {
        k for keys in _load_registry(registry_data).values() for k in keys
    }
    bootstrap = _bootstrap_keys(registry_data)
    identity = _identity_defaults(registry_data)
    extra: list[tuple[str, str, str]] = []
    for key, value in env_values.items():
        if key in bootstrap:
            continue
        if key in identity:
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
        display = "***" if value else "(empty)"
        print(f"    {folder}{key} = {display}")

    if extra:
        print(f"\n  Additional repo-only keys ({len(extra)}):")
        for folder, key, value in extra:
            display = "***" if value else "(empty)"
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
    if not admin_token:
        print(f"ERROR: Admin token file is empty: {_ADMIN_TOKEN_FILE}", file=sys.stderr)
        raise SystemExit(1)

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
    print(
        f"  POSTGRES_DATABASE={repo_name.replace('-', '_')}  # suggested value — verify this matches your actual .env"
    )
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
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Note: after onboarding, a suggested POSTGRES_DATABASE value is printed "
            "as the repo name with hyphens replaced by underscores "
            "(e.g. 'my-repo' → 'my_repo'). "
            "Verify this matches the actual database name before using it."
        ),
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
