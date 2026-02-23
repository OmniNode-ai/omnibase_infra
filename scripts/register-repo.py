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

~/.omnibase/.env contains bootstrap credentials plus shared platform keys.
Bootstrap-only lines (circular Infisical dependency — must stay in .env):
    POSTGRES_PASSWORD=...
    INFISICAL_ADDR=http://localhost:8880
    INFISICAL_CLIENT_ID=...
    INFISICAL_CLIENT_SECRET=...
    INFISICAL_PROJECT_ID=...

All other platform-wide configuration lives in Infisical under /shared/*.

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
import re
import sys
from collections.abc import Callable
from pathlib import Path
from typing import cast
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

# Auth-related substrings that indicate a server-rejected request.
# Used in _upsert_secret to distinguish "secret not found" (recoverable)
# from auth/connection failures (must re-raise).
# NOTE: bare "auth" is intentionally excluded — it is a 4-letter substring
# that matches false positives like "OAuth", "path not found in database auth
# schema", etc.  The specific phrases below are sufficient.
_AUTH_INDICATORS = (
    "unauthorized",
    "forbidden",
    "invalid token",
    "expired token",
    "authentication failed",
    "access denied",
)

# Substrings (case-insensitive) that indicate a key holds sensitive material.
# In dry-run output, values for matching keys are shown as "***" to avoid
# leaking credentials to the terminal.  Non-matching keys show their actual
# value so operators can verify the correct config will be seeded.
_SENSITIVE_KEY_PATTERNS = frozenset(
    {
        "PASSWORD",
        "SECRET",
        # "_KEY" matches keys where KEY appears as a word segment (ENCRYPTION_KEY,
        # REDIS_KEY, SIGNING_KEY, etc.) but does NOT match VALKEY_HOST/PORT/DB
        # where "KEY" is an interior substring of the word "VALKEY".
        "_KEY",
        "TOKEN",
        "CREDENTIAL",
        # "_AUTH" matches keys where AUTH appears as an interior segment
        # (INFISICAL_AUTH_SECRET, VAULT_AUTH_TOKEN, SERVICE_AUTH_TOKEN, etc.)
        # but does NOT match keys where AUTH is a leading word
        # (e.g. AUTH_PROXY_URL) or an interior substring of another word
        # (e.g. OAUTH_CLIENT_ID). Contrast with _AUTH_INDICATORS above, where
        # bare "auth" was intentionally excluded to avoid false positives in
        # error message classification.
        "_AUTH",
        "CERT",
        "PEM",
        "_PAT",
        "WEBHOOK",
    }
)

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
    return data  # type: ignore[no-any-return]  # yaml.safe_load returns Any; runtime isinstance guard above ensures dict, not dict[str, list[str]] — inner types are validated by the _load_registry loop


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
        # A null YAML value (e.g. "consul:" with no items) is parsed by
        # yaml.safe_load as None.  The isinstance(keys, list) guard below
        # catches NoneType before the empty-list check and raises ValueError
        # with 'got NoneType', so null folder entries are treated as an
        # invalid type rather than silently skipped.
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
    return cast("dict[str, list[str]]", shared)


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
    if not keys:
        raise ValueError(
            "bootstrap_only section is empty in shared_key_registry.yaml — "
            "this would allow bootstrap credentials (POSTGRES_PASSWORD, etc.) "
            "to be seeded into Infisical. Add the bootstrap-only keys or remove the section."
        )
    if not all(isinstance(k, str) for k in keys):
        raise ValueError(
            f"[ERROR] registry 'bootstrap_only' entries must be strings in {_REGISTRY_PATH}"
        )
    result = frozenset(keys)
    if "POSTGRES_PASSWORD" not in result:
        raise ValueError(
            "POSTGRES_PASSWORD must be in bootstrap_only — check shared_key_registry.yaml"
        )
    return result


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
    if not keys:
        raise ValueError(
            "identity_defaults section is empty in shared_key_registry.yaml — "
            "at least one identity default key (e.g. POSTGRES_DATABASE) is required."
        )
    if not all(isinstance(k, str) for k in keys):
        raise ValueError(
            f"[ERROR] registry 'identity_defaults' entries must be strings in {_REGISTRY_PATH}"
        )
    return frozenset(keys)


def _service_override_required(
    data: dict[str, object] | None = None,
) -> frozenset[str]:
    """Load keys from the ``service_override_required`` section of the registry.

    These keys are present in ``/shared/`` as platform-wide defaults but MUST
    be overridden per-service under ``/services/<repo>/<transport>/<KEY>``
    before the service starts.  The ``onboard-repo`` command warns if any of
    these keys are absent from the onboarding plan.

    Args:
        data: Pre-loaded registry dict from :func:`_read_registry_data`.  When
            provided the file is not re-read; when omitted the file is read
            once inside this function.
    """
    if data is None:
        data = _read_registry_data()
    if "service_override_required" not in data:
        # Section is optional — older registry files without it are valid.
        return frozenset()
    keys = data["service_override_required"]
    if not isinstance(keys, list):
        raise ValueError(
            f"[ERROR] registry 'service_override_required' must be a list in {_REGISTRY_PATH}"
        )
    if not keys:
        # An empty list is treated the same as an absent section — the registry
        # was intentionally cleaned up and no overrides are currently required.
        # Only non-list/non-None types are invalid.
        return frozenset()
    if not all(isinstance(k, str) for k in keys):
        raise ValueError(
            f"[ERROR] registry 'service_override_required' entries must be strings in {_REGISTRY_PATH}"
        )
    return frozenset(keys)


# Per-repo folders to create under /services/<repo>/
REPO_TRANSPORT_FOLDERS = ("db", "kafka", "env")

# Per-repo keys to seed (sourced from repo .env).
# NOTE: POSTGRES_DATABASE is intentionally excluded — it is an identity_default
# (hardcoded as a Settings class default per repo) and must NOT be seeded into
# Infisical.
# NOTE: POSTGRES_DSN is intentionally excluded — a composite DSN with a
# hardcoded database name silently routes all services to the same database.
# Each service uses POSTGRES_DATABASE (identity default) combined with the
# shared POSTGRES_HOST / POSTGRES_PORT / POSTGRES_USER / POSTGRES_PASSWORD
# keys from /shared/db/ instead.
#
# CONSTRAINT: All keys added here MUST be DATABASE transport keys only.
# The loop in cmd_onboard_repo that iterates over REPO_SECRET_KEYS hardcodes
# the destination path as /services/<repo>/db/ (see: plan.append(..."/db/"...)).
# Adding non-DB keys here (e.g. Kafka, HTTP, or LLM keys) would silently store
# them under /db/ instead of their correct transport path. Non-DB repo-specific
# keys should be handled via the `extra` list in cmd_onboard_repo (which buckets
# them under /services/<repo>/env/) or added to shared_key_registry.yaml.
REPO_SECRET_KEYS: list[str] = []


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_infisical_adapter() -> tuple[object, Callable[[Exception], str]]:
    """Load and initialise the Infisical adapter using env credentials.

    Returns (adapter, sanitize_fn).

    Note:
        INFISICAL_ADDR is validated at two layers (defense-in-depth): command
        entry points (cmd_seed_shared and cmd_onboard_repo) both check for
        presence and a valid scheme before calling this function, AND this
        function repeats those checks to guard callers that bypass the
        entry-point pre-flight.
    """
    from pydantic import SecretStr

    from omnibase_infra.adapters._internal.adapter_infisical import AdapterInfisical
    from omnibase_infra.adapters.models.model_infisical_config import (
        ModelInfisicalAdapterConfig,
    )
    from omnibase_infra.utils.util_error_sanitization import sanitize_error_message

    infisical_addr = os.environ.get("INFISICAL_ADDR", "")
    if not infisical_addr:
        print(
            "Error: INFISICAL_ADDR is not set. "
            "Set it to the Infisical URL (e.g. http://localhost:8880) in your environment "
            "or ~/.omnibase/.env before calling this function.",
            file=sys.stderr,
        )
        raise SystemExit(1)
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

    # Defense-in-depth: command entry points also validate both INFISICAL_ADDR and
    # INFISICAL_PROJECT_ID before calling here. These guards protect callers that
    # bypass the entry-point pre-flight.
    if not infisical_addr.startswith(("http://", "https://")):
        logger.error(
            "INFISICAL_ADDR must start with http:// or https://: got %r", infisical_addr
        )
        raise SystemExit(1)

    # Both command entry points pre-validate INFISICAL_PROJECT_ID UUID format and return 1
    # before reaching here. This re-validates as defense-in-depth for non-command callers;
    # it produces a SystemExit (not return 1) for callers that bypass entry-point validation.
    try:
        project_uuid = UUID(project_id)
    except ValueError:
        # Suppress the ValueError cause chain — this is a user-input validation error and
        # the internal ValueError detail (from UUID.__init__) is not useful to the operator.
        # Consistent with the entry-point validation style (logger.error + return 1) which
        # also does not propagate the ValueError.
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
    try:
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
                    # 409 = folder already exists (idempotent). 400 = bad request — surfaces as real error.
                    if part_resp.status_code not in (200, 201, 409):
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
                    # 409 = folder already exists (idempotent). 400 = bad request — surfaces as real error.
                    if resp.status_code not in (200, 201, 409):
                        resp.raise_for_status()
    except httpx.HTTPError as e:
        logger.exception(
            "Failed to create Infisical folder '%s': %s",
            path_prefix,
            e,
        )
        raise SystemExit(1) from e
    except Exception as e:
        logger.exception(
            "Unexpected error creating Infisical folder '%s': %s",
            path_prefix,
            e,
        )
        raise SystemExit(1) from e
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
) -> str:
    """Create or update a secret. Returns 'created', 'updated', or 'skipped'.

    Note:
        Error-message sanitization is the caller's responsibility.  This
        function raises exceptions as-is so callers retain full control over
        how errors are formatted and logged via their own ``sanitize`` callable.
    """
    from omnibase_infra.errors import (
        InfraConnectionError,
        RuntimeHostError,
        SecretResolutionError,
    )

    existing = None
    try:
        existing = adapter.get_secret(secret_name=key, secret_path=folder)  # type: ignore[attr-defined]
    except RuntimeHostError as _get_exc:
        # The adapter may raise either:
        #   - A RuntimeHostError subclass that is NOT SecretResolutionError
        #     (e.g. InfraAuthenticationError, InfraTimeoutError,
        #     InfraUnavailableError).  These are infrastructure failures that
        #     the outer loop must handle — re-raise immediately so the caller
        #     can abort the seed run.  The auth-indicator string check below
        #     would be dead code for these typed errors without this guard.
        #   - SecretResolutionError (a RuntimeHostError subclass) for two cases:
        #       1. "Adapter not initialized" — programming error, re-raise.
        #       2. "Secret not found" — treat as None so we can create it.
        if not isinstance(_get_exc, SecretResolutionError):
            raise  # infrastructure failure — let the outer loop handle it

        # We are now handling SecretResolutionError only.
        #
        # Only suppress the error when the secret genuinely does not exist yet.
        # Re-raise for any exception that indicates a connection problem, auth
        # failure, or other infrastructure error — those must not be silently
        # swallowed, because the subsequent create_secret call will also fail
        # and the root cause will be lost.
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

        # Step 1: Check for auth indicators — always re-raise these immediately,
        # before attempting any "not found" classification.  Auth errors must
        # never be silently swallowed, regardless of what the message also says.
        err_msg = str(_get_exc).lower()
        has_auth_indicator = any(tok in err_msg for tok in _AUTH_INDICATORS)

        # Also inspect the cause chain: an auth error wrapping a 404-style
        # message must still propagate (e.g. SDK wraps HTTP 401 with a generic
        # "not found" outer message).
        cause = getattr(_get_exc, "__cause__", None)
        cause_msg = str(cause).lower() if cause is not None else ""
        cause_has_auth = (
            any(tok in cause_msg for tok in _AUTH_INDICATORS) if cause_msg else False
        )

        if has_auth_indicator or cause_has_auth:
            raise  # explicit: auth errors always propagate

        # Step 2: Determine if this is "secret not found" (only reached when
        # no auth indicator was detected above).
        is_not_found = (
            "not found" in err_msg or "404" in err_msg or "does not exist" in err_msg
        )
        if not is_not_found and cause_msg:
            # top-level wasn't "not found" — check if cause says "not found"
            is_not_found = (
                "not found" in cause_msg
                or "404" in cause_msg
                or "does not exist" in cause_msg
            )
        if not is_not_found:
            # Three cases reach this raise:
            #   1. "Adapter not initialized" — programming error (adapter.initialize() was not called).
            #   2. Auth/connection failure — not caught by the auth-indicator check above
            #      (e.g. SDK wraps the HTTP error without standard auth wording).
            #   3. Any other unexpected SecretResolutionError that is not "not found".
            # All three must propagate; only the "not found" branch (is_not_found=True) is safe to swallow.
            raise  # Re-raise: not-initialized / unexpected error
    except Exception as _bare_exc:
        # Bare SDK exception (not wrapped as RuntimeHostError).  Wrap it in
        # InfraConnectionError so the outer loop's _is_abort_error check
        # correctly classifies it as an infrastructure-level abort rather than
        # silently counting it as a per-key error.  Auth errors are preserved
        # via the cause chain and the outer loop's string-based fallback check.
        logger.debug(
            "Wrapping bare SDK exception from get_secret(%s, %s) as InfraConnectionError: %s",
            folder,
            key,
            _bare_exc,
        )
        raise InfraConnectionError(
            f"SDK raised unexpected error fetching secret {folder}{key}: {_bare_exc}"
        ) from _bare_exc

    if existing is not None:
        if not overwrite:
            return "skipped"
        # Write errors propagate naturally to the outer loop.
        adapter.update_secret(  # type: ignore[attr-defined]
            secret_name=key,
            secret_path=folder,
            secret_value=value,
        )
        return "updated"

    # Write errors propagate naturally to the outer loop.
    adapter.create_secret(  # type: ignore[attr-defined]
        secret_name=key,
        secret_path=folder,
        secret_value=value,
    )
    return "created"


def _is_abort_error(exc: Exception) -> bool:
    """Return True if *exc* is an infrastructure-level error that must abort a seed run.

    Two categories trigger an abort:

    1. ``RuntimeHostError`` (or any subclass) — typed infra failures raised by
       ``_upsert_secret`` for connection, timeout, auth, and unavailability errors.
       Any ``RuntimeHostError`` that reaches the outer loop was not suppressed
       inside ``_upsert_secret``, which means it is systemic.

    2. Auth-indicator strings — defence-in-depth for SDK exceptions that may not
       be typed as ``RuntimeHostError`` but whose message contains a token from
       ``_AUTH_INDICATORS`` (e.g. ``"unauthorized"``, ``"forbidden"``).

    Returns:
        ``True`` when the caller should re-raise *exc* and abort; ``False`` when
        the error is a per-key failure that can be counted and logged.
    """
    try:
        from omnibase_infra.errors import RuntimeHostError
    except ImportError:
        # omnibase_infra not installed (e.g. script run without uv sync).
        # Fall back to conservative check: only abort on explicit SystemExit.
        return isinstance(exc, SystemExit)

    if isinstance(exc, RuntimeHostError):
        return True
    err_msg = str(exc).lower()
    return any(indicator in err_msg for indicator in _AUTH_INDICATORS)


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

    try:
        registry_data = _read_registry_data()
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        raise SystemExit(1) from e
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        raise SystemExit(1) from e
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

    # NOTE: Intentionally validates BEFORE the dry-run gate (unlike cmd_onboard_repo).
    # This is a design choice for early failure: we always validate credentials even
    # for a preview run, so the operator knows immediately if the connection is
    # misconfigured before seeing the plan.
    #
    # The pre-flight checks below (INFISICAL_ADDR, INFISICAL_PROJECT_ID) will fail
    # fast even when --dry-run is passed — if credentials are missing, the command
    # exits non-zero before printing the plan. This is intentional: dry-run is meant
    # to preview what *would* be seeded, and that preview is only meaningful if the
    # operator has a valid Infisical configuration.
    #
    # Actual Infisical network calls (upsert, folder creation) are still skipped in
    # dry-run — only the credential presence/format is checked here, not connectivity.
    #
    # This differs from cmd_onboard_repo, which defers INFISICAL_ADDR validation to
    # the --execute path so dry-run works without a live Infisical instance. The
    # asymmetry is intentional — seed-shared is a platform-wide operation where an
    # early credential check is more valuable than dry-run accessibility without
    # credentials. Do not "fix" this to match cmd_onboard_repo.
    infisical_addr = os.environ.get("INFISICAL_ADDR")
    if not infisical_addr:
        print(
            "ERROR: INFISICAL_ADDR is not set. "
            "Set it to the Infisical URL before seeding.",
            file=sys.stderr,
        )
        raise SystemExit(1)
    if not infisical_addr.startswith(("http://", "https://")):
        print(
            f"ERROR: INFISICAL_ADDR is not a valid URL: {infisical_addr!r}\n"
            "It must start with http:// or https:// (e.g. http://localhost:8880).",
            file=sys.stderr,
        )
        raise SystemExit(1)
    project_id = os.environ.get("INFISICAL_PROJECT_ID", "")
    if not project_id:
        print(
            "ERROR: INFISICAL_PROJECT_ID is not set. "
            "Set it in your environment or ~/.omnibase/.env before running seed-shared. "
            "You can find the project ID after running scripts/provision-infisical.py.",
            file=sys.stderr,
        )
        raise SystemExit(1)
    try:
        UUID(project_id)
    except ValueError:
        # ValueError here is user input, not a system error — no stacktrace needed
        logger.error("INFISICAL_PROJECT_ID is not a valid UUID: %s", project_id)  # noqa: TRY400
        return 1

    print(f"\n=== seed-shared (env: {env_path}) ===")
    print(f"  {len(plan)} keys with values to write")
    print(f"  {len(missing_value)} keys with no value (will create empty slots)")

    if missing_value:
        print("\n  Keys with no value (empty placeholders):")
        for folder, key in missing_value:
            print(f"    {folder}{key}")

    print("\n  Keys to seed:")
    for folder, key, value in sorted(plan):
        key_upper = key.upper()
        is_sensitive = any(pat in key_upper for pat in _SENSITIVE_KEY_PATTERNS)
        if not value:
            display = "(empty)"
        elif is_sensitive:
            display = "***"
        else:
            display = value
        print(f"    {folder}{key} = {display}")

    if not args.execute:
        print("\n[dry-run] Pass --execute to write to Infisical.")
        return 0

    print("\nWriting to Infisical...")
    try:
        adapter, sanitize = _load_infisical_adapter()
    except SystemExit as e:
        # Integer exit codes (e.g. SystemExit(1)) are not printed — the preceding
        # logger.error already describes the failure. Only string messages add context.
        if isinstance(e.code, str):
            print(e.code, file=sys.stderr)
        return 1

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
                )
                counts[outcome] += 1
                logger.info("  [%s] %s%s", outcome.upper(), folder, key)
                if key == "KAFKA_GROUP_ID" and outcome in (
                    "created",
                    "updated",
                ):
                    logger.warning(
                        "  KAFKA_GROUP_ID seeded to /shared/kafka/ as a placeholder default.\n"
                        "  Each service MUST override this at /services/<repo>/kafka/KAFKA_GROUP_ID\n"
                        "  to avoid shared consumer group collisions."
                    )
            except Exception as exc:
                if _is_abort_error(exc):
                    logger.exception(
                        "Infrastructure error seeding %s%s — aborting: %s",
                        folder,
                        key,
                        sanitize(exc),
                    )
                    raise
                counts["error"] += 1
                logger.warning("  [ERROR] %s%s: %s", folder, key, sanitize(exc))

        # Also create empty placeholders for keys with no value
        for folder, key in missing_value:
            try:
                outcome = _upsert_secret(
                    adapter,
                    key,
                    "",
                    folder,
                    overwrite=False,
                )
                counts[outcome] += 1
                logger.info("  [%s] %s%s (placeholder)", outcome.upper(), folder, key)
            except Exception as exc:
                if _is_abort_error(exc):
                    logger.exception(
                        "Infrastructure error seeding %s%s — aborting: %s",
                        folder,
                        key,
                        sanitize(exc),
                    )
                    raise
                counts["error"] += 1
                logger.warning(
                    "  [ERROR placeholder] %s%s: %s", folder, key, sanitize(exc)
                )
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

    path_prefix = f"/services/{repo_name}"

    # Identify repo-specific secrets to seed.
    # Keys missing from the env file are intentionally seeded as empty strings —
    # they reserve the Infisical slot so the runtime can update_secret without a
    # prior create step. Per-service identity (e.g. POSTGRES_DATABASE) is baked
    # into each repo's Settings class as a default= and is NOT seeded here.
    plan: list[tuple[str, str, str]] = []
    for key in REPO_SECRET_KEYS:
        value = env_values.get(key, "")
        plan.append((f"{path_prefix}/db/", key, value))

    # Any extra keys in the env file that are NOT in shared, NOT bootstrap,
    # and NOT an identity default (per-repo value baked into Settings.default=).
    registry_data = _read_registry_data()
    registry = _load_registry(registry_data)
    shared_keys_flat = {k for keys in registry.values() for k in keys}
    bootstrap = _bootstrap_keys(registry_data)
    identity = _identity_defaults(registry_data)
    # Build a set of keys already in the plan (from REPO_SECRET_KEYS) for O(1)
    # duplicate detection.  Without this, the inner check would be O(n*m).
    planned_keys: set[str] = {pk for _, pk, _ in plan}
    extra: list[tuple[str, str, str]] = []
    for key, value in env_values.items():
        if key in bootstrap:
            continue
        if key in identity:
            continue
        if key in shared_keys_flat:
            continue
        if key in planned_keys:
            continue
        # Extra keys not matching any transport folder are bucketed into /env/.
        # If a key belongs to a specific transport (e.g. kafka, db, http), add it
        # to shared_key_registry.yaml under the appropriate folder, or handle it
        # via onboard-repo per-service overrides. Leaving transport-specific keys
        # here will cause them to be stored under /env/ instead of their transport
        # path, which may confuse config consumers expecting a known Infisical path.
        extra.append((f"{path_prefix}/env/", key, value))

    print(f"\n=== onboard-repo: {repo_name} ===")
    print(f"  Infisical path: {path_prefix}/")
    print(f"  Env file: {env_path}")
    print("\n  Repo-specific keys:")
    for folder, key, value in plan:
        key_upper = key.upper()
        is_sensitive = any(pat in key_upper for pat in _SENSITIVE_KEY_PATTERNS)
        if not value:
            display = "(empty)"
        elif is_sensitive:
            display = "***"
        else:
            display = value
        print(f"    {folder}{key} = {display}")

    if extra:
        print(f"\n  Additional repo-only keys ({len(extra)}):")
        for folder, key, value in extra:
            key_upper = key.upper()
            is_sensitive = any(pat in key_upper for pat in _SENSITIVE_KEY_PATTERNS)
            if not value:
                display = "(empty)"
            elif is_sensitive:
                display = "***"
            else:
                display = value
            print(f"    {folder}{key} = {display}")

    if not args.execute:
        print("\n[dry-run] Pass --execute to create folders and write secrets.")
        return 0

    # Validate Infisical connection config after the dry-run gate — INFISICAL_ADDR
    # and INFISICAL_PROJECT_ID are only required when actually writing to Infisical,
    # so dry-run can work offline without a live Infisical instance.
    infisical_addr = os.environ.get("INFISICAL_ADDR")
    if not infisical_addr:
        print(
            "ERROR: INFISICAL_ADDR is not set. "
            "Set it to the Infisical URL before seeding.",
            file=sys.stderr,
        )
        raise SystemExit(1)
    if not infisical_addr.startswith(("http://", "https://")):
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
    try:
        UUID(project_id)
    except ValueError:
        # ValueError here is user input, not a system error — no stacktrace needed
        logger.error("INFISICAL_PROJECT_ID is not a valid UUID: %s", project_id)  # noqa: TRY400
        return 1

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
        # Integer exit codes (e.g. SystemExit(1)) are not printed — the preceding
        # logger.error already describes the failure. Only string messages add context.
        if isinstance(e.code, str):
            print(e.code, file=sys.stderr)
        return 1

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
                )
                counts[outcome] += 1
                logger.info("  [%s] %s%s", outcome.upper(), folder, key)
            except Exception as exc:
                if _is_abort_error(exc):
                    logger.exception(
                        "Infrastructure error seeding %s%s — aborting: %s",
                        folder,
                        key,
                        sanitize(exc),
                    )
                    raise
                counts["error"] += 1
                logger.warning("  [ERROR] %s%s: %s", folder, key, sanitize(exc))
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

    # Only emit the service_override_required warning block when the seed
    # completed without errors.  If the loop was aborted by a re-raised
    # exception the finally block already ran adapter.shutdown() and the
    # exception is propagating — this code is unreachable in that case.
    # For the non-exception path, gate on zero errors so that a partially
    # failed seed (per-key errors counted but run not aborted) does not
    # print misleading "ACTION REQUIRED" guidance after a broken run.
    if counts.get("error", 0) == 0:
        # Warn for any keys declared as service_override_required that are NOT
        # included in the onboarding plan.  These keys have shared platform defaults
        # that are intentionally wrong for production use; each service must supply
        # its own value before starting.
        all_plan_keys: set[str] = {pk for _, pk, _ in all_secrets}
        override_required = _service_override_required(registry_data)
        for override_key in sorted(override_required - all_plan_keys):
            # Determine the most likely transport folder for this key by scanning
            # the shared registry for which folder it belongs to.
            transport_folder = next(
                (
                    parts[
                        1
                    ]  # e.g. "/shared/kafka/" → parts=["shared","kafka"] → "kafka"
                    for folder, keys in registry.items()
                    if override_key in keys
                    for parts in [folder.strip("/").split("/")]
                    if len(parts) >= 2
                ),
                "<unknown>",  # fallback if key is not found in shared registry
            )
            logger.warning(
                "ACTION REQUIRED: '%s' is declared service_override_required but was not "
                "included in the onboarding plan for repo '%s'. "
                "You MUST manually add /services/%s/%s/%s to Infisical before the service starts. "
                "The shared /shared/%s/%s value is a placeholder default only — "
                "relying on it in production is a misconfiguration.",
                override_key,
                repo_name,
                repo_name,
                transport_folder,
                override_key,
                transport_folder,
                override_key,
            )

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
