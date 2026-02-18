#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Seed Infisical with configuration from ONEX contracts and .env values.

This script scans ONEX contract YAML files, extracts configuration
requirements (transport types, env dependencies), and populates Infisical
with the expected keys. It is designed to be safe by default:

    - ``--dry-run`` (default: true) -- shows what would be done without writing
    - ``--execute`` -- required to actually write to Infisical
    - ``--create-missing-keys`` (default: true) -- creates keys that don't exist
    - ``--set-values`` (default: false) -- sets values from .env if available
    - ``--overwrite-existing`` (default: false) -- overwrites existing values
    - ``--import-env FILE`` -- import values from a .env file
    - ``--export`` -- export current Infisical values to stdout

Usage:
    # Dry run (default) -- show what would happen
    uv run python scripts/seed-infisical.py \\
        --contracts-dir src/omnibase_infra/nodes

    # Create missing keys in Infisical
    uv run python scripts/seed-infisical.py \\
        --contracts-dir src/omnibase_infra/nodes \\
        --create-missing-keys \\
        --execute

    # Import values from .env
    uv run python scripts/seed-infisical.py \\
        --contracts-dir src/omnibase_infra/nodes \\
        --import-env .env \\
        --set-values \\
        --execute

    # Export current Infisical keys (values masked)
    uv run python scripts/seed-infisical.py --export

    # Export with actual values (use with caution -- not for CI logs)
    uv run python scripts/seed-infisical.py --export --reveal

.. versionadded:: 0.10.0
    Created as part of OMN-2287.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

# Ensure project root is on the path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("seed-infisical")


def _parse_env_file(env_path: Path) -> dict[str, str]:
    """Parse a .env file into a key-value dict.

    Handles:
    - Comments (lines starting with #)
    - Empty lines
    - KEY=VALUE format
    - ``export KEY=VALUE`` prefix
    - Quoted values (single and double quotes stripped)
    - Inline comments after values

    Note:
        Multiline values and escaped quotes are not supported.
    """
    values: dict[str, str] = {}
    if not env_path.is_file():
        logger.warning("Env file not found: %s", env_path)
        return values

    for line in env_path.read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        # Handle 'export KEY=VALUE' syntax
        if stripped.startswith("export "):
            stripped = stripped[7:]
        if "=" not in stripped:
            continue
        key, _, value = stripped.partition("=")
        key = key.strip()
        # Value processing order:
        # 1. Strip surrounding whitespace from the raw value.
        # 2. Detect whether the value is quoted (before removing quotes).
        # 3. If quoted, strip the outer quotes -- inline comments inside
        #    quoted strings are preserved as literal text.
        # 4. If NOT quoted, strip inline comments (text after ' #').
        value = value.strip()
        is_quoted = len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"')
        if is_quoted:
            value = value[1:-1]
        if " #" in value and not is_quoted:
            value = value.split(" #")[0].strip()
        if key:
            values[key] = value

    logger.info("Parsed %d values from %s", len(values), env_path)
    return values


def _extract_requirements(
    contracts_dir: Path,
) -> tuple[list[dict[str, str]], list[str]]:
    """Extract config requirements from contracts.

    Returns:
        Tuple of (requirements_list, errors_list) where each requirement
        is a dict with keys: key, transport_type, folder, source.
    """
    from omnibase_infra.runtime.config_discovery.contract_config_extractor import (
        ContractConfigExtractor,
    )
    from omnibase_infra.runtime.config_discovery.transport_config_map import (
        TransportConfigMap,
    )

    extractor = ContractConfigExtractor()
    reqs = extractor.extract_from_paths([contracts_dir])

    transport_map = TransportConfigMap()
    result: list[dict[str, str]] = []
    seen: set[str] = set()

    # Build specs from transport types, excluding bootstrap-only transports.
    # Bootstrap transports (e.g. INFISICAL) must come from the environment,
    # not from Infisical itself -- seeding their keys into Infisical would
    # create a circular dependency (Infisical needs those credentials to start).
    for transport in reqs.transport_types:
        if transport_map.is_bootstrap_transport(transport):
            logger.debug(
                "Skipping bootstrap transport %s in seed (credentials come "
                "from env, not Infisical)",
                transport.value,
            )
            continue
        spec = transport_map.shared_spec(transport)
        for key in spec.keys:
            if key not in seen:
                seen.add(key)
                result.append(
                    {
                        "key": key,
                        "transport_type": transport.value,
                        "folder": spec.infisical_folder,
                        "source": "transport",
                    }
                )

    # Add explicit env dependencies
    for req in reqs.requirements:
        if req.source_field.startswith("dependencies[") and req.key not in seen:
            seen.add(req.key)
            result.append(
                {
                    "key": req.key,
                    "transport_type": "env",
                    "folder": "/shared/env/",
                    "source": str(req.source_contract),
                }
            )

    return result, reqs.errors


def _print_diff_summary(
    requirements: list[dict[str, str]],
    env_values: dict[str, str],
    *,
    create_missing: bool,
    set_values: bool,
    overwrite_existing: bool,
) -> None:
    """Print a diff summary of what would be done."""
    print("\n--- Seed Diff Summary ---")
    print(f"Total keys discovered: {len(requirements)}")
    print(
        f"Values available from .env: {sum(1 for r in requirements if r['key'] in env_values)}"
    )
    if overwrite_existing:
        print("Mode: OVERWRITE (existing keys will be overwritten)")
    print()

    for req in sorted(requirements, key=lambda r: r["key"]):
        key = req["key"]
        folder = req["folder"]
        has_value = key in env_values

        # Determine the action label based on flags.
        # When overwrite_existing is active, existing keys are tagged
        # OVERWRITE instead of SKIP so the user sees the intent clearly.
        action = "OVERWRITE" if overwrite_existing else "SKIP"
        if create_missing:
            action = "CREATE"
        if set_values and has_value:
            action = "OVERWRITE" if overwrite_existing else "SET (if new)"
        elif set_values and not has_value:
            action = "CREATE (no value)"

        value_indicator = " (has .env value)" if has_value else ""
        print(f"  [{action:>16s}] {folder}{key}{value_indicator}")

    print("\n--- End Diff Summary ---\n")


def _load_infisical_credentials() -> tuple[str, str, str, str]:
    """Load Infisical credentials from environment variables.

    Returns:
        Tuple of (infisical_addr, client_id, client_secret, project_id).

    Raises:
        SystemExit: If required credentials are missing.
    """
    infisical_addr = os.environ.get("INFISICAL_ADDR", "http://localhost:8880")
    client_id = os.environ.get("INFISICAL_CLIENT_ID", "")
    client_secret = os.environ.get("INFISICAL_CLIENT_SECRET", "")
    project_id = os.environ.get("INFISICAL_PROJECT_ID", "")

    if not all([client_id, client_secret, project_id]):
        logger.error(
            "Missing Infisical credentials. Set INFISICAL_CLIENT_ID, "
            "INFISICAL_CLIENT_SECRET, and INFISICAL_PROJECT_ID in environment."
        )
        raise SystemExit(1)

    return infisical_addr, client_id, client_secret, project_id


def _do_seed(
    requirements: list[dict[str, str]],
    env_values: dict[str, str],
    *,
    create_missing: bool,
    set_values: bool,
    overwrite_existing: bool,
) -> tuple[int, int, int, int]:
    """Execute the actual seed operation.

    Returns:
        Tuple of (created, updated, skipped, errors) counts.
        ``created`` tracks secrets newly created in Infisical.
        ``updated`` tracks secrets overwritten with new values.
        ``skipped`` tracks secrets left untouched.
        ``errors`` tracks secrets that failed to process.
    """
    from uuid import UUID

    try:
        from pydantic import SecretStr

        from omnibase_infra.adapters._internal.adapter_infisical import (
            AdapterInfisical,
        )
        from omnibase_infra.adapters.models.model_infisical_config import (
            ModelInfisicalAdapterConfig,
        )
    except ImportError:
        logger.exception("Cannot import Infisical adapter")
        return 0, 0, 0, len(requirements)

    # Build adapter config from environment
    try:
        infisical_addr, client_id, client_secret, project_id = (
            _load_infisical_credentials()
        )
    except SystemExit:
        return 0, 0, 0, len(requirements)

    # Validate project_id is a valid UUID
    try:
        project_uuid = UUID(project_id)
    except ValueError:
        logger.exception(
            "INFISICAL_PROJECT_ID is not a valid UUID: '%s'. "
            "Expected format: xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx",
            project_id,
        )
        return 0, 0, 0, len(requirements)

    # Initialize the adapter -- errors must propagate so main() can exit(1).
    try:
        config = ModelInfisicalAdapterConfig(
            host=infisical_addr,
            client_id=SecretStr(client_id),
            client_secret=SecretStr(client_secret),
            project_id=project_uuid,
        )
        adapter = AdapterInfisical(config)
        adapter.initialize()
    except Exception:
        logger.exception("Failed to initialize Infisical adapter")
        raise

    created = 0
    updated = 0
    skipped = 0
    error_count = 0

    try:
        for req in requirements:
            key = req["key"]
            folder = req["folder"]
            has_value = key in env_values
            secret_value = env_values.get(key, "") if set_values else ""

            try:
                # Check if key exists
                existing = None
                try:
                    existing = adapter.get_secret(
                        secret_name=key,
                        secret_path=folder,
                    )
                except Exception as exc:
                    logger.debug("Key check failed for %s at %s: %s", key, folder, exc)

                if existing is not None and not overwrite_existing:
                    skipped += 1
                    logger.debug("Key %s already exists at %s, skipping", key, folder)
                    continue

                if existing is not None and overwrite_existing:
                    # Update existing secret
                    adapter.update_secret(
                        secret_name=key,
                        secret_path=folder,
                        secret_value=secret_value,
                    )
                    updated += 1
                    logger.info(
                        "Updated secret: %s at %s (value %s)",
                        key,
                        folder,
                        "from .env" if has_value and set_values else "empty",
                    )

                elif create_missing and existing is None:
                    # Create new secret
                    adapter.create_secret(
                        secret_name=key,
                        secret_path=folder,
                        secret_value=secret_value,
                    )
                    created += 1
                    logger.info(
                        "Created secret: %s at %s (value %s)",
                        key,
                        folder,
                        "from .env" if has_value and set_values else "empty",
                    )

                else:
                    skipped += 1

            except Exception as exc:
                logger.warning("Error processing %s: %s", key, exc)
                error_count += 1
    finally:
        adapter.shutdown()

    return created, updated, skipped, error_count


def _do_export(*, reveal: bool = False) -> bool:
    """Export current Infisical values to stdout.

    Args:
        reveal: If True, print actual secret values. If False (default),
            print key names with masked placeholders.

    Returns:
        True if export succeeded, False on any error.
    """
    from uuid import UUID

    try:
        from pydantic import SecretStr

        from omnibase_infra.adapters._internal.adapter_infisical import (
            AdapterInfisical,
        )
        from omnibase_infra.adapters.models.model_infisical_config import (
            ModelInfisicalAdapterConfig,
        )
    except ImportError:
        logger.exception("Cannot import Infisical adapter")
        return False

    try:
        infisical_addr, client_id, client_secret, project_id = (
            _load_infisical_credentials()
        )
    except SystemExit:
        return False

    # Validate project_id is a valid UUID
    try:
        project_uuid = UUID(project_id)
    except ValueError:
        logger.exception(
            "INFISICAL_PROJECT_ID is not a valid UUID: '%s'. "
            "Expected format: xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx",
            project_id,
        )
        return False

    # Initialize the adapter -- errors must propagate so main() can exit(1).
    try:
        config = ModelInfisicalAdapterConfig(
            host=infisical_addr,
            client_id=SecretStr(client_id),
            client_secret=SecretStr(client_secret),
            project_id=project_uuid,
        )
        adapter = AdapterInfisical(config)
        adapter.initialize()
    except Exception:
        logger.exception("Failed to initialize Infisical adapter for export")
        raise

    try:
        secrets = adapter.list_secrets()

        if reveal:
            print(
                "WARNING: Secret values are being printed in plaintext. "
                "Do NOT use this output in CI logs or shared terminals.",
                file=sys.stderr,
            )
            print("=" * 72, file=sys.stderr)
            for secret in secrets:
                print(f"{secret.key}={secret.value.get_secret_value()}")
        else:
            for secret in secrets:
                print(f"{secret.key}=****")
        return True
    except Exception:
        logger.exception("Failed to list secrets from Infisical")
        return False
    finally:
        adapter.shutdown()


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Seed Infisical with ONEX contract configuration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--contracts-dir",
        type=Path,
        default=_PROJECT_ROOT / "src" / "omnibase_infra" / "nodes",
        help="Directory to scan for contract.yaml files",
    )
    parser.add_argument(
        "--create-missing-keys",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Create keys that don't exist in Infisical (default: true)",
    )
    parser.add_argument(
        "--set-values",
        action="store_true",
        default=False,
        help="Set values from .env for discovered keys (default: false)",
    )
    parser.add_argument(
        "--overwrite-existing",
        action="store_true",
        default=False,
        help="Overwrite existing Infisical values (default: false)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=True,
        help="Show what would be done without writing (default: true)",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        default=False,
        help="Actually write to Infisical (overrides --dry-run)",
    )
    parser.add_argument(
        "--import-env",
        type=Path,
        default=None,
        help="Import values from a .env file",
    )
    parser.add_argument(
        "--export",
        action="store_true",
        default=False,
        help="Export current Infisical values to stdout (keys only; use --reveal for values)",
    )
    parser.add_argument(
        "--reveal",
        action="store_true",
        default=False,
        help="Show actual secret values in --export output (use with caution)",
    )

    args = parser.parse_args()

    # Handle export mode
    if args.export:
        try:
            ok = _do_export(reveal=args.reveal)
        except Exception:
            logger.exception("Export failed with unhandled error")
            return 1
        return 0 if ok else 1

    # Extract requirements from contracts
    logger.info("Scanning contracts in %s", args.contracts_dir)
    try:
        requirements, errors = _extract_requirements(args.contracts_dir)
    except Exception:
        logger.exception("Failed to extract config requirements from contracts")
        return 1

    if errors:
        for err in errors:
            logger.warning("Extraction error: %s", err)

    if not requirements:
        logger.info("No config requirements found in contracts")
        return 0

    logger.info("Found %d config requirements", len(requirements))

    # Load env values
    env_values: dict[str, str] = {}
    if args.import_env:
        env_values = _parse_env_file(args.import_env)
    else:
        # Use current environment
        env_values = dict(os.environ)

    # Always show diff summary
    _print_diff_summary(
        requirements,
        env_values,
        create_missing=args.create_missing_keys,
        set_values=args.set_values,
        overwrite_existing=args.overwrite_existing,
    )

    # Execute or dry-run
    if args.execute:
        logger.info("Executing seed operation...")
        try:
            created, updated, skipped, error_count = _do_seed(
                requirements,
                env_values,
                create_missing=args.create_missing_keys,
                set_values=args.set_values,
                overwrite_existing=args.overwrite_existing,
            )
        except Exception:
            logger.exception("Seed operation failed with unhandled error")
            return 1
        logger.info(
            "Seed complete: %d created, %d updated, %d skipped, %d errors",
            created,
            updated,
            skipped,
            error_count,
        )
        if error_count > 0:
            logger.error(
                "%d secret(s) failed to process -- exiting with error", error_count
            )
            return 1
    else:
        logger.info("Dry run complete. Use --execute to write to Infisical.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
