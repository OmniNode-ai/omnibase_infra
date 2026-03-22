# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Env var validator for the infrastructure catalog."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field

from omnibase_infra.docker.catalog.manifest_schema import CatalogManifest


@dataclass(frozen=True)
class ValidationResult:
    """Result of env var validation."""

    ok: bool
    missing: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class LocalhostViolation:
    """A single localhost violation found in a catalog manifest."""

    service: str
    env_section: str  # hardcoded_env | operational_defaults | catalog_env
    var_name: str
    value: str


# Pattern that matches localhost in URLs/connection strings.
# Matches "localhost" as a hostname component (preceded by @, //, or start-of-string
# followed by optional port).  Does NOT flag localhost in healthcheck commands or
# CORS origins (those are container-internal and correct).
_LOCALHOST_DB_URL_RE = re.compile(r"(?:@|//)localhost(?::\d+)?(?:/|$)", re.IGNORECASE)

# Env var names that are exempt from the localhost check because they are
# container-internal (e.g. CORS origins listing browser-accessible URLs,
# Keycloak issuer URLs resolved by the browser, not the container).
_LOCALHOST_EXEMPT_PATTERNS: set[str] = {
    "CORS_ORIGINS",
    "KEYCLOAK_ISSUER",
}


def validate_env(required: set[str]) -> ValidationResult:
    """Check that all required env vars are set in the current environment."""
    missing = [var for var in sorted(required) if not os.environ.get(var)]
    return ValidationResult(ok=len(missing) == 0, missing=missing)


def validate_no_localhost_in_container_env(
    manifests: dict[str, CatalogManifest],
) -> list[LocalhostViolation]:
    """Reject localhost in env vars that will be set inside Docker containers.

    DB URLs, Kafka brokers, and other service addresses must use Docker DNS
    names (e.g. ``postgres:5432``, ``redpanda:9092``), never ``localhost``.

    Only checks ``hardcoded_env``, ``operational_defaults``, and ``catalog_env``
    -- these are literal values baked into the compose file.  ``required_env``
    vars are passthrough references (``${VAR:?...}``) and are not checked here
    since their values come from the host environment at compose-up time.

    Returns a list of violations (empty means clean).
    """
    violations: list[LocalhostViolation] = []

    for name, manifest in manifests.items():
        for section_name, section_dict in [
            ("hardcoded_env", manifest.hardcoded_env),
            ("operational_defaults", manifest.operational_defaults),
            ("catalog_env", manifest.catalog_env),
        ]:
            for var_name, value in section_dict.items():
                if var_name in _LOCALHOST_EXEMPT_PATTERNS:
                    continue
                if _LOCALHOST_DB_URL_RE.search(value):
                    violations.append(
                        LocalhostViolation(
                            service=name,
                            env_section=section_name,
                            var_name=var_name,
                            value=value,
                        )
                    )

    return violations
