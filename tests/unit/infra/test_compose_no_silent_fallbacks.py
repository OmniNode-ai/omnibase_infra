# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Tests that no env var uses the empty-string-means-disabled pattern."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

COMPOSE_FILE = (
    Path(__file__).resolve().parents[3] / "docker" / "docker-compose.infra.yml"
)

# ROLE_* passwords in postgres service are intentionally empty-means-skip:
# the init script creates roles only when the corresponding password is set.
# These are NOT feature flags or opt-in toggles.
#
# Infisical vars are intentionally opt-in via empty addr:
#   INFISICAL_ADDR empty = skip Infisical, fall back to env vars (local dev).
#   CLIENT_ID/SECRET/PROJECT_ID are only meaningful when ADDR is set.
#   INFISICAL_REQUIRED is an opt-in strict mode flag; empty = permissive.
#
# OmniMemory vars are intentionally feature-flag style:
#   OMNIMEMORY_ENABLED empty = feature disabled (non-memory deployments unaffected).
#   OMNIMEMORY_DB_URL empty = skipped when memory feature is disabled.
ALLOWED_EMPTY_DEFAULTS = {
    "ROLE_OMNIBASE_PASSWORD",
    "ROLE_OMNICLAUDE_PASSWORD",
    "ROLE_OMNIDASH_PASSWORD",
    "ROLE_OMNIINTELLIGENCE_PASSWORD",
    "ROLE_OMNIMEMORY_PASSWORD",
    "ROLE_OMNINODE_PASSWORD",
    # Infisical opt-in: empty addr = disable Infisical prefetch (local dev without secrets profile)
    "INFISICAL_ADDR",
    "INFISICAL_CLIENT_ID",
    "INFISICAL_CLIENT_SECRET",
    "INFISICAL_PROJECT_ID",
    "INFISICAL_REQUIRED",
    # OmniMemory feature flag: empty = feature disabled for non-memory deployments
    "OMNIMEMORY_ENABLED",
    "OMNIMEMORY_DB_URL",
    # Keycloak auth profile (OMN-3361): secret only needed when auth profile is active
    "KEYCLOAK_ADMIN_CLIENT_SECRET",
}


@pytest.mark.unit
def test_no_empty_default_fallbacks_in_runtime_env() -> None:
    """No env var should use the empty-string-means-disabled pattern.

    Pattern banned: ${VAR:-}  (empty default = silently disabled)
    Pattern allowed: ${VAR:?message}  (required, fails loud)
    Pattern allowed: ${VAR:-value}  (operational default with real value)
    Pattern allowed: "literal"  (hardcoded)

    Exception: ROLE_*_PASSWORD in postgres service (empty = skip role creation).
    """
    with open(COMPOSE_FILE) as f:
        content = f.read()

    # Find all ${VAR:-} patterns (empty default)
    empty_defaults = re.findall(r"\$\{([A-Z_]+):-\}", content)

    # Filter out allowed exceptions
    violations = [v for v in empty_defaults if v not in ALLOWED_EMPTY_DEFAULTS]

    assert len(violations) == 0, (
        f"Found {len(violations)} empty-default fallbacks that should be "
        f"converted to required or removed: {violations}"
    )
