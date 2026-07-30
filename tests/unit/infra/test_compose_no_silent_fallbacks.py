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
ALLOWED_EMPTY_DEFAULTS = {
    "ROLE_OMNIBASE_PASSWORD",
    "ROLE_OMNICLAUDE_PASSWORD",
    "ROLE_OMNIDASH_PASSWORD",
    "ROLE_OMNIINTELLIGENCE_PASSWORD",
    "ROLE_OMNIMEMORY_PASSWORD",
    "ROLE_OMNINODE_PASSWORD",
    # OMN-4316: OMNIMEMORY_* vars are intentionally opt-in (empty = disabled)
    "OMNIMEMORY_ENABLED",
    "OMNIMEMORY_MEMGRAPH_HOST",
    "OMNIMEMORY_DB_URL",
    # Qdrant vector store: opt-in (empty = Qdrant not configured)
    "QDRANT_HOST",
    "QDRANT_API_KEY",
    # Infisical: opt-in config store; empty = fall back to env vars
    "INFISICAL_ADDR",
    "INFISICAL_CLIENT_ID",
    "INFISICAL_CLIENT_SECRET",
    "INFISICAL_PROJECT_ID",
    "INFISICAL_REQUIRED",
    # Local Docker runtime prefetch remains opt-in through prefixed host vars.
    "ONEX_RUNTIME_INFISICAL_ADDR",
    "ONEX_RUNTIME_INFISICAL_CLIENT_ID",
    "ONEX_RUNTIME_INFISICAL_CLIENT_SECRET",
    "ONEX_RUNTIME_INFISICAL_PROJECT_ID",
    "ONEX_RUNTIME_INFISICAL_ENVIRONMENT",
    # Keycloak: opt-in auth; empty = keycloak not configured
    "KEYCLOAK_ADMIN_CLIENT_SECRET",
    # OMN-12196: bifrost source contract path; empty = resolve from omnimarket
    # package via importlib.resources (opt-in override for custom source path)
    "BIFROST_SOURCE_CONTRACT_PATH",
    # OMN-12864: Bifrost local inference endpoints — provided by the committed
    # lane overlay (docker/lane-overlays/dev.bifrost.env) at runtime. Compose-level
    # :? is intentionally omitted so `docker compose config` renders cleanly in
    # CI without the lane overlay pre-loaded. Validation at the Python layer
    # (ModelBifrostLaneOverlay + render_bifrost_delegation_contract) is the
    # enforcement point for dev-lane deployments.
    "BIFROST_LOCAL_CODER_ENDPOINT_URL",
    "BIFROST_LOCAL_REASONER_ENDPOINT_URL",
    "BIFROST_LOCAL_EMBEDDING_ENDPOINT_URL",
    # OMN-15529: OnexBot-OCC-Writer App identity. Opt-in per host — the App key
    # is minted onto a specific runtime host (.201) by the operator, so every
    # other lane must render and run without it. Empty is NOT "silently
    # disabled" here, it is the current, explicit PAT behaviour: both consumers
    # read the mode as `os.environ.get(..., "pat").strip().lower() or "pat"`, and
    # `resolve_api_key(..., required=False)` treats an empty credential as absent
    # and raises GitHubAppCredentialMissingError in app mode instead of falling
    # back to the shared PAT. A `:?` form would wedge every unprovisioned lane.
    "ONEXBOT_OCC_APP_ID",
    "ONEXBOT_OCC_PRIVATE_KEY",
    "OMNI_OCC_GITHUB_AUTH_MODE",
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
