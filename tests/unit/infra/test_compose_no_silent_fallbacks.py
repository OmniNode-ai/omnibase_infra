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
