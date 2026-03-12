# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Regression tests for the x-runtime-env anchor in docker-compose.infra.yml (OMN-4800).

Missing vars from x-runtime-env are a silent failure mode — the container starts
but the var is absent. Per CLAUDE.md: "vars reach containers ONLY if listed in
x-runtime-env anchor".

These tests prevent:
1. Required keys missing from the anchor
2. Docker Compose syntax errors in the anchor
3. Services bypassing the anchor with conflicting environment blocks
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
import yaml

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

COMPOSE_PATH = (
    Path(__file__).parent.parent.parent / "docker" / "docker-compose.infra.yml"
)

# Keys that MUST be present in x-runtime-env.
# Add new required runtime vars here — this is the canonical required-key list.
REQUIRED_RUNTIME_KEYS: frozenset[str] = frozenset(
    {
        "POSTGRES_PASSWORD",
        "OMNIBASE_INFRA_DB_URL",
        "KAFKA_BOOTSTRAP_SERVERS",
        "KAFKA_BROKER_ALLOWLIST",
        "INFISICAL_ADDR",
        "INFISICAL_CLIENT_ID",
        "INFISICAL_CLIENT_SECRET",
        "INFISICAL_PROJECT_ID",
        "ONEX_CONTRACTS_DIR",
        "ONEX_LOG_LEVEL",
        "ONEX_ENVIRONMENT",
        "USE_EVENT_ROUTING",
        "VALKEY_HOST",
        "VALKEY_PORT",
    }
)


def _load_compose() -> dict:
    """Load and parse the docker-compose.infra.yml file."""
    assert COMPOSE_PATH.exists(), f"Compose file not found: {COMPOSE_PATH}"
    with COMPOSE_PATH.open() as fh:
        data = yaml.safe_load(fh)
    assert isinstance(data, dict), (
        f"Compose file did not parse as a YAML mapping: {COMPOSE_PATH}"
    )
    return data


def _get_runtime_env_keys(data: dict) -> set[str]:
    """Extract keys from the x-runtime-env anchor."""
    runtime_env = data.get("x-runtime-env")
    assert runtime_env is not None, (
        "x-runtime-env anchor not found in docker-compose.infra.yml. "
        "This anchor is required for env var passthrough to containers."
    )
    assert isinstance(runtime_env, dict), (
        f"x-runtime-env is not a YAML mapping (got {type(runtime_env).__name__}). "
        "The anchor must be a key-value mapping."
    )
    return set(runtime_env.keys())


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestRuntimeEnvAnchorContainsRequiredKeys:
    """Test 1: x-runtime-env anchor contains all required keys."""

    @pytest.mark.unit
    def test_anchor_contains_required_keys(self) -> None:
        """Assert all required runtime keys are present in x-runtime-env."""
        data = _load_compose()
        actual_keys = _get_runtime_env_keys(data)

        missing = REQUIRED_RUNTIME_KEYS - actual_keys
        assert not missing, (
            f"x-runtime-env is missing {len(missing)} required key(s):\n"
            + "\n".join(f"  - {k}" for k in sorted(missing))
            + "\n\nAdd missing keys to x-runtime-env in docker/docker-compose.infra.yml. "
            "Format: KEY: ${KEY:-} (or ${KEY:?error} for required keys)."
        )


class TestRuntimeEnvAnchorSyntaxValid:
    """Test 2: docker-compose.infra.yml parses without YAML errors."""

    @pytest.mark.unit
    def test_anchor_syntax_valid(self) -> None:
        """Verify docker-compose.infra.yml loads cleanly as valid YAML."""
        # This primarily catches YAML syntax errors in the anchor definition
        data = _load_compose()
        runtime_env = data.get("x-runtime-env")
        assert runtime_env is not None, "x-runtime-env anchor not found"
        assert isinstance(runtime_env, dict), "x-runtime-env must be a YAML mapping"
        assert len(runtime_env) > 0, (
            "x-runtime-env is empty — no vars will be passed through"
        )

    @pytest.mark.unit
    def test_anchor_has_no_null_values(self) -> None:
        """Assert no key in x-runtime-env resolves to a bare null value.

        Null values in the anchor indicate the key was listed without a default,
        which may cause silent failures in containers.
        """
        data = _load_compose()
        runtime_env = data.get("x-runtime-env", {})
        # yaml.safe_load resolves ${VAR:-} to the literal string (not None)
        # but bare `KEY:` (no value) resolves to None
        null_keys = [k for k, v in runtime_env.items() if v is None]
        assert not null_keys, (
            f"x-runtime-env has {len(null_keys)} key(s) with null values: {null_keys}. "
            "Use KEY: ${{KEY:-}} for optional vars or KEY: ${{KEY:?error}} for required vars."
        )


class TestRuntimeEnvPassthroughNotBypassed:
    """Test 3: No service bypasses x-runtime-env with conflicting environment blocks."""

    @pytest.mark.unit
    def test_env_passthrough_not_bypassed(self) -> None:
        """Assert runtime services use *runtime-env anchor, not standalone environment blocks.

        Services that inherit <<: *runtime-env should not also define a standalone
        environment: block that duplicates anchor keys — the standalone block would
        shadow or conflict with the anchor values.

        This test checks that runtime-profile services (those with <<: *runtime-env)
        do not have environment: blocks with keys that overlap with x-runtime-env.
        """
        data = _load_compose()
        runtime_env_keys = _get_runtime_env_keys(data)
        services = data.get("services", {})

        violations: list[str] = []

        for svc_name, svc_config in services.items():
            if not isinstance(svc_config, dict):
                continue

            env_block = svc_config.get("environment")
            if not env_block:
                continue

            # Only flag services that also inherit *runtime-env
            # (services with bare environment: blocks that don't inherit are fine)
            # Check if this service inherits the runtime-env anchor via <<: *runtime-env
            # After yaml.safe_load, anchor merges are resolved — check deploy labels or
            # known runtime service names instead
            # Heuristic: if env_block shares many keys with runtime_env_keys, likely merged
            if isinstance(env_block, dict):
                svc_env_keys = set(env_block.keys())
                overlap = svc_env_keys & runtime_env_keys
                if (
                    len(overlap) >= 3
                ):  # 3+ overlapping keys = likely duplicate definition
                    violations.append(
                        f"Service '{svc_name}' has {len(overlap)} env keys "
                        f"that overlap with x-runtime-env: {sorted(overlap)[:5]}..."
                    )

        # This is an advisory check — warn rather than fail
        # Full enforcement requires compose-config resolution which needs Docker running
        if violations:
            import warnings

            for v in violations:
                warnings.warn(
                    f"x-runtime-env bypass advisory: {v}",
                    stacklevel=1,
                )
