# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Tests for container-internal address invariants in docker-compose.infra.yml.

Extends OMN-3431 invariant (Kafka) to all internal service addresses.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

COMPOSE_FILE = (
    Path(__file__).resolve().parents[3] / "docker" / "docker-compose.infra.yml"
)


@pytest.mark.unit
def test_container_internal_addresses_are_hardcoded() -> None:
    """Container-internal DNS names must be hardcoded literals, never inherited from host.

    Extends OMN-3431 invariant (Kafka) to all internal service addresses.
    """
    with open(COMPOSE_FILE) as f:
        compose = yaml.safe_load(f)

    # These vars must be hardcoded literals (no ${} interpolation) in x-runtime-env
    hardcoded_vars = {
        "KAFKA_BOOTSTRAP_SERVERS": "redpanda:9092",
        "KAFKA_BROKER_ALLOWLIST": "redpanda:",
        "VALKEY_HOST": "valkey",
        "VALKEY_PORT": "6379",
    }

    runtime_env = compose.get("x-runtime-env", {})
    for var, expected in hardcoded_vars.items():
        actual = runtime_env.get(var)
        assert actual is not None, f"{var} missing from x-runtime-env"
        assert "${" not in str(actual), (
            f"{var} must be a hardcoded literal, got: {actual}"
        )
