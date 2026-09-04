# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Explicit pytest injection boundary for the public RSD acceptance suite.

Canonical invocation is intentionally explicit::

    uv run pytest -p omnibase_infra.testing.rsd_postgres_acceptance_plugin \
        --rsd-postgres-acceptance-overlay docker/lane-overlays/dev.rsd-postgres-acceptance.yaml \
        tests/unit/runtime/test_rsd_postgres_acceptance_overlay.py

The ``-p`` opt-in is required; this module is never auto-registered by ordinary
pytest runs.
"""

from __future__ import annotations

from pathlib import Path
from typing import cast

import pytest
import yaml
from pydantic import ValidationError
from yaml import YAMLError

from omnibase_infra.runtime.models.model_rsd_postgres_acceptance_overlay import (
    ModelRsdPostgresAcceptanceOverlay,
)
from omnibase_infra.testing.rsd_postgres_acceptance_capability import (
    CapabilityResolver,
    PostgresLifecycleConnectionFactory,
    RsdPostgresAcceptanceResolutionError,
    resolve_postgres_lifecycle_factory,
)


def load_overlay(path: Path) -> ModelRsdPostgresAcceptanceOverlay:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return ModelRsdPostgresAcceptanceOverlay.model_validate(raw)


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption("--rsd-postgres-acceptance-overlay", action="store", default=None)


@pytest.fixture
def postgres_lifecycle_connection_factory(
    request: pytest.FixtureRequest,
) -> PostgresLifecycleConnectionFactory:
    configured = request.config.getoption("--rsd-postgres-acceptance-overlay")
    if not isinstance(configured, str) or not configured:
        pytest.fail("RSD PostgreSQL acceptance overlay path must be explicit")
    resolver_value = getattr(
        request.config, "rsd_postgres_acceptance_capability_resolver", None
    )
    if not callable(resolver_value):
        pytest.fail("operator capability resolver is not injected")
    resolver = cast("CapabilityResolver", resolver_value)
    try:
        capability_ref = load_overlay(Path(configured)).postgres_capability_ref
    except (OSError, ValidationError, YAMLError):
        pytest.fail("RSD PostgreSQL acceptance overlay is invalid", pytrace=False)
        raise AssertionError("unreachable")
    try:
        return resolve_postgres_lifecycle_factory(resolver, capability_ref)
    except RsdPostgresAcceptanceResolutionError as exc:
        pytest.fail(str(exc), pytrace=False)
        raise AssertionError("unreachable")
