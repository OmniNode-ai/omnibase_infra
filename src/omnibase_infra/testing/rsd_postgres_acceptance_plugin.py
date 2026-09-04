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

from collections.abc import Callable
from contextlib import AbstractContextManager
from pathlib import Path
from typing import cast

import pytest
import yaml

from omnibase_infra.runtime.models.model_rsd_postgres_acceptance_overlay import (
    ModelRsdPostgresAcceptanceOverlay,
)


def load_overlay(path: Path) -> ModelRsdPostgresAcceptanceOverlay:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return ModelRsdPostgresAcceptanceOverlay.model_validate(raw)


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption("--rsd-postgres-acceptance-overlay", action="store", default=None)


@pytest.fixture
def postgres_lifecycle_connection_factory(
    request: pytest.FixtureRequest,
) -> Callable[[], AbstractContextManager[object]]:
    configured = request.config.getoption("--rsd-postgres-acceptance-overlay")
    if not isinstance(configured, str) or not configured:
        pytest.fail("RSD PostgreSQL acceptance overlay path must be explicit")
    resolver = cast(
        "Callable[[str], Callable[[], AbstractContextManager[object]]] | None",
        getattr(request.config, "rsd_postgres_acceptance_capability_resolver", None),
    )
    if not callable(resolver):
        pytest.fail("operator capability resolver is not injected")
    return resolver(load_overlay(Path(configured)).postgres_capability_ref)
