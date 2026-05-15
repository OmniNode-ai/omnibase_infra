# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Integration coverage for catalog ulimit rendering."""

from __future__ import annotations

from pathlib import Path

import pytest

from omnibase_infra.docker.catalog.generator import generate_compose
from omnibase_infra.docker.catalog.resolver import CatalogResolver

CATALOG_DIR = str(Path(__file__).resolve().parents[3] / "docker" / "catalog")


@pytest.mark.integration
def test_redpanda_catalog_ulimits_render_to_compose() -> None:
    resolver = CatalogResolver(catalog_dir=CATALOG_DIR)
    resolved = resolver.resolve(bundles=["core"])
    compose = generate_compose(resolved)

    assert compose["services"]["redpanda"]["ulimits"] == {
        "nofile": {"soft": 65535, "hard": 65535}
    }
