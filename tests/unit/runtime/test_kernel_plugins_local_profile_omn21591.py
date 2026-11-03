# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Laptop plugin selection follows the packages shipped by its runtime image."""

from __future__ import annotations

from pathlib import Path

import pytest

from omnibase_infra.docker.catalog.generator import generate_compose
from omnibase_infra.docker.catalog.resolver import CatalogResolver

pytestmark = pytest.mark.unit

_CATALOG = Path(__file__).resolve().parents[3] / "docker" / "catalog"


@pytest.mark.parametrize(
    ("bundle", "default"),
    [("local", ""), ("local-lane-mirror", ""), ("local-market", "delegation")],
)
def test_laptop_kernel_plugins_match_the_image_packages(
    bundle: str, default: str
) -> None:
    """Infra-only kernels must not require an unannounced market plugin at boot."""
    compose = generate_compose(
        CatalogResolver(catalog_dir=str(_CATALOG)).resolve([bundle])
    )
    services = compose["services"]
    assert isinstance(services, dict)
    for service in ("omninode-runtime", "runtime-effects", "tenant-projection-writer"):
        env = services[service]["environment"]
        assert env["ONEX_RUNTIME_KERNEL_PLUGINS"] == (
            f"${{ONEX_LOCAL_RUNTIME_KERNEL_PLUGINS:-{default}}}"
        )
