# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Catalog prerequisite for the local environment observer."""

import re
from pathlib import Path

import yaml

from omnibase_infra.docker.catalog.resolver import CatalogResolver

_REPO = Path(__file__).resolve().parents[3]
_CATALOG = _REPO / "docker" / "catalog"


def test_runtime_image_ships_resolvable_local_catalog() -> None:
    """The effects observer needs the real catalog in the final image."""
    dockerfile = (_REPO / "docker" / "Dockerfile.runtime").read_text()
    final_stage = re.split(r"^FROM\s+", dockerfile, flags=re.MULTILINE)[-1]
    assert re.search(
        r"^COPY --chown=omniinfra:omniinfra docker/catalog/ /app/docker/catalog/\s*$",
        final_stage,
        flags=re.MULTILINE,
    )
    resolved = CatalogResolver(catalog_dir=str(_CATALOG)).resolve(["local"])
    bundles = yaml.safe_load((_CATALOG / "bundles.yaml").read_text())
    assert set(bundles["local"]["services"]) <= resolved.service_names
    assert {"omninode-runtime", "runtime-effects"} <= resolved.service_names
