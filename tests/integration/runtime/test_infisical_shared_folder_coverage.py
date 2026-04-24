# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Integration coverage for Infisical shared folder provisioning."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest
import yaml

from omnibase_infra.runtime.config_discovery.transport_config_map import (
    TransportConfigMap,
)

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_SCRIPTS_DIR = _PROJECT_ROOT / "scripts"
sys.path.insert(0, str(_SCRIPTS_DIR))

_provision_mod = importlib.import_module("provision-infisical")


def _folder_slug_from_shared_path(path: str) -> str:
    return path.strip("/").split("/")[1]


@pytest.mark.integration
def test_infisical_provisioning_covers_registry_and_runtime_prefetch_folders() -> None:
    """Provisioning must create every folder used by seed-shared and prefetch."""
    provisioned_folders = set(_provision_mod._default_shared_folder_slugs())

    registry_data = yaml.safe_load(
        (_PROJECT_ROOT / "config" / "shared_key_registry.yaml").read_text(
            encoding="utf-8"
        )
    )
    registry_folders = {
        _folder_slug_from_shared_path(path) for path in registry_data["shared"]
    }
    prefetch_folders = {
        _folder_slug_from_shared_path(spec.infisical_folder)
        for spec in TransportConfigMap().all_shared_specs()
    }

    assert registry_folders | prefetch_folders <= provisioned_folders
    assert "filesystem" in provisioned_folders
