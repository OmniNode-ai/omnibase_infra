# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Integration coverage for registry effect contract protocol dependencies."""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.integration

REPO_ROOT = Path(__file__).resolve().parents[2]
CONTRACT_PATH = (
    REPO_ROOT
    / "src"
    / "omnibase_infra"
    / "nodes"
    / "node_registry_effect"
    / "contract.yaml"
)


def test_registry_effect_protocol_dependencies_are_importable() -> None:
    contract = yaml.safe_load(CONTRACT_PATH.read_text(encoding="utf-8"))
    dependencies = {
        dependency["name"]: dependency
        for dependency in contract["dependencies"]
        if dependency.get("type") == "protocol"
    }

    for dependency_name in (
        "protocol_postgres_adapter",
        "protocol_effect_idempotency_store",
    ):
        dependency = dependencies[dependency_name]
        module = importlib.import_module(dependency["module"])
        protocol = getattr(module, dependency["class_name"])

        assert protocol.__name__ == dependency["class_name"]
