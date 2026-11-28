# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Tests for OMN-26616 identity contract models."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

import pytest
import yaml
from pydantic import BaseModel

pytestmark = pytest.mark.unit

CONTRACT_PATH = (
    Path(__file__).resolve().parents[3]
    / "src/omnibase_infra/nodes/node_check_delegation_identity_effect/contract.yaml"
)


@pytest.fixture(scope="module")
def contract() -> dict[str, Any]:
    with CONTRACT_PATH.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


@pytest.mark.parametrize("key", ["input_model", "output_model"])
def test_model_entry_resolves(contract: dict[str, Any], key: str) -> None:
    entry = contract[key]
    module = importlib.import_module(entry["module"])
    assert hasattr(module, entry["name"]), (
        f"Module {entry['module']} missing attribute {entry['name']}"
    )
    obj = getattr(module, entry["name"])
    assert issubclass(obj, BaseModel), (
        f"{entry['name']} is not a pydantic BaseModel subclass"
    )


def test_default_handler_resolves(contract: dict[str, Any]) -> None:
    handler = contract["handler_routing"]["default_handler"]
    module_path, class_name = handler.split(":")
    module = importlib.import_module(module_path)
    assert hasattr(module, class_name), (
        f"Module {module_path} missing class {class_name}"
    )
