# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Loader for ONEX feature-flag contract YAMLs (runtime.contract.yaml, event_bus.contract.yaml)."""

from __future__ import annotations

from pathlib import Path

import yaml

from omnibase_infra.models.contracts.model_service_contract import (
    ModelFeatureFlagContract,
)


def load_service_contract(path: Path) -> ModelFeatureFlagContract:
    """Parse a feature-flag contract YAML into a typed ModelFeatureFlagContract.

    Raises FileNotFoundError if path does not exist.
    Raises pydantic.ValidationError if the YAML does not conform to the schema.
    """
    raw = yaml.safe_load(path.read_text())
    return ModelFeatureFlagContract.model_validate(raw)
