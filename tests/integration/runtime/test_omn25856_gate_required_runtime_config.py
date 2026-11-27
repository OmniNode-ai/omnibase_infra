# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-25856: the shipped runtime config selects the gate_required boot policy.

Reads the real ``contracts/runtime/runtime_config.yaml`` from disk and parses it
through the production runtime config model, so a policy the model refuses or a
config that drifts back to ``in_use`` fails here and not at boot.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from omnibase_infra.models.model_boot_subscriptions_config import (
    ModelBootSubscriptionsConfig,
)

pytestmark = pytest.mark.integration

_CONFIG = Path(__file__).resolve().parents[3] / "contracts/runtime/runtime_config.yaml"


def _shipped_boot_subscriptions() -> ModelBootSubscriptionsConfig:
    raw = yaml.safe_load(_CONFIG.read_text())
    return ModelBootSubscriptionsConfig(**raw["boot_subscriptions"])


def test_shipped_runtime_config_selects_gate_required() -> None:
    config = _shipped_boot_subscriptions()
    assert config.policy == "gate_required"
    assert config.demand_start is True
    assert config.defer_retained is True


def test_in_use_policy_does_not_defer_retained_topics() -> None:
    config = ModelBootSubscriptionsConfig(policy="in_use")
    assert config.demand_start is True
    assert config.defer_retained is False


def test_unknown_policy_is_refused() -> None:
    with pytest.raises(ValueError):
        ModelBootSubscriptionsConfig.model_validate({"policy": "everything"})
