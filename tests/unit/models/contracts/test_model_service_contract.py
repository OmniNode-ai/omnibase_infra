# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Unit tests for ModelFeatureFlagContract and load_service_contract loader."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from omnibase_infra.models.contracts.model_feature_flag import ModelFeatureFlag
from omnibase_infra.models.contracts.model_service_contract import (
    ModelFeatureFlagContract,
)
from omnibase_infra.runtime.service_contract_loader import load_service_contract

_FLAG = {
    "name": "ENABLE_HOT_RELOAD",
    "env_var": "ENABLE_HOT_RELOAD",
    "default": "false",
    "description": "Gate for hot reload",
}


class TestModelFeatureFlagContract:
    """Tests for ModelFeatureFlagContract construction and validation."""

    def test_service_contract_round_trips(self) -> None:
        raw = {
            "name": "runtime",
            "description": "Runtime service contract",
            "feature_flags": [_FLAG],
        }
        m = ModelFeatureFlagContract.model_validate(raw)
        assert m.name == "runtime"
        assert m.feature_flags[0].name == "ENABLE_HOT_RELOAD"
        assert m.feature_flags[0].default == "false"

    def test_service_contract_defaults(self) -> None:
        m = ModelFeatureFlagContract.model_validate({"name": "my_service"})
        assert m.description == ""
        assert m.feature_flags == []

    def test_service_contract_is_frozen(self) -> None:
        m = ModelFeatureFlagContract.model_validate({"name": "svc"})
        with pytest.raises(Exception):
            m.name = "other"  # type: ignore[misc]

    def test_service_contract_rejects_extra_fields(self) -> None:
        with pytest.raises(ValidationError):
            ModelFeatureFlagContract.model_validate(
                {"name": "svc", "unknown_key": "boom"}
            )

    def test_service_contract_requires_name(self) -> None:
        with pytest.raises(ValidationError):
            ModelFeatureFlagContract.model_validate({})

    def test_feature_flag_entry_rejects_extra_fields(self) -> None:
        with pytest.raises(ValidationError):
            ModelFeatureFlag.model_validate({**_FLAG, "bogus": "field"})

    def test_feature_flag_entry_requires_name_and_env_var_and_default(self) -> None:
        with pytest.raises(ValidationError):
            ModelFeatureFlag.model_validate({"name": "X"})


class TestLoadServiceContract:
    """Tests for the load_service_contract path-based loader."""

    def test_load_runtime_contract_yaml(self, tmp_path: Path) -> None:
        contract_file = tmp_path / "runtime.contract.yaml"
        contract_file.write_text(
            "name: runtime_service\n"
            "description: Feature flags for ONEX runtime services\n"
            "feature_flags:\n"
            "  - name: ENABLE_RUNTIME_LOG_BRIDGE\n"
            "    env_var: ENABLE_RUNTIME_LOG_BRIDGE\n"
            "    default: 'false'\n"
            "    description: Gate for runtime log-to-Kafka event bridge\n"
        )
        m = load_service_contract(contract_file)
        assert m.name == "runtime_service"
        assert m.feature_flags[0].name == "ENABLE_RUNTIME_LOG_BRIDGE"
        assert m.feature_flags[0].default == "false"

    def test_load_event_bus_contract_yaml(self, tmp_path: Path) -> None:
        contract_file = tmp_path / "event_bus.contract.yaml"
        contract_file.write_text(
            "name: event_bus_service\n"
            "description: Feature flags for Kafka event bus infrastructure services\n"
            "feature_flags:\n"
            "  - name: ENABLE_CONSUMER_HEALTH_EMITTER\n"
            "    env_var: ENABLE_CONSUMER_HEALTH_EMITTER\n"
            "    default: 'false'\n"
            "    description: Gate for consumer health metric emission\n"
        )
        m = load_service_contract(contract_file)
        assert m.name == "event_bus_service"
        assert m.feature_flags[0].name == "ENABLE_CONSUMER_HEALTH_EMITTER"

    def test_load_raises_on_missing_file(self, tmp_path: Path) -> None:
        missing = tmp_path / "nonexistent.yaml"
        with pytest.raises(FileNotFoundError):
            load_service_contract(missing)

    def test_load_raises_on_invalid_contract(self, tmp_path: Path) -> None:
        bad_file = tmp_path / "bad.yaml"
        bad_file.write_text("unknown_key: boom\n")
        with pytest.raises(ValidationError):
            load_service_contract(bad_file)

    def test_load_raises_on_malformed_yaml(self, tmp_path: Path) -> None:
        bad_file = tmp_path / "malformed.yaml"
        bad_file.write_text("key: [unclosed bracket\n")
        with pytest.raises(ValueError, match="Invalid YAML"):
            load_service_contract(bad_file)
