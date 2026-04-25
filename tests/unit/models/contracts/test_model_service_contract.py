# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Unit tests for ModelFeatureFlagContract and load_service_contract loader."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from omnibase_infra.models.contracts.model_service_contract import (
    ModelFeatureFlagContract,
)
from omnibase_infra.runtime.service_contract_loader import load_service_contract


class TestModelFeatureFlagContract:
    """Tests for ModelFeatureFlagContract construction and validation."""

    def test_service_contract_round_trips(self) -> None:
        raw = {
            "name": "runtime",
            "description": "Runtime service contract",
            "feature_flags": {"enable_hot_reload": False, "enable_debug_mode": False},
        }
        m = ModelFeatureFlagContract.model_validate(raw)
        assert m.name == "runtime"
        assert m.feature_flags["enable_hot_reload"] is False

    def test_service_contract_defaults(self) -> None:
        m = ModelFeatureFlagContract.model_validate({"name": "my_service"})
        assert m.description == ""
        assert m.feature_flags == {}

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

    def test_feature_flags_must_be_bool_values(self) -> None:
        with pytest.raises(ValidationError):
            ModelFeatureFlagContract.model_validate(
                {"name": "svc", "feature_flags": {"flag_a": "not_a_bool"}}
            )


class TestLoadServiceContract:
    """Tests for the load_service_contract path-based loader."""

    def test_load_runtime_contract_yaml(self, tmp_path: pytest.TempPathFactory) -> None:
        contract_file = tmp_path / "runtime.contract.yaml"  # type: ignore[operator]
        contract_file.write_text(
            "name: runtime_service\n"
            "description: Feature flags for ONEX runtime services\n"
            "feature_flags:\n"
            "  ENABLE_RUNTIME_LOG_BRIDGE: false\n"
        )
        m = load_service_contract(contract_file)  # type: ignore[arg-type]
        assert m.name == "runtime_service"
        assert m.feature_flags["ENABLE_RUNTIME_LOG_BRIDGE"] is False

    def test_load_event_bus_contract_yaml(
        self, tmp_path: pytest.TempPathFactory
    ) -> None:
        contract_file = tmp_path / "event_bus.contract.yaml"  # type: ignore[operator]
        contract_file.write_text(
            "name: event_bus_service\n"
            "description: Feature flags for Kafka event bus infrastructure services\n"
            "feature_flags:\n"
            "  ENABLE_CONSUMER_HEALTH_EMITTER: false\n"
        )
        m = load_service_contract(contract_file)  # type: ignore[arg-type]
        assert m.name == "event_bus_service"
        assert m.feature_flags["ENABLE_CONSUMER_HEALTH_EMITTER"] is False

    def test_load_raises_on_missing_file(
        self, tmp_path: pytest.TempPathFactory
    ) -> None:
        missing = tmp_path / "nonexistent.yaml"  # type: ignore[operator]
        with pytest.raises(FileNotFoundError):
            load_service_contract(missing)  # type: ignore[arg-type]

    def test_load_raises_on_invalid_contract(
        self, tmp_path: pytest.TempPathFactory
    ) -> None:
        bad_file = tmp_path / "bad.yaml"  # type: ignore[operator]
        bad_file.write_text("unknown_key: boom\n")
        with pytest.raises(ValidationError):
            load_service_contract(bad_file)  # type: ignore[arg-type]
